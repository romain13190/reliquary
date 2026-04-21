"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BLOCK_TIME_SECONDS,
    BOOTSTRAP_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    CHECKPOINT_STATE_PATH_DEFAULT,
    POLL_INTERVAL_SECONDS,
    SUBNET_START_BLOCK,
    UID_BURN,
    VALIDATOR_HTTP_PORT,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
    WINDOW_TIMEOUT_SECONDS,
)
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RolloutSubmission, WindowState
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.checkpoint import CheckpointStore
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer
from reliquary.validator.state_persistence import ValidatorState
from reliquary.validator.training import train_step
from reliquary.validator.weights import compute_weights_v2

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH


def is_bootstrap_window(window_start: int, subnet_start: int) -> bool:
    """True iff *window_start* is within ``BOOTSTRAP_WINDOWS`` of ``subnet_start``.

    Bootstrap windows use the relaxed zone / cooldown / M values so the
    batch can fill while miner population and env coverage are thin.
    """
    if window_start < subnet_start:
        return False
    return window_start - subnet_start < BOOTSTRAP_WINDOWS


def open_grpo_window(
    window_start: int,
    current_round: int,
    env,
    model,
    *,
    cooldown_map: CooldownMap,
    tokenizer,
    bootstrap: bool = False,
) -> GrpoWindowBatcher:
    """Instantiate a GrpoWindowBatcher for this window.

    ``cooldown_map`` is the validator's long-lived CooldownMap, shared
    across windows. Each window's sealed batch updates it via
    ``GrpoWindowBatcher.seal_batch``.
    """
    def _completion_text(rollout: RolloutSubmission) -> str:
        prompt_len = rollout.commit.get("rollout", {}).get("prompt_length", 0)
        return tokenizer.decode(rollout.tokens[prompt_len:])

    return GrpoWindowBatcher(
        window_start=window_start,
        current_round=current_round,
        env=env,
        model=model,
        cooldown_map=cooldown_map,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
    )


def compute_weights_for_window(
    batch: list[ValidSubmission],
) -> tuple[dict[str, float], float]:
    """Collapse a sealed batch into (miner_weights, burn_weight) via v2 flat formula."""
    return compute_weights_v2(batch_hotkeys=[sub.hotkey for sub in batch])


class ValidationService:
    def __init__(
        self,
        wallet,
        model,
        tokenizer,
        env: Environment,
        netuid: int,
        *,
        use_drand: bool = True,
        http_host: str = "0.0.0.0",
        http_port: int = VALIDATOR_HTTP_PORT,
        external_ip: str | None = None,
        external_port: int | None = None,
    ) -> None:
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.netuid = netuid
        self.use_drand = use_drand
        self.external_ip = external_ip
        self.external_port = external_port

        self._last_processed_window: int = -1
        self._batched_hotkeys: list[str] = []
        self._windows_in_interval: int = 0
        self._cooldown_map = CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        self.server = ValidatorServer(host=http_host, port=http_port)

        # v2.1 state machine infrastructure
        self._state_path = CHECKPOINT_STATE_PATH_DEFAULT
        self._state = ValidatorState(self._state_path)
        self._state.load()
        self._checkpoint_store = CheckpointStore(
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
            staging_dir_path=CHECKPOINT_STAGING_DIR_DEFAULT,
        )
        self._active_batcher = None
        self._current_window_state: WindowState = WindowState.READY

    def _set_state(self, s: WindowState) -> None:
        self._current_window_state = s
        # Also notify the server so /window/state returns the right value.
        try:
            self.server.set_current_state(s)
        except AttributeError:
            # Task 9 adds this method; be tolerant during development.
            pass

    def _open_window(self) -> None:
        """Create a new GrpoWindowBatcher and mark state OPEN.

        Increments ``self._state.window_n`` and wires the active
        checkpoint hash into the batcher so miners on stale checkpoints
        get WRONG_CHECKPOINT rejected before GRAIL compute.
        """
        self._state.window_n += 1
        bootstrap = is_bootstrap_window(
            window_start=self._state.window_n,
            subnet_start=SUBNET_START_BLOCK,
        )
        self._active_batcher = open_grpo_window(
            window_start=self._state.window_n,
            current_round=self._state.window_n,  # placeholder; drand wiring later
            env=self.env, model=self.model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
        cp = self._checkpoint_store.current_manifest()
        self._active_batcher.current_checkpoint_hash = (
            cp.file_hash if cp else ""
        )
        self.server.set_active_batcher(self._active_batcher)
        self._set_state(WindowState.OPEN)

    async def _train_and_publish(self) -> None:
        """TRAINING + PUBLISHING + READY phases."""
        if self._active_batcher is None:
            logger.warning("_train_and_publish called with no active batcher")
            return

        self._set_state(WindowState.TRAINING)
        batch = self._active_batcher.seal_batch()
        for sub in batch:
            self._batched_hotkeys.append(sub.hotkey)

        self.model = train_step(self.model, batch)

        self._set_state(WindowState.PUBLISHING)
        new_n = self._state.checkpoint_n + 1
        try:
            entry = await self._checkpoint_store.publish(
                checkpoint_n=new_n, model=self.model,
            )
            self._state.checkpoint_n = new_n
            self._state.save()
            # Notify server of new manifest (Task 9 adds this method).
            try:
                self.server.set_current_checkpoint(entry)
            except AttributeError:
                pass
        except Exception:
            logger.exception("checkpoint publish failed; staying on previous")

        try:
            await self._archive_window(self._active_batcher, batch)
        except Exception:
            logger.exception("window archive failed")

        self.server.set_active_batcher(None)
        self._active_batcher = None
        self._set_state(WindowState.READY)

    async def _archive_window(self, batcher, batch) -> None:
        archive = {
            "window_start": batcher.window_start,
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": [
                {
                    "hotkey": s.hotkey,
                    "prompt_idx": s.prompt_idx,
                    "signed_round": s.signed_round,
                    "k": s.k,
                }
                for s in batch
            ],
        }
        await storage.upload_window_dataset(
            batcher.window_start, archive,
            validator_hotkey=self.wallet.hotkey.ss58_address,
        )

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._rebuild_cooldown_from_history(subtensor)
        logger.info(
            "Validator started (v2.1): env=%s, netuid=%d, http=%s:%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
        )
        try:
            while True:
                try:
                    self._open_window()
                    try:
                        await asyncio.wait_for(
                            self._active_batcher.seal_event.wait(),
                            timeout=WINDOW_TIMEOUT_SECONDS,
                        )
                        logger.info(
                            "Window %d sealed (B valid received)",
                            self._state.window_n,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Window %d timed out at %ds — sealing partial",
                            self._state.window_n, WINDOW_TIMEOUT_SECONDS,
                        )

                    await self._train_and_publish()

                    # Weight cadence: every ROLLING_WINDOWS sealed windows.
                    if self._state.window_n % ROLLING_WINDOWS == 0:
                        submitted = await self._submit_weights(subtensor)
                        if submitted:
                            self._batched_hotkeys.clear()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Window iteration failed")
                    # Reset to READY so the next iteration doesn't spin on error state.
                    self.server.set_active_batcher(None)
                    self._active_batcher = None
                    self._set_state(WindowState.READY)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            await self.server.stop()

    async def _serve_axon_on_chain(self, subtensor) -> None:
        """Publish this validator's axon (ip:port) to the chain metagraph.

        Miners read `metagraph.axons[uid].ip/port` via `discover_validator_url`
        to route their submissions. Skipped with a warning when no external
        address is configured — miners then need `--validator-url` overrides
        to find this validator.
        """
        if not self.external_ip or not self.external_port:
            logger.warning(
                "serve_axon skipped: no external_ip/external_port provided. "
                "Miners won't discover this validator via metagraph; use "
                "--validator-url on the miner side."
            )
            return
        try:
            import bittensor as bt

            axon = bt.Axon(
                wallet=self.wallet,
                ip=self.external_ip,
                port=self.external_port,
                external_ip=self.external_ip,
                external_port=self.external_port,
            )
            response = await subtensor.serve_axon(
                netuid=self.netuid,
                axon=axon,
                wait_for_inclusion=True,
                wait_for_finalization=False,
                raise_error=False,
            )
            success = getattr(response, "is_success", None)
            if success is False:
                logger.error(
                    "serve_axon failed: %s:%d not published (response=%s). "
                    "Likely: hotkey not registered on netuid %d, or chain rejected.",
                    self.external_ip, self.external_port, response, self.netuid,
                )
                return
            logger.info(
                "serve_axon published: %s:%d announced on netuid %d",
                self.external_ip, self.external_port, self.netuid,
            )
        except Exception:
            logger.exception(
                "serve_axon threw — miners will have to use --validator-url"
            )

    async def _rebuild_cooldown_from_history(self, subtensor) -> None:
        """At startup, reconstruct CooldownMap from the last
        BATCH_PROMPT_COOLDOWN_WINDOWS archived windows on R2.

        R2 is the durable source of truth — each window's sealed batch is
        uploaded by ``_archive_window``. Rebuilding from that history means:
          * local disk state isn't needed (no JSON file to manage)
          * multi-validator consistency: any validator rebuilding from the
            same R2 prefix converges to the same cooldown map
          * a fresh validator joining an active subnet picks up the
            current cooldown state without coordination
        """
        try:
            current_block = await chain.get_current_block(subtensor)
            current_window = self._compute_target_window(current_block)
            archives = await storage.list_recent_datasets(
                hotkey=self.wallet.hotkey.ss58_address,
                current_window=current_window,
                n=BATCH_PROMPT_COOLDOWN_WINDOWS,
            )
            self._cooldown_map.rebuild_from_history(
                archives, current_window=current_window,
            )
            logger.info(
                "Rebuilt cooldown from %d archive windows (current=%d, map size=%d)",
                len(archives), current_window, len(self._cooldown_map),
            )
        except Exception:
            logger.exception(
                "Failed to rebuild cooldown from history; starting with empty state"
            )

    async def _derive_randomness(self, subtensor, target_window: int) -> str:
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain
            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                target_window, chain_info["genesis_time"], chain_info["period"],
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"],
            )
        return chain.compute_window_randomness(block_hash)

    async def _submit_weights(self, subtensor) -> bool:
        """Submit flat 1/B weights + burn."""
        # Aggregate batch members across the interval.
        hotkey_counts: dict[str, int] = defaultdict(int)
        for hk in self._batched_hotkeys:
            hotkey_counts[hk] += 1

        # Each slot is worth 1/(ROLLING_WINDOWS * B_BATCH) of the total
        # emission over the interval. Equivalently, since miner_weights
        # from compute_weights_v2 sums to len(batch)/B per window and we
        # have ROLLING_WINDOWS windows, the normalized per-miner share is
        # count * 1/(ROLLING_WINDOWS * B_BATCH).
        total_slots = ROLLING_WINDOWS * B_BATCH
        if total_slots == 0:
            return True

        miner_weights = {
            hk: count / total_slots for hk, count in hotkey_counts.items()
        }
        used = len(self._batched_hotkeys)
        burn_weight = (total_slots - used) / total_slots

        logger.info(
            "Submitting v2 weights: %d miners, burn=%.4f (used=%d/%d slots)",
            len(miner_weights), burn_weight, used, total_slots,
        )
        for hk, w in sorted(miner_weights.items(), key=lambda x: -x[1])[:10]:
            logger.info("  %s: %.6f", hk[:8], w)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
        uids: list[int] = []
        weight_vals: list[float] = []
        for hk, w in miner_weights.items():
            if hk in hotkey_to_uid and w > 0:
                uids.append(int(hotkey_to_uid[hk]))
                weight_vals.append(w)
        if burn_weight > 0:
            uids.append(UID_BURN)
            weight_vals.append(burn_weight)
        if not uids:
            logger.info("No non-zero weights to submit; nothing to do.")
            return True
        return await chain.set_weights(
            subtensor, self.wallet, self.netuid, uids, weight_vals,
        )

    @staticmethod
    def _compute_target_window(current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH

    # _run_window is superseded by _open_window + _train_and_publish + _archive_window.
    # Kept as dead code for Task 13 cleanup.
    async def _run_window(self, subtensor, target_window: int) -> None:
        pass
