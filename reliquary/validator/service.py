"""Validator main loop — v2.1 batch-driven state machine (OPEN→TRAINING→PUBLISHING→READY)."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BOOTSTRAP_WINDOWS,
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    CHECKPOINT_STAGING_DIR_DEFAULT,
    DEFAULT_HF_REPO_ID,
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
from reliquary.validator.training import train_step

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
        hf_repo_id: str | None = None,
    ) -> None:
        self.wallet = wallet
        self.model = model
        # Enable gradient checkpointing to reduce activation memory.
        # Harmless if already enabled or unsupported by the model.
        try:
            self.model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            logger.warning(
                "model does not support gradient_checkpointing_enable"
            )
        self.tokenizer = tokenizer
        self.env = env
        self.netuid = netuid
        self.use_drand = use_drand
        self.external_ip = external_ip
        self.external_port = external_port

        self._last_processed_window: int = -1
        self._windows_in_interval: int = 0
        self._cooldown_map = CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        self.server = ValidatorServer(host=http_host, port=http_port)

        # v2.1 state machine infrastructure — in-memory only, bootstrapped at
        # startup from R2 + HF (no local JSON state file).
        self._window_n: int = 0
        self._checkpoint_n: int = 0
        self._miner_scores_ema: defaultdict[str, float] = defaultdict(float)
        self._publish_every = CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        self._checkpoint_store = CheckpointStore(
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
            repo_id=hf_repo_id or DEFAULT_HF_REPO_ID,
            staging_dir_path=CHECKPOINT_STAGING_DIR_DEFAULT,
        )
        self._active_batcher = None
        self._current_window_state: WindowState = WindowState.READY

    def _update_ema(self, batch: list) -> None:
        """EMA update on the per-hotkey miner score dict.

        Applied to every hotkey we've ever seen — absent miners this window
        contribute 0 → their EMA decays by factor (1 - α). Active miners
        blend in their fresh contribution.

        The sum across all hotkeys converges to the smoothed fill rate of
        the batch — burn is derived as the complement.
        """
        from reliquary.constants import EMA_ALPHA
        alpha = EMA_ALPHA

        window_contribs: dict[str, int] = defaultdict(int)
        for sub in batch:
            window_contribs[sub.hotkey] += 1

        all_hotkeys = set(self._miner_scores_ema) | set(window_contribs)
        for hk in all_hotkeys:
            fraction = window_contribs.get(hk, 0) / B_BATCH
            old = self._miner_scores_ema[hk]
            self._miner_scores_ema[hk] = alpha * fraction + (1 - alpha) * old

        # Prune near-zero entries to bound the dict size.
        self._miner_scores_ema = defaultdict(float, {
            hk: v for hk, v in self._miner_scores_ema.items() if v > 1e-6
        })

    def _set_state(self, s: WindowState) -> None:
        self._current_window_state = s
        # Also notify the server so /state returns the right value.
        try:
            self.server.set_current_state(s)
        except AttributeError:
            # Task 9 adds this method; be tolerant during development.
            pass

    def _open_window(self) -> None:
        """Create a new GrpoWindowBatcher and mark state OPEN.

        Increments ``self._window_n`` and wires the active checkpoint hash
        into the batcher so miners on stale checkpoints get WRONG_CHECKPOINT
        rejected before GRAIL compute.
        """
        self._window_n += 1
        bootstrap = is_bootstrap_window(
            window_start=self._window_n,
            subnet_start=SUBNET_START_BLOCK,
        )
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            current_round=self._window_n,  # placeholder; drand wiring later
            env=self.env, model=self.model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
        cp = self._checkpoint_store.current_manifest()
        self._active_batcher.current_checkpoint_hash = (
            cp.revision if cp else ""
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
        self._update_ema(batch)

        self.model = train_step(self.model, batch)

        self._set_state(WindowState.PUBLISHING)
        # checkpoint_n only advances on publish; use window_n for cadence.
        next_n = self._checkpoint_n + 1
        # Push to HF every N windows, or immediately if no checkpoint exists yet.
        should_publish = (
            (self._window_n % self._publish_every == 0)
            or self._checkpoint_store.current_manifest() is None
        )
        if should_publish:
            try:
                entry = await self._checkpoint_store.publish(
                    checkpoint_n=next_n, model=self.model,
                )
                self._checkpoint_n = next_n
                try:
                    self.server.set_current_checkpoint(entry)
                except AttributeError:
                    pass
                logger.info(
                    "Published checkpoint %d to %s@%s",
                    entry.checkpoint_n, entry.repo_id, entry.revision[:12],
                )
            except Exception:
                logger.exception("HF publish failed; staying on previous checkpoint")
        else:
            logger.info(
                "Skipping HF publish for window_n=%d (publishing every %d)",
                self._window_n, self._publish_every,
            )

        try:
            await self._archive_window(self._active_batcher, batch)
        except Exception:
            logger.exception("window archive failed")

        self.server.set_active_batcher(None)
        self._active_batcher = None
        self._set_state(WindowState.READY)

    async def _archive_window(self, batcher, batch) -> None:
        batch_entries = []
        for s in batch:
            problem = self.env.get_problem(s.prompt_idx)
            rollouts_payload = [
                {
                    "tokens": r.tokens,
                    "completion_text": text,
                    "reward": r.reward,
                }
                for r, text in zip(s.rollouts, s.completion_texts)
            ]
            batch_entries.append({
                "hotkey": s.hotkey,
                "prompt_idx": s.prompt_idx,
                "signed_round": s.signed_round,
                "sigma": s.sigma,
                "prompt": problem.get("prompt", ""),
                "ground_truth": problem.get("ground_truth", ""),
                "rollouts": rollouts_payload,
            })

        archive = {
            "window_start": batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,  # provenance
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": batch_entries,
        }
        await storage.upload_window_dataset(batcher.window_start, archive)

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._bootstrap_state_from_external()
        await self._rebuild_cooldown_from_history()
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
                            self._window_n,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Window %d timed out at %ds — sealing partial",
                            self._window_n, WINDOW_TIMEOUT_SECONDS,
                        )

                    await self._train_and_publish()

                    # Weight cadence: every ROLLING_WINDOWS sealed windows.
                    if self._window_n % ROLLING_WINDOWS == 0:
                        submitted = await self._submit_weights(subtensor)
                        if not submitted:
                            logger.warning(
                                "set_weights did not succeed — EMA unchanged for next retry"
                            )
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

    async def _bootstrap_state_from_external(self) -> None:
        """Derive window_n, checkpoint_n, and miner_scores_ema from R2 + HF.

        Called once at startup before the main loop. Zero local state required.
        """
        # 1. window_n from R2 archive keys
        try:
            windows = await storage.list_all_window_keys()
            if windows:
                self._window_n = max(windows)
                logger.info("Bootstrapped window_n=%d from R2", self._window_n)
            else:
                logger.info("No archives in R2 — starting from window_n=0")
        except Exception:
            logger.exception("Failed to bootstrap window_n from R2; starting at 0")

        # 2. checkpoint_n from HF commit history
        try:
            from huggingface_hub import HfApi
            repo_id = self._checkpoint_store.repo_id
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id)
            ckpt_count = sum(1 for c in commits if c.title.startswith("checkpoint "))
            self._checkpoint_n = ckpt_count
            logger.info("Bootstrapped checkpoint_n=%d from HF", self._checkpoint_n)
        except Exception:
            logger.exception("Failed to bootstrap checkpoint_n from HF; starting at 0")

        # 3. miner_scores_ema by replaying last K archives
        try:
            archives = await storage.list_recent_datasets(
                current_window=self._window_n + 1,
                n=ROLLING_WINDOWS * 3,  # enough for EMA half-life to converge
            )
            self._miner_scores_ema = self._replay_ema(archives)
            logger.info(
                "Bootstrapped EMA from %d archives, %d hotkeys tracked",
                len(archives), len(self._miner_scores_ema),
            )
        except Exception:
            logger.exception("Failed to bootstrap EMA from R2; starting empty")

    def _replay_ema(self, archives: list[dict]) -> "defaultdict[str, float]":
        """Deterministically derive EMA state from a list of archives."""
        from reliquary.constants import EMA_ALPHA
        ema: defaultdict[str, float] = defaultdict(float)
        alpha = EMA_ALPHA
        # Sort ascending by window_start to replay in order
        for record in sorted(archives, key=lambda r: int(r["window_start"])):
            window_contribs: dict[str, int] = defaultdict(int)
            for entry in record.get("batch", []):
                window_contribs[entry["hotkey"]] += 1
            all_hotkeys = set(ema) | set(window_contribs)
            for hk in all_hotkeys:
                fraction = window_contribs.get(hk, 0) / B_BATCH
                old = ema[hk]
                ema[hk] = alpha * fraction + (1 - alpha) * old
            ema = defaultdict(float, {hk: v for hk, v in ema.items() if v > 1e-6})
        return ema

    async def _rebuild_cooldown_from_history(self) -> None:
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
            current_window = self._window_n
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1,
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
        """Submit weights from the current EMA snapshot. EMA is NOT cleared
        on success — it persists across submits so that any submit within
        an epoch carries the full rolling view of miner contributions.
        """
        miner_weights = dict(self._miner_scores_ema)
        total = sum(miner_weights.values())
        burn_weight = max(0.0, 1.0 - total)

        logger.info(
            "Submitting weights: %d miners (ema_total=%.4f), burn=%.4f",
            len(miner_weights), total, burn_weight,
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

