"""Validator main loop — v2 GRPO market batching via HTTP endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    BLOCK_TIME_SECONDS,
    POLL_INTERVAL_SECONDS,
    UID_BURN,
    VALIDATOR_HTTP_PORT,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.protocol.submission import RolloutSubmission
from reliquary.validator.batcher_v2 import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer
from reliquary.validator.weights import compute_weights_v2

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH


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
        self._empty_batch_slots: int = 0
        self._windows_in_interval: int = 0
        self._cooldown_map = CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        self.server = ValidatorServer(host=http_host, port=http_port)

    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        logger.info(
            "Validator started: env=%s, netuid=%d, http=%s:%d, rolling_windows=%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
            ROLLING_WINDOWS,
        )
        try:
            while True:
                try:
                    current_block = await chain.get_current_block(subtensor)
                    target_window = self._compute_target_window(current_block)
                    if target_window <= self._last_processed_window:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    await self._run_window(subtensor, target_window)
                    self._last_processed_window = target_window
                    self._windows_in_interval += 1
                    if self._windows_in_interval >= ROLLING_WINDOWS:
                        submitted = await self._submit_weights(subtensor)
                        if submitted:
                            self._batched_hotkeys.clear()
                            self._empty_batch_slots = 0
                        else:
                            logger.warning(
                                "set_weights did not succeed — keeping %d "
                                "hotkey slots and %d empty slots for next cycle",
                                len(self._batched_hotkeys), self._empty_batch_slots,
                            )
                        self._windows_in_interval = 0
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Validation loop iteration failed")
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

    async def _run_window(self, subtensor, target_window: int) -> None:
        randomness = await self._derive_randomness(subtensor, target_window)
        # v2: current drand round derived from the beacon, not block hash.
        # For now use target_window as a placeholder logical round counter —
        # the real wiring to drand round is in a follow-up. GrpoWindowBatcher
        # uses this to validate signed_round freshness.
        current_round = target_window

        batcher = open_grpo_window(
            window_start=target_window,
            current_round=current_round,
            env=self.env,
            model=self.model,
            cooldown_map=self._cooldown_map,
            tokenizer=self.tokenizer,
        )
        batcher.randomness = randomness  # pass through for GRAIL sketch verification
        self.server.set_active_batcher(batcher)

        deadline = time.monotonic() + WINDOW_LENGTH * BLOCK_TIME_SECONDS
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(1)

            # Seal the batch at window close — this records cooldowns.
            batch = batcher.seal_batch()
            for sub in batch:
                self._batched_hotkeys.append(sub.hotkey)
            empty = B_BATCH - len(batch)
            self._empty_batch_slots += empty
            logger.info(
                "Window %d sealed: batch=%d/%d, empty=%d, valid_pool=%d",
                target_window, len(batch), B_BATCH, empty,
                len(batcher.valid_submissions()),
            )

            # Archive the valid submissions pool (not just the batch) for audit.
            archive = {
                "window_start": target_window,
                "randomness": randomness,
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
                "valid_submissions": [
                    {
                        "hotkey": s.hotkey,
                        "prompt_idx": s.prompt_idx,
                        "signed_round": s.signed_round,
                        "k": s.k,
                    }
                    for s in batcher.valid_submissions()
                ],
            }
            try:
                await storage.upload_window_dataset(
                    target_window,
                    archive,
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                )
            except Exception:
                logger.exception("Failed to upload window dataset")
        finally:
            self.server.set_active_batcher(None)

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
