"""Validator main loop — synchronous GRPO batching via HTTP endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from reliquary.constants import (
    BLOCK_TIME_SECONDS,
    POLL_INTERVAL_SECONDS,
    PROMPTS_PER_WINDOW,
    SLOT_DEADLINE_SECONDS,
    UID_BURN,
    VALIDATOR_HTTP_PORT,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from reliquary.environment.base import Environment
from reliquary.infrastructure import chain, storage
from reliquary.miner.prompts import derive_window_prompts
from reliquary.validator.batcher import ProblemSlot, WindowBatcher
from reliquary.validator.server import ValidatorServer
from reliquary.validator.weights import compute_weights

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH


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
        self._miner_scores: defaultdict[str, float] = defaultdict(float)
        self._burn_accumulated: float = 0.0
        self._windows_in_interval: int = 0

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
                        await self._submit_weights(subtensor)
                        self._miner_scores.clear()
                        self._burn_accumulated = 0.0
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
        problems = derive_window_prompts(self.env, randomness, PROMPTS_PER_WINDOW)
        slots = [
            ProblemSlot(slot_index=i, prompt_id=p["id"], problem=p)
            for i, p in enumerate(problems)
        ]
        batcher = WindowBatcher(
            window_start=target_window,
            slots=slots,
            randomness=randomness,
            env=self.env,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.server.set_active_batcher(batcher)

        deadline = time.monotonic() + WINDOW_LENGTH * BLOCK_TIME_SECONDS
        try:
            while time.monotonic() < deadline:
                # Trigger per-slot timeouts so the SLOT_DEADLINE_SECONDS cap
                # is actually enforced (slots freeze at 60s regardless of the
                # enclosing window deadline).
                batcher.finalize_due_slots()
                if batcher.is_window_complete():
                    logger.info("Window %d settled (all slots finalized)", target_window)
                    break
                await asyncio.sleep(1)

            # Safety net: force-finalize anything still open. Needed when the
            # window deadline fires before the SLOT deadline (small WINDOW_LENGTH)
            # or when the slot clock and service clock have drifted.
            batcher.finalize_due_slots(now=time.monotonic() + SLOT_DEADLINE_SECONDS)

            scores = batcher.get_miner_scores()
            burn = batcher.get_burn_score()
            for hk, s in scores.items():
                self._miner_scores[hk] += s
            self._burn_accumulated += burn
            logger.info(
                "Window %d scored: %d miners (total %.2f), burn %.2f",
                target_window, len(scores), sum(scores.values()), burn,
            )

            archive = batcher.get_archive_data()
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

    async def _submit_weights(self, subtensor) -> None:
        scores = dict(self._miner_scores)
        burn = self._burn_accumulated
        miner_weights, burn_weight = compute_weights(scores, burn_score=burn)
        non_zero_miners = {hk: w for hk, w in miner_weights.items() if w > 0}
        logger.info(
            "Submitting weights: %d miners + %.4f burn to UID %d "
            "(miner_total=%.2f, burn_score=%.2f)",
            len(non_zero_miners), burn_weight, UID_BURN,
            sum(scores.values()), burn,
        )
        for hk, w in sorted(non_zero_miners.items(), key=lambda x: -x[1])[:10]:
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
        if uids:
            await chain.set_weights(
                subtensor, self.wallet, self.netuid, uids, weight_vals,
            )

    @staticmethod
    def _compute_target_window(current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH
