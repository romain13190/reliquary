"""Weight-only validator — reads R2 archives, submits weights on-chain.

No model, no HTTP server, no HF writes. Meant to be run alongside a
trainer validator; any number of weight-only nodes can participate and
all submit consistent weights because they read the same R2 archives.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from reliquary.constants import (
    B_BATCH,
    EMA_ALPHA,
    POLL_INTERVAL_SECONDS,
    UID_BURN,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain, storage

# ROLLING_WINDOWS mirrors the definition in service.py — kept local to avoid
# circular imports. Both must stay in sync with WEIGHT_SUBMISSION_INTERVAL.
ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH

logger = logging.getLogger(__name__)


class WeightOnlyValidator:
    """Lightweight validator that only sets weights.

    Every ``WEIGHT_SUBMISSION_INTERVAL`` blocks:
      1. Read last K archives from R2
      2. Replay EMA update
      3. Submit weights on-chain via chain.set_weights

    No local state: every submit recomputes from scratch.
    """

    def __init__(self, wallet, netuid: int) -> None:
        self.wallet = wallet
        self.netuid = netuid
        self._last_submit_block: int = 0

    async def run(self, subtensor) -> None:
        logger.info(
            "Weight-only validator started (netuid=%d, hotkey=%s)",
            self.netuid, self.wallet.hotkey.ss58_address,
        )
        while True:
            try:
                current_block = await chain.get_current_block(subtensor)
                if current_block - self._last_submit_block < WEIGHT_SUBMISSION_INTERVAL:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Read + replay + submit
                windows = await storage.list_all_window_keys()
                if not windows:
                    logger.info("No archives yet; nothing to submit")
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                archives = await storage.list_recent_datasets(
                    current_window=max(windows) + 1,
                    n=ROLLING_WINDOWS * 3,
                )
                ema = self._replay_ema(archives)
                miner_weights = dict(ema)
                total = sum(miner_weights.values())
                burn_weight = max(0.0, 1.0 - total)

                submitted = await self._submit_weights(
                    subtensor, miner_weights, burn_weight,
                )
                if submitted:
                    self._last_submit_block = current_block
                    logger.info(
                        "Submitted weights: %d miners (total=%.4f), burn=%.4f",
                        len(miner_weights), total, burn_weight,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("weight-only loop iteration failed")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    @staticmethod
    def _replay_ema(archives: list[dict]) -> dict[str, float]:
        ema: dict[str, float] = {}
        alpha = EMA_ALPHA
        for record in sorted(archives, key=lambda r: int(r["window_start"])):
            window_contribs: dict[str, int] = defaultdict(int)
            for entry in record.get("batch", []):
                window_contribs[entry["hotkey"]] += 1
            all_hotkeys = set(ema) | set(window_contribs)
            for hk in all_hotkeys:
                fraction = window_contribs.get(hk, 0) / B_BATCH
                ema[hk] = alpha * fraction + (1 - alpha) * ema.get(hk, 0.0)
            ema = {hk: v for hk, v in ema.items() if v > 1e-6}
        return ema

    async def _submit_weights(
        self, subtensor, miner_weights: dict[str, float], burn_weight: float,
    ) -> bool:
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
