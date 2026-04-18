"""Bittensor chain interactions for Reliquary."""

import asyncio
import hashlib
import logging
import os
from typing import Any

from reliquary.constants import BLOCK_TIME_SECONDS

logger = logging.getLogger(__name__)

NETUID = int(os.getenv("NETUID", "81"))
NETWORK = os.getenv("BT_NETWORK", "finney")


async def get_subtensor():
    """Create async subtensor."""
    import bittensor as bt

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


async def get_metagraph(subtensor, netuid: int = NETUID):
    """Get subnet metagraph."""
    return await subtensor.metagraph(netuid)


async def get_block_hash(subtensor, block_number: int) -> str:
    """Get block hash for a given block number."""
    return await subtensor.get_block_hash(block_number)


async def get_current_block(subtensor) -> int:
    """Get current block number."""
    return await subtensor.get_current_block()


async def set_weights(
    subtensor,
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
) -> bool:
    """Submit weights on-chain.

    Returns True only when the chain reports success. A rejected extrinsic
    (no validator permit, rate limit, bad weights, etc.) does NOT raise — it
    comes back as ``ExtrinsicResponse(success=False, message=...)`` — so we
    inspect the response explicitly instead of trusting "no exception ==
    success".

    bittensor's own ``set_weights`` already retries transient failures up to
    ``max_attempts=5`` times across consecutive blocks, so external retry
    wrapping here would be redundant.
    """
    try:
        response = await subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
        )
    except Exception as e:
        logger.error("set_weights raised: %s", e, exc_info=True)
        return False

    success = bool(getattr(response, "success", False))
    if success:
        logger.info("set_weights OK — %d uids committed", len(uids))
        return True
    msg = getattr(response, "message", None) or "<no message>"
    err = getattr(response, "error", None)
    logger.error(
        "set_weights rejected by chain: %s (error=%r)", msg, err,
    )
    return False


def compute_drand_round_for_window(
    window_start_block: int, genesis_time: int, period: int
) -> int:
    """Deterministically compute which drand round to use for a window.

    Both miner and validator call this with the same inputs to agree
    on a single round — no "latest" fetch needed.

    Returns:
        The drand round number (>= 1).
    """
    window_timestamp = window_start_block * BLOCK_TIME_SECONDS
    if window_timestamp < genesis_time:
        return 1  # Clamp: can't go before round 1
    return 1 + (window_timestamp - genesis_time) // period


def compute_window_randomness(
    block_hash: str,
    drand_randomness: str | None = None,
    drand_round: int | None = None,
) -> str:
    """Combine block hash, drand randomness, and round into window randomness.

    Including the round number prevents a miner from choosing a round
    whose randomness is favorable.
    """
    clean_hash = block_hash.replace("0x", "")
    if drand_randomness:
        material = bytes.fromhex(clean_hash) + bytes.fromhex(drand_randomness)
        if drand_round is not None:
            material += drand_round.to_bytes(8, "big")
        combined = hashlib.sha256(material).hexdigest()
        return combined
    return clean_hash
