"""Weight computation for Reliquary — flat 1/B v2 GRPO market formula."""


def compute_weights_v2(
    batch_hotkeys: list[str],
) -> tuple[dict[str, float], float]:
    """Flat 1/B payment per batch member; unused slots burn.

    Flat 1/B payment per batch member. Each of the (at most) B batch
    members receives exactly ``1 / B_BATCH`` of the window emission; the
    remainder ``(B - len(batch)) / B`` is routed to ``UID_BURN`` as a
    protocol-inefficiency signal.

    The flat share (not ``1/len(batch)``) is intentional:
      * signals shortfall via burn rate rather than masking it
      * removes the "lone survivor" incentive (miners can't profit by
        DoSing competitors — others' failures only grow the burn)

    Duplicate hotkeys in ``batch_hotkeys`` (possible if the same miner
    wins batch slots on two distinct prompts) are summed into one entry.

    Args:
        batch_hotkeys: the hotkeys of the batch members, in selection
            order. Length must be in ``[0, B_BATCH]``.

    Returns:
        (miner_weights, burn_weight) where:
          * miner_weights maps each unique hotkey to its share
          * burn_weight = (B - len(batch)) / B
          * miner_weights.values().sum() + burn_weight == 1.0
    """
    from reliquary.constants import B_BATCH

    n = len(batch_hotkeys)
    if n > B_BATCH:
        raise ValueError(
            f"batch size {n} exceeds B_BATCH={B_BATCH}; caller must cap"
        )

    per_slot = 1.0 / B_BATCH
    miner_weights: dict[str, float] = {}
    for hk in batch_hotkeys:
        miner_weights[hk] = miner_weights.get(hk, 0.0) + per_slot

    burn_weight = (B_BATCH - n) / B_BATCH
    return miner_weights, burn_weight
