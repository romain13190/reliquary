"""Weight computation for Reliquary — advantage-based + explicit UID_BURN burn share."""

from reliquary.constants import SUPERLINEAR_EXPONENT


def compute_weights(
    miner_scores: dict[str, float],
    burn_score: float = 0.0,
    superlinear_exponent: float = SUPERLINEAR_EXPONENT,
) -> tuple[dict[str, float], float]:
    """Normalise raw miner scores + unclaimed burn into on-chain weights.

    Economic model:
      * ``miner_scores`` holds per-hotkey cumulative |advantage| signal
        across the rolling window interval.
      * ``burn_score`` holds the window's unclaimed budget (notional budget
        minus Σ miner_scores) — signal that degenerate or imbalanced slots
        failed to emit and that we route to ``UID_BURN``.
      * Burn is treated LINEARLY (no superlinear) because it represents a
        protocol-level inefficiency share, not a competitive outcome.
      * Miner shares get the superlinear exponent applied INSIDE their
        remaining pool ``(1 − burn_fraction)`` — concentrating emission on
        the top information producers while preserving the burn cut.

    Args:
        miner_scores: {hotkey: cumulative advantage sum in this interval}
        burn_score:   unclaimed budget (≥ 0).
        superlinear_exponent: sybil-concentration exponent (default 4.0).
            Applied only to miners, not to burn.

    Returns:
        (miner_weights, burn_weight) where:
          * miner_weights sums to ``(1 − burn_weight)``
          * total (miners + burn) sums to 1.0 iff any signal exists, else
            every returned value is 0.0 (no emission to submit).
    """
    miner_linear_total = sum(max(0.0, s) for s in miner_scores.values())
    burn_linear = max(0.0, burn_score)
    total_linear = miner_linear_total + burn_linear

    if total_linear == 0.0:
        return {hk: 0.0 for hk in miner_scores}, 0.0

    burn_weight = burn_linear / total_linear
    miner_share = 1.0 - burn_weight

    raw = {hk: max(0.0, s) ** superlinear_exponent for hk, s in miner_scores.items()}
    raw_total = sum(raw.values())
    if raw_total == 0.0:
        # Only burn has signal — entire emission goes to UID_BURN.
        return {hk: 0.0 for hk in miner_scores}, burn_weight

    miner_weights = {hk: (r / raw_total) * miner_share for hk, r in raw.items()}
    return miner_weights, burn_weight


def compute_weights_v2(
    batch_hotkeys: list[str],
) -> tuple[dict[str, float], float]:
    """Flat 1/B payment per batch member; unused slots burn.

    v2 replacement for ``compute_weights``. Each of the (at most) B batch
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
