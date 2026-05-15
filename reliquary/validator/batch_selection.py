"""Drand-round-anchored batch selection + emission distribution (design A').

Called once per window at seal time. Two things happen here:

1. **Pick the B distinct prompts that go into the training step.** Order is
   driven by the drand round each miner attached at submit time. Submissions
   are bucketed into 3-second slots (one drand quicknet round per slot);
   earlier slots fill the batch first. Within a slot, distinct prompts are
   admitted in a canonical hash order so two validators with different
   in-memory orderings still agree.

2. **Compute the emission distribution.** Each filled slot is worth
   ``pool / B``. Within a slot, the K miners who submitted for that
   ``(round, prompt)`` split that share equally. So a miner sybiling the
   same prompt with N hotkeys in the same drand round earns N × (1/N) =
   identical total payout to a single hotkey — registration costs make
   sybiling strictly wasteful.

Unfilled slots burn their share (no redistribution). This keeps emission
math simple and stable; partial fills happen when miner activity is low,
and the burn appears in the on-chain weight setter as missing weight for
that window.

v2.3+: replaces the v2.2 pure-FIFO ``select_batch``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from reliquary.validator.cooldown import CooldownMap


class _SubmissionLike(Protocol):
    """Duck-typed submission — works with any class exposing these attrs."""

    hotkey: str
    prompt_idx: int
    merkle_root: bytes
    drand_round: int


def _within_slot_key(sub: _SubmissionLike) -> bytes:
    """Canonical sort key for tie-breaking within a (round, prompt) bucket.

    Two validators may receive submissions in different network order; both
    sort by this key to agree on which submission represents the slot when
    multiple prompts compete inside the same drand round.
    """
    h = hashlib.sha256()
    h.update(sub.hotkey.encode())
    h.update(sub.prompt_idx.to_bytes(8, "big", signed=False))
    h.update(sub.merkle_root)
    return h.digest()


def _prompt_canonical_key(prompt_idx: int) -> bytes:
    """Canonical order for prompts seen in the same drand round."""
    return hashlib.sha256(prompt_idx.to_bytes(8, "big", signed=False)).digest()


def select_batch_and_distribute(
    submissions: list[Any],
    *,
    b: int,
    cooldown_map: CooldownMap,
    current_window: int,
    pool: float = 1.0,
) -> tuple[list[Any], dict[str, float]]:
    """Pick the training batch and the reward distribution.

    Args:
        submissions: all GRAIL-validated submissions for the window, in
            arbitrary order. Each carries an attached ``drand_round`` that
            was already validated at accept time.
        b: training batch size (= B_BATCH). Equals the number of slots in
            the emission pool.
        cooldown_map: read-only view of which prompts are in cooldown.
            Prompts in cooldown are skipped entirely.
        current_window: needed to evaluate cooldown.
        pool: total emission budget for the window. Each filled slot pays
            ``pool / b`` (NOT ``pool / filled_slots`` — under-filled
            windows burn unused share rather than redistributing).

    Returns:
        (training_batch, rewards_by_hotkey)
        - training_batch: up to ``b`` submissions, one per filled slot. The
          canonical representative for each ``(round, prompt)`` slot is the
          submission with the smallest ``_within_slot_key`` (= deterministic
          across validators).
        - rewards_by_hotkey: dict hotkey → float. Sums to (slots_filled / b)
          × pool, with unfilled slots burned.

    Does NOT mutate ``cooldown_map`` — the caller records post-selection.
    """
    if b <= 0 or not submissions:
        return [], {}

    # Group: drand_round → prompt_idx → list[submission]
    by_round: dict[int, dict[int, list[Any]]] = {}
    for sub in submissions:
        if cooldown_map.is_in_cooldown(sub.prompt_idx, current_window):
            continue
        prompts = by_round.setdefault(sub.drand_round, {})
        prompts.setdefault(sub.prompt_idx, []).append(sub)
    if not by_round:
        return [], {}

    slot_share = pool / b
    training_batch: list[Any] = []
    rewards: dict[str, float] = {}
    claimed_prompts: set[int] = set()
    rounds_ascending = sorted(by_round)
    for round_no in rounds_ascending:
        if len(training_batch) >= b:
            break
        prompts_in_round = by_round[round_no]
        # Order prompts within the same round canonically — two validators
        # with different network-arrival orders still pick the same prompts.
        prompts_ordered = sorted(
            prompts_in_round, key=_prompt_canonical_key,
        )
        for prompt_idx in prompts_ordered:
            if len(training_batch) >= b:
                break
            if prompt_idx in claimed_prompts:
                continue
            slot_subs = prompts_in_round[prompt_idx]
            claimed_prompts.add(prompt_idx)
            k_p = len(slot_subs)
            per_miner = slot_share / k_p
            for sub in slot_subs:
                rewards[sub.hotkey] = rewards.get(sub.hotkey, 0.0) + per_miner
            # Canonical pick for the training step.
            representative = min(slot_subs, key=_within_slot_key)
            training_batch.append(representative)

    return training_batch, rewards
