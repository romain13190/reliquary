"""Pure batch selection: FIFO by signed_round, distinct prompts, cooldown-aware.

Called once per window to pick the B submissions that go into the training
step. Separated from the orchestrator (``WindowBatcher``) to make the
selection logic trivially testable in isolation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from reliquary.validator.cooldown import CooldownMap


class _SubmissionLike(Protocol):
    """Duck-typed submission — works with any class exposing these attrs."""

    hotkey: str
    prompt_idx: int
    signed_round: int
    merkle_root: bytes


def _tiebreak_key(sub: _SubmissionLike) -> bytes:
    """Deterministic, consensus-safe tiebreak within a drand round.

    Hash of (hotkey, prompt_idx, merkle_root) — miner-controlled but stable
    across validators. Any validator seeing the same set of submissions
    produces the same ordering.
    """
    h = hashlib.sha256()
    h.update(sub.hotkey.encode())
    h.update(sub.prompt_idx.to_bytes(8, "big", signed=False))
    h.update(sub.merkle_root)
    return h.digest()


def select_batch(
    submissions: list[Any],
    *,
    b: int,
    current_window: int,
    cooldown_map: CooldownMap,
) -> list[Any]:
    """Return the ordered list of at most *b* batch members.

    Rules:
        1. Sort by ``(signed_round, tiebreak_hash)`` — deterministic FIFO.
        2. For each submission in order:
           - skip if its ``prompt_idx`` is already represented in the batch
             (diversity constraint — one prompt per batch slot)
           - skip if its ``prompt_idx`` is in cooldown per ``cooldown_map``
           - otherwise append; stop when ``len(batch) == b``
        3. Does NOT mutate ``cooldown_map`` — the caller records
           post-selection (via ``record_batched``) once the batch is final.
    """
    if b <= 0:
        return []

    ordered = sorted(submissions, key=lambda s: (s.signed_round, _tiebreak_key(s)))

    batch: list[Any] = []
    seen_prompts: set[int] = set()
    for sub in ordered:
        if len(batch) >= b:
            break
        if sub.prompt_idx in seen_prompts:
            continue
        if cooldown_map.is_in_cooldown(sub.prompt_idx, current_window):
            continue
        batch.append(sub)
        seen_prompts.add(sub.prompt_idx)
    return batch
