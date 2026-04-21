"""CooldownMap: tracks last batch-membership window per prompt_idx.

A prompt that just entered a training batch is ineligible for the batch
for ``cooldown_windows`` following windows. This forces the curriculum
to rotate so the policy has time to shift between reuses of the same
prompt.
"""

from __future__ import annotations


class CooldownMap:
    """Per-prompt "last batched at window N" store + eligibility predicate.

    The cooldown window is a half-open interval:
        ``[last_batched, last_batched + cooldown_windows)`` → ineligible.
    At ``current_window == last_batched + cooldown_windows`` the prompt
    becomes eligible again.
    """

    def __init__(self, cooldown_windows: int) -> None:
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self._cooldown_windows = cooldown_windows
        self._last_batched: dict[int, int] = {}

    def record_batched(self, prompt_idx: int, window: int) -> None:
        """Mark *prompt_idx* as having entered the batch at *window*."""
        if prompt_idx < 0:
            raise ValueError("prompt_idx must be non-negative")
        if window < 0:
            raise ValueError("window must be non-negative")
        self._last_batched[prompt_idx] = window

    def is_in_cooldown(self, prompt_idx: int, current_window: int) -> bool:
        """True iff *prompt_idx* was batched within the cooldown horizon."""
        if self._cooldown_windows == 0:
            return False
        last = self._last_batched.get(prompt_idx)
        if last is None:
            return False
        return current_window - last < self._cooldown_windows

    def current_cooldown_set(self, current_window: int) -> set[int]:
        """All prompt_idx that are currently in cooldown."""
        if self._cooldown_windows == 0:
            return set()
        return {
            idx for idx, last in self._last_batched.items()
            if current_window - last < self._cooldown_windows
        }

    def __len__(self) -> int:
        return len(self._last_batched)
