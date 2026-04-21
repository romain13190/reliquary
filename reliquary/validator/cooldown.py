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

    # ---------- persistence ----------

    def save(self, path) -> None:
        """Serialise to JSON at *path*. Atomic via tmp-file + rename."""
        import json
        import os
        import tempfile

        path = str(path)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".cooldown.", dir=os.path.dirname(path) or "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "cooldown_windows": self._cooldown_windows,
                        "last_batched": self._last_batched,
                    },
                    f,
                )
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load(self, path) -> None:
        """Load state from JSON at *path*. No-op if file doesn't exist."""
        import json
        import os

        path = str(path)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        # JSON object keys are strings — coerce back to int.
        self._last_batched = {int(k): int(v) for k, v in data["last_batched"].items()}

    # ---------- rebuild from archived window data ----------

    def rebuild_from_history(
        self,
        archived_windows: list[dict],
        current_window: int,
    ) -> None:
        """Rebuild state from the last N archived windows' batch records.

        *archived_windows* is a list of dicts, each with ``window_start``
        (int) and ``batch`` (list of {prompt_idx: int, ...}). Typically
        fetched from the R2 dataset archive at validator startup.

        Only windows within ``cooldown_windows`` of *current_window* matter —
        older entries have already expired and are skipped.
        """
        self._last_batched.clear()
        horizon = current_window - self._cooldown_windows
        for record in archived_windows:
            w = int(record["window_start"])
            if w <= horizon:
                continue
            for entry in record.get("batch", []):
                idx = int(entry["prompt_idx"])
                # Keep the most recent window for each prompt.
                if self._last_batched.get(idx, -1) < w:
                    self._last_batched[idx] = w
