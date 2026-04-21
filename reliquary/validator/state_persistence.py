"""ValidatorState: tiny JSON store for window_n + checkpoint_n counters.

Local-first: validator restart loads from disk. Cooldown rebuild already
covers R2 fallback, so this file is purely a hot-path optimisation —
losing it just means starting counters from 0 (the rebuild loop will
then advance them through observed history).
"""

from __future__ import annotations

import json
import os
import tempfile


class ValidatorState:
    """Tiny counters store. Mutate fields directly, then call ``save()``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.window_n: int = 0
        self.checkpoint_n: int = 0
        self.miner_scores_ema: dict[str, float] = {}

    def load(self) -> None:
        """Load from disk if present; otherwise leave defaults."""
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            data = json.load(f)
        self.window_n = int(data.get("window_n", 0))
        self.checkpoint_n = int(data.get("checkpoint_n", 0))
        self.miner_scores_ema = {
            str(k): float(v) for k, v in data.get("miner_scores_ema", {}).items()
        }

    def save(self) -> None:
        """Atomic write via tmp + rename. Creates parent dir if needed."""
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".s.", dir=parent)
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "window_n": self.window_n,
                        "checkpoint_n": self.checkpoint_n,
                        "miner_scores_ema": self.miner_scores_ema,
                    },
                    f,
                )
            os.replace(tmp_path, self.path)
        except Exception:
            os.unlink(tmp_path)
            raise
