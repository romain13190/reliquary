"""Deterministic window-prompt derivation for Reliquary miners.

Both miner and validator must call derive_window_prompts with the same
randomness and n to produce the same ordered sequence of problems.
"""

from __future__ import annotations

import hashlib

from reliquary.environment.base import Environment


def derive_window_prompts(env: Environment, randomness: str, n: int) -> list[dict]:
    """Deterministically pick *n* problems from *env* using SHA256(randomness || slot).

    For each slot index in [0, n) compute:
        h = SHA256(randomness_bytes || slot.to_bytes(4, 'big'))
        idx = int.from_bytes(h[:8], 'big') % len(env)

    This is the canonical selection algorithm agreed by miner and validator.
    """
    randomness_bytes = bytes.fromhex(randomness)
    problems: list[dict] = []
    for slot in range(n):
        h = hashlib.sha256(randomness_bytes + slot.to_bytes(4, "big")).digest()
        idx = int.from_bytes(h[:8], "big") % len(env)
        problems.append(env.get_problem(idx))
    return problems
