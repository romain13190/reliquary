"""Base Environment protocol for Reliquary verifiable inference.

Defines the interface that all concrete environments must satisfy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Environment(Protocol):
    """An environment that produces (prompt, ground_truth) pairs and scores completions.

    Implementations must be deterministic: the same index always returns
    the same problem, and the same (problem, completion) pair always returns
    the same reward.
    """

    name: str

    def __len__(self) -> int:
        """Return the number of problems in this environment."""
        ...

    def get_problem(self, index: int) -> dict:
        """Return the problem at the given index.

        Returns a dict with keys:
            prompt (str): the question / instruction shown to the model
            ground_truth (str | int): the correct answer
            id (str): stable 16-char hex string = sha256(prompt)[:16]

        Indices wrap modulo len(self) so out-of-range values are safe.
        """
        ...

    def compute_reward(self, problem: dict, completion: str) -> float:
        """Score a completion against the ground truth.

        Returns a float in [0, 1].  Never raises on malformed input.
        """
        ...
