"""Tests for reliquary.miner.prompts — deterministic window-prompt derivation."""

from __future__ import annotations

import pytest

from reliquary.miner.prompts import derive_window_prompts


# ---------------------------------------------------------------------------
# Minimal fake environment (no I/O, no external deps)
# ---------------------------------------------------------------------------

class _FakeEnv:
    name = "fake"

    def __init__(self, size: int = 10):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def get_problem(self, idx: int) -> dict:
        return {
            "prompt": f"Q{idx % self._size}",
            "ground_truth": str(idx),
            "id": f"id-{idx % self._size}",
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RANDOMNESS_A = "a1b2c3d4" * 8  # 64 hex chars (32 bytes)
_RANDOMNESS_B = "deadbeef" * 8


@pytest.fixture
def env10():
    return _FakeEnv(size=10)


@pytest.fixture
def env3():
    return _FakeEnv(size=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_deterministic_same_randomness_same_prompts(env10):
    """Same randomness always produces the same ordered list of prompts."""
    result_1 = derive_window_prompts(env10, _RANDOMNESS_A, 8)
    result_2 = derive_window_prompts(env10, _RANDOMNESS_A, 8)
    assert result_1 == result_2


def test_different_randomness_different_prompts(env10):
    """Different randomness almost certainly produces a different sequence."""
    result_a = derive_window_prompts(env10, _RANDOMNESS_A, 8)
    result_b = derive_window_prompts(env10, _RANDOMNESS_B, 8)
    # At least one problem must differ (astronomically unlikely to collide)
    ids_a = [p["id"] for p in result_a]
    ids_b = [p["id"] for p in result_b]
    assert ids_a != ids_b


def test_count_exact(env10):
    """Returns exactly N entries when requested."""
    for n in (1, 4, 8, 16):
        result = derive_window_prompts(env10, _RANDOMNESS_A, n)
        assert len(result) == n, f"Expected {n} problems, got {len(result)}"


def test_indices_within_env_range(env10):
    """Every problem returned has valid structure and id comes from the env."""
    valid_ids = {f"id-{i}" for i in range(len(env10))}
    problems = derive_window_prompts(env10, _RANDOMNESS_A, 8)
    for p in problems:
        assert "prompt" in p
        assert "ground_truth" in p
        assert "id" in p
        assert p["id"] in valid_ids, f"id {p['id']!r} not in env"


def test_handles_small_env(env3):
    """With an env of length 3, requesting 8 still returns 8 (modulo wrapping)."""
    problems = derive_window_prompts(env3, _RANDOMNESS_A, 8)
    assert len(problems) == 8
    valid_ids = {f"id-{i}" for i in range(len(env3))}
    for p in problems:
        assert p["id"] in valid_ids, f"id {p['id']!r} out of range for size-3 env"
