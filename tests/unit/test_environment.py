"""Tests for the Reliquary environment module.

TDD: tests written first, then implementation.
"""

import hashlib
import pytest


# ---------------------------------------------------------------------------
# Loader / factory tests
# ---------------------------------------------------------------------------

def test_loader_returns_gsm8k_instance():
    from reliquary.environment import load_environment
    from reliquary.environment.gsm8k import GSM8KEnvironment

    env = load_environment("gsm8k")
    assert isinstance(env, GSM8KEnvironment)


def test_loader_raises_on_unknown():
    from reliquary.environment import load_environment

    with pytest.raises(ValueError, match="Unknown environment: nope"):
        load_environment("nope")


# ---------------------------------------------------------------------------
# Dataset-touching tests (may skip if network is unavailable)
# ---------------------------------------------------------------------------

def _load_gsm8k_env():
    """Helper: load GSM8K env, skip on any dataset error."""
    try:
        from reliquary.environment.gsm8k import GSM8KEnvironment
        return GSM8KEnvironment()
    except Exception as exc:
        pytest.skip(f"Could not load GSM8K dataset: {exc}")


def test_dataset_loaded_and_indexable():
    env = _load_gsm8k_env()
    assert len(env) > 0

    problem = env.get_problem(0)
    assert isinstance(problem["prompt"], str) and len(problem["prompt"]) > 0
    assert isinstance(problem["ground_truth"], str) and len(problem["ground_truth"]) > 0


def test_problem_id_is_stable_sha_prefix():
    env = _load_gsm8k_env()

    p1 = env.get_problem(0)
    p2 = env.get_problem(0)

    assert p1["id"] == p2["id"], "id must be stable across calls"
    assert len(p1["id"]) == 16, "id must be 16 hex chars"
    # Verify it equals sha256(prompt)[:16]
    expected = hashlib.sha256(p1["prompt"].encode()).hexdigest()[:16]
    assert p1["id"] == expected


def test_index_wraps_modulo_length():
    env = _load_gsm8k_env()

    p0 = env.get_problem(0)
    p_wrap = env.get_problem(len(env))
    assert p_wrap["id"] == p0["id"], "index should wrap modulo length"


# ---------------------------------------------------------------------------
# Reward extraction tests — no dataset needed
# ---------------------------------------------------------------------------

from reliquary.environment.gsm8k import _compute_gsm8k_reward  # noqa: E402 — after skippable imports


def test_reward_exact_numeric_match():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, "The answer is 42") == 1.0


def test_reward_boxed_takes_priority():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, r"I think 17 but actually \boxed{42}") == 1.0


def test_reward_last_boxed_wins():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, r"\boxed{17} no wait \boxed{42}") == 1.0


def test_reward_last_number_when_no_boxed():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, "Step 1: 17. Step 2: 25. Final: 42.") == 1.0


def test_reward_thousands_separator_in_completion():
    problem = {"ground_truth": "1234"}
    assert _compute_gsm8k_reward(problem, "answer is 1,234") == 1.0


def test_reward_thousands_separator_in_ground_truth():
    problem = {"ground_truth": "1,234"}
    assert _compute_gsm8k_reward(problem, "answer is 1234") == 1.0


def test_reward_decimal_zero_normalization():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, "42.0") == 1.0


def test_reward_wrong_number():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, "answer is 41") == 0.0


def test_reward_no_number_in_completion():
    problem = {"ground_truth": "42"}
    assert _compute_gsm8k_reward(problem, "I don't know") == 0.0


def test_reward_negative_number():
    problem = {"ground_truth": "-5"}
    assert _compute_gsm8k_reward(problem, "answer is -5") == 1.0


def test_reward_handles_garbage_gt():
    problem = {"ground_truth": "not a number"}
    # Should return 0.0 without raising
    result = _compute_gsm8k_reward(problem, "42")
    assert result == 0.0
