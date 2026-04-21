"""GRPO training helpers — pure-math tests (no model)."""

import math

import pytest

from reliquary.validator.training import _compute_advantages


def test_compute_advantages_symmetric_group():
    # Half success, half failure → advantages symmetric around zero.
    advs = _compute_advantages([1, 1, 1, 1, 0, 0, 0, 0])
    assert len(advs) == 8
    # mean=0.5, std=0.5 → advantages = ±1
    for a in advs[:4]:
        assert abs(a - 1.0) < 1e-9
    for a in advs[4:]:
        assert abs(a + 1.0) < 1e-9


def test_compute_advantages_all_same_is_zero():
    assert _compute_advantages([1, 1, 1, 1, 1, 1, 1, 1]) == [0.0] * 8
    assert _compute_advantages([0, 0, 0, 0]) == [0.0] * 4


def test_compute_advantages_sum_is_zero():
    """Normalized advantages always sum to 0 (within float tolerance)."""
    for rewards in ([1, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0.5, 0.3, 0.1]):
        advs = _compute_advantages(rewards)
        assert abs(sum(advs)) < 1e-9


def test_compute_advantages_empty():
    assert _compute_advantages([]) == []


def test_compute_advantages_single_element():
    # One element, variance=0 → advantage=0.
    assert _compute_advantages([1.0]) == [0.0]
