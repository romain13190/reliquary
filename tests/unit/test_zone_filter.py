"""Zone filter: k ∈ [ZONE_K_MIN, ZONE_K_MAX] (binary in/out, no scoring)."""

from reliquary.constants import (
    BOOTSTRAP_ZONE_K_MAX, BOOTSTRAP_ZONE_K_MIN,
    M_ROLLOUTS, ZONE_K_MAX, ZONE_K_MIN,
)
from reliquary.validator.verifier import is_in_zone, rewards_to_k


def test_k_below_min_rejected():
    assert is_in_zone(k=ZONE_K_MIN - 1) is False


def test_k_min_accepted():
    assert is_in_zone(k=ZONE_K_MIN) is True


def test_k_max_accepted():
    assert is_in_zone(k=ZONE_K_MAX) is True


def test_k_above_max_rejected():
    assert is_in_zone(k=ZONE_K_MAX + 1) is False


def test_k_all_zeros_rejected():
    assert is_in_zone(k=0) is False


def test_k_all_ones_rejected():
    assert is_in_zone(k=M_ROLLOUTS) is False


def test_bootstrap_mode_wider_zone():
    assert is_in_zone(k=1, bootstrap=True) is True
    assert is_in_zone(k=1, bootstrap=False) is False
    assert is_in_zone(k=7, bootstrap=True) is True
    assert is_in_zone(k=7, bootstrap=False) is False


def test_bootstrap_still_rejects_k_0_and_M():
    assert is_in_zone(k=0, bootstrap=True) is False
    assert is_in_zone(k=M_ROLLOUTS, bootstrap=True) is False


def test_rewards_to_k_binary():
    assert rewards_to_k([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) == 4


def test_rewards_to_k_with_tolerance():
    assert rewards_to_k([0.999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 1


def test_rewards_to_k_all_zero():
    assert rewards_to_k([0.0] * M_ROLLOUTS) == 0


def test_rewards_to_k_all_one():
    assert rewards_to_k([1.0] * M_ROLLOUTS) == M_ROLLOUTS
