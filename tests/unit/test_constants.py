"""Sanity checks on v2 constants — catches accidental edits."""

from reliquary import constants as C


def test_v2_zone_bounds():
    assert C.ZONE_K_MIN == 2
    assert C.ZONE_K_MAX == 6
    assert C.ZONE_K_MIN < C.ZONE_K_MAX


def test_v2_group_sizes():
    assert C.M_ROLLOUTS == 8
    assert C.B_BATCH == 8


def test_v2_temperature_fixed_nonzero():
    assert 0.5 < C.T_PROTO <= 1.0


def test_v2_cooldown_values():
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 50
    assert C.BOOTSTRAP_WINDOWS == 100


def test_v2_bootstrap_zone_is_wider_than_steady():
    # k ∈ [1, 7] during bootstrap vs [2, 6] steady
    assert C.BOOTSTRAP_ZONE_K_MIN < C.ZONE_K_MIN
    assert C.BOOTSTRAP_ZONE_K_MAX > C.ZONE_K_MAX
