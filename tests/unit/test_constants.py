"""Sanity checks on v2 constants — catches accidental edits."""

from reliquary import constants as C


def test_v2_sigma_bounds():
    assert C.SIGMA_MIN == 0.43
    assert C.BOOTSTRAP_SIGMA_MIN == 0.33
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN


def test_v2_group_sizes():
    assert C.M_ROLLOUTS == 8
    assert C.B_BATCH == 8


def test_v2_temperature_fixed_nonzero():
    assert 0.5 < C.T_PROTO <= 1.0


def test_v2_cooldown_values():
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 50
    assert C.BOOTSTRAP_WINDOWS == 100


def test_v2_bootstrap_sigma_lower_than_steady():
    # Bootstrap accepts groups with lower σ (σ ≥ 0.33) vs steady (σ ≥ 0.43)
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN


def test_wandb_constants_present():
    assert C.WANDB_PROJECT == "reliquary-validator"
    assert C.WANDB_TRAINING_VERSION == "v1"
