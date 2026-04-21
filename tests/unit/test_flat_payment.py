"""compute_weights_v2: flat 1/B payment + UID_BURN on unused slots."""

from reliquary.constants import B_BATCH, UID_BURN
from reliquary.validator.weights import compute_weights_v2


def test_empty_batch_full_burn():
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=[])
    assert miner_weights == {}
    assert burn_weight == 1.0


def test_full_batch_no_burn():
    batch = [f"hk{i}" for i in range(B_BATCH)]
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    assert set(miner_weights.keys()) == set(batch)
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    assert abs(burn_weight) < 1e-9
    assert abs(sum(miner_weights.values()) + burn_weight - 1.0) < 1e-9


def test_partial_batch_partial_burn():
    batch = ["hk0", "hk1", "hk2", "hk3", "hk4"]  # 5/8
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    assert len(miner_weights) == 5
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    assert abs(burn_weight - 3.0 / B_BATCH) < 1e-9
    assert abs(sum(miner_weights.values()) + burn_weight - 1.0) < 1e-9


def test_weights_sum_to_one_always():
    for n in range(B_BATCH + 1):
        batch = [f"hk{i}" for i in range(n)]
        miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
        total = sum(miner_weights.values()) + burn_weight
        assert abs(total - 1.0) < 1e-9, f"sum != 1.0 at n={n}"


def test_duplicate_hotkey_still_single_payment():
    """Batch may contain the same hotkey twice if the miner won batch slots
    on two distinct prompts. Merge into one entry per hotkey."""
    batch = ["alice", "alice", "bob"]
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    assert abs(miner_weights["alice"] - 2.0 / B_BATCH) < 1e-9
    assert abs(miner_weights["bob"] - 1.0 / B_BATCH) < 1e-9
    assert abs(burn_weight - (B_BATCH - 3) / B_BATCH) < 1e-9


def test_over_full_batch_raises():
    batch = [f"hk{i}" for i in range(B_BATCH + 1)]
    import pytest
    with pytest.raises(ValueError, match="batch size"):
        compute_weights_v2(batch_hotkeys=batch)
