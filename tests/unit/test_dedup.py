"""Unit tests for hash-dedup primitives."""

from reliquary.protocol.submission import RejectReason


def test_hash_duplicate_reject_reason_exists():
    assert RejectReason.HASH_DUPLICATE.value == "hash_duplicate"


def test_compute_rollout_hash_returns_32_bytes():
    from reliquary.validator.dedup import compute_rollout_hash
    h = compute_rollout_hash([1, 2, 3, 4])
    assert isinstance(h, bytes)
    assert len(h) == 32


def test_compute_rollout_hash_deterministic():
    from reliquary.validator.dedup import compute_rollout_hash
    h1 = compute_rollout_hash([100, 200, 300, 400, 500])
    h2 = compute_rollout_hash([100, 200, 300, 400, 500])
    assert h1 == h2


def test_compute_rollout_hash_differs_on_single_token_change():
    from reliquary.validator.dedup import compute_rollout_hash
    a = compute_rollout_hash([10, 20, 30, 40, 50])
    b = compute_rollout_hash([10, 20, 31, 40, 50])  # single token diff
    assert a != b


def test_compute_rollout_hash_rejects_negative_tokens():
    import pytest
    from reliquary.validator.dedup import compute_rollout_hash
    with pytest.raises(ValueError):
        compute_rollout_hash([1, -2, 3])


def test_hashset_empty_does_not_contain():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    assert compute_rollout_hash([1, 2, 3]) not in s
    assert len(s) == 0


def test_hashset_add_then_contains():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([10, 20, 30])
    s.add(h, window=100)
    assert h in s
    assert len(s) == 1


def test_hashset_add_duplicate_is_idempotent():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([10, 20, 30])
    s.add(h, window=100)
    s.add(h, window=110)
    assert len(s) == 1  # same hash → one entry, latest window kept


def test_hashset_negative_window_rejected():
    import pytest
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([1, 2, 3])
    with pytest.raises(ValueError):
        s.add(h, window=-1)


def test_hashset_negative_retention_rejected():
    import pytest
    from reliquary.validator.dedup import RolloutHashSet
    with pytest.raises(ValueError):
        RolloutHashSet(retention_windows=-1)


def test_hashset_prune_drops_expired_entries():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    old = compute_rollout_hash([1, 1, 1])
    recent = compute_rollout_hash([2, 2, 2])
    s.add(old, window=100)
    s.add(recent, window=145)
    # At window 151: window 100 is 51 away (>= 50) → drop
    s.prune(current_window=151)
    assert old not in s
    assert recent in s


def test_hashset_prune_keeps_boundary_at_minus_one():
    """An entry at window=100 with retention=50 must stay until current=150."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([3, 3, 3])
    s.add(h, window=100)
    s.prune(current_window=149)
    assert h in s
    s.prune(current_window=150)
    assert h not in s


def test_hashset_prune_zero_retention_drops_everything():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=0)
    h = compute_rollout_hash([4, 4, 4])
    s.add(h, window=100)
    s.prune(current_window=100)
    assert h not in s


def test_rebuild_from_history_indexes_hash_field():
    """When archives carry an explicit `hash` field per rollout, use it."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h_a = compute_rollout_hash([1, 2, 3]).hex()
    h_b = compute_rollout_hash([4, 5, 6]).hex()
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 42,
                    "rollouts": [
                        {"tokens": [1, 2, 3], "hash": h_a, "reward": 1.0},
                        {"tokens": [4, 5, 6], "hash": h_b, "reward": 0.0},
                    ],
                }
            ],
        }
    ]
    s.rebuild_from_history(archives, current_window=110)
    assert bytes.fromhex(h_a) in s
    assert bytes.fromhex(h_b) in s
    assert len(s) == 2


def test_rebuild_from_history_recomputes_when_hash_missing():
    """Backwards-compat: pre-feature archives have only `tokens`, no `hash`."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 42,
                    "rollouts": [
                        {"tokens": [7, 8, 9], "reward": 1.0},  # no hash key
                    ],
                }
            ],
        }
    ]
    s.rebuild_from_history(archives, current_window=110)
    assert compute_rollout_hash([7, 8, 9]) in s


def test_rebuild_from_history_skips_expired_windows():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    archives = [
        {
            "window_start": 40,  # expired at current=100 (50 horizon)
            "batch": [{"prompt_idx": 1, "rollouts": [{"tokens": [9, 9]}]}],
        },
        {
            "window_start": 90,
            "batch": [{"prompt_idx": 2, "rollouts": [{"tokens": [8, 8]}]}],
        },
    ]
    s.rebuild_from_history(archives, current_window=100)
    assert compute_rollout_hash([9, 9]) not in s
    assert compute_rollout_hash([8, 8]) in s


def test_rebuild_from_history_clears_previous_state():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    stale = compute_rollout_hash([1])
    s.add(stale, window=100)
    s.rebuild_from_history([], current_window=110)
    assert stale not in s
    assert len(s) == 0
