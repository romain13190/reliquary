"""Tests for deterministic beacon round selection.

Miner and validator must agree on the same drand round for a given window.
The round is derived from the window_start block number, not fetched as 'latest'.
"""

from reliquary.constants import BLOCK_TIME_SECONDS, WINDOW_LENGTH


class TestDeterministicBeaconRound:
    def test_compute_window_randomness_includes_round(self):
        """Window randomness must bind the drand round number to prevent
        a miner from choosing a favorable round."""
        from reliquary.infrastructure.chain import compute_window_randomness

        block_hash = "aa" * 32
        drand_rand = "bb" * 32

        r1 = compute_window_randomness(block_hash, drand_rand, drand_round=100)
        r2 = compute_window_randomness(block_hash, drand_rand, drand_round=101)
        r_no_round = compute_window_randomness(block_hash, drand_rand, drand_round=None)

        assert r1 != r2, "Different rounds must produce different randomness"
        assert r1 != r_no_round, "Providing a round must change the result"

    def test_compute_drand_round_for_window(self):
        """Round selection must be deterministic from window_start and chain params."""
        from reliquary.infrastructure.chain import compute_drand_round_for_window

        genesis_time = 1000
        period = 3

        # Window at block 100: timestamp = 100 * 12 = 1200
        # Expected round = 1 + (1200 - 1000) // 3 = 1 + 66 = 67
        r = compute_drand_round_for_window(100, genesis_time, period)
        assert r == 67

        # Same input = same output (deterministic)
        assert compute_drand_round_for_window(100, genesis_time, period) == 67

        # Different window = different round
        assert compute_drand_round_for_window(130, genesis_time, period) != 67

    def test_compute_drand_round_before_genesis_returns_1(self):
        from reliquary.infrastructure.chain import compute_drand_round_for_window

        # Window at block 10: timestamp = 120, before genesis at 1000
        r = compute_drand_round_for_window(10, 1000, 3)
        assert r == 1  # Clamp to round 1 (the first valid round)
