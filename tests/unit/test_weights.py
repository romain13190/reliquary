from reliquary.validator.weights import compute_weights


class TestComputeWeights:
    def test_single_miner_gets_weight_one(self):
        miner_w, burn_w = compute_weights({"miner_a": 42.0})
        assert abs(miner_w["miner_a"] - 1.0) < 1e-9
        assert burn_w == 0.0

    def test_higher_reward_gets_higher_weight(self):
        miner_w, _ = compute_weights({"a": 200.0, "b": 100.0})
        assert miner_w["a"] > miner_w["b"]

    def test_superlinear_bigger_miner_gets_more_than_90_pct(self):
        """200 vs 100 with exponent=4: 200^4/(200^4+100^4) ≈ 0.941."""
        miner_w, _ = compute_weights({"a": 200.0, "b": 100.0})
        assert miner_w["a"] > 0.9

    def test_zero_reward_gives_zero_weight(self):
        miner_w, _ = compute_weights({"a": 0.0, "b": 10.0})
        assert miner_w["a"] == 0.0

    def test_negative_reward_treated_as_zero(self):
        miner_w, _ = compute_weights({"a": -5.0, "b": 10.0})
        assert miner_w["a"] == 0.0
        assert miner_w["b"] > 0.0

    def test_multiple_miners_sum_to_one(self):
        scores = {"a": 100.0, "b": 200.0, "c": 50.0}
        miner_w, burn_w = compute_weights(scores)
        assert abs(sum(miner_w.values()) + burn_w - 1.0) < 1e-6

    def test_empty_input_returns_empty_dict(self):
        miner_w, burn_w = compute_weights({})
        assert miner_w == {}
        assert burn_w == 0.0

    def test_all_zeros_returns_all_zero_weights(self):
        miner_w, burn_w = compute_weights({"a": 0.0, "b": 0.0})
        assert all(w == 0.0 for w in miner_w.values())
        assert set(miner_w.keys()) == {"a", "b"}
        assert burn_w == 0.0


class TestBurnRouting:
    def test_burn_only_when_no_miner_signal(self) -> None:
        """All miners 0 but burn > 0 → full emission goes to UID_BURN."""
        miner_w, burn_w = compute_weights({"a": 0.0, "b": 0.0}, burn_score=100.0)
        assert burn_w == 1.0
        assert all(w == 0.0 for w in miner_w.values())

    def test_all_zero_when_neither_miners_nor_burn(self) -> None:
        miner_w, burn_w = compute_weights({"a": 0.0}, burn_score=0.0)
        assert burn_w == 0.0
        assert miner_w == {"a": 0.0}

    def test_burn_linear_miners_superlinear(self) -> None:
        """Burn stays linear while miner pool applies superlinear (^4).

        miner_a = 200, miner_b = 100, burn = 300
        → burn_frac = 300 / (300 + 200) = 0.5
        → miner pool = 0.5
        → within miners, a^4 / (a^4 + b^4) × 0.5 = (200^4 / 170e8) × 0.5 ≈ 0.941 × 0.5 ≈ 0.4706
        """
        miner_w, burn_w = compute_weights(
            {"a": 200.0, "b": 100.0}, burn_score=300.0
        )
        assert abs(burn_w - 0.5) < 1e-6
        assert abs(sum(miner_w.values()) - 0.5) < 1e-6
        # miner_a should dominate the miner pool.
        assert miner_w["a"] > 0.45
        assert miner_w["b"] < 0.05

    def test_total_always_sums_to_one_when_signal(self) -> None:
        miner_w, burn_w = compute_weights(
            {"a": 42.0, "b": 13.0, "c": 7.0}, burn_score=88.0
        )
        assert abs(sum(miner_w.values()) + burn_w - 1.0) < 1e-9

    def test_no_burn_matches_previous_behaviour(self) -> None:
        """burn_score=0 → burn_w=0, miner distribution identical to legacy code."""
        scores = {"a": 100.0, "b": 200.0, "c": 50.0}
        miner_w, burn_w = compute_weights(scores, burn_score=0.0)
        assert burn_w == 0.0
        assert abs(sum(miner_w.values()) - 1.0) < 1e-9

    def test_negative_burn_treated_as_zero(self) -> None:
        miner_w, burn_w = compute_weights({"a": 10.0}, burn_score=-5.0)
        assert burn_w == 0.0
        assert abs(miner_w["a"] - 1.0) < 1e-9
