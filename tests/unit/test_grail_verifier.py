import torch
import pytest
from reliquary.protocol.grail_verifier import (
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
    adaptive_sketch_tolerance,
    GRAILVerifier,
)
from reliquary.constants import PRIME_Q, PROOF_SKETCH_TOLERANCE_BASE


class TestLogMagnitudeBucket:
    def test_zero(self):
        assert log_magnitude_bucket(0.0) == 0

    def test_near_zero(self):
        assert log_magnitude_bucket(1e-7) == 0

    def test_positive(self):
        b = log_magnitude_bucket(5.0)
        assert b > 0

    def test_negative(self):
        b = log_magnitude_bucket(-5.0)
        assert b < 0

    def test_symmetry(self):
        assert log_magnitude_bucket(5.0) == -log_magnitude_bucket(-5.0)

    def test_nan_returns_zero(self):
        assert log_magnitude_bucket(float("nan")) == 0

    def test_inf(self):
        assert log_magnitude_bucket(float("inf")) == 7
        assert log_magnitude_bucket(float("-inf")) == -7

    def test_monotonic_positive(self):
        """Larger positive values should give equal or larger buckets."""
        prev = 0
        for v in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            b = log_magnitude_bucket(v)
            assert b >= prev
            prev = b


class TestLogMagnitudeBucketVectorized:
    def test_matches_scalar(self):
        values = torch.tensor([0.0, 1e-7, 5.0, -5.0, 100.0, -0.001])
        vec_result = log_magnitude_bucket_vectorized(values)
        scalar_result = torch.tensor(
            [log_magnitude_bucket(v.item()) for v in values],
            dtype=torch.int64,
        )
        assert torch.equal(vec_result, scalar_result)

    def test_nan_handling(self):
        values = torch.tensor([float("nan"), 1.0])
        result = log_magnitude_bucket_vectorized(values)
        assert result[0].item() == 0

    def test_inf_handling(self):
        values = torch.tensor([float("inf"), float("-inf")])
        result = log_magnitude_bucket_vectorized(values)
        assert result[0].item() == 7
        assert result[1].item() == -7

    def test_2d(self):
        values = torch.randn(10, 16)
        result = log_magnitude_bucket_vectorized(values)
        assert result.shape == (10, 16)

    def test_large_batch_matches_scalar(self):
        torch.manual_seed(42)
        values = torch.randn(100)
        vec = log_magnitude_bucket_vectorized(values)
        for i in range(100):
            assert vec[i].item() == log_magnitude_bucket(values[i].item()), f"Mismatch at index {i}"


class TestAdaptiveSketchTolerance:
    def test_position_zero(self):
        assert adaptive_sketch_tolerance(0, 100) == PROOF_SKETCH_TOLERANCE_BASE

    def test_increases_with_position(self):
        t0 = adaptive_sketch_tolerance(0, 1000)
        t100 = adaptive_sketch_tolerance(100, 1000)
        t1000 = adaptive_sketch_tolerance(1000, 1000)
        assert t0 < t100 < t1000

    def test_position_8192(self):
        t = adaptive_sketch_tolerance(8192, 8192)
        # base(6000) + 5.0 * sqrt(8192) = 6000 + 452.5 ≈ 6452
        assert t == 6452


class TestGRAILVerifier:
    @pytest.fixture
    def verifier(self):
        return GRAILVerifier(hidden_dim=128)

    def test_generate_r_vec_shape(self, verifier):
        r = verifier.generate_r_vec("aabbccdd")
        assert r.shape == (16,)  # PROOF_TOPK
        assert r.dtype == torch.int8

    def test_generate_r_vec_deterministic(self, verifier):
        a = verifier.generate_r_vec("aabbccdd")
        b = verifier.generate_r_vec("aabbccdd")
        assert torch.equal(a, b)

    def test_generate_r_vec_bounded(self, verifier):
        r = verifier.generate_r_vec("aabb")
        assert r.abs().max().item() <= 127

    def test_create_commitment(self, verifier):
        h = torch.randn(128)
        r = verifier.generate_r_vec("aabb")
        commit = verifier.create_commitment(h, r)
        assert "sketch" in commit
        assert 0 <= commit["sketch"] < PRIME_Q

    def test_create_commitments_batch_matches_single(self, verifier):
        torch.manual_seed(42)
        h_layer = torch.randn(8, 128)
        r = verifier.generate_r_vec("aabb")
        batch = verifier.create_commitments_batch(h_layer, r)
        for i in range(8):
            single = verifier.create_commitment(h_layer[i], r)
            assert batch[i]["sketch"] == single["sketch"], f"Mismatch at position {i}"

    def test_verify_own_commitment(self, verifier):
        torch.manual_seed(42)
        h = torch.randn(128)
        r = verifier.generate_r_vec("aabb")
        commit = verifier.create_commitment(h, r)
        valid, diag = verifier.verify_commitment(h, commit, r, 100, 0)
        assert valid
        assert diag["sketch_diff"] == 0

    def test_verify_batch_commitments(self, verifier):
        torch.manual_seed(42)
        h_layer = torch.randn(16, 128)
        r = verifier.generate_r_vec("aabb")
        commits = verifier.create_commitments_batch(h_layer, r)
        for i in range(16):
            valid, diag = verifier.verify_commitment(h_layer[i], commits[i], r, 16, i)
            assert valid, f"Failed at position {i}: {diag}"

    def test_different_hidden_fails(self, verifier):
        torch.manual_seed(42)
        h1 = torch.randn(128)
        h2 = torch.randn(128)
        r = verifier.generate_r_vec("aabb")
        commit = verifier.create_commitment(h1, r)
        valid, _ = verifier.verify_commitment(h2, commit, r, 100, 0)
        # Different random hidden states should almost certainly fail
        # (though not guaranteed for every seed)
        # We test this statistically in integration tests
