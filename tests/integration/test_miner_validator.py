"""Integration test: miner creates proof → validator verifies it.

Tests the core GRAIL proof roundtrip without requiring GPU or real models.
"""
import torch
import pytest
from reliquary.protocol.crypto import indices_from_root
from reliquary.protocol.grail_verifier import GRAILVerifier
from reliquary.constants import CHALLENGE_K


class TestMinerValidatorRoundtrip:
    def test_proof_roundtrip_synthetic(self):
        """Same hidden states → proof verifies at all challenge positions."""
        hidden_dim = 256
        seq_len = 64
        torch.manual_seed(42)

        # Miner and validator see the same hidden states
        hidden_states = torch.randn(seq_len, hidden_dim)
        tokens = list(range(seq_len))
        randomness = "aabbccddee112233"

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        # Miner: create commitments
        commitments = verifier.create_commitments_batch(hidden_states, r_vec)
        assert len(commitments) == seq_len

        # Validator: verify at challenge positions
        challenge_indices = indices_from_root(
            tokens, randomness, seq_len, min(CHALLENGE_K, seq_len)
        )

        for idx in challenge_indices:
            valid, diag = verifier.verify_commitment(
                hidden_states[idx], commitments[idx], r_vec, seq_len, idx
            )
            assert valid, f"Failed at position {idx}: {diag}"

    def test_different_hidden_states_produce_different_sketches(self):
        """Different hidden states produce different sketch values.

        The production tolerance (6000) is intentionally generous to handle
        cross-GPU FP drift. Security comes from K=32 positions combined, not
        individual checks. Here we verify that different hidden states produce
        different sketch values (even if they fall within tolerance).
        """
        hidden_dim = 256
        seq_len = 32
        torch.manual_seed(42)
        miner_hidden = torch.randn(seq_len, hidden_dim)
        torch.manual_seed(99)
        validator_hidden = torch.randn(seq_len, hidden_dim)
        randomness = "aabbccddee112233"

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        miner_commits = verifier.create_commitments_batch(miner_hidden, r_vec)
        validator_commits = verifier.create_commitments_batch(validator_hidden, r_vec)

        # At least some positions should have different sketch values
        differences = 0
        for i in range(seq_len):
            if miner_commits[i]["sketch"] != validator_commits[i]["sketch"]:
                differences += 1

        assert differences > 0, "All sketch values identical — hidden states should diverge"

    def test_batch_vs_single_consistency(self):
        """Batch and single commitment creation must be identical."""
        hidden_dim = 512
        seq_len = 32
        torch.manual_seed(123)
        h_layer = torch.randn(seq_len, hidden_dim)

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec("deadbeef")

        batch_commits = verifier.create_commitments_batch(h_layer, r_vec)
        for i in range(seq_len):
            single = verifier.create_commitment(h_layer[i], r_vec)
            assert batch_commits[i]["sketch"] == single["sketch"], (
                f"Batch/single mismatch at pos {i}: "
                f"batch={batch_commits[i]['sketch']} single={single['sketch']}"
            )

    def test_full_sequence_verification(self):
        """Verify ALL positions, not just challenge indices."""
        hidden_dim = 128
        seq_len = 50
        torch.manual_seed(77)
        h = torch.randn(seq_len, hidden_dim)

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec("cafebabe")

        commits = verifier.create_commitments_batch(h, r_vec)

        for i in range(seq_len):
            valid, diag = verifier.verify_commitment(h[i], commits[i], r_vec, seq_len, i)
            assert valid, f"Self-verification failed at pos {i}: diff={diag['sketch_diff']}"
