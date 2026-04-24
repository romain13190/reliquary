"""Tests for the behavioural validators ported from the original GRAIL.

Covers:
- ProofResult wrapping (Task 3): verify_commitment_proofs returns cached logits
- verify_logprobs_claim (Task 5)
- evaluate_token_distribution (Task 6)
"""

from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Task 3 — ProofResult with cached logits
# ---------------------------------------------------------------------------


def test_verify_commitment_proofs_returns_logits(monkeypatch):
    """verify_commitment_proofs must now expose the logits tensor so the
    downstream Logprob / Distribution checks can read it without re-running
    the forward pass.
    """
    from reliquary.validator import verifier

    seq_len = 10
    hidden_dim = 8
    vocab_size = 100
    fake_hidden = torch.zeros(1, seq_len, hidden_dim)
    fake_logits = torch.zeros(1, seq_len, vocab_size)

    def fake_forward(model, ids, attn, layer):
        return fake_hidden, fake_logits

    monkeypatch.setattr(
        "reliquary.shared.forward.forward_single_layer",
        fake_forward,
    )
    monkeypatch.setattr(
        "reliquary.shared.hf_compat.resolve_hidden_size",
        lambda m: hidden_dim,
    )

    class FakeVerifier:
        def __init__(self, hidden_dim):
            pass

        def generate_r_vec(self, r):
            return torch.zeros(hidden_dim)

        def verify_commitment(self, h, commit, r, seq, idx):
            return True, None

    monkeypatch.setattr(
        "reliquary.protocol.grail_verifier.GRAILVerifier",
        FakeVerifier,
    )

    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])

    commit = {
        "tokens": list(range(seq_len)),
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
    }

    result = verifier.verify_commitment_proofs(commit, model, "a" * 64)

    assert hasattr(result, "logits"), "ProofResult must expose .logits"
    assert isinstance(result.logits, torch.Tensor)
    assert result.logits.shape == (seq_len, vocab_size)
    assert hasattr(result, "all_passed")
    assert result.all_passed is True
