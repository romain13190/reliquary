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


# ---------------------------------------------------------------------------
# Task 5 — verify_logprobs_claim (IS-median deviation check)
# ---------------------------------------------------------------------------


def test_verify_logprobs_claim_honest_passes():
    """When miner_logprobs match validator-recomputed logprobs, the median
    IS-deviation is ~0 and the check passes.
    """
    from reliquary.validator import verifier
    import math

    vocab = 50
    seq_len = 40
    logits = torch.zeros(seq_len, vocab)  # flat → each token lp = -log(vocab)
    uniform_lp = -math.log(vocab)

    tokens = list(range(seq_len))
    prompt_len = 5
    # Full-sequence layout (length == len(tokens)).
    claimed = [0.0] * prompt_len + [uniform_lp] * (seq_len - prompt_len)

    ok, median_dev = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=seq_len - prompt_len,
        claimed_logprobs=claimed,
        logits=logits,
        challenge_randomness="b" * 64,
    )
    assert ok is True, f"honest pair must pass, got median_dev={median_dev}"


def test_verify_logprobs_claim_cheater_fails():
    """When miner lies about logprobs, median IS-deviation exceeds 0.10."""
    from reliquary.validator import verifier

    vocab = 50
    seq_len = 40
    logits = torch.zeros(seq_len, vocab)

    tokens = list(range(seq_len))
    prompt_len = 5
    # Miner claims implausibly good logprobs (-0.5 vs true ~-3.9)
    # → dev = exp(|−0.5 − (−3.9)|) − 1 ≈ exp(3.4) − 1 ≈ 28.
    claimed = [0.0] * prompt_len + [-0.5] * (seq_len - prompt_len)

    ok, median_dev = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=seq_len - prompt_len,
        claimed_logprobs=claimed,
        logits=logits,
        challenge_randomness="b" * 64,
    )
    assert ok is False
    assert median_dev > 0.10


def test_verify_logprobs_claim_too_short_rejects():
    """Completion shorter than CHALLENGE_K cannot be challenged → reject."""
    from reliquary.validator import verifier

    vocab = 50
    seq_len = 20  # completion only 15, < CHALLENGE_K (32)
    logits = torch.zeros(seq_len, vocab)
    tokens = list(range(seq_len))

    ok, _ = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=5,
        completion_length=15,
        claimed_logprobs=[0.0] * seq_len,
        logits=logits,
        challenge_randomness="c" * 64,
    )
    assert ok is False


def test_verify_logprobs_claim_completion_only_layout():
    """Miner's token_logprobs payload has length == completion_length
    (no prompt padding). The helper must accept both shapes."""
    from reliquary.validator import verifier
    import math

    vocab = 50
    seq_len = 40
    logits = torch.zeros(seq_len, vocab)
    uniform_lp = -math.log(vocab)

    tokens = list(range(seq_len))
    prompt_len = 5
    claimed = [uniform_lp] * (seq_len - prompt_len)  # completion-only

    ok, _ = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=seq_len - prompt_len,
        claimed_logprobs=claimed,
        logits=logits,
        challenge_randomness="d" * 64,
    )
    assert ok is True


# ---------------------------------------------------------------------------
# Task 6 — evaluate_token_distribution (chosen-token probability stats)
# ---------------------------------------------------------------------------


def test_evaluate_token_distribution_honest_passes():
    """Honest miner sampling from our (peaked) distribution: the chosen
    token has high probability under the validator's model →
    median > SAMPLING_MEDIAN_LOW_MAX and q10 > SAMPLING_LOW_Q10_MAX.

    Mimics a realistic LLM where the sampled token is typically in the
    top of the distribution. (The thresholds were calibrated on that
    regime — uniform distributions over a large vocab would look
    'suspicious' to them, by design.)
    """
    from reliquary.validator import verifier

    vocab = 100
    seq_len = 50
    prompt_len = 5
    # Every position t picks a designated "chosen" token and gives it
    # high logit (5.0) while the rest are at -1. Softmax at T=0.9 yields
    # p(chosen) ≈ 0.99 → median and q10 both ≈ 0.99.
    logits = torch.full((seq_len, vocab), -1.0)
    tokens = [i % vocab for i in range(seq_len)]
    for t in range(1, seq_len):
        logits[t - 1, tokens[t]] = 5.0

    ok, metrics = verifier.evaluate_token_distribution(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=seq_len - prompt_len,
        logits=logits,
        temperature=0.9,
    )
    assert ok is True
    assert metrics["median"] > 0.30


def test_evaluate_token_distribution_cheater_fails():
    """Cheater: chose tokens that are very low-probability under the
    validator's model → median and q10 both collapse below thresholds.
    """
    from reliquary.validator import verifier

    vocab = 100
    seq_len = 50
    prompt_len = 5
    # Logits favour token 0 massively; every other token has ~0 probability.
    logits = torch.full((seq_len, vocab), -10.0)
    logits[:, 0] = 10.0
    # Miner "chose" tokens 50-94 — all very low-probability under this model.
    tokens = [0] * prompt_len + list(range(50, 95))

    ok, metrics = verifier.evaluate_token_distribution(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=seq_len - prompt_len,
        logits=logits,
        temperature=0.9,
    )
    assert ok is False
    assert metrics["median"] < 0.30
    assert metrics["q10"] < 0.025


def test_evaluate_token_distribution_too_short_skips():
    """Completion shorter than SAMPLING_MIN_STEPS returns (None, {}) —
    not enough data to decide, caller defaults to accept."""
    from reliquary.validator import verifier

    vocab = 10
    seq_len = 20  # completion only 15, < SAMPLING_MIN_STEPS (30)
    logits = torch.zeros(seq_len, vocab)
    tokens = [0] * seq_len

    ok, _ = verifier.evaluate_token_distribution(
        tokens=tokens,
        prompt_length=5,
        completion_length=15,
        logits=logits,
        temperature=0.9,
    )
    assert ok is None


