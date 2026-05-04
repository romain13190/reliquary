"""verify_termination — strict EOS-only termination check.

The miner must end every rollout with the tokenizer's EOS token, AND the
model must have assigned probability >= MIN_EOS_PROBABILITY to EOS at the
position that produced it. No max-tokens-fallback branch — see spec for
RL-context rationale.
"""

import pytest
import torch

from reliquary.constants import MIN_EOS_PROBABILITY
from reliquary.validator.verifier import verify_termination


class _FakeTokenizer:
    eos_token_id = 99


def _commit(tokens: list[int]) -> dict:
    """Minimal commit dict — verify_termination only reads ``tokens``."""
    return {"tokens": tokens}


def _make_logits(seq_len: int, vocab_size: int = 100, eos_logit: float = 5.0):
    """Logits where EOS token (id 99) has high probability at every position."""
    logits = torch.zeros(seq_len, vocab_size)
    logits[:, 99] = eos_logit
    return logits


def test_accepts_when_ends_with_eos_at_high_prob():
    tokens = [10, 20, 30, 99]  # last token = EOS
    logits = _make_logits(seq_len=4, eos_logit=5.0)  # p(EOS) ~ 0.97
    assert verify_termination(_commit(tokens), _FakeTokenizer(), logits) is True


def test_rejects_when_does_not_end_with_eos():
    tokens = [10, 20, 30, 40]  # last token != EOS
    logits = _make_logits(seq_len=4)
    assert verify_termination(_commit(tokens), _FakeTokenizer(), logits) is False


def test_rejects_when_eos_prob_below_threshold():
    tokens = [10, 20, 30, 99]
    # logits where EOS is wildly improbable: id 99 gets large negative logit
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0  # p(EOS) ~ 4.5e-5, well below 0.02
    assert verify_termination(_commit(tokens), _FakeTokenizer(), logits) is False


def test_rejects_when_tokenizer_has_no_eos():
    tokens = [10, 20, 30, 99]
    logits = _make_logits(seq_len=4)

    class NoEosTokenizer:
        eos_token_id = None

    assert verify_termination(_commit(tokens), NoEosTokenizer(), logits) is False


def test_uses_logits_at_second_to_last_position():
    """The probability is read from logits[-2] (the position that PRODUCED tokens[-1])."""
    tokens = [10, 20, 30, 99]
    # Make EOS unlikely everywhere EXCEPT at position -2
    logits = torch.zeros(4, 100)
    logits[:, 99] = -10.0
    logits[-2, 99] = 5.0  # p(EOS|context-at-pos-2) ~ 0.97
    assert verify_termination(_commit(tokens), _FakeTokenizer(), logits) is True
