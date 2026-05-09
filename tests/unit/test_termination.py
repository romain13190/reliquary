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


# ---------------------------------------------------------------------
# Path 1 — max-length termination based on total context length.
# Honest miners running under a `max_model_len` ceiling (e.g. vLLM) cap
# at prompt_length + completion_length = max_model_len, so completion_length
# alone never reaches MAX_NEW_TOKENS_PROTOCOL_CAP. Path 1 must check the
# total to accept these.
# ---------------------------------------------------------------------


def _commit_with_lengths(tokens: list[int], prompt_length: int, completion_length: int) -> dict:
    return {
        "tokens": tokens,
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
        },
    }


def test_path1_accepts_max_model_len_bound_termination():
    """Miner with max_model_len=cap and prompt_length>0 hits prompt+compl=cap.
    Last token is not EOS (model drifted), p_stop is ~0 — Path 2 fails — but
    Path 1 must accept on total-length grounds."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length  # exactly fills the cap
    seq_len = prompt_length + completion_length
    # Token 42 is not EOS (eos_token_id=99 in _FakeTokenizer)
    tokens = [42] * seq_len
    # Logits assign vanishing probability to EOS — Path 2 would reject
    logits = torch.zeros(seq_len, 100)
    logits[:, 99] = -20.0
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), logits,
    ) is True


def test_path1_accepts_when_completion_alone_meets_cap():
    """Backwards-compat: a miner running pure max_new_tokens=cap (no
    max_model_len constraint) still passes Path 1."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 0
    completion_length = MAX_NEW_TOKENS_PROTOCOL_CAP
    seq_len = completion_length
    tokens = [42] * seq_len
    logits = torch.zeros(seq_len, 100)
    logits[:, 99] = -20.0  # Path 2 fails
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), logits,
    ) is True


def test_path1_rejects_short_truncation_below_cap():
    """A miner who truncates well below the cap and forges a non-EOS last
    token must be rejected — this is the gaming-safe property of Path 1."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    prompt_length = 33
    completion_length = 100  # total = 133, way below the cap
    seq_len = prompt_length + completion_length
    tokens = [42] * seq_len  # last token NOT EOS
    logits = torch.zeros(seq_len, 100)
    logits[:, 99] = -20.0  # p_stop ~ 0
    assert prompt_length + completion_length < MAX_NEW_TOKENS_PROTOCOL_CAP
    assert verify_termination(
        _commit_with_lengths(tokens, prompt_length, completion_length),
        _FakeTokenizer(), logits,
    ) is False
