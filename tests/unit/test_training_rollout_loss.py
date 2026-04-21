"""_rollout_loss + train_step — uses sshleifer/tiny-gpt2 for CPU testing.

These tests require transformers but not a GPU. They verify the math
runs end-to-end on a real (tiny) model.
"""

import pytest

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Preload attempt to fail fast if no network / no cache
    _check = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    del _check
except Exception:
    pytest.skip("tiny-gpt2 not available", allow_module_level=True)

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from reliquary.validator.training import (
    _rollout_loss, _compute_advantages, train_step, reset_training_state,
)


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Load sshleifer/tiny-gpt2 (≈ 500KB) for fast CPU tests."""
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    return model, tokenizer


@dataclass
class _FakeRollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _FakeGroup:
    rollouts: list
    prompt_idx: int = 0


def _build_rollout(tokens, reward, prompt_length):
    """Build a fake rollout with synthetic (but length-correct) old_logprobs."""
    n_completion = len(tokens) - prompt_length
    return _FakeRollout(
        tokens=tokens,
        reward=reward,
        commit={
            "rollout": {
                "prompt_length": prompt_length,
                "token_logprobs": [-1.0] * n_completion,  # arbitrary baseline
            },
        },
    )


def test_rollout_loss_zero_advantage_gives_zero_ppo_loss(tiny_model_and_tokenizer):
    """With advantage=0, both surr1 and surr2 are 0 → ppo_loss = 0."""
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    from reliquary.validator.training import _lazy_init
    _lazy_init(model)
    from reliquary.validator import training
    ref = training._ref_model

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6],
        reward=0.5,
        prompt_length=2,
    )
    device = next(model.parameters()).device
    ppo_loss, kl = _rollout_loss(model, ref, rollout, advantage=0.0, device=device)
    assert abs(ppo_loss.item()) < 1e-6
    # KL is advantage-independent; just check it's non-negative (by construction)
    assert kl.item() >= -1e-6


def test_rollout_loss_produces_finite_values(tiny_model_and_tokenizer):
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    from reliquary.validator.training import _lazy_init
    _lazy_init(model)
    from reliquary.validator import training
    ref = training._ref_model

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        reward=1.0,
        prompt_length=3,
    )
    device = next(model.parameters()).device
    ppo_loss, kl = _rollout_loss(model, ref, rollout, advantage=1.0, device=device)
    assert torch.isfinite(ppo_loss)
    assert torch.isfinite(kl)


def test_train_step_updates_optimizer(tiny_model_and_tokenizer):
    """train_step should take at least one optimizer step when there's signal."""
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], r, 2) for r in [1, 1, 0, 0]]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)

    # Take a snapshot of one parameter to verify it changed
    sample_param = next(model.parameters())
    before = sample_param.detach().clone()

    result = train_step(model, [group])
    assert result is model

    # Parameter should have changed (tiny-gpt2 is tiny, but non-zero grad)
    after = next(model.parameters()).detach().clone()
    diff = (before - after).abs().max().item()
    assert diff > 0.0, "expected some parameter change after optimizer step"


def test_train_step_empty_batch_noop(tiny_model_and_tokenizer):
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    result = train_step(model, [])
    assert result is model


def test_train_step_degenerate_groups_skipped(tiny_model_and_tokenizer):
    """All-same-reward groups contribute zero signal → no optimizer step."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    # All rewards identical → advantages all zero → no backward pass
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2) for _ in range(4)]
    group = _FakeGroup(rollouts=rollouts)

    before = next(model.parameters()).detach().clone()
    train_step(model, [group])
    after = next(model.parameters()).detach().clone()
    # No update should happen
    assert torch.equal(before, after)
