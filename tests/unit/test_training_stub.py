"""train_step — basic interface tests (no real model required).

These tests cover the stable interface contract: empty-batch is a no-op,
the model reference is always returned, and a non-empty batch with no
valid rollouts still returns cleanly. The full math is tested in
test_training_grpo.py and test_training_rollout_loss.py.
"""

import logging
from unittest.mock import MagicMock, patch

from reliquary.validator.training import train_step, reset_training_state


def test_train_step_with_empty_batch():
    reset_training_state()
    model = MagicMock()
    result = train_step(model=model, batch=[])
    assert result is model


def test_train_step_empty_batch_logs(caplog):
    reset_training_state()
    caplog.set_level(logging.INFO, logger="reliquary.validator.training")
    train_step(model=MagicMock(), batch=[])
    assert any("empty batch" in rec.message for rec in caplog.records)


def test_train_step_returns_model_on_all_degenerate_groups():
    """If every group is degenerate (all-same reward), no optimizer step
    and the original model is still returned."""
    reset_training_state()

    import torch
    import reliquary.validator.training as _t

    # Build a tiny linear model so _lazy_init and device resolution work.
    model = torch.nn.Linear(2, 2)

    # All rollouts have identical reward → advantages all zero → skipped.
    rollout = MagicMock()
    rollout.reward = 1.0
    group = MagicMock()
    group.rollouts = [rollout] * 4
    group.prompt_idx = 0

    result = train_step(model=model, batch=[group])
    assert result is model

    # Optimizer state should have been initialised but step not taken
    # (n_processed == 0 → early return before optimizer.step).
    assert _t._optimizer is not None
    # Verify the optimizer step count is still 0.
    assert _t._scheduler.last_epoch == 0
