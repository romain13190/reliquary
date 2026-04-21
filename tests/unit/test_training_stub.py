"""train_step stub — does not modify weights, returns model unchanged."""

import logging
from unittest.mock import MagicMock

from reliquary.validator.training import train_step


def test_train_step_returns_same_model():
    model = MagicMock(name="model")
    batch = [MagicMock(name="batch_member") for _ in range(8)]
    result = train_step(model=model, batch=batch)
    assert result is model


def test_train_step_with_empty_batch():
    model = MagicMock()
    result = train_step(model=model, batch=[])
    assert result is model


def test_train_step_logs_batch_size(caplog):
    caplog.set_level(logging.INFO)
    train_step(model=MagicMock(), batch=[MagicMock() for _ in range(5)])
    assert any("5" in rec.message for rec in caplog.records)
