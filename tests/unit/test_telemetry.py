"""Validator-side wandb telemetry — fail-soft, opt-in, no-op by default."""

import pytest

from reliquary.validator import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry(monkeypatch):
    """Ensure each test starts with WANDB_API_KEY unset and module state
    fresh — the module caches `_enabled` / `_run` as globals."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("RELIQUARY_WANDB_VERSION", raising=False)
    telemetry._reset_for_tests()
    yield
    telemetry._reset_for_tests()


def test_is_active_false_before_init():
    assert telemetry.is_active() is False


def test_init_noop_when_no_api_key():
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    assert telemetry.is_active() is False


def test_log_training_step_is_safe_when_disabled():
    telemetry.log_training_step({"train/lr": 1e-5}, step=3)  # must not raise


def test_finish_is_safe_when_disabled():
    telemetry.finish()  # must not raise
