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


from unittest.mock import MagicMock


def test_init_happy_path_with_mocked_wandb(monkeypatch):
    """With WANDB_API_KEY set and wandb.init mocked, telemetry activates
    and passes the expected kwargs."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    fake_wandb = MagicMock()
    fake_wandb.init.return_value = MagicMock(id="fakerun")
    # Inject fake wandb into sys.modules so the lazy `import wandb` inside
    # init() picks it up.
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    telemetry.init(hotkey_ss58=hotkey, config={"learning_rate": 1e-5})

    assert telemetry.is_active() is True
    fake_wandb.init.assert_called_once()
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["project"] == "reliquary-validator"
    assert kwargs["id"] == f"{hotkey[:8]}-v1"
    assert kwargs["resume"] == "allow"
    assert kwargs["config"]["learning_rate"] == 1e-5


def test_init_reads_wandb_project_env_override(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_PROJECT", "my-custom-project")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert fake_wandb.init.call_args.kwargs["project"] == "my-custom-project"


def test_init_reads_wandb_entity_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert fake_wandb.init.call_args.kwargs["entity"] == "my-team"


def test_init_version_env_override_changes_run_id(monkeypatch):
    """RELIQUARY_WANDB_VERSION overrides the constant, producing a
    different run id — this is how the operator starts a new run."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("RELIQUARY_WANDB_VERSION", "v9")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    telemetry.init(hotkey_ss58=hotkey, config={})

    assert fake_wandb.init.call_args.kwargs["id"] == f"{hotkey[:8]}-v9"
