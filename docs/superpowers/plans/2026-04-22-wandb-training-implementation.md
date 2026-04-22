# Wandb Telemetry for Validator Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in wandb telemetry to the validator's GRPO training step. Per-operator, fail-soft, no impact on validators that don't install the extra.

**Architecture:** New `reliquary/validator/telemetry.py` module with `init / log_training_step / finish / is_active` API. `wandb` is a lazy, optional import. Opt-in via `WANDB_API_KEY` env var. Run id derived from `{hotkey[:8]}-{version}` with `resume="allow"`, so restarts resume the same run and bumping `WANDB_TRAINING_VERSION` starts a new one. `training.py` and `service.py` are the only call sites.

**Tech Stack:** Python 3.11+, `wandb>=0.16` (optional dependency), `pytest` with monkeypatch for mocking.

**Spec:** `docs/superpowers/specs/2026-04-22-wandb-training-design.md`

---

## File Structure

**Create:**
- `reliquary/validator/telemetry.py` — module (~80 LoC)
- `tests/unit/test_telemetry.py` — all telemetry unit tests

**Modify:**
- `reliquary/constants.py` — add `WANDB_PROJECT`, `WANDB_TRAINING_VERSION`
- `reliquary/validator/training.py` — add `window_index` param to `train_step`, add `n_skipped` counter, build metrics dict, call `telemetry.log_training_step`
- `reliquary/validator/service.py` — call `telemetry.init` in `__init__`, pass `window_index` into `train_step`, call `telemetry.finish` in shutdown
- `pyproject.toml` — add `wandb` to `[project.optional-dependencies]`
- `tests/unit/test_constants.py` — assert new constants are present
- `tests/unit/test_training_stub.py` — assert train_step forwards metrics to telemetry

---

## Task 1: Add WANDB_PROJECT and WANDB_TRAINING_VERSION constants

**Files:**
- Modify: `reliquary/constants.py`
- Test: `tests/unit/test_constants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_constants.py`:

```python
def test_wandb_constants_present():
    assert C.WANDB_PROJECT == "reliquary-validator"
    assert C.WANDB_TRAINING_VERSION == "v1"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_constants.py::test_wandb_constants_present -v`
Expected: FAIL with `AttributeError: module 'reliquary.constants' has no attribute 'WANDB_PROJECT'`.

- [ ] **Step 3: Add the constants**

Append to `reliquary/constants.py` (group them together at the bottom with a one-line section header — follow whatever style the file already uses for grouping):

```python
# Wandb telemetry (opt-in, validator-only)
WANDB_PROJECT = "reliquary-validator"
WANDB_TRAINING_VERSION = "v1"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_constants.py::test_wandb_constants_present -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_constants.py
git commit -m "feat(constants): WANDB_PROJECT + WANDB_TRAINING_VERSION"
```

---

## Task 2: Telemetry module — disabled-path scaffold

Builds the empty shell: `is_active`, `init`, `log_training_step`, `finish`, all safe when `WANDB_API_KEY` is unset. No wandb import yet.

**Files:**
- Create: `reliquary/validator/telemetry.py`
- Create: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_telemetry.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary.validator.telemetry'`.

- [ ] **Step 3: Create the module scaffold**

Create `reliquary/validator/telemetry.py`:

```python
"""Validator-side wandb telemetry.

Opt-in: requires ``WANDB_API_KEY`` env var. Fails soft on any exception
and on missing ``wandb`` package. Never propagates to the validator main
loop. Run id is deterministic (``{hotkey[:8]}-{version}``) so restarts
resume the same wandb run; bump ``WANDB_TRAINING_VERSION`` (or set
``RELIQUARY_WANDB_VERSION``) to start a new run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from reliquary.constants import WANDB_PROJECT, WANDB_TRAINING_VERSION

logger = logging.getLogger(__name__)

_run: Any = None
_enabled: bool = False
_log_warned: bool = False  # suppress repeated log-path warnings


def is_active() -> bool:
    return _enabled


def init(hotkey_ss58: str, config: dict) -> None:
    """Initialise wandb if ``WANDB_API_KEY`` is set. No-op otherwise.

    Fail-soft: any exception (ImportError, network, auth) disables
    telemetry for the rest of the session and logs a warning.
    """
    global _run, _enabled
    if not os.getenv("WANDB_API_KEY"):
        logger.info("wandb: WANDB_API_KEY not set, telemetry disabled")
        return
    # Active paths added in later tasks.


def log_training_step(metrics: dict, step: int | None) -> None:
    """Forward a metrics dict to wandb.log. No-op if disabled. Fail-soft."""
    if not _enabled:
        return
    # Active path added in later task.


def finish() -> None:
    """Close the wandb run. No-op if disabled. Fail-soft."""
    if not _enabled:
        return
    # Active path added in later task.


def _reset_for_tests() -> None:
    """Test-only helper: reset module state so each test starts fresh."""
    global _run, _enabled, _log_warned
    _run = None
    _enabled = False
    _log_warned = False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): scaffold module with disabled no-op path"
```

---

## Task 3: Telemetry init — active path with mocked wandb

Covers the happy path: `WANDB_API_KEY` set, `wandb.init` succeeds, run id format is correct, `resume="allow"`.

**Files:**
- Modify: `reliquary/validator/telemetry.py`
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: 3 new tests FAIL (`is_active()` returns False because the active path is not implemented).

- [ ] **Step 3: Implement the active init path**

Edit `reliquary/validator/telemetry.py`, replace the body of `init()`:

```python
def init(hotkey_ss58: str, config: dict) -> None:
    """Initialise wandb if ``WANDB_API_KEY`` is set. No-op otherwise.

    Fail-soft: any exception (ImportError, network, auth) disables
    telemetry for the rest of the session and logs a warning.
    """
    global _run, _enabled
    if not os.getenv("WANDB_API_KEY"):
        logger.info("wandb: WANDB_API_KEY not set, telemetry disabled")
        return

    version = os.getenv("RELIQUARY_WANDB_VERSION", WANDB_TRAINING_VERSION)
    project = os.getenv("WANDB_PROJECT", WANDB_PROJECT)
    entity = os.getenv("WANDB_ENTITY")
    run_id = f"{hotkey_ss58[:8]}-{version}"

    try:
        import wandb  # lazy: optional dependency
        _run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="allow",
            config=config,
        )
        _enabled = True
        logger.info(
            "wandb: initialised (project=%s id=%s entity=%s)",
            project, run_id, entity or "<default>",
        )
    except Exception as e:  # noqa: BLE001 — fail-soft by design
        logger.warning("wandb: init failed (%s), telemetry disabled", e)
        _enabled = False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: all tests PASS (7 so far).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): init happy path — resume-by-id + env overrides"
```

---

## Task 4: Telemetry init — version env override starts a new run id

Covers the UX of bumping the version without touching code.

**Files:**
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/unit/test_telemetry.py::test_init_version_env_override_changes_run_id -v`
Expected: PASS (Task 3 already implements the override — this is a regression guard).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_telemetry.py
git commit -m "test(telemetry): lock in RELIQUARY_WANDB_VERSION override behaviour"
```

---

## Task 5: Telemetry init — fail-soft on wandb.init exception

**Files:**
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
def test_init_fail_soft_on_wandb_exception(monkeypatch, caplog):
    """If wandb.init raises (network, auth, quota), telemetry stays
    disabled and the validator keeps running. No exception propagates."""
    import logging as _logging
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    fake_wandb = MagicMock()
    fake_wandb.init.side_effect = RuntimeError("network down")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})  # must not raise

    assert telemetry.is_active() is False
    assert any("init failed" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/unit/test_telemetry.py::test_init_fail_soft_on_wandb_exception -v`
Expected: PASS (Task 3's `except Exception` already covers this — regression guard).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_telemetry.py
git commit -m "test(telemetry): lock in fail-soft on wandb.init exception"
```

---

## Task 6: Telemetry init — fail-soft on ImportError when wandb not installed

Validators that don't install `reliquary[wandb]` must keep working.

**Files:**
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
def test_init_fail_soft_when_wandb_not_installed(monkeypatch, caplog):
    """If `import wandb` raises ImportError (package not installed),
    telemetry stays disabled. No exception propagates."""
    import builtins
    import logging as _logging

    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert telemetry.is_active() is False
    assert any("init failed" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/unit/test_telemetry.py::test_init_fail_soft_when_wandb_not_installed -v`
Expected: PASS (the broad `except Exception` in `init()` catches `ImportError`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_telemetry.py
git commit -m "test(telemetry): lock in fail-soft when wandb isn't installed"
```

---

## Task 7: Telemetry log_training_step — forwards to wandb.log when active

**Files:**
- Modify: `reliquary/validator/telemetry.py`
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
def test_log_training_step_forwards_metrics(monkeypatch):
    """When active, log_training_step calls wandb.log with the same dict
    and step."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    assert telemetry.is_active() is True

    metrics = {"train/lr": 1e-5, "train/ppo_loss": 0.42}
    telemetry.log_training_step(metrics, step=7)

    fake_wandb.log.assert_called_once_with(metrics, step=7)


def test_log_training_step_accepts_none_step(monkeypatch):
    """step=None is valid — wandb picks its own counter."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.log_training_step({"x": 1}, step=None)

    fake_wandb.log.assert_called_once_with({"x": 1}, step=None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: 2 new tests FAIL (`wandb.log` never called — the function body is still a no-op).

- [ ] **Step 3: Implement the active log path**

Edit `reliquary/validator/telemetry.py`, replace `log_training_step`:

```python
def log_training_step(metrics: dict, step: int | None) -> None:
    """Forward a metrics dict to wandb.log. No-op if disabled. Fail-soft.

    Only the first exception is logged — subsequent failures in the same
    session are silenced to avoid flooding the log on prolonged outages.
    """
    global _log_warned
    if not _enabled:
        return
    try:
        import wandb  # already imported successfully in init()
        wandb.log(metrics, step=step)
    except Exception as e:  # noqa: BLE001
        if not _log_warned:
            logger.warning("wandb: log failed (%s), suppressing further warnings", e)
            _log_warned = True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): log_training_step forwards metrics to wandb.log"
```

---

## Task 8: Telemetry log_training_step — fail-soft on wandb.log exception

Guards against transient network issues mid-run.

**Files:**
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
def test_log_training_step_fail_soft_on_log_exception(monkeypatch, caplog):
    """If wandb.log raises, the call returns silently (first warning
    only) — the validator must keep training."""
    import logging as _logging
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    fake_wandb.log.side_effect = RuntimeError("network down")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.log_training_step({"x": 1}, step=1)  # must not raise
    telemetry.log_training_step({"x": 2}, step=2)  # must not raise

    warnings = [rec for rec in caplog.records if "log failed" in rec.message]
    assert len(warnings) == 1  # second failure is suppressed
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/unit/test_telemetry.py::test_log_training_step_fail_soft_on_log_exception -v`
Expected: PASS (Task 7's `try/except` + `_log_warned` flag already covers this — regression guard).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_telemetry.py
git commit -m "test(telemetry): lock in log-path fail-soft + warning suppression"
```

---

## Task 9: Telemetry finish — calls wandb.finish when active

**Files:**
- Modify: `reliquary/validator/telemetry.py`
- Modify: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_telemetry.py`:

```python
def test_finish_calls_wandb_finish_when_active(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.finish()

    fake_wandb.finish.assert_called_once()
    assert telemetry.is_active() is False  # state cleared


def test_finish_fail_soft_on_exception(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    fake_wandb.finish.side_effect = RuntimeError("boom")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.finish()  # must not raise
    assert telemetry.is_active() is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: 2 new tests FAIL (`fake_wandb.finish` never called; `is_active()` still True after `finish()`).

- [ ] **Step 3: Implement finish**

Edit `reliquary/validator/telemetry.py`, replace `finish`:

```python
def finish() -> None:
    """Close the wandb run. No-op if disabled. Fail-soft."""
    global _run, _enabled
    if not _enabled:
        return
    try:
        import wandb
        wandb.finish()
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb: finish failed (%s)", e)
    finally:
        _run = None
        _enabled = False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_telemetry.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): finish closes the wandb run and clears state"
```

---

## Task 10: train_step — emit metrics to telemetry

Adds an optional `window_index` parameter, a `n_skipped` counter for degenerate groups, builds a flat metrics dict, and calls `telemetry.log_training_step(metrics, step=window_index)` after a successful step.

**Files:**
- Modify: `reliquary/validator/training.py`
- Modify: `tests/unit/test_training_stub.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_training_stub.py`:

```python
def test_train_step_forwards_metrics_to_telemetry(monkeypatch):
    """When train_step completes a successful step, it calls
    telemetry.log_training_step with the GRPO metrics dict and the
    window_index as the step."""
    from unittest.mock import MagicMock
    import torch

    import reliquary.validator.training as _t
    from reliquary.validator import telemetry

    _t.reset_training_state()

    captured = {}

    def fake_log(metrics, step):
        captured["metrics"] = metrics
        captured["step"] = step

    monkeypatch.setattr(telemetry, "log_training_step", fake_log)

    # Build a tiny model and a batch with a non-degenerate group so a
    # real optimizer step runs. We reuse the same minimal rollout shape
    # the other stub tests use but with varied rewards.
    model = torch.nn.Linear(2, 2)

    def _mk_rollout(reward, prompt_len=1):
        r = MagicMock()
        r.reward = float(reward)
        r.tokens = [0, 1]
        r.commit = {"rollout": {"prompt_length": prompt_len, "token_logprobs": [0.0]}}
        return r

    group = MagicMock()
    group.rollouts = [_mk_rollout(1.0), _mk_rollout(0.0)]
    group.prompt_idx = 0

    # The test doesn't care whether the _rollout_loss forward pass
    # actually runs; monkeypatch it to return loss tensors connected to
    # model parameters (so loss.backward() works) but numerically zero.
    # Must return fresh tensors each call — the graph is freed after each
    # backward pass.
    def fake_loss(model, ref_model, rollout, advantage, device):
        p = next(model.parameters())
        zero_loss = (p.sum() * 0.0)
        return zero_loss, zero_loss.detach()

    monkeypatch.setattr(_t, "_rollout_loss", fake_loss)

    _t.train_step(model=model, batch=[group], window_index=7)

    assert captured["step"] == 7
    m = captured["metrics"]
    for key in (
        "train/lr", "train/ppo_loss", "train/kl", "train/grad_norm",
        "train/rollouts_processed", "train/rollouts_total",
        "train/valid_rollout_ratio",
        "rewards/mean", "rewards/std", "rewards/min", "rewards/max",
        "batch/n_groups", "batch/n_degenerate_groups", "batch/degenerate_ratio",
    ):
        assert key in m, f"missing metric {key}"
    assert m["batch/n_groups"] == 1
    assert m["batch/n_degenerate_groups"] == 0
    assert m["rewards/min"] == 0.0
    assert m["rewards/max"] == 1.0


def test_train_step_counts_degenerate_groups(monkeypatch):
    """A batch of only degenerate groups reports n_degenerate_groups ==
    n_groups and does not emit metrics (no successful step — early
    return before the metrics branch)."""
    from unittest.mock import MagicMock
    import torch

    import reliquary.validator.training as _t
    from reliquary.validator import telemetry

    _t.reset_training_state()

    called = []
    monkeypatch.setattr(
        telemetry, "log_training_step",
        lambda metrics, step: called.append((metrics, step)),
    )

    model = torch.nn.Linear(2, 2)
    rollout = MagicMock()
    rollout.reward = 1.0
    group = MagicMock()
    group.rollouts = [rollout] * 4
    group.prompt_idx = 0

    _t.train_step(model=model, batch=[group], window_index=3)

    # No successful step → no metrics emitted.
    assert called == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_training_stub.py::test_train_step_forwards_metrics_to_telemetry tests/unit/test_training_stub.py::test_train_step_counts_degenerate_groups -v`
Expected: FAIL — `train_step` does not accept `window_index` and does not call `telemetry.log_training_step`.

- [ ] **Step 3: Modify train_step**

Edit `reliquary/validator/training.py`:

1. Add import near the top (after the `torch` imports):

```python
from reliquary.validator import telemetry
```

2. Change the signature of `train_step` (line 205) to accept an optional `window_index`:

```python
def train_step(model, batch: list, window_index: int | None = None) -> Any:
```

3. Update the docstring's first line to mention the new param (keep the rest of the docstring):

```python
    """Run one GRPO step on *batch* (list of ValidSubmission).

    If *window_index* is provided, it is used as the wandb step when
    telemetry is enabled — aligning the x-axis of wandb charts with the
    subnet window index. Safe to omit in tests.
    ...
    """
```

4. Replace the body of the `for group in batch:` loop's degenerate branch and the final block (after the optimizer step). Here is the exact replacement, covering from line 230 (the `n_total_rollouts = ...` line) through the end of the function:

```python
    n_total_rollouts = sum(len(g.rollouts) for g in batch)
    total_ppo = 0.0
    total_kl = 0.0
    n_processed = 0
    n_skipped = 0  # degenerate groups (std==0 → no signal)

    for group in batch:
        rewards = [r.reward for r in group.rollouts]
        advantages = _compute_advantages(rewards)
        if all(a == 0.0 for a in advantages):
            n_skipped += 1
            logger.debug("skipping degenerate group on prompt_idx=%d", group.prompt_idx)
            continue

        for rollout, adv in zip(group.rollouts, advantages):
            try:
                ppo_loss, kl = _rollout_loss(
                    model=model, ref_model=_ref_model,
                    rollout=rollout, advantage=adv, device=device,
                )
            except ValueError as e:
                logger.warning("rollout skipped: %s", e)
                continue
            loss = (ppo_loss + KL_BETA * kl) / n_total_rollouts
            loss.backward()
            total_ppo += ppo_loss.item()
            total_kl += kl.item()
            n_processed += 1

    if n_processed == 0:
        logger.info("train_step: no valid rollouts processed")
        return model

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    _optimizer.step()
    _scheduler.step()
    lr = _scheduler.get_last_lr()[0]

    logger.info(
        "train_step: lr=%.2e ppo=%.4f kl=%.4f grad_norm=%.3f rollouts=%d/%d",
        lr, total_ppo / n_processed, total_kl / n_processed,
        float(grad_norm), n_processed, n_total_rollouts,
    )

    # Emit structured metrics to wandb (no-op if telemetry disabled).
    all_rewards = [r.reward for g in batch for r in g.rollouts]
    n_rewards = len(all_rewards)
    reward_mean = sum(all_rewards) / n_rewards
    reward_var = sum((r - reward_mean) ** 2 for r in all_rewards) / n_rewards
    reward_std = reward_var ** 0.5
    n_groups = len(batch)
    metrics = {
        "train/lr": lr,
        "train/ppo_loss": total_ppo / n_processed,
        "train/kl": total_kl / n_processed,
        "train/grad_norm": float(grad_norm),
        "train/rollouts_processed": n_processed,
        "train/rollouts_total": n_total_rollouts,
        "train/valid_rollout_ratio": n_processed / n_total_rollouts,
        "rewards/mean": reward_mean,
        "rewards/std": reward_std,
        "rewards/min": min(all_rewards),
        "rewards/max": max(all_rewards),
        "batch/n_groups": n_groups,
        "batch/n_degenerate_groups": n_skipped,
        "batch/degenerate_ratio": n_skipped / n_groups,
    }
    telemetry.log_training_step(metrics, step=window_index)

    return model
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest tests/unit/test_training_stub.py -v`
Expected: all tests in the file PASS (including pre-existing ones — signatures and behaviour unchanged for callers that omit `window_index`).

- [ ] **Step 5: Run the full training test suite to catch regressions**

Run: `pytest tests/unit/test_training_stub.py tests/unit/test_training_grpo.py tests/unit/test_training_rollout_loss.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/training.py tests/unit/test_training_stub.py
git commit -m "feat(training): emit GRPO metrics to telemetry per window"
```

---

## Task 11: Wire telemetry into ValidatorService

Initialise telemetry at service construction time, pass `window_index` into `train_step`, and close the run on shutdown.

**Files:**
- Modify: `reliquary/validator/service.py`

- [ ] **Step 1: Read the current service.py head and shutdown path**

Run: `grep -n "def __init__\|self.wallet = wallet\|train_step(\|def stop\|def shutdown\|def close\|asyncio.CancelledError" reliquary/validator/service.py`
Expected: shows `__init__` around line 84-97, the `train_step(self.model, batch)` call around line 225, and whatever stop/shutdown hook exists. Note the exact line numbers — use them in the edits below.

- [ ] **Step 2: Add the telemetry import**

At the top of `reliquary/validator/service.py`, with the other `reliquary.validator` imports (near line 29), add:

```python
from reliquary.validator import telemetry
```

- [ ] **Step 3: Add telemetry.init() call in __init__**

In `ValidatorService.__init__`, **immediately after** `self.wallet = wallet` (line 97), insert:

```python
        telemetry.init(
            hotkey_ss58=wallet.hotkey.ss58_address,
            config={
                "learning_rate": LEARNING_RATE,
                "kl_beta": KL_BETA,
                "ppo_clip_epsilon": PPO_CLIP_EPSILON,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "lr_warmup_windows": LR_WARMUP_WINDOWS,
                "lr_cosine_max_windows": LR_COSINE_MAX_WINDOWS,
                "b_batch": B_BATCH,
                "m_rollouts_per_prompt": M_ROLLOUTS_PER_PROMPT,
                "window_length": WINDOW_LENGTH,
                "wandb_training_version": WANDB_TRAINING_VERSION,
            },
        )
```

Add any constants not already imported at the top of the file (verify with grep — `LEARNING_RATE`, `KL_BETA`, `PPO_CLIP_EPSILON`, `GRAD_CLIP_NORM`, `LR_WARMUP_WINDOWS`, `LR_COSINE_MAX_WINDOWS`, `M_ROLLOUTS_PER_PROMPT`, `WANDB_TRAINING_VERSION`). Merge them into the existing `from reliquary.constants import (...)` block alphabetically.

- [ ] **Step 4: Pass window_index to train_step**

In `_train_and_publish`, change the line (around 225):

```python
            self.model = train_step(self.model, batch)
```

to:

```python
            self.model = train_step(self.model, batch, window_index=self._window_n)
```

- [ ] **Step 5: Call telemetry.finish() on shutdown**

Find the service's shutdown path (from Step 1). If there's a `stop()` / `shutdown()` / `close()` method, add `telemetry.finish()` at the end of its body (in a `try/finally` if the method has other teardown calls).

If there is no explicit shutdown method, register an `atexit` handler at the bottom of `ValidatorService.__init__` — **after** the `telemetry.init(...)` call:

```python
        import atexit
        atexit.register(telemetry.finish)
```

Prefer the explicit shutdown path if it exists; `atexit` is the fallback.

- [ ] **Step 6: Run the validator tests to verify nothing regresses**

Run: `pytest tests/unit/ -v -x`
Expected: all PASS. Existing service tests (if any) must keep passing — telemetry.init is a no-op without `WANDB_API_KEY`.

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/service.py
git commit -m "feat(validator): wire telemetry init/finish + window_index into train_step"
```

---

## Task 12: Declare wandb as an optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject.toml**

In `pyproject.toml`, extend the `[project.optional-dependencies]` section:

Before:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

After:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
wandb = [
    "wandb>=0.16",
]
```

- [ ] **Step 2: Verify the package still installs and resolves the extra**

Run: `pip install -e ".[wandb]" --dry-run`
Expected: resolves `wandb>=0.16` and its dependencies without errors.

(If the local env doesn't have network, skip this step — the TOML edit is syntactically trivial and the install command is sufficient verification at CI time.)

- [ ] **Step 3: Run the full test suite one final time**

Run: `pytest tests/ -v`
Expected: all tests PASS. (No wandb imports happen at import time, so the extra does not need to be installed for tests to pass.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: wandb as optional extra (pip install reliquary[wandb])"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS. If anything fails, fix before reporting the feature done.

- [ ] **Step 2: Manual smoke test on your validator (optional, requires a wandb account)**

```bash
pip install -e ".[wandb]"
export WANDB_API_KEY=<your key>
# Optionally: export WANDB_ENTITY=<your team>
reliquary validate ...  # or whatever your normal launch command is
```

Expected: log line `wandb: initialised (project=reliquary-validator id=<hotkey-prefix>-v1 entity=<default|...>)`. After the first full window trains, metrics appear in the wandb UI under run `<hotkey-prefix>-v1`. Restart the validator — the same run resumes. Set `RELIQUARY_WANDB_VERSION=v2` and relaunch — a new run appears.

- [ ] **Step 3: Confirm disabled path still works**

```bash
unset WANDB_API_KEY
reliquary validate ...
```

Expected: log line `wandb: WANDB_API_KEY not set, telemetry disabled`. Validator runs as before; no wandb calls happen.
