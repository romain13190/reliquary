# Wandb integration for validator training

**Date:** 2026-04-22
**Status:** Draft — awaiting user review
**Scope:** Validator-side GRPO training telemetry only.

## Motivation

The GRPO training step in `reliquary/validator/training.py` currently emits a single log line per window (`lr / ppo / kl / grad_norm / rollouts`). That's enough to tail a terminal but not enough to inspect training health over days or compare runs. Adding Weights & Biases gives a persistent, queryable record of training progression.

This is **opt-in** and per-operator: each validator opts in on its own machine by setting `WANDB_API_KEY`. The subnet does not mandate wandb, does not share a project, and the validator must keep running if wandb is unreachable or uninstalled.

## Non-goals

- No changes to persistence of optimizer state or reference model. Restarts still reset `_optimizer` / `_ref_model`; wandb runs may show a visible discontinuity at restart. This is accepted.
- No telemetry on the miner side. Only validator training.
- No shared / central wandb project. Each operator logs to their own.
- No tracking of verifier, server, or submission paths. Only `train_step`.

## Design

### Module layout

New module `reliquary/validator/telemetry.py` (~60 LoC). `training.py` and `service.py` are the only call sites. No other file imports `wandb`.

```
service.py  ── telemetry.init(hotkey, config)   # once at startup
              telemetry.finish()                 # at shutdown
training.py ── telemetry.log_training_step(metrics, step=window_index)
```

`wandb` is imported lazily inside `telemetry.init()`. If the package is not installed, init logs a warning and the module stays in disabled state — no ImportError propagates.

### Public API

```python
# reliquary/validator/telemetry.py

def init(hotkey_ss58: str, config: dict) -> None:
    """No-op if WANDB_API_KEY is unset. Otherwise wandb.init with a
    deterministic run id derived from hotkey + version. Fail-soft: any
    exception disables telemetry for the session."""

def log_training_step(metrics: dict, step: int | None) -> None:
    """No-op if init did not activate. wandb.log(metrics, step=step).
    Fail-soft."""

def finish() -> None:
    """wandb.finish() if active, else no-op."""

def is_active() -> bool:
    """True iff init succeeded."""
```

### Config

New constants in `reliquary/constants.py`:

```python
WANDB_PROJECT = "reliquary-validator"
WANDB_TRAINING_VERSION = "v1"
```

Environment variables read by `telemetry.init()`:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | **Single opt-in gate.** Absent → telemetry disabled, silent no-op. |
| `WANDB_PROJECT` | Override `WANDB_PROJECT` constant. Optional. |
| `WANDB_ENTITY` | Wandb user/team. Optional. |
| `RELIQUARY_WANDB_VERSION` | Override `WANDB_TRAINING_VERSION`. Optional. |

### Run identity

```python
run_id = f"{hotkey_ss58[:8]}-{version}"
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    id=run_id,
    resume="allow",
    config=config,  # see "Config snapshot sent to wandb.init" below
)
```

`resume="allow"` means: create the run on first call, resume the same run on subsequent restarts as long as `run_id` is stable. The id is stable across restarts because it is derived from the hotkey (constant) and the version constant (operator-controlled).

**To start a fresh run, bump `WANDB_TRAINING_VERSION`** (in `constants.py`) or set `RELIQUARY_WANDB_VERSION=v2` at launch.

**Known behavior:** the validator loses optimizer momentum and the frozen reference model at restart (they are module-level globals in `training.py`). Wandb will continue the same run across restarts, but training curves will show a visible discontinuity. This is accepted — widening the scope to persist optimizer state is out of scope.

### Config snapshot sent to `wandb.init`

```python
{
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
    "reliquary_version": importlib.metadata.version("reliquary"),
}
```

### Metrics logged per training step

`train_step` gains an optional parameter `window_index: int | None = None`. At the end of a successful step (current log line is already there), it builds a metrics dict and calls `telemetry.log_training_step(metrics, step=window_index)`.

```python
{
    # existing scalars
    "train/lr": lr,
    "train/ppo_loss": total_ppo / n_processed,
    "train/kl": total_kl / n_processed,
    "train/grad_norm": float(grad_norm),
    "train/rollouts_processed": n_processed,
    "train/rollouts_total": n_total_rollouts,
    "train/valid_rollout_ratio": n_processed / n_total_rollouts,

    # reward distribution across the batch
    "rewards/mean": mean(all_rewards),
    "rewards/std": std(all_rewards),
    "rewards/min": min(all_rewards),
    "rewards/max": max(all_rewards),

    # batch health
    "batch/n_groups": len(batch),
    "batch/n_degenerate_groups": n_skipped,
    "batch/degenerate_ratio": n_skipped / len(batch),
}
```

`all_rewards = [r.reward for g in batch for r in g.rollouts]` — flat across the batch.

`n_skipped` requires a counter in the existing degenerate-group branch of `train_step` (training.py:238) — increment when `all(a == 0.0 for a in advantages)`.

If `window_index is None` (tests with no caller passing it), `wandb.log` is called without `step=` and wandb picks its own monotonic counter. No failure.

### Lifecycle & fail-soft

**`init()` call site:** `ValidatorService.__init__` in `reliquary/validator/service.py`, right after `self.wallet = wallet` (line 97). At that point hotkey + all constants are available.

**`finish()` call site:** the service shutdown path. If the validator crashes without calling `finish()`, wandb's server-side keepalive still closes the run. Not critical.

**Fail-soft behavior — any of:**

- `WANDB_API_KEY` unset → `init()` logs info, returns silently, `_enabled = False`.
- `wandb` package not installed → ImportError caught in `init()`, warning logged, `_enabled = False`.
- `wandb.init()` raises (network, auth, quota) → exception caught, warning logged, `_enabled = False`.
- `wandb.log()` raises → caught, warning logged **once** (suppress subsequent warnings with a flag to avoid log spam), step skipped.
- `wandb.finish()` raises → caught, warning logged.

None of these ever propagate to the validator main loop.

### Dependency

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
wandb = ["wandb>=0.16"]
```

Install via `pip install "reliquary[wandb]"`. Validators that do not install the extra pay zero runtime cost — the lazy import inside `init()` raises ImportError, which is caught.

## Testing

New `tests/unit/test_telemetry.py`:

1. `test_init_noop_when_no_api_key` — unset `WANDB_API_KEY`, `init()` returns silently, `is_active() == False`, subsequent `log_training_step` and `finish` do not raise.
2. `test_init_fail_soft_on_wandb_exception` — monkeypatch `wandb.init` to raise; `is_active() == False`, no exception propagates.
3. `test_init_fail_soft_on_import_error` — simulate `wandb` not installed; `is_active() == False`.
4. `test_init_run_id_format` — set api key + hotkey + version, mock `wandb.init`, assert the `id=` kwarg matches `{hotkey[:8]}-{version}` and `resume="allow"`.
5. `test_init_version_env_override` — `RELIQUARY_WANDB_VERSION=v9` changes the run id.
6. `test_log_step_forwards_metrics` — active init, `log_training_step({"a": 1}, step=42)` calls `wandb.log({"a": 1}, step=42)`.
7. `test_log_step_noop_when_disabled` — inactive init, `log_training_step` does nothing, no exception.

Addition to `tests/unit/test_training.py` (or new file `test_training_telemetry.py`):

8. `test_train_step_logs_telemetry_when_enabled` — monkeypatch `telemetry.log_training_step`, run a train_step on a fake batch with `window_index=7`, assert the logger was called with a dict containing `train/ppo_loss`, `rewards/mean`, `batch/n_degenerate_groups`, and `step=7`.

No integration test against real wandb. The critical path is the fail-soft envelope and metric construction — both covered by unit tests.

## Impact on existing code

- `reliquary/constants.py` — two new constants.
- `reliquary/validator/telemetry.py` — new module (~60 LoC).
- `reliquary/validator/training.py` — add optional `window_index` param to `train_step`, add `n_skipped` counter in the degenerate branch, build metrics dict, one call to `telemetry.log_training_step`. The bulk of the existing logic is untouched.
- `reliquary/validator/service.py` — add `telemetry.init(...)` in `__init__`, pass `window_index=self._window_n` in the existing `train_step(...)` call, call `telemetry.finish()` in shutdown.
- `pyproject.toml` — new optional-dependency group `wandb`.

Existing `train_step` tests use `MagicMock` models; `window_index` is optional and telemetry is disabled by default in tests → zero test churn.

## Out-of-scope / future work

- Persisting optimizer/ref-model to disk to remove restart discontinuity.
- Logging on miner (rollout quality, generation time).
- Logging on verifier (pass rates, GRAIL rejection reasons).
- Per-layer gradient histograms, per-token KL, activation stats (option C from brainstorming).
- Artifact uploads (checkpoints to wandb). The subnet already ships checkpoints via R2; wandb artifact duplication adds bandwidth for no gain today.
