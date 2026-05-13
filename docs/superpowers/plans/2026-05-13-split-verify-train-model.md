# Split verify_model / train_model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the model used by `verify_commitment_proofs` from the model trained by `train_step`, so miners' GRAIL commits (computed against the last *published* checkpoint H_pub) are always verified against exactly H_pub, regardless of how many train_steps the validator has executed since.

**Architecture:** `ValidationService` holds two model objects on cuda:0:
- `self.verify_model` — frozen (eval mode, `requires_grad=False`) snapshot of the last published checkpoint. Used by the batcher (`verify_commitment_proofs`, `verify_termination`) AND as the KL reference inside `train_step`.
- `self.train_model` — trainable copy. Mutated by `train_step` every window.

At construction, `verify_model = deepcopy(train_model)`. After every successful publish (which uploads `train_model` to HF), the validator copies the trained weights back into `verify_model` *in place* via `load_state_dict`, so VRAM is not re-allocated. Between publishes (windows 1-9 of every 10), `verify_model` remains stable while `train_model` drifts forward.

This also obsoletes the `_ref_model = copy.deepcopy(model)` deep-copy currently done inside `training.py:_lazy_init` — the KL reference is now `verify_model`, passed in from the service.

**Tech Stack:** Python 3.11, PyTorch (bf16, cuda:0), transformers `AutoModelForCausalLM`, HuggingFace Hub for publish, pytest for tests.

---

## File map

- `reliquary/validator/service.py` — owns both model objects; orchestrates train + publish + verify-model refresh.
- `reliquary/validator/training.py` — drop the internal `_ref_model` deep-copy; `train_step` now takes `ref_model` as a required kwarg.
- `reliquary/validator/batcher.py` — no semantic change (already takes a single `model` param); the value it receives is now `verify_model`.
- `tests/unit/test_training_rollout_loss.py` — update `train_step` calls to pass `ref_model` explicitly; add a test that `_lazy_init` no longer deep-copies.
- `tests/integration/test_v21_window_loop.py` — add a test that verifies `verify_model` is refreshed at publish time (and not before).
- `tests/unit/test_service_v2.py` — assert `verify_model` and `train_model` are separate objects after construction.
- `reliquary/cli/main.py` — no change (still passes a single `model`; service handles cloning).

---

### Task 1: Update `training.py` so `train_step` takes ref_model as a required kwarg

**Files:**
- Modify: `reliquary/validator/training.py`
- Modify: `tests/unit/test_training_rollout_loss.py`

**Rationale:** Move the responsibility of providing the KL reference out of `training.py` (which currently does `copy.deepcopy(model)` once and never refreshes) and into the caller (`ValidationService`, which will pass `self.verify_model`). This removes a stale-reference bug independently of the rest of the refactor.

- [ ] **Step 1: Write the failing test for the new `train_step` signature**

Edit `tests/unit/test_training_rollout_loss.py`. Replace the four `train_step(...)` calls and add a signature test:

```python
def test_train_step_requires_ref_model_kwarg(tiny_model_and_tokenizer):
    """train_step must receive ref_model as an explicit kwarg — it no longer
    deep-copies internally."""
    import inspect
    sig = inspect.signature(train_step)
    assert "ref_model" in sig.parameters
    assert sig.parameters["ref_model"].kind == inspect.Parameter.KEYWORD_ONLY


def test_train_step_uses_caller_provided_ref(tiny_model_and_tokenizer):
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    import copy
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], r, 2) for r in [1, 1, 0, 0]]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)
    before = next(model.parameters()).detach().clone()
    train_step(model, [group], ref_model=ref)
    after = next(model.parameters()).detach().clone()
    assert (before - after).abs().max().item() > 0.0
```

Also update the existing four `train_step(model, ...)` calls in this file:
- `test_train_step_updates_optimizer`: `train_step(model, [group])` → `train_step(model, [group], ref_model=_make_ref(model))`
- `test_train_step_empty_batch_noop`: `train_step(model, [])` → `train_step(model, [], ref_model=_make_ref(model))`
- `test_train_step_degenerate_groups_skipped`: same pattern.

Add this helper near the top of the test file (just below the imports):

```python
def _make_ref(model):
    import copy
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref
```

Also: the two `_rollout_loss` tests currently access `training._ref_model` (a module global that will disappear). Replace them with locally-constructed refs:

```python
def test_rollout_loss_zero_advantage_gives_zero_ppo_loss(tiny_model_and_tokenizer):
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6],
        reward=0.5,
        prompt_length=2,
    )
    device = next(model.parameters()).device
    ppo_loss, kl = _rollout_loss(model, ref, rollout, advantage=0.0, device=device)
    assert abs(ppo_loss.item()) < 1e-6
    assert kl.item() >= -1e-6


def test_rollout_loss_produces_finite_values(tiny_model_and_tokenizer):
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        reward=1.0,
        prompt_length=3,
    )
    device = next(model.parameters()).device
    ppo_loss, kl = _rollout_loss(model, ref, rollout, advantage=1.0, device=device)
    assert torch.isfinite(ppo_loss)
    assert torch.isfinite(kl)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/unit/test_training_rollout_loss.py::test_train_step_requires_ref_model_kwarg tests/unit/test_training_rollout_loss.py::test_train_step_uses_caller_provided_ref -v`

Expected: FAIL (the current `train_step` doesn't have a `ref_model` kwarg).

- [ ] **Step 3: Refactor `training.py` — remove `_ref_model` global, make `ref_model` a required kwarg**

Edit `reliquary/validator/training.py`:

Remove `_ref_model` and `_model_id` from the module-level globals. Update `_lazy_init` to no longer build a reference model:

```python
# Module-global state — persists across train_step calls for the same model
_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
_optimizer_model_id: Optional[int] = None
```

Replace `_lazy_init` with:

```python
def _lazy_init(model) -> bool:
    """Create optimizer + scheduler on first call for a given model. No-op
    on subsequent calls with the same model. The reference model used for
    KL is no longer built here — it's passed in by the caller (typically
    ``ValidationService.verify_model``) and refreshed externally on each
    publish.
    """
    global _optimizer, _scheduler, _optimizer_model_id
    if _optimizer_model_id == id(model):
        return True

    try:
        params = list(model.parameters())
    except (AttributeError, TypeError):
        logger.warning("_lazy_init: model has no .parameters(); skipping init")
        return False
    if not params:
        logger.warning("_lazy_init: model.parameters() is empty; skipping init")
        return False

    _optimizer = _build_optimizer(params)

    def _lr_lambda(step: int) -> float:
        if step < LR_WARMUP_WINDOWS:
            return (step + 1) / LR_WARMUP_WINDOWS
        progress = (step - LR_WARMUP_WINDOWS) / max(
            1, LR_COSINE_MAX_WINDOWS - LR_WARMUP_WINDOWS
        )
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    _scheduler = torch.optim.lr_scheduler.LambdaLR(_optimizer, _lr_lambda)
    _optimizer_model_id = id(model)
    logger.info("Training state initialised (optimizer, scheduler)")
    return True
```

Replace `reset_training_state` with:

```python
def reset_training_state() -> None:
    """Clear the module-level singletons. Used by tests to start fresh.

    Production code should never call this — it throws away optimiser
    momentum.
    """
    global _optimizer, _scheduler, _optimizer_model_id
    _optimizer = None
    _scheduler = None
    _optimizer_model_id = None
```

Update `train_step` signature and body:

```python
def train_step(
    model,
    batch: list,
    *,
    ref_model,
    window_index: int | None = None,
) -> Any:
    """Run one GRPO step on *batch* (list of ValidSubmission).

    *ref_model* is the frozen reference policy for the KL penalty. The
    caller is responsible for keeping it up to date (in production:
    ``ValidationService.verify_model``, refreshed at every successful
    publish).

    *window_index* is used as the wandb step when telemetry is enabled.
    Safe to omit in tests.
    """
    if not batch:
        logger.info("train_step: empty batch, skipping")
        return model

    if not _lazy_init(model):
        logger.info("train_step: model not initializable (non-torch?), skipping")
        return model
    assert _optimizer is not None and _scheduler is not None

    model.train()
    device = next(model.parameters()).device

    _optimizer.zero_grad()

    n_total_rollouts = sum(len(g.rollouts) for g in batch)
    total_ppo = 0.0
    total_kl = 0.0
    n_processed = 0
    n_skipped = 0

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
                    model=model, ref_model=ref_model,
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

Also delete the `import copy` at the top of the file (no longer used).

- [ ] **Step 4: Run training tests to verify they pass**

Run: `pytest tests/unit/test_training_rollout_loss.py -v`
Expected: 9 passed (5 original + 2 chunked-logprobs tests from earlier + 2 new ref-kwarg tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/training.py tests/unit/test_training_rollout_loss.py
git commit -m "refactor(training): make ref_model a required kwarg on train_step

Removes the module-level _ref_model deep-copy from _lazy_init. The KL
reference policy is now provided by the caller, which lets the validator
service refresh it at every publish (instead of freezing it forever at
the first train_step). No semantic change to the math — _rollout_loss
already took ref_model as a parameter."
```

---

### Task 2: Add `verify_model` and `train_model` to `ValidationService`

**Files:**
- Modify: `reliquary/validator/service.py:142-222` (the `__init__` body)
- Modify: `tests/unit/test_service_v2.py`

**Rationale:** Carve out the two-model split inside the service constructor. Don't wire it into the batcher or training loop yet — that's Tasks 3 and 4. This step is just data structure.

- [ ] **Step 1: Write the failing test asserting both models exist as separate objects**

Edit `tests/unit/test_service_v2.py`. Append at the end of the file:

```python
def test_service_has_separate_verify_and_train_models():
    """ValidationService keeps verify_model and train_model as distinct
    PyTorch objects. The verify model is frozen (eval mode,
    requires_grad=False); the train model is trainable.
    """
    import torch.nn as nn
    from reliquary.validator.service import ValidationService

    train = nn.Linear(4, 4)
    svc = ValidationService(
        wallet=MagicMock(hotkey=MagicMock(ss58_address="x")),
        model=train,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    assert svc.train_model is train
    assert svc.verify_model is not train
    assert all(not p.requires_grad for p in svc.verify_model.parameters())
    assert not svc.verify_model.training  # eval mode

    # In-place refresh works: mutate train, copy into verify, check.
    import torch
    with torch.no_grad():
        for p in svc.train_model.parameters():
            p.add_(1.0)
    svc.verify_model.load_state_dict(svc.train_model.state_dict())
    for p_t, p_v in zip(svc.train_model.parameters(), svc.verify_model.parameters()):
        assert torch.equal(p_t, p_v)
```

The `gradient_checkpointing_enable` call inside `__init__` will log a warning for `nn.Linear` (no checkpointing support) — harmless, the `except (AttributeError, NotImplementedError)` swallow it.

- [ ] **Step 2: Run the new test to verify it fails**

Run: `pytest tests/unit/test_service_v2.py::test_service_has_separate_verify_and_train_models -v`
Expected: FAIL (`AttributeError: 'ValidationService' object has no attribute 'train_model'` or `verify_model`).

- [ ] **Step 3: Edit `ValidationService.__init__` in `service.py`**

Replace the block at `service.py:182-190`:

```python
        self.model = model
        # Enable gradient checkpointing to reduce activation memory.
        # Harmless if already enabled or unsupported by the model.
        try:
            self.model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            logger.warning(
                "model does not support gradient_checkpointing_enable"
            )
```

with:

```python
        import copy
        # Two-model architecture (see docs/superpowers/plans/2026-05-13-...).
        # train_model: trainable, mutated by train_step every window.
        # verify_model: frozen snapshot of the last published checkpoint —
        # used by batcher.verify_commitment_proofs and as the KL reference
        # inside train_step. Refreshed in-place after every successful
        # publish via load_state_dict.
        self.train_model = model
        if model is not None:
            try:
                self.verify_model = copy.deepcopy(model)
                self.verify_model.eval()
                for p in self.verify_model.parameters():
                    p.requires_grad = False
            except (AttributeError, TypeError):
                # Test fixtures (e.g. MagicMock) — fall back to sharing the
                # same object. Tests don't exercise the train/verify split
                # in this case.
                self.verify_model = model
        else:
            self.verify_model = None

        # Enable gradient checkpointing on the train model only.
        try:
            self.train_model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            logger.warning(
                "train_model does not support gradient_checkpointing_enable"
            )
```

Keep `self.model` as a *property* aliased to `self.train_model` for one task, so other call sites still compile while we migrate them in Task 3. Add this right below `__init__`:

```python
    @property
    def model(self):
        """Deprecated. Use ``self.train_model`` or ``self.verify_model``
        explicitly. Kept as an alias during the verify/train split
        migration; will be removed once all call sites are updated."""
        return self.train_model

    @model.setter
    def model(self, value):
        self.train_model = value
```

- [ ] **Step 4: Run all service tests**

Run: `pytest tests/unit/test_service_v2.py tests/integration/test_v21_window_loop.py -v`
Expected: All passing (existing tests still use `svc.model` via the property; new test passes).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_service_v2.py
git commit -m "feat(validator): introduce verify_model / train_model split

ValidationService now holds two model objects on construction:
- self.train_model: trainable, mutated by train_step.
- self.verify_model: frozen deep-copy of the initial weights.

self.model is kept as a temporary property alias for back-compat;
call sites migrate in the next commit."
```

---

### Task 3: Wire `verify_model` into the batcher; wire `verify_model` as KL ref into `train_step`

**Files:**
- Modify: `reliquary/validator/service.py` (the `_open_window` call, the `train_step` call, the publish path)
- Modify: `tests/integration/test_v21_window_loop.py`

- [ ] **Step 1: Write the failing test asserting verify_model is passed to the batcher (not train_model)**

Edit `tests/integration/test_v21_window_loop.py`. Add a new test after `test_one_window_lap_bumps_counters`:

```python
@pytest.mark.asyncio
async def test_open_window_passes_verify_model_to_batcher(monkeypatch):
    """The batcher must receive verify_model (not train_model) so
    verify_commitment_proofs runs against the published checkpoint."""
    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()

    # Tag the two models so we can identify which one reaches the batcher.
    svc.train_model = MagicMock(name="train_model_sentinel")
    svc.verify_model = MagicMock(name="verify_model_sentinel")

    captured = {}

    import reliquary.validator.service as svc_mod
    real_open = svc_mod.open_grpo_window

    def _capture_open(window_start, env, model, *, cooldown_map, tokenizer, bootstrap=False):
        captured["model"] = model
        # Return a minimal batcher stub so the rest of _open_window doesn't crash
        from reliquary.validator.batcher import GrpoWindowBatcher
        return GrpoWindowBatcher(
            window_start=window_start, env=env, model=model,
            cooldown_map=cooldown_map, bootstrap=bootstrap,
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            completion_text_fn=lambda r: "",
        )

    with patch.object(svc_mod, "open_grpo_window", side_effect=_capture_open):
        svc._open_window()

    assert captured["model"] is svc.verify_model
    assert captured["model"] is not svc.train_model
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `pytest tests/integration/test_v21_window_loop.py::test_open_window_passes_verify_model_to_batcher -v`
Expected: FAIL (`assert captured["model"] is svc.verify_model` — currently it's `svc.model` which aliases `train_model`).

- [ ] **Step 3: Edit `_open_window` in `service.py:306-311`**

Replace:

```python
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            env=self.env, model=self.model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
```

with:

```python
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            env=self.env, model=self.verify_model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
```

- [ ] **Step 4: Edit the train_step call in `service.py:407-410`**

Replace:

```python
            try:
                self.model = train_step(
                    self.model, batch, window_index=self._window_n,
                )
```

with:

```python
            try:
                self.train_model = train_step(
                    self.train_model, batch,
                    ref_model=self.verify_model,
                    window_index=self._window_n,
                )
```

- [ ] **Step 5: Edit the publish call in `service.py:444-446`**

Replace:

```python
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.model,
                    )
```

with:

```python
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.train_model,
                    )
```

- [ ] **Step 6: Edit `_apply_resume_from` in `service.py:261`**

Replace:

```python
        # Load weights — this replaces the base model loaded at __init__.
        self.model = self._load_model_fn(local_path)
```

with:

```python
        # Load weights — this replaces both models loaded at __init__.
        # verify_model gets the resumed weights too (so the batcher
        # verifies miners against the resumed checkpoint, which is what
        # they have access to via HF).
        self.train_model = self._load_model_fn(local_path)
        try:
            self.train_model.gradient_checkpointing_enable()
        except (AttributeError, NotImplementedError):
            pass
        if self.verify_model is not None:
            self.verify_model.load_state_dict(self.train_model.state_dict())
        else:
            import copy
            self.verify_model = copy.deepcopy(self.train_model)
            self.verify_model.eval()
            for p in self.verify_model.parameters():
                p.requires_grad = False
```

- [ ] **Step 7: Run the integration tests**

Run: `pytest tests/integration/test_v21_window_loop.py -v`
Expected: All passing (existing tests + the new `test_open_window_passes_verify_model_to_batcher`).

- [ ] **Step 8: Commit**

```bash
git add reliquary/validator/service.py tests/integration/test_v21_window_loop.py
git commit -m "feat(validator): batcher and KL ref now use verify_model

_open_window passes verify_model (not train_model) to the batcher, so
verify_commitment_proofs runs against the published checkpoint. train_step
receives verify_model as ref_model for KL. publish/resume paths updated."
```

---

### Task 4: Refresh `verify_model` in-place after every successful publish

**Files:**
- Modify: `reliquary/validator/service.py` (the publish path inside `_train_and_publish`)
- Modify: `tests/integration/test_v21_window_loop.py`

- [ ] **Step 1: Write the failing test — verify_model state_dict matches train_model only AFTER a publish**

Edit `tests/integration/test_v21_window_loop.py`. Add:

```python
@pytest.mark.asyncio
async def test_verify_model_refreshed_only_after_publish(monkeypatch):
    """verify_model.load_state_dict(train_model.state_dict()) must run
    after a successful publish, and ONLY after a publish (not on windows
    where publish is skipped)."""
    import torch
    import torch.nn as nn

    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()

    # Real (tiny) models so load_state_dict has something to copy.
    train = nn.Linear(4, 4)
    verify = nn.Linear(4, 4)
    # Force the two to start identical so divergence is measurable.
    verify.load_state_dict(train.state_dict())
    svc.train_model = train
    svc.verify_model = verify

    # Stub train_step to mutate train_model (simulate a real grad step).
    def _fake_train_step(model, batch, *, ref_model, window_index=None):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        return model

    import reliquary.validator.service as svc_mod
    monkeypatch.setattr(svc_mod, "train_step", _fake_train_step)

    # publish_every=1 by default in _make_service → publish runs every window.
    with _patch_open_grpo_window(svc):
        svc._open_window()
    svc._active_batcher.seal_event.set()
    # Pretend the seal produced a full batch so trained=True.
    svc._active_batcher.seal_batch = MagicMock(return_value=[MagicMock()] * 100)

    await svc._train_and_publish()

    # After publish, verify_model should equal the mutated train_model.
    for p_t, p_v in zip(train.parameters(), verify.parameters()):
        assert torch.equal(p_t, p_v), "verify_model not refreshed after publish"


@pytest.mark.asyncio
async def test_verify_model_NOT_refreshed_when_publish_skipped(monkeypatch):
    """When the publish-interval gate fails (e.g. window % 10 != 0),
    verify_model must keep its previous weights even though train_step ran."""
    import torch
    import torch.nn as nn

    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()
    svc._publish_every = 10  # default-like cadence
    svc._window_n = 5  # not a multiple of 10 → publish skipped

    train = nn.Linear(4, 4)
    verify = nn.Linear(4, 4)
    verify.load_state_dict(train.state_dict())
    svc.train_model = train
    svc.verify_model = verify
    verify_snapshot = {k: v.detach().clone() for k, v in verify.state_dict().items()}

    def _fake_train_step(model, batch, *, ref_model, window_index=None):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        return model

    import reliquary.validator.service as svc_mod
    monkeypatch.setattr(svc_mod, "train_step", _fake_train_step)

    with _patch_open_grpo_window(svc):
        svc._open_window()  # bumps to 6
    svc._active_batcher.seal_event.set()
    svc._active_batcher.seal_batch = MagicMock(return_value=[MagicMock()] * 100)

    await svc._train_and_publish()

    # verify_model must be unchanged
    for k, before in verify_snapshot.items():
        assert torch.equal(verify.state_dict()[k], before), \
            f"verify_model param '{k}' changed despite publish being skipped"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/integration/test_v21_window_loop.py::test_verify_model_refreshed_only_after_publish tests/integration/test_v21_window_loop.py::test_verify_model_NOT_refreshed_when_publish_skipped -v`

Expected: FAIL (first test fails because refresh isn't wired yet; second test passes vacuously).

- [ ] **Step 3: Add the refresh in `_train_and_publish` (`service.py:442-454`)**

Replace:

```python
            if should_publish:
                try:
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.train_model,
                    )
                    self._checkpoint_n = next_n
                    self.server.set_current_checkpoint(entry)
                    logger.info(
                        "Published checkpoint %d to %s@%s",
                        entry.checkpoint_n, entry.repo_id, entry.revision[:12],
                    )
                except Exception:
                    logger.exception("HF publish failed; staying on previous checkpoint")
```

with:

```python
            if should_publish:
                try:
                    entry = await self._checkpoint_store.publish(
                        checkpoint_n=next_n, model=self.train_model,
                    )
                    self._checkpoint_n = next_n
                    self.server.set_current_checkpoint(entry)
                    # Refresh verify_model in-place so the next window's
                    # batcher verifies miners against the just-published
                    # checkpoint. In-place copy: no new allocation.
                    try:
                        self.verify_model.load_state_dict(
                            self.train_model.state_dict()
                        )
                    except (AttributeError, RuntimeError):
                        logger.exception(
                            "verify_model refresh failed; verify_model now "
                            "stale wrt checkpoint %d", entry.checkpoint_n,
                        )
                    logger.info(
                        "Published checkpoint %d to %s@%s and refreshed verify_model",
                        entry.checkpoint_n, entry.repo_id, entry.revision[:12],
                    )
                except Exception:
                    logger.exception("HF publish failed; staying on previous checkpoint")
```

- [ ] **Step 4: Run all integration tests**

Run: `pytest tests/integration/test_v21_window_loop.py -v`
Expected: All passing.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py tests/integration/test_v21_window_loop.py
git commit -m "feat(validator): refresh verify_model in-place after publish

After a successful HF publish, copy the trained weights into verify_model
via load_state_dict (in-place, no new VRAM alloc). Between publishes
(windows where publish is skipped) verify_model keeps the previously
published weights — which is exactly what miners' GRAIL commits target."
```

---

### Task 5: Remove the `self.model` property alias and clean up

**Files:**
- Modify: `reliquary/validator/service.py`

**Rationale:** The property was a migration scaffold. After Tasks 2-4 every internal call site uses `self.train_model` or `self.verify_model` directly, so the alias can go. Verify no external readers exist.

- [ ] **Step 1: Grep for remaining `self.model` usage in the codebase**

Run: `grep -rn "\.model\b" reliquary/ tests/ --include="*.py" | grep -v "self\.train_model\|self\.verify_model\|model\.config\|model\.parameters\|gradient_checkpointing"`

Inspect the output. Any line of the form `svc.model` or `self.model` in `reliquary/validator/` is a leftover and must be migrated to the appropriate explicit name. Test fixtures using `MagicMock().model = ...` are fine to leave.

- [ ] **Step 2: Remove the property**

In `service.py`, delete the `@property def model` and `@model.setter` block added in Task 2 Step 3.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All passing.

- [ ] **Step 4: Commit**

```bash
git add reliquary/validator/service.py
git commit -m "refactor(validator): drop the self.model alias property

All internal call sites now use self.train_model or self.verify_model
explicitly. The migration scaffolding can go."
```

---

### Task 6: Sanity check — VRAM footprint and full test pass

- [ ] **Step 1: Confirm VRAM accounting**

Read through the final `service.py` `__init__` and convince yourself that the two-model layout is **net neutral** vs the pre-refactor `self.model + _ref_model` layout:

- Pre-refactor: `self.model` (8 GiB bf16) + `_ref_model = copy.deepcopy(self.model)` from `_lazy_init` (8 GiB bf16). 16 GiB total.
- Post-refactor: `self.train_model` (8 GiB bf16) + `self.verify_model = copy.deepcopy(self.train_model)` (8 GiB bf16). 16 GiB total.

Same VRAM budget, but verify_model is now the KL reference too (no extra deep-copy) and is refreshed at every publish (not frozen at startup).

- [ ] **Step 2: Run the full unit + integration suite**

Run: `pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 3: Read the diff against `main`**

Run: `git log --oneline main..HEAD` then `git diff main -- reliquary/ tests/ | wc -l`. Eyeball the diff: every change should fall under one of the 4 logical buckets:
1. `train_step` ref_model kwarg + `_ref_model` removal.
2. Two model objects in `ValidationService`.
3. Call sites use the right one.
4. `verify_model` refresh at publish.

If you see anything else (e.g. accidental rename of unrelated functions, drift in unrelated tests), back it out.

- [ ] **Step 4: Final commit if anything was cleaned up; otherwise done**

---

## Out of scope (do NOT do in this plan)

- Don't deploy / rebuild the Docker image. That happens after this plan lands.
- Don't change `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` (currently 10). Tuning the publish cadence is a separate decision.
- Don't tune `KL_BETA`. KL is now against a more-recently-published reference (instead of frozen at startup), which is the conventional GRPO semantic, but the constant stays as-is.
- Don't add a "pause verifier during train_step" gate. The two-model split makes it unnecessary: verify_model is read-only and unaffected by train_step, so there's no race.
- Don't touch `reliquary/cli/main.py` — it still passes a single `model` to `ValidationService`; cloning is internal.
- Don't pre-emptively offload `verify_model` to CPU or quantize it. If a VRAM regression appears in prod *after* this plan ships, that's a separate follow-up.
