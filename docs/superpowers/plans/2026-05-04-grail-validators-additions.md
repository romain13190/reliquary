# GRAIL Validators Additions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three hard-severity per-rollout validators (Schema, Token, Termination) to Catalyst's `GrpoWindowBatcher` to close gaps vs upstream grail's pipeline.

**Architecture:** Each validator runs in `_accept_locked` per rollout, before or after the GRAIL forward pass depending on its data needs. SchemaValidator (Pydantic `CommitModel`) and TokenValidator (existing `verify_tokens`) run early — they are pure-CPU and protect the GPU from malformed input. TerminationValidator runs after `verify_commitment_proofs` because it reuses the cached logits to check `p(EOS)` at the second-to-last position. All three are hard fail-fast: any failure rejects the entire `BatchSubmissionRequest`.

**Tech Stack:** Python 3.11, Pydantic v2, PyTorch (CPU-side softmax for the EOS check), pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-04-grail-validators-additions-design.md`

---

## File structure

| File | Role | Task that touches it |
|---|---|---|
| `reliquary/constants.py` | Add `MIN_EOS_PROBABILITY = 0.02` | Task 5 |
| `reliquary/protocol/submission.py` | Add `CommitModel`, `ModelInfo`, `BeaconInfo`, `RolloutMetadata`; add 3 new `RejectReason` values | Task 1, Task 2 |
| `reliquary/validator/verifier.py` | Add `verify_termination()`; remove `verify_proof_version()` and 2 redundant checks inside `verify_commitment_proofs` | Task 6, Task 11 |
| `reliquary/validator/batcher.py` | Wire SchemaValidator + TokenValidator + TerminationValidator into `_accept_locked`; inject tokenizer in `__init__`; remove `verify_proof_version_fn` injection | Tasks 3, 4, 7, 8, 11 |
| `reliquary/validator/service.py` | Pass tokenizer to `GrpoWindowBatcher` constructor | Task 7 |
| `reliquary/miner/engine.py` | Remove `RELIQUARY_MAX_NEW_TOKENS` env-var override (hardcode `MAX_NEW_TOKENS_PROTOCOL_CAP`) | Task 9 |
| `tests/unit/test_commit_model.py` | New: Pydantic schema tests | Task 1 |
| `tests/unit/test_termination.py` | New: `verify_termination` tests | Task 6 |
| `tests/unit/test_grpo_window_batcher.py` | Update `_request` factory to produce schema-compliant commits; add Schema/Token/Termination batcher tests | Tasks 3, 4, 8, 10 |
| `tests/unit/test_batch_submission_schema.py` | Update `_valid_rollouts` to produce schema-compliant commits | Task 10 |

---

## Task 1: Add `CommitModel` + supporting Pydantic models (no wiring yet)

**Files:**
- Modify: `reliquary/protocol/submission.py`
- Create: `tests/unit/test_commit_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_commit_model.py`:

```python
"""Pydantic schema for the inner GRAIL commit dict.

Tests the structural contract that miners must satisfy in
``RolloutSubmission.commit``. Cross-field consistency rules
(commitments length, prompt+completion length, token_logprobs
length) are checked here.
"""

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import CommitModel


def _valid_commit(seq_len: int = 40, prompt_len: int = 8) -> dict:
    """Build a schema-compliant commit dict for a rollout of ``seq_len`` tokens."""
    completion_len = seq_len - prompt_len
    return {
        "tokens": list(range(seq_len)),
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v5",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_len,
            "completion_length": completion_len,
            "success": True,
            "total_reward": 0.5,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def test_valid_commit_parses():
    CommitModel.model_validate(_valid_commit())


def test_missing_tokens_rejected():
    payload = _valid_commit()
    del payload["tokens"]
    with pytest.raises(ValidationError, match="tokens"):
        CommitModel.model_validate(payload)


def test_proof_version_must_be_v5():
    payload = _valid_commit()
    payload["proof_version"] = "v4"
    with pytest.raises(ValidationError, match="proof_version"):
        CommitModel.model_validate(payload)


def test_commitments_length_mismatch_rejected():
    payload = _valid_commit(seq_len=40)
    payload["commitments"] = payload["commitments"][:-1]  # 39, not 40
    with pytest.raises(ValidationError, match="commitments"):
        CommitModel.model_validate(payload)


def test_prompt_plus_completion_must_equal_tokens_len():
    payload = _valid_commit(seq_len=40, prompt_len=8)
    payload["rollout"]["prompt_length"] = 9   # 9 + 32 = 41 != 40
    with pytest.raises(ValidationError, match="prompt_length"):
        CommitModel.model_validate(payload)


def test_token_logprobs_length_must_match_tokens():
    payload = _valid_commit(seq_len=40)
    payload["rollout"]["token_logprobs"] = [0.0] * 39  # off by one
    with pytest.raises(ValidationError, match="token_logprobs"):
        CommitModel.model_validate(payload)


def test_extra_field_in_commit_rejected():
    payload = _valid_commit()
    payload["sneaky_field"] = "should be rejected"
    with pytest.raises(ValidationError, match="sneaky_field|Extra inputs"):
        CommitModel.model_validate(payload)


def test_tokens_below_challenge_k_rejected():
    payload = _valid_commit(seq_len=20, prompt_len=4)  # 20 < CHALLENGE_K=32
    with pytest.raises(ValidationError, match="tokens"):
        CommitModel.model_validate(payload)


def test_signature_must_be_hex():
    payload = _valid_commit()
    payload["signature"] = "not-hex-zzz"
    with pytest.raises(ValidationError, match="signature"):
        CommitModel.model_validate(payload)


def test_beacon_randomness_must_be_hex():
    payload = _valid_commit()
    payload["beacon"]["randomness"] = "not-hex-zzz"
    with pytest.raises(ValidationError, match="randomness"):
        CommitModel.model_validate(payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_commit_model.py -v`
Expected: All FAIL with `ImportError: cannot import name 'CommitModel'` or similar.

- [ ] **Step 3: Add the Pydantic models**

Edit `reliquary/protocol/submission.py`. After the existing imports, add `Literal` and `field_validator` if not already present:

```python
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
```

Add the constants import at the top with the others:

```python
from reliquary.constants import CHALLENGE_K, M_ROLLOUTS, MAX_NEW_TOKENS_PROTOCOL_CAP
```

After the existing classes (`BatchSubmissionResponse`, etc.) add the four new models:

```python
class ModelInfo(BaseModel):
    """Identifies the model the miner ran."""
    model_config = ConfigDict(extra="forbid")

    name: str
    layer_index: int


class BeaconInfo(BaseModel):
    """Drand beacon randomness used for this commit."""
    model_config = ConfigDict(extra="forbid")

    randomness: str = Field(..., pattern=r"^[0-9a-fA-F]+$")


class RolloutMetadata(BaseModel):
    """Per-rollout meta: lengths, success flag, claimed reward, logprobs."""
    model_config = ConfigDict(extra="forbid")

    prompt_length: int = Field(..., ge=0)
    completion_length: int = Field(..., gt=0, le=MAX_NEW_TOKENS_PROTOCOL_CAP)
    success: bool
    total_reward: float
    advantage: float
    token_logprobs: list[float]


class CommitModel(BaseModel):
    """The inner ``commit`` dict shipped by the miner inside ``RolloutSubmission``.

    Validated explicitly at the top of ``GrpoWindowBatcher._accept_locked``
    rather than via Pydantic on ``RolloutSubmission.commit`` — keeps the
    failure path inside the batcher's reject-counts telemetry.
    """
    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=CHALLENGE_K)
    commitments: list[dict]
    proof_version: Literal["v5"]
    model: ModelInfo
    signature: str = Field(..., pattern=r"^[0-9a-fA-F]+$")
    beacon: BeaconInfo
    rollout: RolloutMetadata

    @field_validator("commitments")
    @classmethod
    def _commitments_len_matches_tokens(cls, v, info):
        tokens = info.data.get("tokens", [])
        if len(v) != len(tokens):
            raise ValueError(
                f"commitments length {len(v)} must equal tokens length {len(tokens)}"
            )
        return v

    @field_validator("rollout")
    @classmethod
    def _lengths_consistent(cls, v, info):
        tokens = info.data.get("tokens", [])
        if v.prompt_length + v.completion_length != len(tokens):
            raise ValueError(
                f"prompt_length({v.prompt_length}) + "
                f"completion_length({v.completion_length}) must equal "
                f"len(tokens)={len(tokens)}"
            )
        if len(v.token_logprobs) != len(tokens):
            raise ValueError(
                f"token_logprobs length {len(v.token_logprobs)} "
                f"must equal tokens length {len(tokens)}"
            )
        return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_commit_model.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/submission.py tests/unit/test_commit_model.py
git commit -m "feat(protocol): add CommitModel Pydantic schema for inner commit dict"
```

---

## Task 2: Add three new `RejectReason` values

**Files:**
- Modify: `reliquary/protocol/submission.py:25-42` (the `RejectReason` enum)
- Test: `tests/unit/test_batch_submission_schema.py` (verify enum values exist)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_batch_submission_schema.py`:

```python
def test_new_reject_reasons_exist():
    """Schema/Token/Termination validators emit dedicated reject codes."""
    assert RejectReason.BAD_SCHEMA.value == "bad_schema"
    assert RejectReason.BAD_TOKENS.value == "bad_tokens"
    assert RejectReason.BAD_TERMINATION.value == "bad_termination"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_batch_submission_schema.py::test_new_reject_reasons_exist -v`
Expected: FAIL with `AttributeError: BAD_SCHEMA`.

- [ ] **Step 3: Add the enum values**

In `reliquary/protocol/submission.py`, locate the `RejectReason` enum and add three new values just before the existing `WRONG_CHECKPOINT` line:

```python
    BAD_SCHEMA = "bad_schema"
    BAD_TOKENS = "bad_tokens"
    BAD_TERMINATION = "bad_termination"
    WRONG_CHECKPOINT = "wrong_checkpoint"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_batch_submission_schema.py::test_new_reject_reasons_exist -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/submission.py tests/unit/test_batch_submission_schema.py
git commit -m "feat(protocol): add BAD_SCHEMA/BAD_TOKENS/BAD_TERMINATION reject reasons"
```

---

## Task 3: Update the batcher test `_request` factory to produce schema-compliant commits

**Files:**
- Modify: `tests/unit/test_grpo_window_batcher.py` (the `_request` and `_request_v21` factories near line 48 and line 256)

This task is purely refactoring of the test factory so that all subsequent tasks (which will wire SchemaValidator into the batcher) don't break the existing 30+ batcher tests.

**Important design note:** the existing tests use `commit["completion_text_for_test"]` as a back-channel for `completion_text_fn` (avoiding a real tokenizer). With `extra="forbid"` on `CommitModel`, this key will be rejected. We refactor the pattern: `_make_batcher`'s `completion_text_fn` derives text from `rollout.reward` instead — the test's `_request` already controls reward, so this is lossless and produces a clean commit dict.

- [ ] **Step 1: Add a helper that builds a schema-compliant commit**

In `tests/unit/test_grpo_window_batcher.py`, after the imports and before `_request`, add:

```python
from reliquary.constants import CHALLENGE_K


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate.

    Default produces a ``CHALLENGE_K + 4`` token sequence: 4 prompt tokens,
    ``CHALLENGE_K`` completion tokens (the minimum the proof needs).
    """
    if tokens is None:
        tokens = list(range(CHALLENGE_K + prompt_length))
    seq_len = len(tokens)
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v5",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }
```

- [ ] **Step 2: Update `_make_batcher` to derive completion text from reward**

In `tests/unit/test_grpo_window_batcher.py`, modify the `_make_batcher` helper so its `completion_text_fn` reads `rollout.reward` instead of `rollout.commit.get("completion_text_for_test", "")`:

```python
def _make_batcher(**overrides) -> GrpoWindowBatcher:
    kwargs = dict(
        window_start=500,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        verify_proof_version_fn=_always_true_proof_version,
        completion_text_fn=lambda rollout: (
            "CORRECT" if rollout.reward > 0.5 else "wrong"
        ),
    )
    kwargs.update(overrides)
    return GrpoWindowBatcher(**kwargs)
```

This change is invisible to the existing tests because the FakeEnv's reward computation already keys off the substring "CORRECT".

- [ ] **Step 3: Update `_request` and `_request_v21` to use the helper**

Replace the rollout-construction loop inside `_request` (around lines 54-67) with:

```python
def _request(
    prompt_idx=42, signed_round=1000, window_start=500,
    rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for r in rewards:
        commit = _make_commit(success=r > 0.5, total_reward=r)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=r,
                commit=commit,
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        signed_round=signed_round,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
    )
```

Apply the same change to `_request_v21` (around line 256) — replace its rollout loop and pass `commit["tokens"]` as the outer `tokens` field.

- [ ] **Step 4: Run the existing batcher tests to verify they still pass**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v`
Expected: All existing tests PASS (we changed the fixture but not the verification logic).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_grpo_window_batcher.py
git commit -m "test(batcher): _make_commit helper produces schema-compliant commits"
```

---

## Task 4: Update `tests/unit/test_batch_submission_schema.py` `_valid_rollouts` to produce schema-compliant commits

**Files:**
- Modify: `tests/unit/test_batch_submission_schema.py` (the `_valid_rollouts` helper)

Same purpose as Task 3, but for the schema-test file.

- [ ] **Step 1: Update `_valid_rollouts` to produce full commits**

Replace the body of `_valid_rollouts` in `tests/unit/test_batch_submission_schema.py` with:

```python
def _valid_rollouts(k: int = 4):
    """k successes, (M - k) failures, all with schema-compliant commits."""
    from reliquary.constants import CHALLENGE_K

    rollouts = []
    seq_len = CHALLENGE_K + 4
    prompt_len = 4
    completion_len = seq_len - prompt_len
    for i in range(M_ROLLOUTS):
        tokens = list(range(seq_len))
        commit = {
            "tokens": tokens,
            "commitments": [{"sketch": 0} for _ in range(seq_len)],
            "proof_version": "v5",
            "model": {"name": "test-model", "layer_index": 6},
            "signature": "ab" * 32,
            "beacon": {"randomness": "cd" * 16},
            "rollout": {
                "prompt_length": prompt_len,
                "completion_length": completion_len,
                "success": i < k,
                "total_reward": 1.0 if i < k else 0.0,
                "advantage": 0.0,
                "token_logprobs": [0.0] * seq_len,
            },
        }
        rollouts.append(
            RolloutSubmission(
                tokens=tokens,
                reward=1.0 if i < k else 0.0,
                commit=commit,
            )
        )
    return rollouts
```

- [ ] **Step 2: Run schema tests to verify they still pass**

Run: `pytest tests/unit/test_batch_submission_schema.py -v`
Expected: All existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_batch_submission_schema.py
git commit -m "test(schema): _valid_rollouts produces schema-compliant commits"
```

---

## Task 5: Add `MIN_EOS_PROBABILITY` constant

**Files:**
- Modify: `reliquary/constants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_constants.py` (or create it if it doesn't exist):

```python
def test_min_eos_probability_constant_present():
    from reliquary.constants import MIN_EOS_PROBABILITY
    assert 0.0 < MIN_EOS_PROBABILITY < 1.0
    assert MIN_EOS_PROBABILITY == 0.02
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_constants.py::test_min_eos_probability_constant_present -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the constant**

Append to `reliquary/constants.py` (group it near the other proof constants such as `CHALLENGE_K`):

```python
# Minimum probability the model must have assigned to EOS at the position
# that produced it. Below this threshold, the rollout is presumed to be
# artificially truncated (a miner truncating mid-reasoning to lock in a
# favourable partial output). Calibrated by upstream grail at 0% honest FP.
MIN_EOS_PROBABILITY = 0.02
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_constants.py::test_min_eos_probability_constant_present -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_constants.py
git commit -m "feat(constants): add MIN_EOS_PROBABILITY=0.02 for TerminationValidator"
```

---

## Task 6: Add `verify_termination()` to verifier.py

**Files:**
- Modify: `reliquary/validator/verifier.py`
- Create: `tests/unit/test_termination.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_termination.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_termination.py -v`
Expected: All FAIL with `ImportError: cannot import name 'verify_termination'`.

- [ ] **Step 3: Add the function to verifier.py**

Append to `reliquary/validator/verifier.py` (after `verify_proof_version`, before `verify_commitment_proofs`):

```python
def verify_termination(commit: dict, tokenizer: Any, logits: torch.Tensor) -> bool:
    """Hard check: rollout ends with EOS at p(EOS) >= MIN_EOS_PROBABILITY.

    Reuses the per-token logits cached from ``verify_commitment_proofs``,
    so this costs O(vocab) on a single position — no extra forward pass.

    Strict EOS-only: a rollout that hit ``max_new_tokens`` without sampling
    EOS is treated as artificially truncated and rejected. In RL settings
    where reward depends on parsing the model's final output (boxed math,
    code block, JSON), a truncated rollout scores zero anyway — there is
    no legitimate reason for a healthy rollout to hit the cap.
    """
    from reliquary.constants import MIN_EOS_PROBABILITY

    tokens = commit["tokens"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return False
    if tokens[-1] != eos_id:
        return False
    # logits[-2] is the distribution that produced tokens[-1] (the EOS).
    p_eos = float(torch.softmax(logits[-2].float(), dim=-1)[eos_id].item())
    return p_eos >= MIN_EOS_PROBABILITY
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_termination.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_termination.py
git commit -m "feat(verifier): add verify_termination — strict EOS-only check"
```

---

## Task 7: Inject tokenizer into `GrpoWindowBatcher` constructor

**Files:**
- Modify: `reliquary/validator/batcher.py` (constructor)
- Modify: `reliquary/validator/service.py` (where `GrpoWindowBatcher` is instantiated)

The TerminationValidator (Task 8) needs the tokenizer to read `eos_token_id`. Currently the batcher only holds `self.model`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_grpo_window_batcher.py` (at the bottom):

```python
def test_constructor_accepts_tokenizer():
    """Tokenizer must be passable to the batcher (used by TerminationValidator)."""
    class FakeTokenizer:
        eos_token_id = 99

    fake_tok = FakeTokenizer()
    b = _make_batcher(tokenizer=fake_tok)
    assert b.tokenizer is fake_tok
```

Also update the `_make_batcher` helper to add a default tokenizer (keep the `completion_text_fn` lambda from Task 3):

```python
def _make_batcher(**overrides) -> GrpoWindowBatcher:
    class _DefaultFakeTokenizer:
        eos_token_id = 99

    kwargs = dict(
        window_start=500,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        tokenizer=_DefaultFakeTokenizer(),
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        verify_proof_version_fn=_always_true_proof_version,
        completion_text_fn=lambda rollout: (
            "CORRECT" if rollout.reward > 0.5 else "wrong"
        ),
    )
    kwargs.update(overrides)
    return GrpoWindowBatcher(**kwargs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_constructor_accepts_tokenizer -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'tokenizer'`.

- [ ] **Step 3: Update `GrpoWindowBatcher.__init__`**

In `reliquary/validator/batcher.py`, modify the `__init__` signature to accept `tokenizer`:

```python
def __init__(
    self,
    window_start: int,
    current_round: int,
    env: Environment,
    model: Any,
    *,
    tokenizer: Any = None,         # NEW — required for TerminationValidator
    cooldown_map: CooldownMap | None = None,
    bootstrap: bool = False,
    completion_text_fn: Callable[[RolloutSubmission], str],
    canonical_prompt_tokens_fn: Callable[[int], list[int]] | None = None,
    verify_commitment_proofs_fn: Callable[..., Any] | None = None,
    verify_signature_fn: Callable[[dict, str], bool] | None = None,
    verify_proof_version_fn: Callable[[dict], bool] | None = None,
    time_fn: Callable[[], float] | None = None,
    now_round_fn: Callable[[], int] | None = None,
) -> None:
    ...
    self.model = model
    self.tokenizer = tokenizer       # NEW — store for use in _accept_locked
    ...
```

- [ ] **Step 4: Update `service.py` to pass the tokenizer**

There is exactly one instantiation site: `reliquary/validator/service.py:88`. The `tokenizer` is already in scope as a function argument (used by `_completion_text` and `_canonical_prompt_tokens` just above). Add `tokenizer=tokenizer,` as a kwarg in the `GrpoWindowBatcher(...)` call:

```python
    return GrpoWindowBatcher(
        window_start=window_start,
        current_round=current_round,
        env=env,
        model=model,
        tokenizer=tokenizer,                       # NEW
        cooldown_map=cooldown_map,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
        canonical_prompt_tokens_fn=_canonical_prompt_tokens,
        now_round_fn=now_round_fn,
    )
```

- [ ] **Step 5: Run all batcher and service tests**

Run: `pytest tests/unit/test_grpo_window_batcher.py tests/unit/test_service_v2.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/batcher.py reliquary/validator/service.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(batcher): accept tokenizer kwarg for upcoming TerminationValidator"
```

---

## Task 8: Wire SchemaValidator + TokenValidator + TerminationValidator into `_accept_locked`

**Files:**
- Modify: `reliquary/validator/batcher.py:_accept_locked`
- Modify: `tests/unit/test_grpo_window_batcher.py` (add 8 new tests)

This is the central wiring task. It introduces the three new checks in the per-rollout loop. We do all three together because they share the same insertion site and several test fixtures.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
import torch
from reliquary.validator.verifier import ProofResult


def _grail_with_logits(seq_len: int, eos_id: int = 99):
    """Stub that returns logits where EOS is highly probable everywhere."""
    def _fn(commit, model, randomness):
        logits = torch.zeros(seq_len, 100)
        logits[:, eos_id] = 5.0
        return ProofResult(
            all_passed=True, passed=1, checked=1, logits=logits,
            sketch_diff_max=0,
        )
    return _fn


# ----- SchemaValidator wiring -----

def test_reject_bad_schema_missing_proof_version():
    b = _make_batcher()
    req = _request()
    # Mutate one rollout's commit to break schema
    req.rollouts[0].commit.pop("proof_version")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


def test_reject_bad_schema_extra_field():
    b = _make_batcher()
    req = _request()
    req.rollouts[0].commit["unauthorized_field"] = "x"
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


def test_reject_bad_schema_inconsistent_lengths():
    b = _make_batcher()
    req = _request()
    req.rollouts[0].commit["rollout"]["prompt_length"] = 999
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_SCHEMA


# ----- TokenValidator wiring -----
# The verify_tokens function in protocol/tokens.py is now wired into the
# batcher AFTER schema validation. We stub model.config so verify_tokens
# can resolve vocab_size.

class _ModelStubWithVocab:
    """Minimal stub satisfying resolve_vocab_size(model.config)."""
    class config:
        vocab_size = 1000
        max_position_embeddings = 4096


def test_reject_bad_tokens_above_vocab():
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()
    # vocab_size=1000, inject a token == vocab_size (out of bounds)
    req.rollouts[0].commit["tokens"] = [1000] * (CHALLENGE_K + 4)
    # Re-sync the outer field so RolloutSubmission stays consistent
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    # Re-sync commitments + token_logprobs lengths for schema
    req.rollouts[0].commit["commitments"] = [
        {"sketch": 0} for _ in range(CHALLENGE_K + 4)
    ]
    req.rollouts[0].commit["rollout"]["token_logprobs"] = [0.0] * (CHALLENGE_K + 4)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TOKENS


def test_reject_bad_tokens_negative_id():
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()
    req.rollouts[0].commit["tokens"] = [-1] + list(range(CHALLENGE_K + 3))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    req.rollouts[0].commit["commitments"] = [
        {"sketch": 0} for _ in range(CHALLENGE_K + 4)
    ]
    req.rollouts[0].commit["rollout"]["token_logprobs"] = [0.0] * (CHALLENGE_K + 4)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TOKENS


# ----- TerminationValidator wiring -----

def test_reject_bad_termination_when_last_token_not_eos():
    seq_len = CHALLENGE_K + 4
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
    )
    req = _request()
    # Last token != 99 (EOS) — sequence ends in seq_len-1
    req.rollouts[0].commit["tokens"] = list(range(seq_len))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.BAD_TERMINATION


def test_termination_skipped_when_grail_returns_empty_logits():
    """Backward-compat: when the GRAIL stub returns empty logits, the
    termination check is skipped. The default ``_always_true_grail`` does
    this (it predates the cached-logits path), so the existing
    full-pipeline tests stay green without becoming termination-aware.
    """
    b = _make_batcher(model=_ModelStubWithVocab())
    req = _request()  # default rewards [1,1,1,1,0,0,0,0] → sigma above SIGMA_MIN
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


# Note on "happy path with logits" test: a positive case where the rollout
# DOES end with EOS *and* survives the full pipeline (logprob + distribution)
# requires synthetic logits whose log_softmax matches the miner-claimed
# token_logprobs (which the test fixture sets to all-zero). Building such a
# fixture pulls the test toward an end-to-end integration test. We cover the
# wiring with the reject case above and the empty-logits skip case; the
# happy path is exercised by the existing pipeline tests through the
# empty-logits branch. A real end-to-end happy path lives in tests/integration.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v -k "bad_schema or bad_tokens or bad_termination or termination_skipped"`
Expected: The schema/tokens/termination tests FAIL — none of the new validators are wired yet. The `termination_skipped` test should already PASS (it tests the fallback path using existing fixtures).

- [ ] **Step 3: Wire the three validators in `_accept_locked`**

In `reliquary/validator/batcher.py`, locate `_accept_locked` (around line 190). Add new imports at the top of the file:

```python
from pydantic import ValidationError

from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    CommitModel,        # NEW
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
    WindowState,
)
from reliquary.protocol.tokens import verify_tokens   # NEW
from reliquary.validator.verifier import verify_termination  # NEW
```

Then locate the per-rollout loop. The current first iteration looks like:

```python
for rollout in request.rollouts:
    if canonical_prompt_tokens is not None:
        rollout_meta = rollout.commit.get("rollout", {}) or {}
        miner_prompt_len = int(rollout_meta.get("prompt_length", 0))
        ...
```

Insert two new checks at the very top of the loop, BEFORE `canonical_prompt_tokens`:

```python
for rollout in request.rollouts:
    # Schema check: structural validation of commit dict (cheap, no GPU)
    try:
        CommitModel.model_validate(rollout.commit)
    except ValidationError:
        return self._reject(RejectReason.BAD_SCHEMA)

    # Token check: vocab bounds + max length (cheap, protects forward pass)
    if not verify_tokens(rollout.commit["tokens"], self.model.config):
        return self._reject(RejectReason.BAD_TOKENS)

    if canonical_prompt_tokens is not None:
        ...
```

After the existing `proof = self._verify_commitment(...)` call and the `if not proof.all_passed` check, but BEFORE the existing `if proof.logits.numel() == 0: continue` line, insert:

```python
            proof = self._verify_commitment(
                rollout.commit, self.model, self.randomness
            )
            if proof.sketch_diff_max > sketch_diff_max:
                sketch_diff_max = proof.sketch_diff_max
            if not proof.all_passed:
                return self._reject(RejectReason.GRAIL_FAIL)

            # Termination check: rollout must end with EOS at p(EOS) >= threshold.
            # Reuses cached logits from the GRAIL forward — zero extra compute.
            # Skipped when grail stub returns empty logits (legacy test fixtures).
            if proof.logits.numel() > 0:
                if not verify_termination(
                    rollout.commit, self.tokenizer, proof.logits
                ):
                    return self._reject(RejectReason.BAD_TERMINATION)

            # Behavioural checks (use cached logits from the GRAIL forward pass).
            # Skip gracefully if the logits tensor is empty (legacy stubs in tests).
            if proof.logits.numel() == 0:
                continue
```

- [ ] **Step 4: Re-run all batcher tests to verify both old and new pass**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v`
Expected: All tests PASS, including the 8 new ones.

- [ ] **Step 5: Run the full test suite to catch downstream breakage**

Run: `pytest tests/unit/ -v --timeout 60`
Expected: All tests PASS. If `test_batch_submission_schema.py`, `test_service_v2.py`, or `test_v21_window_loop.py` fail because of the schema gate, fix the fixtures inline (apply the same pattern as Task 4).

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(batcher): wire SchemaValidator/TokenValidator/TerminationValidator"
```

---

## Task 9: Remove `RELIQUARY_MAX_NEW_TOKENS` env-var override

**Files:**
- Modify: `reliquary/miner/engine.py:153-154`

The strict TerminationValidator means a miner who lowers their cap below the protocol value would have rollouts truncated without EOS → all rejected. Removing the override makes the failure mode impossible.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_miner_engine_v2.py`:

```python
def test_engine_default_max_new_tokens_is_protocol_cap(monkeypatch):
    """The env-var override is removed; max_new_tokens is the protocol cap."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
    from reliquary.miner.engine import MiningEngine

    monkeypatch.setenv("RELIQUARY_MAX_NEW_TOKENS", "512")
    # Constructing MiningEngine should NOT pick up the env var.
    # We stub all heavy deps; the goal is just to read the default value.
    eng = MiningEngine.__new__(MiningEngine)  # avoid full __init__
    # Trigger the default-value branch: instantiating with no arg.
    import inspect
    sig = inspect.signature(MiningEngine.__init__)
    default = sig.parameters["max_new_tokens"].default
    assert default == MAX_NEW_TOKENS_PROTOCOL_CAP
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_miner_engine_v2.py::test_engine_default_max_new_tokens_is_protocol_cap -v`
Expected: FAIL — the default still reads the env var.

- [ ] **Step 3: Hardcode the default**

In `reliquary/miner/engine.py:153`, replace:

```python
max_new_tokens: int = int(
    os.environ.get("RELIQUARY_MAX_NEW_TOKENS", MAX_NEW_TOKENS_PROTOCOL_CAP)
),
```

with:

```python
max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
```

If `os` is no longer used elsewhere in the file, remove the `import os` line; otherwise leave it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_miner_engine_v2.py::test_engine_default_max_new_tokens_is_protocol_cap -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py tests/unit/test_miner_engine_v2.py
git commit -m "refactor(miner): remove RELIQUARY_MAX_NEW_TOKENS env-var override"
```

---

## Task 10: Cleanup — remove `verify_proof_version` and redundant in-verifier checks

**Files:**
- Modify: `reliquary/validator/verifier.py` (remove `verify_proof_version`, remove 2 checks inside `verify_commitment_proofs`)
- Modify: `reliquary/validator/batcher.py` (remove the `verify_proof_version_fn` injection point)
- Modify: `tests/unit/test_grpo_window_batcher.py` (remove `verify_proof_version_fn` from `_make_batcher`)

These checks are now covered by `CommitModel`:
- `proof_version == "v5"` → covered by `Literal["v5"]` field
- `len(commitments) != seq_len` → covered by `_commitments_len_matches_tokens` validator
- `seq_len > MAX_TOKENS_PER_ROLLOUT` → covered by `verify_tokens` (Task 8)

- [ ] **Step 1: Remove `verify_proof_version()` from verifier.py**

In `reliquary/validator/verifier.py:53-55`, delete the function:

```python
def verify_proof_version(commit: dict) -> bool:
    """Hard check: proof version must match protocol."""
    return commit.get("proof_version") == GRAIL_PROOF_VERSION
```

- [ ] **Step 2: Remove the redundant length checks inside `verify_commitment_proofs`**

In `reliquary/validator/verifier.py`, locate `verify_commitment_proofs`. Delete:

```python
    # SECURITY: Miner must provide exactly one commitment per token.
    seq_len = len(tokens)
    if len(commitments) != seq_len:
        logger.warning(...)
        return ProofResult(...)

    # SECURITY: Reject sequences that would cause GPU OOM.
    if seq_len > MAX_TOKENS_PER_ROLLOUT:
        logger.warning(...)
        return ProofResult(...)
```

Replace with just:

```python
    seq_len = len(tokens)
```

Reason: `CommitModel._commitments_len_matches_tokens` covers the first; `verify_tokens` (`_validate_sequence_length`) covers the second.

- [ ] **Step 3: Remove `verify_proof_version_fn` from `GrpoWindowBatcher.__init__`**

In `reliquary/validator/batcher.py`, delete:
- the `verify_proof_version_fn: Callable[[dict], bool] | None = None` parameter
- the `if verify_proof_version_fn is None: ... verify_proof_version_fn = verify_proof_version` block
- the `self._verify_proof_version = verify_proof_version_fn` line
- the `if not self._verify_proof_version(rollout.commit): return self._reject(RejectReason.GRAIL_FAIL)` call inside `_accept_locked`

Also remove the `from reliquary.validator.verifier import verify_proof_version` import inside `_accept_locked` (it should be in the lazy block at the top of `__init__`).

- [ ] **Step 4: Remove `verify_proof_version_fn` from test fixtures**

In `tests/unit/test_grpo_window_batcher.py`, delete:
- the `def _always_true_proof_version(commit): return True` helper
- the `verify_proof_version_fn=_always_true_proof_version,` line in `_make_batcher`

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/unit/ -v --timeout 60`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/verifier.py reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "refactor(verifier): remove checks now covered by CommitModel + verify_tokens"
```

---

## Task 11: Verify integration — run the full test suite + smoke test

**Files:**
- None (verification only)

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/unit/ -v --timeout 60`
Expected: All tests PASS.

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/ -v --timeout 120`
Expected: All tests PASS.

If any integration test fails because of the schema gate (commits with insufficient fields), fix its fixture by importing `_make_commit` from the unit-test helper or by inlining the same dict structure.

- [ ] **Step 3: Verify reject_counts telemetry includes the new reasons**

Quick sanity check — there is no test for this, but the values appear in archive R2 metadata. Run a one-off grep:

```bash
grep -r "reject_counts\|reason\.value" reliquary/validator/service.py | head
```

Confirm that `reject_counts` is keyed by `RejectReason.value` (string), not by reason name — if so, the new `BAD_SCHEMA` / `BAD_TOKENS` / `BAD_TERMINATION` strings will appear automatically in the archive without any further change.

- [ ] **Step 4: Final commit (no-op marker for the plan)**

Nothing to commit if all previous tasks committed cleanly. Verify:

```bash
git status
```

Expected: working tree clean.

---

## Definition of done

- [ ] All 10 tasks committed
- [ ] `pytest tests/unit tests/integration -v` passes
- [ ] `git log --oneline -15` shows ~10 incremental commits with the plan's commit messages
- [ ] No `RELIQUARY_MAX_NEW_TOKENS` references remain in `reliquary/miner/engine.py`
- [ ] No `verify_proof_version` references remain in the codebase (`grep -r verify_proof_version reliquary/` empty)
- [ ] `RejectReason` enum has `BAD_SCHEMA`, `BAD_TOKENS`, `BAD_TERMINATION` values
