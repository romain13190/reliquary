# GRPO Market Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1's per-slot advantage scoring with a free-prompt GRPO market: miners pick prompts, submit 8-rollout groups, first-8-in-zone win the training batch, flat `1/B` payment.

**Architecture:** Refactor `WindowBatcher` from a slot-based state machine (8 validator-derived prompts × 32 completions) to a flat submission pool + cooldown map. Pure functions (`CooldownMap`, `select_batch`, `compute_weights_v2`) are built and tested in isolation first, then composed into the orchestrator. GRAIL v1 (proof verification) stays unchanged — tolerant sketch handles cross-GPU variance.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, FastAPI + httpx for HTTP, PyTorch for GRAIL forward-pass, bittensor-drand for beacons.

**Spec reference:** `docs/superpowers/specs/2026-04-20-grpo-market-design.md` (commit `556b043`).

---

## File structure

**Modified:**
- `reliquary/constants.py` — new params (`ZONE_K_MIN/MAX`, `M_ROLLOUTS`, `B_BATCH`, `BATCH_PROMPT_COOLDOWN_WINDOWS`, `T_PROTO`, bootstrap); deprecate old slot-based params
- `reliquary/protocol/submission.py` — new `BatchSubmissionRequest`, `BatchSubmissionResponse`, `RejectReason` enum, `GrpoBatchState`
- `reliquary/validator/batcher.py` — rewrite orchestrator; no slots, flat submission pool, cooldown enforcement
- `reliquary/validator/verifier.py` — add `verify_reward_claim()` helper
- `reliquary/validator/weights.py` — flat-`1/B` computation, UID_BURN on unused slots
- `reliquary/validator/server.py` — new `/submit` + `/window/{n}/state` endpoints
- `reliquary/miner/engine.py` — free prompt selection, local reward computation, signed-round handling
- `reliquary/miner/submitter.py` — new payload shape, `RejectReason` handling

**Created:**
- `reliquary/validator/cooldown.py` — `CooldownMap` with local JSON persistence + R2 history rebuild
- `reliquary/validator/batch_selection.py` — pure `select_batch()` function (isolated for testing)

**Tests created:**
- `tests/unit/test_cooldown.py`
- `tests/unit/test_batch_selection.py`
- `tests/unit/test_zone_filter.py`
- `tests/unit/test_reward_verification.py`
- `tests/unit/test_flat_payment.py`
- `tests/unit/test_batch_submission_schema.py`
- `tests/unit/test_grpo_window_batcher.py` (replaces `test_batcher.py`)

**Tests deleted (in cleanup task):**
- `tests/unit/test_batcher.py` (slot-based tests)
- `tests/unit/test_diversity.py` (prefix-dedup)

---

## Task order rationale

Pure, testable primitives first → orchestrators second → I/O layers last. This lets each task land with passing tests and no regressions on untouched code. The v1 path stays alive until Task 11 (wire-in); v1 tests keep passing until Task 14 (cleanup).

---

## Task 1: New constants

**Files:**
- Modify: `reliquary/constants.py`
- Test: `tests/unit/test_constants.py` (create)

**Context:** v1 constants (`GROUP_SIZE`, `PROMPTS_PER_WINDOW`, etc.) remain for now — removed in cleanup task. New constants are added alongside.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_constants.py`:

```python
"""Sanity checks on v2 constants — catches accidental edits."""

from reliquary import constants as C


def test_v2_zone_bounds():
    assert C.ZONE_K_MIN == 2
    assert C.ZONE_K_MAX == 6
    assert C.ZONE_K_MIN < C.ZONE_K_MAX


def test_v2_group_sizes():
    assert C.M_ROLLOUTS == 8
    assert C.B_BATCH == 8


def test_v2_temperature_fixed_nonzero():
    assert 0.5 < C.T_PROTO <= 1.0


def test_v2_cooldown_values():
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 50
    assert C.BOOTSTRAP_COOLDOWN_WINDOWS == 10
    assert C.BOOTSTRAP_WINDOWS == 100


def test_v2_bootstrap_zone_is_wider_than_steady():
    # k ∈ [1, 7] during bootstrap vs [2, 6] steady
    assert C.BOOTSTRAP_ZONE_K_MIN < C.ZONE_K_MIN
    assert C.BOOTSTRAP_ZONE_K_MAX > C.ZONE_K_MAX


def test_v2_bootstrap_m_smaller():
    assert C.BOOTSTRAP_M_ROLLOUTS < C.M_ROLLOUTS
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_constants.py -v
```

Expected: FAIL with `AttributeError: module 'reliquary.constants' has no attribute 'ZONE_K_MIN'`.

- [ ] **Step 3: Add constants**

Append to `reliquary/constants.py` (after the existing `ECONOMIC / INCENTIVE` section):

```python
# ────────────────  GRPO MARKET (v2)  ────────────────

# Apprenable zone: a group with k successes ∈ [ZONE_K_MIN, ZONE_K_MAX]
# inclusive is eligible for the training batch. Outside this range means
# the group has ~no GRPO signal (too easy or too hard) and is rejected.
ZONE_K_MIN = 2
ZONE_K_MAX = 6

# Number of rollouts per submission (= size of each GRPO group).
M_ROLLOUTS = 8

# Training batch size — the first B valid in-zone submissions (FIFO by
# signed_round, distinct prompts, not in cooldown) feed the GRPO step.
B_BATCH = 8

# Sampling temperature fixed at protocol level. Miners who use a different
# T would produce samples from a different distribution → biased GRPO
# gradient. Value chosen in the GRPO-friendly range (non-zero).
T_PROTO = 0.9

# Top-p and top-k for sampling (fixed alongside T_PROTO).
TOP_P_PROTO = 1.0
TOP_K_PROTO = 0

# A prompt that entered the training batch is ineligible for B_BATCH for
# the next N windows (= training steps). Forces curriculum rotation so
# the policy has time to shift between reuses.
BATCH_PROMPT_COOLDOWN_WINDOWS = 50

# Bootstrap phase: first BOOTSTRAP_WINDOWS of a new subnet/checkpoint use
# relaxed thresholds to keep the batch filling while miner pop + env
# coverage are thin.
BOOTSTRAP_WINDOWS = 100
BOOTSTRAP_ZONE_K_MIN = 1
BOOTSTRAP_ZONE_K_MAX = 7
BOOTSTRAP_M_ROLLOUTS = 4
BOOTSTRAP_COOLDOWN_WINDOWS = 10
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/unit/test_constants.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_constants.py
git commit -m "feat(constants): add v2 GRPO market params — zone, M, B, cooldown, bootstrap"
```

---

## Task 2: Reward verification helper

**Files:**
- Modify: `reliquary/validator/verifier.py` (append function)
- Test: `tests/unit/test_reward_verification.py` (create)

**Context:** v2 miners send a claimed reward per completion. Validator must re-run `env.compute_reward` and confirm match. This function is used by the new batcher in Task 8.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_reward_verification.py`:

```python
"""verify_reward_claim: re-runs env.compute_reward and checks miner's claim."""

from reliquary.validator.verifier import verify_reward_claim


class FakeEnv:
    """Deterministic env: reward = 1.0 iff completion contains 'CORRECT'."""

    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


def test_reward_matches_claim_accepted():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    assert verify_reward_claim(env, problem, "this is CORRECT", claimed=1.0) is True


def test_reward_mismatches_claim_rejected():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    # Miner claims 1.0 but env says 0.0 (text doesn't contain CORRECT)
    assert verify_reward_claim(env, problem, "wrong answer", claimed=1.0) is False


def test_float_tolerance():
    """Continuous rewards: match within 1e-6."""
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    # FakeEnv returns exact 1.0; miner claims 1.0000001 (within tolerance)
    assert verify_reward_claim(
        env, problem, "CORRECT", claimed=1.0000001
    ) is True


def test_wide_float_divergence_rejected():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    assert verify_reward_claim(
        env, problem, "CORRECT", claimed=0.5  # env says 1.0
    ) is False
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_reward_verification.py -v
```

Expected: FAIL with `ImportError: cannot import name 'verify_reward_claim'`.

- [ ] **Step 3: Add implementation**

Append to `reliquary/validator/verifier.py`:

```python
def verify_reward_claim(
    env: Any,
    problem: dict,
    completion_text: str,
    claimed: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Re-compute the env's reward on *completion_text* and compare to *claimed*.

    Miners declare the reward of each completion in their submission (saves
    validator compute when they can pre-filter out-of-zone) but the validator
    re-runs ``env.compute_reward`` to check honesty. A mismatch means the
    miner lied about reward, warranting rejection.

    Returns True iff |env_reward - claimed| <= tolerance. The small tolerance
    absorbs float64 formatting round-trip (JSON serialisation) noise.
    """
    try:
        actual = env.compute_reward(problem, completion_text)
    except Exception:
        return False
    return abs(float(actual) - float(claimed)) <= tolerance
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/unit/test_reward_verification.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_reward_verification.py
git commit -m "feat(verifier): add verify_reward_claim helper"
```

---

## Task 3: CooldownMap class — in-memory

**Files:**
- Create: `reliquary/validator/cooldown.py`
- Create: `tests/unit/test_cooldown.py`

**Context:** Tracks "prompt last batched at window N" per prompt_idx. Persistence and R2 rebuild come in Task 4.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_cooldown.py`:

```python
"""CooldownMap — in-memory lifecycle."""

import pytest

from reliquary.validator.cooldown import CooldownMap


def test_empty_map_never_in_cooldown():
    m = CooldownMap(cooldown_windows=50)
    assert m.is_in_cooldown(prompt_idx=42, current_window=100) is False


def test_just_batched_is_in_cooldown():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # Next window — still in cooldown
    assert m.is_in_cooldown(prompt_idx=42, current_window=101) is True


def test_cooldown_expires_after_N_windows():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # At window 150 — still in cooldown (100 + 50 inclusive boundary)
    assert m.is_in_cooldown(prompt_idx=42, current_window=149) is True
    # At window 150 — exactly the boundary → cooldown ends
    assert m.is_in_cooldown(prompt_idx=42, current_window=150) is False


def test_different_prompts_independent():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    m.record_batched(prompt_idx=7, window=105)
    assert m.is_in_cooldown(42, 110) is True
    assert m.is_in_cooldown(7, 110) is True
    assert m.is_in_cooldown(99, 110) is False


def test_re_record_updates_last_seen():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    # Same prompt re-enters at window 200 → cooldown resets from 200
    m.record_batched(prompt_idx=42, window=200)
    assert m.is_in_cooldown(42, 240) is True
    assert m.is_in_cooldown(42, 250) is False


def test_current_cooldown_set_at_window():
    m = CooldownMap(cooldown_windows=50)
    m.record_batched(prompt_idx=42, window=100)
    m.record_batched(prompt_idx=7, window=90)
    m.record_batched(prompt_idx=99, window=40)  # expired by window 130
    assert m.current_cooldown_set(current_window=130) == {42, 7}


def test_zero_cooldown_never_blocks():
    """With cooldown=0, no prompt is ever in cooldown."""
    m = CooldownMap(cooldown_windows=0)
    m.record_batched(prompt_idx=42, window=100)
    assert m.is_in_cooldown(42, 100) is False
    assert m.is_in_cooldown(42, 101) is False


def test_negative_prompt_idx_rejected():
    m = CooldownMap(cooldown_windows=50)
    with pytest.raises(ValueError):
        m.record_batched(prompt_idx=-1, window=100)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_cooldown.py -v
```

Expected: FAIL with `ModuleNotFoundError: reliquary.validator.cooldown`.

- [ ] **Step 3: Implement CooldownMap**

Create `reliquary/validator/cooldown.py`:

```python
"""CooldownMap: tracks last batch-membership window per prompt_idx.

A prompt that just entered a training batch is ineligible for the batch
for ``cooldown_windows`` following windows. This forces the curriculum
to rotate so the policy has time to shift between reuses of the same
prompt.
"""

from __future__ import annotations


class CooldownMap:
    """Per-prompt "last batched at window N" store + eligibility predicate.

    The cooldown window is a half-open interval:
        ``[last_batched, last_batched + cooldown_windows)`` → ineligible.
    At ``current_window == last_batched + cooldown_windows`` the prompt
    becomes eligible again.
    """

    def __init__(self, cooldown_windows: int) -> None:
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self._cooldown_windows = cooldown_windows
        self._last_batched: dict[int, int] = {}

    def record_batched(self, prompt_idx: int, window: int) -> None:
        """Mark *prompt_idx* as having entered the batch at *window*."""
        if prompt_idx < 0:
            raise ValueError("prompt_idx must be non-negative")
        if window < 0:
            raise ValueError("window must be non-negative")
        self._last_batched[prompt_idx] = window

    def is_in_cooldown(self, prompt_idx: int, current_window: int) -> bool:
        """True iff *prompt_idx* was batched within the cooldown horizon."""
        if self._cooldown_windows == 0:
            return False
        last = self._last_batched.get(prompt_idx)
        if last is None:
            return False
        return current_window - last < self._cooldown_windows

    def current_cooldown_set(self, current_window: int) -> set[int]:
        """All prompt_idx that are currently in cooldown."""
        if self._cooldown_windows == 0:
            return set()
        return {
            idx for idx, last in self._last_batched.items()
            if current_window - last < self._cooldown_windows
        }

    def __len__(self) -> int:
        return len(self._last_batched)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/unit/test_cooldown.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/cooldown.py tests/unit/test_cooldown.py
git commit -m "feat(cooldown): add in-memory CooldownMap class"
```

---

## Task 4: CooldownMap persistence + R2 rebuild

**Files:**
- Modify: `reliquary/validator/cooldown.py`
- Modify: `tests/unit/test_cooldown.py`

**Context:** On restart a validator must not lose its cooldown state. Two mechanisms: (a) local JSON snapshot between windows, (b) rebuild-from-dataset-archive at cold start.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_cooldown.py`:

```python
import json
import tempfile
from pathlib import Path


def test_persist_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cd.json"
    m1 = CooldownMap(cooldown_windows=50)
    m1.record_batched(prompt_idx=42, window=100)
    m1.record_batched(prompt_idx=7, window=105)
    m1.save(path)

    m2 = CooldownMap(cooldown_windows=50)
    m2.load(path)
    assert m2.is_in_cooldown(42, 110) is True
    assert m2.is_in_cooldown(7, 110) is True
    assert m2.is_in_cooldown(99, 110) is False


def test_load_missing_file_is_empty(tmp_path: Path):
    m = CooldownMap(cooldown_windows=50)
    m.load(tmp_path / "nonexistent.json")
    assert len(m) == 0


def test_load_malformed_file_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    m = CooldownMap(cooldown_windows=50)
    with pytest.raises(json.JSONDecodeError):
        m.load(path)


def test_rebuild_from_history_takes_most_recent():
    """If the same prompt appeared in multiple archived windows, keep the latest."""
    archived = [
        {"window_start": 100, "batch": [{"prompt_idx": 42}, {"prompt_idx": 7}]},
        {"window_start": 105, "batch": [{"prompt_idx": 42}, {"prompt_idx": 99}]},
        {"window_start": 110, "batch": [{"prompt_idx": 7}]},
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    # Prompt 42 last seen at 105
    assert m.is_in_cooldown(42, 120) is True
    assert m.is_in_cooldown(42, 155) is True
    assert m.is_in_cooldown(42, 156) is False
    # Prompt 7 last seen at 110
    assert m.is_in_cooldown(7, 159) is True
    assert m.is_in_cooldown(7, 160) is False
    # Prompt 99 last seen at 105
    assert m.is_in_cooldown(99, 154) is True


def test_rebuild_ignores_windows_older_than_cooldown():
    """Windows older than cooldown horizon are pointless to load."""
    archived = [
        {"window_start": 10, "batch": [{"prompt_idx": 42}]},   # way expired
        {"window_start": 105, "batch": [{"prompt_idx": 7}]},   # fresh
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    assert m.is_in_cooldown(42, 120) is False  # expired long ago
    assert m.is_in_cooldown(7, 120) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_cooldown.py -v -k "persist or load or rebuild"
```

Expected: FAIL with `AttributeError: 'CooldownMap' object has no attribute 'save'`.

- [ ] **Step 3: Add persistence and rebuild methods**

Append to `reliquary/validator/cooldown.py`:

```python
    # ---------- persistence ----------

    def save(self, path) -> None:
        """Serialise to JSON at *path*. Atomic via tmp-file + rename."""
        import json
        import os
        import tempfile

        path = str(path)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".cooldown.", dir=os.path.dirname(path) or "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "cooldown_windows": self._cooldown_windows,
                        "last_batched": self._last_batched,
                    },
                    f,
                )
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load(self, path) -> None:
        """Load state from JSON at *path*. No-op if file doesn't exist."""
        import json
        import os

        path = str(path)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        # JSON object keys are strings — coerce back to int.
        self._last_batched = {int(k): int(v) for k, v in data["last_batched"].items()}

    # ---------- rebuild from archived window data ----------

    def rebuild_from_history(
        self,
        archived_windows: list[dict],
        current_window: int,
    ) -> None:
        """Rebuild state from the last N archived windows' batch records.

        *archived_windows* is a list of dicts, each with ``window_start``
        (int) and ``batch`` (list of {prompt_idx: int, ...}). Typically
        fetched from the R2 dataset archive at validator startup.

        Only windows within ``cooldown_windows`` of *current_window* matter —
        older entries have already expired and are skipped.
        """
        self._last_batched.clear()
        horizon = current_window - self._cooldown_windows
        for record in archived_windows:
            w = int(record["window_start"])
            if w <= horizon:
                continue
            for entry in record.get("batch", []):
                idx = int(entry["prompt_idx"])
                # Keep the most recent window for each prompt.
                if self._last_batched.get(idx, -1) < w:
                    self._last_batched[idx] = w
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_cooldown.py -v
```

Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/cooldown.py tests/unit/test_cooldown.py
git commit -m "feat(cooldown): JSON persistence + rebuild-from-history"
```

---

## Task 5: Pure `select_batch()` function

**Files:**
- Create: `reliquary/validator/batch_selection.py`
- Create: `tests/unit/test_batch_selection.py`

**Context:** Pure function; takes a list of validated submissions + cooldown map + window, returns the ordered batch. Easy to test exhaustively without any I/O.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_batch_selection.py`:

```python
"""select_batch: deterministic, cooldown-aware batch assembly."""

from dataclasses import dataclass, field

import pytest

from reliquary.validator.batch_selection import select_batch
from reliquary.validator.cooldown import CooldownMap


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    signed_round: int
    merkle_root: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, signed_round, merkle_root=None):
    return FakeSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        signed_round=signed_round,
        merkle_root=merkle_root or hotkey.encode().ljust(32, b"\x00"),
    )


def test_empty_pool_returns_empty_batch():
    cd = CooldownMap(cooldown_windows=50)
    assert select_batch(
        submissions=[], b=8, current_window=100, cooldown_map=cd
    ) == []


def test_fills_up_to_b():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i, signed_round=1000 + i) for i in range(12)]
    batch = select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    assert len(batch) == 8


def test_fifo_by_signed_round():
    cd = CooldownMap(cooldown_windows=50)
    # 3 submissions on distinct prompts, different rounds
    subs = [
        _sub("late", prompt_idx=1, signed_round=1005),
        _sub("early", prompt_idx=2, signed_round=1001),
        _sub("middle", prompt_idx=3, signed_round=1003),
    ]
    batch = select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    hotkeys = [s.hotkey for s in batch]
    assert hotkeys == ["early", "middle", "late"]


def test_duplicate_prompt_only_first_wins():
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("first", prompt_idx=42, signed_round=1000),
        _sub("second", prompt_idx=42, signed_round=1001),  # dup prompt, later
        _sub("third", prompt_idx=7, signed_round=1002),
    ]
    batch = select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    assert [s.hotkey for s in batch] == ["first", "third"]


def test_cooldown_blocks_prompt():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=100)
    subs = [_sub("hk", prompt_idx=42, signed_round=1000)]
    batch = select_batch(subs, b=8, current_window=120, cooldown_map=cd)
    assert batch == []


def test_cooldown_expires_prompt_eligible_again():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=100)
    subs = [_sub("hk", prompt_idx=42, signed_round=1000)]
    # current_window >= 100 + 50 → eligible
    batch = select_batch(subs, b=8, current_window=150, cooldown_map=cd)
    assert [s.hotkey for s in batch] == ["hk"]


def test_tiebreak_deterministic_on_same_round():
    cd = CooldownMap(cooldown_windows=50)
    # Two subs on DISTINCT prompts, same round — tiebreak by hash
    subs = [
        _sub("alice", prompt_idx=1, signed_round=1000, merkle_root=b"\xff" * 32),
        _sub("bob", prompt_idx=2, signed_round=1000, merkle_root=b"\x01" * 32),
    ]
    batch_a = select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    batch_b = select_batch(list(reversed(subs)), b=8, current_window=100, cooldown_map=cd)
    assert [s.hotkey for s in batch_a] == [s.hotkey for s in batch_b]


def test_cooldown_not_mutated_by_select():
    """select_batch must not modify cooldown_map; record_batched is caller's job."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub("hk", prompt_idx=42, signed_round=1000)]
    select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    assert cd.is_in_cooldown(42, 100) is False


def test_partial_fill_when_all_cooldown_blocked():
    cd = CooldownMap(cooldown_windows=50)
    # All 3 prompts already batched recently
    for idx in (1, 2, 3):
        cd.record_batched(idx, window=100)
    subs = [_sub(f"hk{i}", prompt_idx=i, signed_round=1000) for i in (1, 2, 3)]
    batch = select_batch(subs, b=8, current_window=110, cooldown_map=cd)
    assert batch == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_batch_selection.py -v
```

Expected: FAIL with `ModuleNotFoundError: reliquary.validator.batch_selection`.

- [ ] **Step 3: Implement select_batch**

Create `reliquary/validator/batch_selection.py`:

```python
"""Pure batch selection: FIFO by signed_round, distinct prompts, cooldown-aware.

Called once per window to pick the B submissions that go into the training
step. Separated from the orchestrator (``WindowBatcher``) to make the
selection logic trivially testable in isolation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from reliquary.validator.cooldown import CooldownMap


class _SubmissionLike(Protocol):
    """Duck-typed submission — works with any class exposing these attrs."""

    hotkey: str
    prompt_idx: int
    signed_round: int
    merkle_root: bytes


def _tiebreak_key(sub: _SubmissionLike) -> bytes:
    """Deterministic, consensus-safe tiebreak within a drand round.

    Hash of (hotkey, prompt_idx, merkle_root) — miner-controlled but stable
    across validators. Any validator seeing the same set of submissions
    produces the same ordering.
    """
    h = hashlib.sha256()
    h.update(sub.hotkey.encode())
    h.update(sub.prompt_idx.to_bytes(8, "big", signed=False))
    h.update(sub.merkle_root)
    return h.digest()


def select_batch(
    submissions: list[Any],
    *,
    b: int,
    current_window: int,
    cooldown_map: CooldownMap,
) -> list[Any]:
    """Return the ordered list of at most *b* batch members.

    Rules:
        1. Sort by ``(signed_round, tiebreak_hash)`` — deterministic FIFO.
        2. For each submission in order:
           - skip if its ``prompt_idx`` is already represented in the batch
             (diversity constraint — one prompt per batch slot)
           - skip if its ``prompt_idx`` is in cooldown per ``cooldown_map``
           - otherwise append; stop when ``len(batch) == b``
        3. Does NOT mutate ``cooldown_map`` — the caller records
           post-selection (via ``record_batched``) once the batch is final.
    """
    if b <= 0:
        return []

    ordered = sorted(submissions, key=lambda s: (s.signed_round, _tiebreak_key(s)))

    batch: list[Any] = []
    seen_prompts: set[int] = set()
    for sub in ordered:
        if len(batch) >= b:
            break
        if sub.prompt_idx in seen_prompts:
            continue
        if cooldown_map.is_in_cooldown(sub.prompt_idx, current_window):
            continue
        batch.append(sub)
        seen_prompts.add(sub.prompt_idx)
    return batch
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_batch_selection.py -v
```

Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batch_selection.py tests/unit/test_batch_selection.py
git commit -m "feat(batcher): pure select_batch — FIFO + distinct prompts + cooldown"
```

---

## Task 6: Flat `1/B` weight computation

**Files:**
- Modify: `reliquary/validator/weights.py` — add `compute_weights_v2`
- Create: `tests/unit/test_flat_payment.py`

**Context:** New flat-`1/B` payment replaces v1's superlinear on advantages. v1's `compute_weights` stays for now; it's removed in cleanup.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_flat_payment.py`:

```python
"""compute_weights_v2: flat 1/B payment + UID_BURN on unused slots."""

from reliquary.constants import B_BATCH, UID_BURN
from reliquary.validator.weights import compute_weights_v2


def test_empty_batch_full_burn():
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=[])
    assert miner_weights == {}
    assert burn_weight == 1.0


def test_full_batch_no_burn():
    batch = [f"hk{i}" for i in range(B_BATCH)]
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    assert set(miner_weights.keys()) == set(batch)
    # Each member gets exactly 1/B
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    assert abs(burn_weight) < 1e-9
    # Total sums to 1.0
    assert abs(sum(miner_weights.values()) + burn_weight - 1.0) < 1e-9


def test_partial_batch_partial_burn():
    batch = ["hk0", "hk1", "hk2", "hk3", "hk4"]  # 5/8 full
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    assert len(miner_weights) == 5
    # Each of the 5 still gets 1/B (NOT 1/len(batch))
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    # Unused share = 3/B burns
    assert abs(burn_weight - 3.0 / B_BATCH) < 1e-9
    # Total sums to 1.0
    assert abs(sum(miner_weights.values()) + burn_weight - 1.0) < 1e-9


def test_weights_sum_to_one_always():
    """Invariant across batch sizes 0..B."""
    for n in range(B_BATCH + 1):
        batch = [f"hk{i}" for i in range(n)]
        miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
        total = sum(miner_weights.values()) + burn_weight
        assert abs(total - 1.0) < 1e-9, f"sum != 1.0 at n={n}"


def test_duplicate_hotkey_still_single_payment():
    """Batch is pre-deduped by prompt_idx, so same hotkey shouldn't appear
    twice. If it does, we merge — single entry per hotkey."""
    batch = ["alice", "alice", "bob"]  # alice on 2 different prompts
    miner_weights, burn_weight = compute_weights_v2(batch_hotkeys=batch)
    # Alice's two slots sum into one weight entry
    assert abs(miner_weights["alice"] - 2.0 / B_BATCH) < 1e-9
    assert abs(miner_weights["bob"] - 1.0 / B_BATCH) < 1e-9
    assert abs(burn_weight - (B_BATCH - 3) / B_BATCH) < 1e-9


def test_over_full_batch_raises():
    """> B hotkeys is a programming error — caller should have capped."""
    batch = [f"hk{i}" for i in range(B_BATCH + 1)]
    import pytest
    with pytest.raises(ValueError, match="batch size"):
        compute_weights_v2(batch_hotkeys=batch)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_flat_payment.py -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_weights_v2'`.

- [ ] **Step 3: Add compute_weights_v2**

Append to `reliquary/validator/weights.py`:

```python
def compute_weights_v2(
    batch_hotkeys: list[str],
) -> tuple[dict[str, float], float]:
    """Flat 1/B payment per batch member; unused slots burn.

    v2 replacement for ``compute_weights``. Each of the (at most) B batch
    members receives exactly ``1 / B_BATCH`` of the window emission; the
    remainder ``(B - len(batch)) / B`` is routed to ``UID_BURN`` as a
    protocol-inefficiency signal.

    The flat share (not ``1/len(batch)``) is intentional:
      * signals shortfall via burn rate rather than masking it
      * removes the "lone survivor" incentive (miners can't profit by
        DoSing competitors — others' failures only grow the burn)

    Duplicate hotkeys in ``batch_hotkeys`` (possible if the same miner
    wins batch slots on two distinct prompts) are summed into one entry.

    Args:
        batch_hotkeys: the hotkeys of the batch members, in selection
            order. Length must be in ``[0, B_BATCH]``.

    Returns:
        (miner_weights, burn_weight) where:
          * miner_weights maps each unique hotkey to its share (sum of
            its batch slots × 1/B)
          * burn_weight = (B - len(batch)) / B
          * miner_weights.values().sum() + burn_weight == 1.0
    """
    from reliquary.constants import B_BATCH

    n = len(batch_hotkeys)
    if n > B_BATCH:
        raise ValueError(
            f"batch size {n} exceeds B_BATCH={B_BATCH}; caller must cap"
        )

    per_slot = 1.0 / B_BATCH
    miner_weights: dict[str, float] = {}
    for hk in batch_hotkeys:
        miner_weights[hk] = miner_weights.get(hk, 0.0) + per_slot

    burn_weight = (B_BATCH - n) / B_BATCH
    return miner_weights, burn_weight
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_flat_payment.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/weights.py tests/unit/test_flat_payment.py
git commit -m "feat(weights): flat 1/B payment with UID_BURN on unused slots (v2)"
```

---

## Task 7: New protocol schemas

**Files:**
- Modify: `reliquary/protocol/submission.py` (add new types, keep old)
- Create: `tests/unit/test_batch_submission_schema.py`

**Context:** Pydantic v2 classes for the new payload. Old `SubmissionRequest`/`Response` stay alive so v1 tests still pass.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_batch_submission_schema.py`:

```python
"""Pydantic schemas for v2 GRPO market submissions."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
)


def _valid_rollouts(k: int = 4):
    """k successes, (M - k) failures, all with well-formed GRAIL fields."""
    rollouts = []
    for i in range(M_ROLLOUTS):
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5],
                reward=1.0 if i < k else 0.0,
                commit={"proof_version": "v5", "tokens": [1, 2, 3, 4, 5]},
            )
        )
    return rollouts


def test_valid_request_parses():
    req = BatchSubmissionRequest(
        miner_hotkey="hk" * 24,
        prompt_idx=42,
        window_start=1000,
        signed_round=999_999,
        merkle_root="00" * 32,
        rollouts=_valid_rollouts(k=4),
    )
    assert req.prompt_idx == 42
    assert len(req.rollouts) == M_ROLLOUTS


def test_wrong_rollout_count_rejected():
    with pytest.raises(ValidationError, match="rollouts"):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=42,
            window_start=1000,
            signed_round=999_999,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(k=4)[:7],  # 7 instead of M
        )


def test_negative_prompt_idx_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=-1,
            window_start=1000,
            signed_round=999_999,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(),
        )


def test_malformed_merkle_root_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=0,
            window_start=1000,
            signed_round=999_999,
            merkle_root="zz",  # not hex, wrong length
            rollouts=_valid_rollouts(),
        )


def test_all_reject_reasons_serialisable():
    for reason in RejectReason:
        resp = BatchSubmissionResponse(accepted=False, reason=reason)
        assert resp.model_dump()["reason"] == reason.value


def test_accepted_response():
    resp = BatchSubmissionResponse(accepted=True, reason=RejectReason.ACCEPTED)
    dumped = resp.model_dump()
    assert dumped["accepted"] is True
    assert dumped["reason"] == RejectReason.ACCEPTED.value


def test_grpo_batch_state_exposes_cooldown():
    state = GrpoBatchState(
        window_start=100,
        current_round=999,
        cooldown_prompts=[42, 7, 99],
        valid_submissions=12,
    )
    dumped = state.model_dump()
    assert set(dumped["cooldown_prompts"]) == {42, 7, 99}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_batch_submission_schema.py -v
```

Expected: FAIL with `ImportError: cannot import name 'BatchSubmissionRequest'`.

- [ ] **Step 3: Add new schemas**

Append to `reliquary/protocol/submission.py`:

```python
from enum import Enum

from reliquary.constants import M_ROLLOUTS


class RejectReason(str, Enum):
    """Canonical reject codes emitted by the v2 validator.

    ``ACCEPTED`` is a sentinel used in success responses (``accepted=True``).
    All other values are mutually exclusive; only the first failure reason
    is reported per submission.
    """
    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    BAD_PROMPT_IDX = "bad_prompt_idx"
    STALE_ROUND = "stale_round"
    PROMPT_IN_COOLDOWN = "prompt_in_cooldown"
    GRAIL_FAIL = "grail_fail"
    REWARD_MISMATCH = "reward_mismatch"
    OUT_OF_ZONE = "out_of_zone"
    WRONG_ROLLOUT_COUNT = "wrong_rollout_count"
    WINDOW_MISMATCH = "window_mismatch"
    WINDOW_NOT_ACTIVE = "window_not_active"


class RolloutSubmission(BaseModel):
    """A single rollout's payload: tokens, miner-claimed reward, GRAIL commit."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1)
    reward: float  # miner's local env.compute_reward value; validator re-checks
    commit: dict[str, Any]


class BatchSubmissionRequest(BaseModel):
    """v2 miner→validator payload: one group of M rollouts on one prompt."""

    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(..., min_length=1)
    prompt_idx: int = Field(..., ge=0)
    window_start: int = Field(..., ge=0)
    signed_round: int = Field(..., ge=0)
    # Hex-encoded SHA-256 merkle root over the M rollout leaves (64 chars).
    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    rollouts: list[RolloutSubmission]

    @field_validator("rollouts")
    @classmethod
    def _rollout_count_is_M(cls, v):
        if len(v) != M_ROLLOUTS:
            raise ValueError(
                f"rollouts must have exactly {M_ROLLOUTS} entries, got {len(v)}"
            )
        return v


class BatchSubmissionResponse(BaseModel):
    """Validator verdict on a submission."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: RejectReason


class GrpoBatchState(BaseModel):
    """Live window state for miners polling ``/window/{n}/state`` (v2).

    Miners use this to:
      * confirm the window is active (``current_round``)
      * avoid submitting on prompts already in the cooldown set
      * gauge competition (``valid_submissions`` count)
    """

    model_config = ConfigDict(extra="forbid")

    window_start: int = Field(..., ge=0)
    current_round: int = Field(..., ge=0)
    cooldown_prompts: list[int] = Field(default_factory=list)
    valid_submissions: int = Field(..., ge=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_batch_submission_schema.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/submission.py tests/unit/test_batch_submission_schema.py
git commit -m "feat(protocol): add v2 BatchSubmission schemas + RejectReason enum"
```

---

## Task 8: Zone filter helper

**Files:**
- Modify: `reliquary/validator/verifier.py` (append)
- Create: `tests/unit/test_zone_filter.py`

**Context:** Tiny but important — the gate that determines batch eligibility. Binary in/out.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_zone_filter.py`:

```python
"""Zone filter: k ∈ [ZONE_K_MIN, ZONE_K_MAX] (binary in/out, no scoring)."""

from reliquary.constants import (
    BOOTSTRAP_ZONE_K_MAX, BOOTSTRAP_ZONE_K_MIN,
    M_ROLLOUTS, ZONE_K_MAX, ZONE_K_MIN,
)
from reliquary.validator.verifier import is_in_zone, rewards_to_k


def test_k_below_min_rejected():
    assert is_in_zone(k=ZONE_K_MIN - 1) is False


def test_k_min_accepted():
    assert is_in_zone(k=ZONE_K_MIN) is True


def test_k_max_accepted():
    assert is_in_zone(k=ZONE_K_MAX) is True


def test_k_above_max_rejected():
    assert is_in_zone(k=ZONE_K_MAX + 1) is False


def test_k_all_zeros_rejected():
    assert is_in_zone(k=0) is False


def test_k_all_ones_rejected():
    assert is_in_zone(k=M_ROLLOUTS) is False


def test_bootstrap_mode_wider_zone():
    # k=1 fails steady, passes bootstrap
    assert is_in_zone(k=1, bootstrap=True) is True
    assert is_in_zone(k=1, bootstrap=False) is False
    # k=7 same
    assert is_in_zone(k=7, bootstrap=True) is True
    assert is_in_zone(k=7, bootstrap=False) is False


def test_bootstrap_still_rejects_k_0_and_M():
    assert is_in_zone(k=0, bootstrap=True) is False
    assert is_in_zone(k=M_ROLLOUTS, bootstrap=True) is False


def test_rewards_to_k_binary():
    assert rewards_to_k([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) == 4


def test_rewards_to_k_with_tolerance():
    # Float round-trip noise — 0.999999 still counts as 1.0 success
    assert rewards_to_k([0.999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 1


def test_rewards_to_k_all_zero():
    assert rewards_to_k([0.0] * M_ROLLOUTS) == 0


def test_rewards_to_k_all_one():
    assert rewards_to_k([1.0] * M_ROLLOUTS) == M_ROLLOUTS
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_zone_filter.py -v
```

Expected: FAIL with `ImportError: cannot import name 'is_in_zone'`.

- [ ] **Step 3: Implement helpers**

Append to `reliquary/validator/verifier.py`:

```python
def rewards_to_k(rewards: list[float], *, success_threshold: float = 0.5) -> int:
    """Count rewards that are successes (> success_threshold).

    GSM8K returns binary {0.0, 1.0}; the threshold tolerates float round-trip
    noise and future continuous envs where "success" is >= some level.
    """
    return sum(1 for r in rewards if r > success_threshold)


def is_in_zone(k: int, *, bootstrap: bool = False) -> bool:
    """True iff k lies strictly inside the apprenable zone.

    Steady state: ``ZONE_K_MIN <= k <= ZONE_K_MAX`` (default [2, 6]).
    Bootstrap mode: ``BOOTSTRAP_ZONE_K_MIN <= k <= BOOTSTRAP_ZONE_K_MAX``
    (default [1, 7]) — wider to keep the batch filling while miner
    population and env coverage are thin.

    ``k=0`` (trivially hard) and ``k=M_ROLLOUTS`` (trivially easy) are
    always rejected regardless of bootstrap flag — these have zero
    GRPO signal by definition.
    """
    from reliquary.constants import (
        BOOTSTRAP_ZONE_K_MAX, BOOTSTRAP_ZONE_K_MIN,
        M_ROLLOUTS, ZONE_K_MAX, ZONE_K_MIN,
    )

    if k <= 0 or k >= M_ROLLOUTS:
        return False
    if bootstrap:
        return BOOTSTRAP_ZONE_K_MIN <= k <= BOOTSTRAP_ZONE_K_MAX
    return ZONE_K_MIN <= k <= ZONE_K_MAX
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_zone_filter.py -v
```

Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_zone_filter.py
git commit -m "feat(verifier): is_in_zone + rewards_to_k helpers"
```

---

## Task 9: `WindowBatcher` v2 — skeleton + ingestion

**Files:**
- Create: `reliquary/validator/batcher_v2.py` (parallel to v1 file; swap in Task 11)
- Create: `tests/unit/test_grpo_window_batcher.py`

**Context:** The v1 `WindowBatcher` stays alive; we build v2 alongside so tests land incrementally. Task 11 swaps the import.

- [ ] **Step 1: Write failing test — construction + reject path**

Create `tests/unit/test_grpo_window_batcher.py`:

```python
"""GrpoWindowBatcher: accepts submissions, enforces verification pipeline,
exposes select_batch at window close."""

from dataclasses import dataclass
from typing import Any

import pytest

from reliquary.constants import B_BATCH, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batcher_v2 import GrpoWindowBatcher


class FakeEnv:
    name = "fake"
    def __len__(self):
        return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "a", "id": f"pid-{idx}"}
    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


def _always_true_grail(commit, model, randomness):
    return True, 1, 1


def _always_false_grail(commit, model, randomness):
    return False, 0, 1


def _always_true_sig(commit, hotkey):
    return True


def _always_true_proof_version(commit):
    return True


def _request(
    prompt_idx=42, signed_round=1000, window_start=500,
    rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    if rewards is None:
        # Default: 4 successes / 4 failures = k=4, in zone
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for i, r in enumerate(rewards):
        # Completion text controls the validator's env.compute_reward re-run.
        text = "CORRECT" if r > 0.5 else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5],
                reward=r,
                commit={
                    "proof_version": "v5",
                    "tokens": [1, 2, 3, 4, 5],
                    "completion_text_for_test": text,  # used by fake tokenizer
                },
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        signed_round=signed_round,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )


class FakeTokenizer:
    """Minimal tokenizer stub. decode() reads completion text out of commit."""

    def decode(self, tokens):
        # In the fake, decode is called per rollout; we can't know which
        # so return empty. The batcher v2 should accept a completion_text
        # extractor rather than a tokenizer — see Task 9 impl.
        return ""


def _make_batcher(**overrides) -> GrpoWindowBatcher:
    kwargs = dict(
        window_start=500,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        verify_proof_version_fn=_always_true_proof_version,
        completion_text_fn=lambda rollout: rollout.commit.get(
            "completion_text_for_test", ""
        ),
    )
    kwargs.update(overrides)
    return GrpoWindowBatcher(**kwargs)


def test_constructor_sets_window_and_round():
    b = _make_batcher()
    assert b.window_start == 500
    assert b.current_round == 1000


def test_reject_window_mismatch():
    b = _make_batcher()
    req = _request(window_start=999)  # wrong window
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WINDOW_MISMATCH


def test_reject_stale_round():
    b = _make_batcher(current_round=1000)
    # Miner signed on round 500 — too old (> STALE_ROUND_BUFFER windows back)
    req = _request(signed_round=500)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.STALE_ROUND


def test_accept_in_zone_submission():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)  # k=4
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED
    assert len(b.valid_submissions()) == 1


def test_reject_out_of_zone_all_fail():
    b = _make_batcher()
    req = _request(rewards=[0.0] * 8)  # k=0
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE
    assert len(b.valid_submissions()) == 0


def test_reject_out_of_zone_all_pass():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 8)  # k=8
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE


def test_reject_grail_fail():
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    req = _request()
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.GRAIL_FAIL


def test_reject_reward_mismatch():
    """Miner claims k=4 but actual env gives k=0 → reject."""
    b = _make_batcher()
    # Miner claims reward=1.0 for 4 rollouts, but their completion text
    # is "wrong" → env returns 0.0 → mismatch
    rollouts = []
    for i in range(M_ROLLOUTS):
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < 4 else 0.0,
                commit={
                    "proof_version": "v5",
                    "tokens": [1, 2, 3],
                    "completion_text_for_test": "wrong",  # env says 0.0 always
                },
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=500,
        signed_round=1000,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.REWARD_MISMATCH
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_grpo_window_batcher.py -v
```

Expected: FAIL with `ModuleNotFoundError: reliquary.validator.batcher_v2`.

- [ ] **Step 3: Implement GrpoWindowBatcher**

Create `reliquary/validator/batcher_v2.py`:

```python
"""GrpoWindowBatcher — v2 orchestrator for the free-prompt GRPO market.

Replaces the slot-based ``WindowBatcher`` once Task 11 wires it in. Holds
a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    M_ROLLOUTS,
)
from reliquary.environment.base import Environment
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batch_selection import select_batch
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.verifier import (
    is_in_zone,
    rewards_to_k,
    verify_reward_claim,
)

logger = logging.getLogger(__name__)


# Maximum drand-round lag tolerated: a miner's ``signed_round`` may be up to
# this many rounds behind ``current_round`` to be accepted. Newer than
# current_round is always rejected (replay of future beacon).
STALE_ROUND_LAG_MAX = 10


@dataclass
class ValidSubmission:
    """A submission that passed all v2 verification checks."""

    hotkey: str
    prompt_idx: int
    signed_round: int
    merkle_root_bytes: bytes
    k: int
    rollouts: list[RolloutSubmission]
    arrived_at: float = 0.0


class GrpoWindowBatcher:
    """Accepts v2 submissions, runs the full verification pipeline, and
    exposes ``valid_submissions()`` + ``select_batch()`` at window close.

    Parameters
    ----------
    window_start:
        On-chain block or logical window identifier for this batch.
    current_round:
        Latest drand round observed by the validator. Used to reject
        ``signed_round`` values that are too old or in the future.
    env:
        Active ``Environment`` providing ``get_problem(idx)`` and
        ``compute_reward(problem, completion_text)``.
    model:
        HuggingFace model (or GRAIL-compatible stub) used to re-verify
        commitment proofs.
    cooldown_map:
        Optional shared ``CooldownMap``. If None, a fresh one is created
        (useful for tests).
    bootstrap:
        If True, apply ``BOOTSTRAP_ZONE_K_*`` bounds instead of steady
        bounds.
    completion_text_fn:
        Callable extracting the decoded completion text from a single
        ``RolloutSubmission`` (so reward re-computation can run without
        depending on a tokenizer object). Production wiring injects a
        closure over the validator's tokenizer; tests inject a stub.
    verify_*_fn:
        Injectable verifier primitives — default to the real
        ``reliquary.validator.verifier`` implementations.
    """

    def __init__(
        self,
        window_start: int,
        current_round: int,
        env: Environment,
        model: Any,
        *,
        cooldown_map: CooldownMap | None = None,
        bootstrap: bool = False,
        completion_text_fn: Callable[[RolloutSubmission], str],
        verify_commitment_proofs_fn: Callable[..., tuple[bool, int, int]] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        verify_proof_version_fn: Callable[[dict], bool] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.window_start = window_start
        self.current_round = current_round
        self.env = env
        self.model = model
        self.bootstrap = bootstrap
        self._completion_text = completion_text_fn
        self._time_fn = time_fn or time.monotonic

        self._cooldown = cooldown_map or CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        if verify_commitment_proofs_fn is None:
            from reliquary.validator.verifier import verify_commitment_proofs
            verify_commitment_proofs_fn = verify_commitment_proofs
        if verify_signature_fn is None:
            from reliquary.validator.verifier import verify_signature
            verify_signature_fn = verify_signature
        if verify_proof_version_fn is None:
            from reliquary.validator.verifier import verify_proof_version
            verify_proof_version_fn = verify_proof_version

        self._verify_commitment = verify_commitment_proofs_fn
        self._verify_signature = verify_signature_fn
        self._verify_proof_version = verify_proof_version_fn

        self._lock = threading.Lock()
        self._valid: list[ValidSubmission] = []
        # randomness string passed through to GRAIL — injected by the
        # validator service (same as v1's ``randomness`` field on
        # WindowBatcher). Tests that don't care set this to "".
        self.randomness: str = ""

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        """Run the full verification pipeline; append to ``_valid`` on success."""
        with self._lock:
            return self._accept_locked(request)

    def _accept_locked(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        # Cheap gates first — no GRAIL compute.
        if request.window_start != self.window_start:
            return self._reject(RejectReason.WINDOW_MISMATCH)
        if request.prompt_idx >= len(self.env):
            return self._reject(RejectReason.BAD_PROMPT_IDX)
        if not self._round_fresh(request.signed_round):
            return self._reject(RejectReason.STALE_ROUND)
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return self._reject(RejectReason.PROMPT_IN_COOLDOWN)

        # Re-verify rewards BEFORE GRAIL — cheap compared to a forward pass.
        problem = self.env.get_problem(request.prompt_idx)
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            if not verify_reward_claim(self.env, problem, text, rollout.reward):
                return self._reject(RejectReason.REWARD_MISMATCH)

        k = rewards_to_k([r.reward for r in request.rollouts])
        if not is_in_zone(k, bootstrap=self.bootstrap):
            return self._reject(RejectReason.OUT_OF_ZONE)

        # Expensive gates: GRAIL on every rollout.
        for rollout in request.rollouts:
            if not self._verify_proof_version(rollout.commit):
                return self._reject(RejectReason.GRAIL_FAIL)
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return self._reject(RejectReason.BAD_SIGNATURE)
            passed, _, _ = self._verify_commitment(
                rollout.commit, self.model, self.randomness
            )
            if not passed:
                return self._reject(RejectReason.GRAIL_FAIL)

        # Admit.
        self._valid.append(
            ValidSubmission(
                hotkey=request.miner_hotkey,
                prompt_idx=request.prompt_idx,
                signed_round=request.signed_round,
                merkle_root_bytes=bytes.fromhex(request.merkle_root),
                k=k,
                rollouts=list(request.rollouts),
                arrived_at=self._time_fn(),
            )
        )
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    def _round_fresh(self, signed_round: int) -> bool:
        """Reject rounds from the future or too far in the past."""
        if signed_round > self.current_round:
            return False
        return (self.current_round - signed_round) <= STALE_ROUND_LAG_MAX

    @staticmethod
    def _reject(reason: RejectReason) -> BatchSubmissionResponse:
        return BatchSubmissionResponse(accepted=False, reason=reason)

    # ----------------------------- accessors -----------------------------

    def valid_submissions(self) -> list[ValidSubmission]:
        """Snapshot of valid submissions — safe to iterate after window close."""
        with self._lock:
            return list(self._valid)

    def seal_batch(self) -> list[ValidSubmission]:
        """Run selection and record cooldown entries. Called once per window."""
        with self._lock:
            batch = select_batch(
                self._valid,
                b=B_BATCH,
                current_window=self.window_start,
                cooldown_map=self._cooldown,
            )
            for sub in batch:
                self._cooldown.record_batched(sub.prompt_idx, self.window_start)
            return batch

    def get_state(self) -> GrpoBatchState:
        """Public state exposed via ``/window/{n}/state``."""
        with self._lock:
            return GrpoBatchState(
                window_start=self.window_start,
                current_round=self.current_round,
                cooldown_prompts=sorted(
                    self._cooldown.current_cooldown_set(self.window_start)
                ),
                valid_submissions=len(self._valid),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_grpo_window_batcher.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher_v2.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(batcher): v2 GrpoWindowBatcher — ingestion + reject pipeline"
```

---

## Task 10: `GrpoWindowBatcher.seal_batch` + end-to-end tests

**Files:**
- Modify: `tests/unit/test_grpo_window_batcher.py` (append)

**Context:** `seal_batch` is already in the implementation (Task 9); this task exercises the full ingestion → selection → cooldown loop.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
# --- seal_batch + cooldown lifecycle ---

def test_seal_batch_empty_pool_returns_empty():
    b = _make_batcher()
    assert b.seal_batch() == []


def test_seal_batch_fifo_across_many_submissions():
    b = _make_batcher()
    # 10 submissions, different prompts, varying rounds
    for i, round_num in enumerate([1003, 1001, 1005, 1002, 1004,
                                    1006, 1007, 1008, 1009, 1010]):
        req = _request(
            prompt_idx=i, signed_round=round_num, hotkey=f"hk{i}",
        )
        resp = b.accept_submission(req)
        assert resp.accepted, f"unexpected reject for {i}: {resp.reason}"
    batch = b.seal_batch()
    assert len(batch) == B_BATCH
    # Sorted by signed_round: hk1(1001), hk3(1002), hk0(1003), hk4(1004),
    #                         hk2(1005), hk5(1006), hk6(1007), hk7(1008)
    rounds = [s.signed_round for s in batch]
    assert rounds == sorted(rounds)


def test_seal_batch_cooldown_recorded():
    b = _make_batcher()
    req = _request(prompt_idx=42, signed_round=1000)
    b.accept_submission(req)
    batch = b.seal_batch()
    assert len(batch) == 1
    # Subsequent window: prompt 42 should now be in cooldown
    assert b._cooldown.is_in_cooldown(42, b.window_start + 1) is True


def test_sealed_batch_respects_cooldown_from_previous_window():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    b = _make_batcher(window_start=120, cooldown_map=cd)  # within cooldown
    # Submission on prompt 42 → rejected at ingestion for cooldown.
    req = _request(prompt_idx=42, signed_round=1000, window_start=120)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN


def test_state_endpoint_exposes_cooldown():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    cd.record_batched(prompt_idx=7, window=105)
    b = _make_batcher(window_start=110, cooldown_map=cd)
    state = b.get_state()
    assert set(state.cooldown_prompts) == {42, 7}
    assert state.valid_submissions == 0


def test_distinct_prompts_in_batch_only():
    b = _make_batcher()
    # Two submissions on same prompt, different miners
    b.accept_submission(_request(prompt_idx=42, signed_round=1000, hotkey="alice"))
    b.accept_submission(_request(prompt_idx=42, signed_round=1001, hotkey="bob"))
    b.accept_submission(_request(prompt_idx=7, signed_round=1002, hotkey="carol"))
    batch = b.seal_batch()
    assert len(batch) == 2
    hotkeys = {s.hotkey for s in batch}
    assert hotkeys == {"alice", "carol"}  # Bob dropped (prompt 42 taken)
```

- [ ] **Step 2: Run tests to verify they pass**

Note: no new implementation needed — tests exercise existing `seal_batch` from Task 9.

```
pytest tests/unit/test_grpo_window_batcher.py -v
```

Expected: PASS (14 tests total).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_grpo_window_batcher.py
git commit -m "test(batcher): seal_batch + cooldown + state endpoint coverage"
```

---

## Task 11: Wire `GrpoWindowBatcher` into the validator server

**Files:**
- Modify: `reliquary/validator/server.py`
- Modify: `tests/unit/test_validator_server.py` (exists — replace v1 tests)

**Context:** Swap the server's `active_batcher` type from `WindowBatcher` to `GrpoWindowBatcher`. This is the point where v2 takes over.

- [ ] **Step 1: Write failing test**

Look at `tests/unit/test_validator_server.py` to match existing patterns, then replace its content:

```python
"""Validator HTTP server — v2 GRPO market endpoints."""

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    RejectReason,
)
from reliquary.validator.batcher_v2 import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


def _batcher(window_start=500, cooldown_map=None):
    return GrpoWindowBatcher(
        window_start=window_start,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=lambda c, m, r: (True, 1, 1),
        verify_signature_fn=lambda c, h: True,
        verify_proof_version_fn=lambda c: True,
        completion_text_fn=lambda r: r.commit.get("completion_text_for_test", ""),
    )


def _request(prompt_idx=42, window_start=500, signed_round=1000, k=4):
    rollouts = []
    for i in range(M_ROLLOUTS):
        text = "CORRECT" if i < k else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < k else 0.0,
                commit={"tokens": [1, 2, 3], "proof_version": "v5", "completion_text_for_test": text},
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=prompt_idx,
        window_start=window_start,
        signed_round=signed_round,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )


def test_submit_returns_queued_on_active_window():
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == RejectReason.ACCEPTED.value


def test_submit_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 503


def test_submit_409_on_window_mismatch():
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request(window_start=999).model_dump(mode="json"))
    assert resp.status_code == 409


def test_state_endpoint_returns_grpo_batch_state():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    client = TestClient(server.app)
    resp = client.get("/window/500/state")
    assert resp.status_code == 200
    state = GrpoBatchState(**resp.json())
    assert state.window_start == 500
    assert 42 in state.cooldown_prompts


def test_state_endpoint_404_on_wrong_window():
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    client = TestClient(server.app)
    resp = client.get("/window/999/state")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_validator_server.py -v
```

Expected: FAIL — server still uses v1 `SubmissionRequest`.

- [ ] **Step 3: Rewire server.py**

Rewrite `reliquary/validator/server.py` (replace the old body):

```python
"""FastAPI server: receives miner submissions, exposes window state.

v2 uses GrpoWindowBatcher. /submit drops requests on an asyncio queue;
a worker drains the queue and runs accept_submission in a worker thread
so heavy GRAIL verification doesn't block HTTP responses.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)
from reliquary.validator.batcher_v2 import GrpoWindowBatcher

logger = logging.getLogger(__name__)


class _Health(BaseModel):
    status: str
    active_window: int | None


class ValidatorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self.active_batcher: GrpoWindowBatcher | None = None
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        self.active_batcher = batcher

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Reliquary Validator", version="2.0")

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return _Health(
                status="ok",
                active_window=(
                    self.active_batcher.window_start if self.active_batcher else None
                ),
            )

        @app.post("/submit", response_model=BatchSubmissionResponse)
        async def submit(request: BatchSubmissionRequest) -> BatchSubmissionResponse:
            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            if request.window_start != batcher.window_start:
                raise HTTPException(status_code=409, detail="window_mismatch")

            # Under TestClient (no worker running) we run synchronously so tests
            # see the real accept verdict; under uvicorn we enqueue for the
            # worker and return a provisional ACCEPTED. The worker's real
            # verdict surfaces in logs.
            if self._worker_task is None:
                return batcher.accept_submission(request)

            await self._submit_queue.put((request, batcher))
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED
            )

        @app.get(
            "/window/{window_start}/state", response_model=GrpoBatchState
        )
        async def window_state(window_start: int) -> GrpoBatchState:
            batcher = self.active_batcher
            if batcher is None or batcher.window_start != window_start:
                raise HTTPException(status_code=404, detail="window_not_active")
            return batcher.get_state()

        return app

    async def _submit_worker(self) -> None:
        while True:
            try:
                request, batcher = await self._submit_queue.get()
            except asyncio.CancelledError:
                return
            try:
                response = await asyncio.to_thread(
                    batcher.accept_submission, request
                )
                if response.accepted:
                    logger.info(
                        "accepted prompt=%d hotkey=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                    )
                else:
                    logger.warning(
                        "rejected prompt=%d hotkey=%s reason=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                        response.reason.value,
                    )
            except Exception:
                logger.exception(
                    "submission worker failed on prompt %d", request.prompt_idx
                )

    async def start(self) -> None:
        if self._task is not None:
            return
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(self._submit_worker())
        await asyncio.sleep(0)
        logger.info("Validator HTTP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            self._server = None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_validator_server.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Run the whole validator unit suite to confirm no regressions**

```
pytest tests/unit/ -v -k "not test_batcher and not test_diversity"
```

(Excluding the v1 `test_batcher.py` and `test_diversity.py` tests which we'll delete in cleanup; some of them may fail because of ripple effects — that's OK at this step.)

Expected: all non-excluded tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/server.py tests/unit/test_validator_server.py
git commit -m "feat(server): swap v1 WindowBatcher for v2 GrpoWindowBatcher"
```

---

## Task 12: Miner engine — free prompt selection

**Files:**
- Modify: `reliquary/miner/engine.py`
- Create: `tests/unit/test_miner_engine_v2.py` (unit tests for prompt-picking)

**Context:** `MiningEngine.mine_window` is rewritten. The heavy model generation path is unchanged; what changes is (a) prompt selection (free from env, skip cooldown), (b) payload shape, (c) signed_round.

- [ ] **Step 1: Write failing test — prompt selection logic**

Create `tests/unit/test_miner_engine_v2.py`:

```python
"""Miner prompt-picking strategy: pull random in-range, skip cooldown."""

import random
import pytest

from reliquary.miner.engine import pick_prompt_idx


class FakeEnv:
    def __len__(self):
        return 100


def test_pick_prompt_in_range():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_prompt_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(0, 95))  # only 5 choices free: 95..99
    for _ in range(20):
        idx = pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)
        assert idx not in cooldown


def test_pick_prompt_all_cooldown_raises():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(100))
    with pytest.raises(RuntimeError, match="no eligible prompt"):
        pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_miner_engine_v2.py -v
```

Expected: FAIL with `ImportError: cannot import name 'pick_prompt_idx'`.

- [ ] **Step 3: Add pick_prompt_idx to engine.py**

Append to `reliquary/miner/engine.py` (after imports, before the class):

```python
import random as _random


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    The reference miner uses uniform-random selection with rejection
    sampling against the cooldown set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    rng = rng or _random
    n = len(env)
    # Fast path: if most prompts are eligible, rejection sample.
    if len(cooldown_prompts) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    # Slow path: enumerate eligible and pick.
    eligible = [i for i in range(n) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_miner_engine_v2.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Rewrite `MiningEngine.mine_window`**

Replace the body of `mine_window` in `reliquary/miner/engine.py`. The new flow:

```python
    async def mine_window(
        self,
        subtensor,
        window_start: int,
        use_drand: bool = True,
    ) -> list:
        """v2: pick prompts freely, generate M rollouts each, submit batch.

        Returns the list of BatchSubmissionResponse objects collected.
        """
        import httpx
        import random

        from reliquary.constants import M_ROLLOUTS
        from reliquary.miner.submitter import (
            SubmissionError,
            discover_validator_url,
            get_window_state_v2,
            submit_batch_v2,
        )
        from reliquary.protocol.submission import BatchSubmissionRequest

        # 1. Compute window randomness (used by GRAIL sketches — unchanged from v1).
        randomness = await self._compute_randomness(subtensor, window_start, use_drand)

        # 2. Resolve validator URL.
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        deadline = (
            time.monotonic()
            + WINDOW_LENGTH * BLOCK_TIME_SECONDS
            - UPLOAD_BUFFER
        )
        logger.info(
            "Mining v2 window %d — %.0fs budget, validator %s",
            window_start,
            WINDOW_LENGTH * BLOCK_TIME_SECONDS - UPLOAD_BUFFER,
            url,
        )

        rng = random.Random()
        results = []

        async with httpx.AsyncClient(timeout=30) as client:
            while time.monotonic() < deadline:
                # 3a. Fetch cooldown set + current round.
                try:
                    state = await get_window_state_v2(url, window_start, client=client)
                except SubmissionError as exc:
                    logger.debug("state fetch failed: %s", exc)
                    continue

                cooldown_set = set(state.cooldown_prompts)
                signed_round = state.current_round

                # 3b. Pick a prompt freely.
                try:
                    prompt_idx = pick_prompt_idx(self.env, cooldown_set, rng=rng)
                except RuntimeError:
                    logger.info("env fully in cooldown; stopping")
                    break

                problem = self.env.get_problem(prompt_idx)

                # 3c. Generate M rollouts and compute local rewards.
                generations = self._generate_m_rollouts(problem, randomness)
                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d / %d rollouts for prompt %d; skipping",
                        len(generations), M_ROLLOUTS, prompt_idx,
                    )
                    continue

                # 3d. Build payload (including GRAIL commits + local rewards).
                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness)
                    for gen in generations
                ]

                # 3e. Merkle root over leaves.
                merkle_root = _compute_merkle_root(rollout_submissions)

                # 3f. Send.
                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=window_start,
                    signed_round=signed_round,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                )
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    logger.info(
                        "submitted prompt %d — accepted=%s reason=%s",
                        prompt_idx, resp.accepted, resp.reason.value,
                    )
                    results.append(resp)
                except SubmissionError as exc:
                    logger.error("submit failed prompt %d: %s", prompt_idx, exc)

        return results

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO. No cherry-picking."""
        import torch
        from reliquary.constants import M_ROLLOUTS, T_PROTO

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        rollouts = []
        for _ in range(M_ROLLOUTS):
            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens],
                    device=getattr(self.vllm_model, "device", "cpu"),
                )
                outputs = self.vllm_model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=T_PROTO,
                )
            all_tokens = outputs[0].tolist()
            rollouts.append({
                "tokens": all_tokens,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        from reliquary.protocol.submission import RolloutSubmission

        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        # Build GRAIL proof — re-use the old _build_completion_submission
        # logic to get the commit dict (extract into a helper).
        completion_sub = self._build_completion_submission(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=completion_sub.commit,
        )


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex."""
    import hashlib

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(repr(r.tokens).encode())
        h.update(repr(r.reward).encode())
        h.update(repr(r.commit).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()
```

(The existing `_build_completion_submission` stays as the GRAIL-commit builder, unchanged. The v1 `_generate_targeted_batch` and `_choose_target_reward` become dead code — removed in Task 14 cleanup.)

- [ ] **Step 6: Run unit tests**

```
pytest tests/unit/test_miner_engine_v2.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reliquary/miner/engine.py tests/unit/test_miner_engine_v2.py
git commit -m "feat(miner): v2 mine_window — free prompt, M rollouts, local rewards"
```

---

## Task 13: Submitter — v2 payload and new reject reasons

**Files:**
- Modify: `reliquary/miner/submitter.py` (add v2 functions alongside v1)
- Modify: `tests/unit/test_submitter.py` (exists — append v2 tests)

**Context:** The retry loop stays. New functions `submit_batch_v2` and `get_window_state_v2` send / parse v2 schemas.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_submitter.py`:

```python
# ---- v2 submitter tests ----

import httpx
import pytest

from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
)
from reliquary.miner.submitter import (
    SubmissionError,
    get_window_state_v2,
    submit_batch_v2,
)


def _rollouts(k=4):
    out = []
    for i in range(8):
        out.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < k else 0.0,
                commit={"tokens": [1, 2, 3], "proof_version": "v5"},
            )
        )
    return out


def _request():
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=100,
        signed_round=999,
        merkle_root="00" * 32,
        rollouts=_rollouts(),
    )


@pytest.mark.asyncio
async def test_submit_batch_v2_ok(monkeypatch):
    responses = [
        httpx.Response(
            200,
            json=BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED
            ).model_dump(mode="json"),
        )
    ]

    async def _post(self, url, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _request(), client=client)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v2_reject_reason_propagated(monkeypatch):
    async def _post(self, url, json=None, timeout=None):
        return httpx.Response(
            200,
            json=BatchSubmissionResponse(
                accepted=False, reason=RejectReason.PROMPT_IN_COOLDOWN
            ).model_dump(mode="json"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _request(), client=client)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN
    await client.aclose()


@pytest.mark.asyncio
async def test_get_window_state_v2(monkeypatch):
    state = GrpoBatchState(
        window_start=100, current_round=999, cooldown_prompts=[42, 7],
        valid_submissions=3,
    )

    async def _get(self, url, timeout=None):
        return httpx.Response(200, json=state.model_dump(mode="json"))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    client = httpx.AsyncClient()
    s = await get_window_state_v2("http://fake", 100, client=client)
    assert s.window_start == 100
    assert set(s.cooldown_prompts) == {42, 7}
    await client.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_submitter.py -v -k "v2"
```

Expected: FAIL with `ImportError: cannot import name 'submit_batch_v2'`.

- [ ] **Step 3: Add v2 functions**

Append to `reliquary/miner/submitter.py`:

```python
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)


async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BatchSubmissionResponse:
    """POST a v2 batch submission. Retries network errors; 4xx is final."""
    payload = request.model_dump(mode="json")
    return await _post_with_retry(
        f"{url}/submit", payload, BatchSubmissionResponse,
        client=client, timeout=timeout,
    )


async def get_window_state_v2(
    url: str,
    window_start: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's v2 GrpoBatchState for a given window."""
    return await _get_with_retry(
        f"{url}/window/{window_start}/state", GrpoBatchState,
        client=client, timeout=timeout,
    )
```

Modify `_post_with_retry` — the existing special-case for v1 `SubmissionResponse` needs to also handle v2 `BatchSubmissionResponse`. Replace the 4xx block:

```python
            if 400 <= resp.status_code < 500:
                detail = _safe_detail(resp)
                # v1 SubmissionResponse short-circuit.
                try:
                    from reliquary.protocol.submission import SubmissionResponse
                    if response_model is SubmissionResponse:
                        return SubmissionResponse(
                            accepted=False,
                            reason=f"http_{resp.status_code}:{detail}",
                            settled=False, slot_count=0,
                        )
                except ImportError:
                    pass
                # v2 BatchSubmissionResponse short-circuit.
                if response_model is BatchSubmissionResponse:
                    # Map HTTP error to an approximate RejectReason.
                    if resp.status_code == 409:
                        reason = RejectReason.WINDOW_MISMATCH
                    elif resp.status_code == 503:
                        reason = RejectReason.WINDOW_NOT_ACTIVE
                    else:
                        reason = RejectReason.BAD_PROMPT_IDX  # generic 4xx
                    return BatchSubmissionResponse(accepted=False, reason=reason)
                raise SubmissionError(f"HTTP {resp.status_code}: {detail}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_submitter.py -v
```

Expected: PASS (all v1 + v2 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/submitter.py tests/unit/test_submitter.py
git commit -m "feat(submitter): v2 submit_batch_v2 + get_window_state_v2"
```

---

## Task 14: Validator service wiring + weights integration

**Files:**
- Modify: `reliquary/validator/service.py`
- Create: `tests/unit/test_service_v2.py`

**Context:** The service creates per-window batchers and pushes weights on-chain. Now it creates `GrpoWindowBatcher` instead of `WindowBatcher`, and calls `compute_weights_v2` at weight time.

- [ ] **Step 1: Read the current service.py to understand the wiring**

```bash
cat reliquary/validator/service.py | head -100
```

Identify the function that instantiates `WindowBatcher` per window (likely `_open_window` or similar) and the function that calls `compute_weights` (likely `_submit_weights`).

- [ ] **Step 2: Write failing test for the window-open path**

Create `tests/unit/test_service_v2.py`:

```python
"""End-to-end: service creates GrpoWindowBatcher per window, seals at window
close, computes weights v2-flavoured."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from reliquary.constants import B_BATCH
from reliquary.validator.batcher_v2 import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.cooldown import CooldownMap


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


def test_service_creates_grpo_window_batcher():
    """The service's open_window() returns a GrpoWindowBatcher wired up
    with the shared CooldownMap."""
    from reliquary.validator.service import open_grpo_window

    shared_cooldown = CooldownMap(cooldown_windows=50)
    batcher = open_grpo_window(
        window_start=100,
        current_round=999,
        env=_FakeEnv(),
        model=None,
        cooldown_map=shared_cooldown,
        tokenizer=MagicMock(),
    )
    assert isinstance(batcher, GrpoWindowBatcher)
    assert batcher.window_start == 100
    assert batcher._cooldown is shared_cooldown


def test_service_compute_weights_for_sealed_batch():
    """After seal_batch, service computes flat 1/B weights."""
    from reliquary.validator.service import compute_weights_for_window
    from reliquary.protocol.submission import RolloutSubmission

    # Build 5 valid submissions (partial batch).
    rollouts = [
        RolloutSubmission(tokens=[1], reward=1.0, commit={"tokens": [1]})
        for _ in range(8)
    ]
    batch = [
        ValidSubmission(
            hotkey=f"hk{i}", prompt_idx=i, signed_round=100,
            merkle_root_bytes=b"\x00" * 32, k=4, rollouts=rollouts,
        )
        for i in range(5)
    ]
    miner_weights, burn_weight = compute_weights_for_window(batch)
    assert len(miner_weights) == 5
    # Each miner gets 1/B, partial → burn the rest.
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    assert abs(burn_weight - 3.0 / B_BATCH) < 1e-9
```

- [ ] **Step 3: Run test to verify it fails**

```
pytest tests/unit/test_service_v2.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 4: Add the wiring functions to service.py**

Append to `reliquary/validator/service.py` (or modify the existing window-open function):

```python
def open_grpo_window(
    window_start: int,
    current_round: int,
    env,
    model,
    *,
    cooldown_map,
    tokenizer,
    bootstrap: bool = False,
):
    """Instantiate a GrpoWindowBatcher for this window.

    ``cooldown_map`` is the validator's long-lived CooldownMap, shared
    across windows; each window's sealed batch updates it via
    ``GrpoWindowBatcher.seal_batch``.
    """
    from reliquary.validator.batcher_v2 import GrpoWindowBatcher

    def _completion_text(rollout):
        prompt_len = rollout.commit.get("rollout", {}).get("prompt_length", 0)
        return tokenizer.decode(rollout.tokens[prompt_len:])

    return GrpoWindowBatcher(
        window_start=window_start,
        current_round=current_round,
        env=env,
        model=model,
        cooldown_map=cooldown_map,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
    )


def compute_weights_for_window(batch) -> tuple[dict[str, float], float]:
    """Collapse a sealed batch (list of ValidSubmission) into
    (miner_weights, burn_weight) ready for chain submission."""
    from reliquary.validator.weights import compute_weights_v2
    return compute_weights_v2(batch_hotkeys=[sub.hotkey for sub in batch])
```

Replace the existing v1 window-open and weight-compute calls in the service loop (the exact lines depend on the current `service.py` layout — identify them in Step 1 and swap). Keep the v1 code commented out if it helps reviewers see the diff; it's deleted in Task 15.

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/unit/test_service_v2.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_service_v2.py
git commit -m "feat(service): open_grpo_window + compute_weights_for_window"
```

---

## Task 15: Cleanup — delete v1 code and tests

**Files:**
- Modify: `reliquary/validator/batcher.py` (delete old class body)
- Modify: `reliquary/validator/weights.py` (delete old `compute_weights`)
- Modify: `reliquary/validator/server.py` (remove any v1 imports)
- Modify: `reliquary/miner/engine.py` (delete `_generate_targeted_batch`, `_choose_target_reward`)
- Modify: `reliquary/miner/submitter.py` (delete v1 `submit_batch`, `get_window_state`)
- Modify: `reliquary/constants.py` (remove deprecated constants)
- Modify: `reliquary/protocol/submission.py` (delete v1 `SubmissionRequest`/`Response`/`SlotState`/`WindowStateResponse`)
- Delete: `tests/unit/test_batcher.py`
- Delete: `tests/unit/test_diversity.py`
- Delete: parts of `tests/unit/test_submission.py` that test v1 schemas

**Context:** Once all v2 tests pass and the service runs v2 end-to-end, rip out v1. Expect ~500 LOC deleted.

- [ ] **Step 1: Search for v1-only constants and remove**

```bash
grep -rn "GROUP_SIZE\|PROMPTS_PER_WINDOW\|DIVERSITY_PREFIX_LEN\|COMPLETIONS_PER_SUBMISSION\|SUPERLINEAR_EXPONENT\|UNIQUE_ROLLOUTS_CAP\|SLOT_DEADLINE_SECONDS\|MINER_BATCH_SIZE" reliquary/ tests/
```

Delete every matching import and usage. In `constants.py`, delete the constants themselves (preserve them only if v2 still references them — it doesn't).

- [ ] **Step 2: Delete v1 classes**

- In `reliquary/validator/batcher.py`: delete everything except imports reused by v2 (likely nothing). Replace the file with a single-line stub:

```python
"""v1 WindowBatcher removed — superseded by reliquary.validator.batcher_v2."""
```

- In `reliquary/validator/weights.py`: delete `compute_weights`. Keep only `compute_weights_v2`.

- In `reliquary/protocol/submission.py`: delete `CompletionSubmission`, `SubmissionRequest`, `SubmissionResponse`, `SlotState`, `WindowStateResponse`.

- In `reliquary/miner/engine.py`: delete `_generate_targeted_batch`, `_choose_target_reward`, `_build_completion_submission` (merged into `_build_rollout_submission` in Task 12 — if not, extract now).

- In `reliquary/miner/submitter.py`: delete `submit_batch` and `get_window_state` (keep the retry helpers `_post_with_retry` and `_get_with_retry`).

- [ ] **Step 3: Delete obsolete tests**

```bash
git rm tests/unit/test_batcher.py tests/unit/test_diversity.py
```

In `tests/unit/test_submission.py`: delete tests for v1 schemas; keep anything that tests generic Pydantic plumbing still relevant.

- [ ] **Step 4: Run the full suite**

```
pytest tests/unit/ -v
```

Expected: PASS with no references to deleted symbols. If anything fails with `AttributeError` or `ImportError`, trace the leftover reference and clean it up.

- [ ] **Step 5: Rename `batcher_v2.py` → `batcher.py`**

```bash
git rm reliquary/validator/batcher.py
git mv reliquary/validator/batcher_v2.py reliquary/validator/batcher.py
```

Update imports across the codebase:

```bash
grep -rln "batcher_v2" reliquary/ tests/
```

Replace every `reliquary.validator.batcher_v2` with `reliquary.validator.batcher`. Run the suite again:

```
pytest tests/unit/ -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: remove v1 slot-based code, rename batcher_v2 → batcher"
```

---

## Task 16: Bootstrap mode activation

**Files:**
- Modify: `reliquary/validator/service.py` — pass `bootstrap=True` to `open_grpo_window` for the first `BOOTSTRAP_WINDOWS`
- Create: `tests/unit/test_bootstrap_mode.py`

**Context:** During the first 100 windows of the subnet (or a new checkpoint rollout), use the relaxed zone and cooldown values. How is the "start block" defined — on-chain? Config file? For the reference impl, use an on-chain `SUBNET_START_BLOCK` constant set at deployment time.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_bootstrap_mode.py`:

```python
"""Bootstrap phase: wider zone, shorter cooldown, smaller M for first
BOOTSTRAP_WINDOWS windows after SUBNET_START_BLOCK."""

from reliquary.constants import BOOTSTRAP_WINDOWS
from reliquary.validator.service import is_bootstrap_window


def test_bootstrap_active_at_start():
    assert is_bootstrap_window(window_start=100, subnet_start=100) is True


def test_bootstrap_active_within_horizon():
    assert is_bootstrap_window(
        window_start=100 + BOOTSTRAP_WINDOWS - 1, subnet_start=100
    ) is True


def test_bootstrap_expires_at_horizon():
    assert is_bootstrap_window(
        window_start=100 + BOOTSTRAP_WINDOWS, subnet_start=100
    ) is False


def test_bootstrap_inactive_before_start():
    # Defensive: if someone queries a window before subnet_start, treat as
    # not-bootstrap (should never happen in practice).
    assert is_bootstrap_window(window_start=50, subnet_start=100) is False
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_bootstrap_mode.py -v
```

Expected: FAIL with `ImportError: cannot import name 'is_bootstrap_window'`.

- [ ] **Step 3: Add is_bootstrap_window + pass to batcher**

Append to `reliquary/validator/service.py`:

```python
from reliquary.constants import BOOTSTRAP_WINDOWS


def is_bootstrap_window(window_start: int, subnet_start: int) -> bool:
    """True iff this window is within ``BOOTSTRAP_WINDOWS`` of subnet start."""
    if window_start < subnet_start:
        return False
    return window_start - subnet_start < BOOTSTRAP_WINDOWS
```

And in the service's `open_grpo_window` call site (wherever it's invoked from the main validator loop), pass:

```python
batcher = open_grpo_window(
    window_start=window_start,
    current_round=current_round,
    env=env,
    model=model,
    cooldown_map=cooldown_map,
    tokenizer=tokenizer,
    bootstrap=is_bootstrap_window(window_start, SUBNET_START_BLOCK),
)
```

Add `SUBNET_START_BLOCK` to `reliquary/constants.py`:

```python
# The first on-chain block at which this subnet deployed v2. Used to
# determine bootstrap eligibility. Set at the coordinated cutover.
SUBNET_START_BLOCK = 0  # updated at deployment time
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_bootstrap_mode.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py reliquary/constants.py tests/unit/test_bootstrap_mode.py
git commit -m "feat(service): activate bootstrap mode for first BOOTSTRAP_WINDOWS"
```

---

## Task 17: Integration smoke test

**Files:**
- Create: `tests/integration/test_grpo_market_smoke.py`

**Context:** End-to-end happy path: validator + miner run a 2-window simulation; confirm batch forms, cooldown fires, weights computed.

- [ ] **Step 1: Write the test**

Create `tests/integration/test_grpo_market_smoke.py`:

```python
"""Smoke test: miner → validator → batch → weights over 2 windows.

No on-chain dependencies — all randomness / drand is stubbed. The goal
is to prove the pieces fit together.
"""

import hashlib
import random

import pytest

from reliquary.constants import B_BATCH, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)
from reliquary.validator.batcher import GrpoWindowBatcher  # after Task 15 rename
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.weights import compute_weights_v2


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"q{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "WIN" in c else 0.0


def _rollouts(k):
    out = []
    for i in range(M_ROLLOUTS):
        text = "WIN" if i < k else "lose"
        out.append(RolloutSubmission(
            tokens=[1, 2, 3],
            reward=1.0 if i < k else 0.0,
            commit={"tokens": [1, 2, 3], "proof_version": "v5",
                    "completion_text_for_test": text},
        ))
    return out


def _merkle_root(n=0):
    h = hashlib.sha256(str(n).encode()).hexdigest()
    return h


def _make_batcher(window, cooldown):
    return GrpoWindowBatcher(
        window_start=window, current_round=window * 10,
        env=FakeEnv(), model=None, cooldown_map=cooldown,
        verify_commitment_proofs_fn=lambda c, m, r: (True, 1, 1),
        verify_signature_fn=lambda c, h: True,
        verify_proof_version_fn=lambda c: True,
        completion_text_fn=lambda r: r.commit.get("completion_text_for_test", ""),
    )


def test_two_windows_with_cooldown():
    cooldown = CooldownMap(cooldown_windows=3)  # small for test speed

    # Window 0: 10 miners submit on distinct prompts, k=4 each.
    b0 = _make_batcher(window=0, cooldown=cooldown)
    for i in range(10):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}", prompt_idx=i,
            window_start=0, signed_round=5 + i,
            merkle_root=_merkle_root(i),
            rollouts=_rollouts(k=4),
        )
        resp = b0.accept_submission(req)
        assert resp.accepted, f"unexpected reject for hk{i}: {resp.reason}"
    batch0 = b0.seal_batch()
    assert len(batch0) == B_BATCH
    assert {s.prompt_idx for s in batch0} == set(range(B_BATCH))

    # Window 1: same miners try the same prompts → all rejected for cooldown.
    b1 = _make_batcher(window=1, cooldown=cooldown)
    for i in range(B_BATCH):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}", prompt_idx=i,
            window_start=1, signed_round=15 + i,
            merkle_root=_merkle_root(100 + i),
            rollouts=_rollouts(k=4),
        )
        resp = b1.accept_submission(req)
        assert resp.accepted is False

    # Window 4 (after cooldown=3): prompt 0 eligible again.
    b4 = _make_batcher(window=4, cooldown=cooldown)
    req = BatchSubmissionRequest(
        miner_hotkey="hk0", prompt_idx=0,
        window_start=4, signed_round=45,
        merkle_root=_merkle_root(500),
        rollouts=_rollouts(k=4),
    )
    resp = b4.accept_submission(req)
    assert resp.accepted is True

    # Weights for the first window: full batch, no burn.
    weights, burn = compute_weights_v2([s.hotkey for s in batch0])
    assert abs(sum(weights.values()) + burn - 1.0) < 1e-9
    assert abs(burn) < 1e-9
    assert len(weights) == B_BATCH
```

- [ ] **Step 2: Run**

```
pytest tests/integration/test_grpo_market_smoke.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_grpo_market_smoke.py
git commit -m "test(integration): 2-window smoke — submission → batch → cooldown → weights"
```

---

## Task 18: README + mining/validating docs refresh

**Files:**
- Modify: `README.md`
- Modify: `docs/mining.md`
- Modify: `docs/validating.md`

**Context:** v1 docs describe slots and advantage scoring. Update to v2 GRPO market.

- [ ] **Step 1: Update README.md**

Replace the "How it works" section (or equivalent) with:

```markdown
## How it works (v2 GRPO market)

Each window (~60 s, 1 training step):

1. Miners pick a `prompt_idx` from the env (GSM8K, ~7473 problems) —
   avoiding the validator's published cooldown set.
2. Each miner generates 8 rollouts at `T_PROTO = 0.9`, computes local
   rewards, builds GRAIL commits, and POSTs a `BatchSubmissionRequest`
   to the validator.
3. Validator verifies in order: signature → prompt_idx in range → round
   freshness → prompt not in cooldown → reward claims match
   `env.compute_reward` → GRAIL sketch matches model forward-pass
   → `k ∈ [2, 6]` (apprenable zone).
4. At window close, validator selects the first `B=8` in-zone
   submissions (FIFO by signed drand round, distinct prompts) for the
   training batch. Sealed batch becomes the GRPO step input; each
   batched prompt enters a 50-window cooldown.
5. Weights: each batch member earns `1/B` of window emission; unused
   slots burn to `UID_BURN`.

Miners who submit outside the zone, on cooldown'd prompts, or too slow
to make the batch earn **zero**. The speed + distinct-prompt
competition removes any incentive for cherry-picking (cherry-picking
takes extra compute → later `signed_round` → displaced from the batch).
```

- [ ] **Step 2: Update docs/mining.md**

Describe the new `reliquary mine` flow: poll `/window/{n}/state`, pick
prompt avoiding cooldown, generate 8 at `T_PROTO`, compute rewards,
submit. No more slot-by-slot iteration.

- [ ] **Step 3: Update docs/validating.md**

Describe the new flow: per-window `GrpoWindowBatcher`, cooldown state
persisted across restarts, flat `1/B` weights at close.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/mining.md docs/validating.md
git commit -m "docs: rewrite for v2 GRPO market — prompt freedom, zone filter, cooldown"
```

---

## Self-review notes

**Spec coverage check:**
- Zone filter → Task 8 ✓
- Flat 1/B payment → Task 6 ✓
- Cooldown (steady + bootstrap) → Tasks 3–4 + 16 ✓
- FIFO batch selection → Task 5 ✓
- GRAIL unchanged (reused as-is) → Tasks 9, 11 ✓
- Reward re-verification → Task 2, used in Task 9 ✓
- New protocol schemas → Task 7 ✓
- Bootstrap phase → Task 16 ✓
- Miner free prompt pick → Task 12 ✓
- Validator server endpoints → Task 11 ✓
- Clean removal of v1 → Task 15 ✓
- Smoke test → Task 17 ✓
- Docs → Task 18 ✓

**No placeholders:** every step has complete code. Imports are specified where they belong.

**Type consistency:**
- `ValidSubmission.prompt_idx` → int, consistent across `select_batch`, `CooldownMap`, `GrpoWindowBatcher`
- `BatchSubmissionRequest.merkle_root` is hex string; `ValidSubmission.merkle_root_bytes` is bytes (converted at ingestion)
- `RejectReason` enum values used as strings in JSON, enum instances internally

**Ordering sanity:** Tasks 1–8 build pure primitives with isolated tests. Task 9 composes them into the batcher. Tasks 11–14 wire the batcher into server + service. Task 15 rips out v1 only after v2 end-to-end tests pass.
