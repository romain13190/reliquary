# Switch Environment from GSM8K to Hendrycks MATH — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GSM8K environment with Hendrycks competition_math so the zone filter (σ ≥ 0.33 bootstrap / 0.43 steady) sees enough mid-difficulty problems to pass Qwen3-4B-Instruct-2507 submissions and trigger GRPO training.

**Architecture:** Environments implement the existing `Environment` Protocol in `reliquary/environment/base.py` (`name`, `__len__`, `get_problem`, `compute_reward`). We add a new `MATHEnvironment` in `reliquary/environment/math.py` backed by the `hendrycks/competition_math` HF dataset (7 500 train problems, 5 difficulty levels, 7 subjects), expose it through `load_environment("math")`, make `"math"` the default in `constants.py`, delete the GSM8K env + tests, and update docs. Reward extraction uses Hendrycks' `last_boxed_only_string` with balanced-brace parsing; answers are string-compared after a conservative LaTeX normalization pass.

**Tech Stack:** Python 3.12, pytest, HuggingFace `datasets`, stdlib `re` / `hashlib`, no new runtime deps.

---

## File Structure

**Create:**
- `reliquary/environment/math.py` — full MATH env (extractor, normalizer, reward fn, env class)
- `tests/unit/test_math_environment.py` — unit tests for extractor / normalizer / reward / loader

**Modify:**
- `reliquary/environment/__init__.py` — swap GSM8K import for MATH, update `load_environment`
- `reliquary/constants.py` — `ENVIRONMENT_NAME = "math"`
- `docs/concepts.md`, `docs/deployment.md`, `docs/mining.md`, `docs/validating.md`, `docs/training.md` — update every GSM8K reference

**Delete:**
- `reliquary/environment/gsm8k.py`
- `tests/unit/test_environment.py` (replaced by the new file — the old one is 100 % GSM8K-specific)

---

### Task 1: New module skeleton + last_boxed extractor

Balanced-brace parsing for LaTeX `\boxed{...}` extraction — the only reliable way to grab the answer when the inner expression itself contains `{...}` (fractions, sets, etc.).

**Files:**
- Create: `reliquary/environment/math.py`
- Test: `tests/unit/test_math_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_math_environment.py
"""Tests for the Hendrycks MATH environment."""

import hashlib
import pytest


def test_last_boxed_only_string_simple():
    from reliquary.environment.math import _last_boxed_only_string
    assert _last_boxed_only_string(r"The answer is \boxed{42}.") == r"\boxed{42}"


def test_last_boxed_only_string_nested_braces():
    from reliquary.environment.math import _last_boxed_only_string
    s = r"So \boxed{\frac{1}{2}}"
    assert _last_boxed_only_string(s) == r"\boxed{\frac{1}{2}}"


def test_last_boxed_only_string_multiple_returns_last():
    from reliquary.environment.math import _last_boxed_only_string
    s = r"First try \boxed{3}, corrected to \boxed{4}."
    assert _last_boxed_only_string(s) == r"\boxed{4}"


def test_last_boxed_only_string_none_when_absent():
    from reliquary.environment.math import _last_boxed_only_string
    assert _last_boxed_only_string("no boxed here") is None


def test_last_boxed_only_string_unbalanced_returns_none():
    from reliquary.environment.math import _last_boxed_only_string
    # Opens \boxed{ but never closes — should fail gracefully.
    assert _last_boxed_only_string(r"\boxed{unclosed") is None


def test_fbox_alias_accepted():
    """Hendrycks data sometimes uses \\fbox{} for the final answer."""
    from reliquary.environment.math import _last_boxed_only_string
    assert _last_boxed_only_string(r"answer: \fbox{7}") == r"\fbox{7}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_math_environment.py::test_last_boxed_only_string_simple -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary.environment.math'`

- [ ] **Step 3: Create the module with the extractor**

```python
# reliquary/environment/math.py
"""Hendrycks MATH environment for Reliquary verifiable inference.

Loads the hendrycks/competition_math dataset (train split, 7 500 problems)
and scores completions by extracting the final \\boxed{...} answer and
comparing it to the ground truth after LaTeX normalization.
"""

from __future__ import annotations

import hashlib
import re
from typing import ClassVar, Optional


# ---------------------------------------------------------------------------
# Answer extraction — balanced-brace parsing of the last \boxed{...} / \fbox{...}
# ---------------------------------------------------------------------------

def _last_boxed_only_string(text: str) -> Optional[str]:
    """Return the last \\boxed{...} / \\fbox{...} substring (including the
    wrapper), or None if no balanced wrapper is found.

    This is the Hendrycks-style parser: it walks braces to handle nested
    expressions like \\boxed{\\frac{1}{2}} that a regex cannot match.
    """
    # Find the last occurrence of either \boxed{ or \fbox{.
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None

    # Walk from the opening brace counting depth.
    open_idx = text.index("{", idx)
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[idx : j + 1]
    # Never balanced — unterminated expression.
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_math_environment.py -v -k boxed`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/math.py tests/unit/test_math_environment.py
git commit -m "feat(env/math): last_boxed extractor with balanced-brace parsing"
```

---

### Task 2: Strip `\boxed{}` wrapper

Small helper so the reward path doesn't have to know the brace positions.

**Files:**
- Modify: `reliquary/environment/math.py`
- Test: `tests/unit/test_math_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/unit/test_math_environment.py

def test_strip_boxed_wrapper_boxed():
    from reliquary.environment.math import _strip_boxed_wrapper
    assert _strip_boxed_wrapper(r"\boxed{42}") == "42"


def test_strip_boxed_wrapper_fbox():
    from reliquary.environment.math import _strip_boxed_wrapper
    assert _strip_boxed_wrapper(r"\fbox{x+y}") == "x+y"


def test_strip_boxed_wrapper_nested_inner():
    from reliquary.environment.math import _strip_boxed_wrapper
    assert _strip_boxed_wrapper(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_strip_boxed_wrapper_non_boxed_input_returned_unchanged():
    from reliquary.environment.math import _strip_boxed_wrapper
    # Not wrapped — returned as-is (lets the reward fn handle raw GT strings).
    assert _strip_boxed_wrapper("42") == "42"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_math_environment.py -v -k strip_boxed`
Expected: 4 FAIL with `ImportError`

- [ ] **Step 3: Add the helper**

```python
# Append to reliquary/environment/math.py

def _strip_boxed_wrapper(s: str) -> str:
    """If ``s`` starts with \\boxed{ or \\fbox{ and ends with a matching },
    return the inner content. Otherwise return ``s`` unchanged.
    """
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_math_environment.py -v -k strip_boxed`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/math.py tests/unit/test_math_environment.py
git commit -m "feat(env/math): strip_boxed_wrapper helper"
```

---

### Task 3: Conservative LaTeX answer normalization

A minimal, well-tested normalization pipeline: covers the transforms that actually show up in MATH ground truths (`\left`/`\right`, `\dfrac`/`\tfrac`, `\text{...}`, `$` math delimiters, whitespace) without trying to be a CAS.

**Files:**
- Modify: `reliquary/environment/math.py`
- Test: `tests/unit/test_math_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/unit/test_math_environment.py

def test_normalize_strips_left_right():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"\left(x+1\right)") == "(x+1)"


def test_normalize_dfrac_tfrac_to_frac():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert _normalize_answer(r"\tfrac{3}{4}") == r"\frac{3}{4}"


def test_normalize_strips_dollar_delimiters():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"$42$") == "42"


def test_normalize_strips_text_wrapper():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"5\text{ cm}") == "5cm"


def test_normalize_strips_mbox_wrapper():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"3\mbox{ units}") == "3units"


def test_normalize_removes_latex_whitespace_macros():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(r"1\,234") == "1234"
    assert _normalize_answer(r"5\!6") == "56"


def test_normalize_strips_trailing_period_and_whitespace():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer(" 42 .  ") == "42"


def test_normalize_collapses_spaces():
    from reliquary.environment.math import _normalize_answer
    assert _normalize_answer("x + 1") == "x+1"


def test_normalize_is_idempotent():
    from reliquary.environment.math import _normalize_answer
    once = _normalize_answer(r"\left(\dfrac{1}{2}\right)")
    twice = _normalize_answer(once)
    assert once == twice
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_math_environment.py -v -k normalize`
Expected: 9 FAIL with `ImportError`

- [ ] **Step 3: Implement the normalizer**

```python
# Append to reliquary/environment/math.py

_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\{([^}]*)\}")


def _normalize_answer(s: str) -> str:
    """Conservative LaTeX normalization for equality comparison.

    Intentionally string-level only (no CAS): the rules below cover the
    transforms that actually occur in Hendrycks MATH ground truths without
    changing the meaning of the expression.
    """
    if s is None:
        return ""
    # Drop LaTeX spacing macros first so downstream rules see clean text.
    for macro in (r"\!", r"\,", r"\ ", r"\;", r"\:"):
        s = s.replace(macro, "")
    # Drop \left / \right size modifiers — they're presentational.
    s = s.replace(r"\left", "").replace(r"\right", "")
    # Canonicalize fraction macros.
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # Strip \text{...} and \mbox{...} wrappers (keep inner content).
    s = _TEXT_RE.sub(r"\1", s)
    s = _MBOX_RE.sub(r"\1", s)
    # Strip math-mode delimiters.
    s = s.replace(r"\$", "").replace("$", "")
    # Strip trailing period / whitespace.
    s = s.strip().rstrip(".").strip()
    # Collapse whitespace (MATH answers should be whitespace-insensitive).
    s = re.sub(r"\s+", "", s)
    return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_math_environment.py -v -k normalize`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/math.py tests/unit/test_math_environment.py
git commit -m "feat(env/math): conservative LaTeX answer normalizer"
```

---

### Task 4: `_compute_math_reward` — pure reward function (no dataset)

Combines extraction, strip, normalization, compare. Kept as a module-level function so it can be unit-tested without constructing the env.

**Files:**
- Modify: `reliquary/environment/math.py`
- Test: `tests/unit/test_math_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/unit/test_math_environment.py

def test_reward_exact_boxed_match():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": "42"}
    assert _compute_math_reward(problem, r"...\boxed{42}") == 1.0


def test_reward_frac_vs_dfrac_equivalent():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": r"\frac{1}{2}"}
    assert _compute_math_reward(problem, r"\boxed{\dfrac{1}{2}}") == 1.0


def test_reward_left_right_stripped():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": r"(x+1)"}
    assert _compute_math_reward(problem, r"\boxed{\left(x+1\right)}") == 1.0


def test_reward_whitespace_insensitive():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": "x+1"}
    assert _compute_math_reward(problem, r"\boxed{x + 1}") == 1.0


def test_reward_wrong_answer_is_zero():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": "42"}
    assert _compute_math_reward(problem, r"\boxed{41}") == 0.0


def test_reward_no_boxed_falls_back_to_last_token():
    from reliquary.environment.math import _compute_math_reward
    # MATH expects \boxed{}; missing it → 0.0 (strict). No numeric fallback
    # because MATH answers are often non-numeric (polynomials, sets, etc.).
    problem = {"ground_truth": "42"}
    assert _compute_math_reward(problem, "the answer is 42") == 0.0


def test_reward_handles_empty_completion():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": "42"}
    assert _compute_math_reward(problem, "") == 0.0


def test_reward_handles_unbalanced_boxed():
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": "42"}
    assert _compute_math_reward(problem, r"\boxed{42") == 0.0


def test_reward_handles_gt_already_boxed():
    """Hendrycks solution fields contain \\boxed{...}; the env loader is
    expected to strip it, but the reward fn should still behave if not.
    """
    from reliquary.environment.math import _compute_math_reward
    problem = {"ground_truth": r"\boxed{42}"}
    assert _compute_math_reward(problem, r"\boxed{42}") == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_math_environment.py -v -k "reward"`
Expected: 9 FAIL with `ImportError`

- [ ] **Step 3: Implement the reward**

```python
# Append to reliquary/environment/math.py

def _compute_math_reward(problem: dict, completion: str) -> float:
    """Score a MATH completion.

    Returns 1.0 when the last ``\\boxed{...}`` in the completion, stripped
    and normalized, equals the ground-truth answer (also stripped/normalized).
    Returns 0.0 otherwise. Never raises.
    """
    try:
        boxed = _last_boxed_only_string(completion)
        if boxed is None:
            return 0.0
        candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
        gt_raw = str(problem.get("ground_truth", ""))
        gt = _normalize_answer(_strip_boxed_wrapper(gt_raw))
        return 1.0 if candidate == gt and gt != "" else 0.0
    except Exception:
        return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_math_environment.py -v -k "reward"`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/math.py tests/unit/test_math_environment.py
git commit -m "feat(env/math): compute_math_reward — boxed-only strict comparison"
```

---

### Task 5: `MATHEnvironment` class with HF dataset loading

The `Environment` protocol instance. Mirrors the GSM8K env structure: class-level dataset cache, `get_problem` builds `{prompt, ground_truth, id}` from a row, `compute_reward` delegates to the module function.

**Files:**
- Modify: `reliquary/environment/math.py`
- Test: `tests/unit/test_math_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/unit/test_math_environment.py

def _load_math_env():
    """Helper: load MATH env, skip on any dataset error."""
    try:
        from reliquary.environment.math import MATHEnvironment
        return MATHEnvironment()
    except Exception as exc:
        pytest.skip(f"Could not load MATH dataset: {exc}")


def test_math_env_name():
    from reliquary.environment.math import MATHEnvironment
    assert MATHEnvironment.name == "math"


def test_math_dataset_loaded_and_indexable():
    env = _load_math_env()
    assert len(env) > 0
    problem = env.get_problem(0)
    assert isinstance(problem["prompt"], str) and len(problem["prompt"]) > 0
    assert isinstance(problem["ground_truth"], str) and len(problem["ground_truth"]) > 0
    assert "id" in problem and len(problem["id"]) == 16


def test_math_ground_truth_is_stripped_of_boxed_wrapper():
    env = _load_math_env()
    problem = env.get_problem(0)
    # get_problem must present the bare answer; reward_fn also strips but
    # storing the bare form keeps logs/diagnostics readable.
    assert not problem["ground_truth"].startswith("\\boxed{")


def test_math_problem_id_is_stable_sha_prefix():
    env = _load_math_env()
    p1 = env.get_problem(0)
    p2 = env.get_problem(0)
    assert p1["id"] == p2["id"]
    expected = hashlib.sha256(p1["prompt"].encode()).hexdigest()[:16]
    assert p1["id"] == expected


def test_math_index_wraps_modulo_length():
    env = _load_math_env()
    p0 = env.get_problem(0)
    p_wrap = env.get_problem(len(env))
    assert p_wrap["id"] == p0["id"]


def test_math_compute_reward_on_real_row():
    env = _load_math_env()
    problem = env.get_problem(0)
    gt = problem["ground_truth"]
    # Feed the env its own ground truth inside a \boxed{} — must score 1.0.
    fake_completion = f"Working...\n\\boxed{{{gt}}}"
    assert env.compute_reward(problem, fake_completion) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_math_environment.py -v -k "math_env or math_dataset or math_ground or math_problem or math_index or math_compute"`
Expected: 6 FAIL with `ImportError` on `MATHEnvironment`

- [ ] **Step 3: Implement the class**

```python
# Append to reliquary/environment/math.py


class MATHEnvironment:
    """Environment backed by the Hendrycks MATH train split (7 500 problems).

    Ground truths are extracted once from the ``solution`` field by taking
    the content of the last \\boxed{...}; completions are scored with the
    same extraction against the completion text.
    """

    name: str = "math"

    # Class-level cache: populated on first instantiation so tests in the
    # same process share the one HF download.
    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        if MATHEnvironment._dataset_cache is None:
            import datasets as hf_datasets  # local import keeps module importable w/o datasets
            MATHEnvironment._dataset_cache = hf_datasets.load_dataset(
                "hendrycks/competition_math", split="train", trust_remote_code=True
            )
        self._dataset = MATHEnvironment._dataset_cache

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        """Return problem at *index* (wraps modulo dataset length)."""
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        question: str = row["problem"]
        solution: str = row["solution"]
        boxed = _last_boxed_only_string(solution)
        gt_str = _strip_boxed_wrapper(boxed) if boxed else ""
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        return {
            "prompt": question,
            "ground_truth": gt_str,
            "id": problem_id,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        """Score a completion using MATH boxed-answer reward."""
        return _compute_math_reward(problem, completion)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_math_environment.py -v`
Expected: all tests passed (dataset-touching tests skip only if HF is unreachable — run on a box with network)

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/math.py tests/unit/test_math_environment.py
git commit -m "feat(env/math): MATHEnvironment backed by hendrycks/competition_math"
```

---

### Task 6: Wire the loader + switch the default + delete GSM8K

Atomic swap so the rest of the system talks to MATH from one commit. GSM8K env and its tests are removed — no compatibility shim (YAGNI; the env is a subnet-wide protocol constant, not a per-user knob).

**Files:**
- Modify: `reliquary/environment/__init__.py`
- Modify: `reliquary/constants.py:94`
- Delete: `reliquary/environment/gsm8k.py`
- Delete: `tests/unit/test_environment.py`
- Test: (reuse) `tests/unit/test_math_environment.py` + a new loader test

- [ ] **Step 1: Write the failing loader test**

```python
# Add to tests/unit/test_math_environment.py

def test_loader_returns_math_instance():
    from reliquary.environment import load_environment
    from reliquary.environment.math import MATHEnvironment
    env = load_environment("math")
    assert isinstance(env, MATHEnvironment)


def test_loader_raises_on_unknown():
    from reliquary.environment import load_environment
    with pytest.raises(ValueError, match="Unknown environment: nope"):
        load_environment("nope")


def test_loader_rejects_legacy_gsm8k():
    from reliquary.environment import load_environment
    with pytest.raises(ValueError, match="Unknown environment: gsm8k"):
        load_environment("gsm8k")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_math_environment.py -v -k loader`
Expected: FAIL — `load_environment("math")` raises `ValueError` (not wired yet).

- [ ] **Step 3: Swap the loader**

Replace the whole `reliquary/environment/__init__.py`:

```python
"""Reliquary environment module.

Provides the Environment protocol and a factory function to instantiate
concrete environments by name.
"""

from reliquary.environment.base import Environment
from reliquary.environment.math import MATHEnvironment


def load_environment(name: str) -> Environment:
    """Return a concrete Environment instance for the given *name*.

    Raises:
        ValueError: if *name* is not a recognised environment.
    """
    if name == "math":
        return MATHEnvironment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = [
    "Environment",
    "load_environment",
]
```

- [ ] **Step 4: Switch the default in constants**

Modify `reliquary/constants.py:94`:

```python
ENVIRONMENT_NAME = "math"
```

- [ ] **Step 5: Delete the old env + its tests**

```bash
rm reliquary/environment/gsm8k.py
rm tests/unit/test_environment.py
```

- [ ] **Step 6: Run the full unit suite**

Run: `pytest tests/unit/ -v`
Expected: all green. If anything in `tests/unit/` imports `reliquary.environment.gsm8k` directly, fix it to import `reliquary.environment.math` (or remove the test if it was only testing the old implementation).

- [ ] **Step 7: Commit**

```bash
git add reliquary/environment/ reliquary/constants.py tests/unit/
git commit -m "feat(env): replace GSM8K with Hendrycks MATH as the default env

Delete reliquary/environment/gsm8k.py and its tests. Wire
MATHEnvironment through load_environment('math') and make 'math'
the default in ENVIRONMENT_NAME. No compatibility shim: the env is
a subnet-wide protocol constant — miners and the validator must be
on the same one, so the default flip IS the migration."
```

---

### Task 7: Integration smoke — end-to-end env wiring

One test that proves the swap didn't break the system-level contract: `ENVIRONMENT_NAME` resolves via `load_environment` and yields an `Environment`-protocol instance whose reward fn is callable.

**Files:**
- Test: `tests/integration/test_math_env_smoke.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_math_env_smoke.py
"""End-to-end smoke test: the default environment name resolves to a
working Environment and its reward function runs on real dataset rows.
"""

import pytest


def test_default_environment_is_math():
    from reliquary.constants import ENVIRONMENT_NAME
    assert ENVIRONMENT_NAME == "math"


def test_default_environment_loads_and_scores():
    from reliquary.constants import ENVIRONMENT_NAME
    from reliquary.environment import load_environment, Environment

    try:
        env = load_environment(ENVIRONMENT_NAME)
    except Exception as exc:
        pytest.skip(f"Could not load default environment: {exc}")

    assert isinstance(env, Environment)
    assert len(env) > 0

    problem = env.get_problem(0)
    assert env.compute_reward(problem, r"\boxed{" + problem["ground_truth"] + "}") == 1.0
    assert env.compute_reward(problem, "definitely wrong") == 0.0
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/integration/test_math_env_smoke.py -v`
Expected: PASS (or skip on network error).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_math_env_smoke.py
git commit -m "test(env): integration smoke — default env loads + scores"
```

---

### Task 8: Documentation — replace every GSM8K mention

All user-facing docs currently say "GSM8K". This task rewrites each to MATH in one sweep so the docs match the code.

**Files:**
- Modify: `docs/concepts.md`
- Modify: `docs/deployment.md`
- Modify: `docs/mining.md`
- Modify: `docs/validating.md`
- Modify: `docs/training.md`

- [ ] **Step 1: Inventory every GSM8K reference**

Run: `grep -rn -i "gsm8k" docs/ README.md`
Expected: a list — every line is a call site to update.

- [ ] **Step 2: Update each doc**

For each occurrence:

1. **Environment name**: `gsm8k` → `math`
2. **Dataset size**: `~7 473 problems` → `~7 500 problems`
3. **Dataset description**: "GSM8K grade-school arithmetic" → "Hendrycks MATH (competition math, 5 difficulty levels, 7 subjects)"
4. **Zone filter explainer**: the "binary {0,1} rewards → 2..6 correct out of 8" framing still applies (MATH rewards are also binary), so keep it — but remove the "grade-school arithmetic" flavour text.
5. **Reward format**: "extract last numeric token" → "extract the final `\boxed{...}` answer and compare after LaTeX normalization"
6. **Example prompts in sample logs**: keep numeric `prompt=4821` — those are just indices.

Do not introduce new headings or restructure. Edit in place.

- [ ] **Step 3: Verify no GSM8K mentions remain outside historical plans/specs**

Run: `grep -rn -i "gsm8k" docs/ README.md`
Expected: matches ONLY under `docs/superpowers/plans/` and `docs/superpowers/specs/` (those are historical artifacts — do not rewrite history).

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: update env references from GSM8K to Hendrycks MATH"
```

---

### Task 9: Deploy on the running testnet boxes

After the code lands on main, pull on the validator box and the miner box and restart both. This is the step that actually proves the plan worked.

**Files:** none (ops step).

- [ ] **Step 1: Pull on both boxes**

Run on your dev box (the two hosts use the same SSH key):

```bash
ssh -i ~/.ssh/distill_h100 wrk-0nkyr4av9l9d@ssh.deployments.targon.com \
    "cd /root/reliquary && git pull origin main"
ssh -i ~/.ssh/distill_h100 wrk-qq2ylgzjgory@ssh.deployments.targon.com \
    "cd /root/reliquary && git pull origin main"
```

Expected: `Fast-forward` on both.

- [ ] **Step 2: Restart validator**

```bash
ssh -i ~/.ssh/distill_h100 wrk-0nkyr4av9l9d@ssh.deployments.targon.com \
    "kill \$(cat /root/reliquary/validator.pid) 2>/dev/null; sleep 2; \
     nohup bash -c 'exec /root/reliquary/run_validator.sh' \
         > /root/reliquary/validator.log 2>&1 & \
     echo \$! > /root/reliquary/validator.pid; \
     sleep 3; tail -15 /root/reliquary/validator.log | grep -vE 'httpx|huggingface_hub'"
```

Expected lines:
```
Validator HTTP server listening on 0.0.0.0:8080
Bootstrapped window_n=<N> from R2
Bootstrapped checkpoint_n=<M> from HF
Validator started (v2.1): env=math, ...
```

Note `env=math`. If it still says `env=gsm8k`, the launcher script or env var is pinning it — check `/root/reliquary/run_validator.sh` for a stale `--environment gsm8k` flag.

- [ ] **Step 3: Restart miner**

```bash
ssh -i ~/.ssh/distill_h100 wrk-qq2ylgzjgory@ssh.deployments.targon.com \
    "kill \$(cat /root/reliquary/miner.pid) 2>/dev/null; sleep 2; \
     nohup bash -c 'exec /root/reliquary/run_miner.sh' \
         > /root/reliquary/miner.log 2>&1 & \
     echo \$! > /root/reliquary/miner.pid; \
     sleep 3; tail -15 /root/reliquary/miner.log | grep -vE 'httpx|huggingface_hub'"
```

Expected: "Miner ready. Entering main loop." within ~30 s after model load. The miner inherits the env from the validator's `/state` response (via `--environment math` default).

- [ ] **Step 4: Watch for the first acceptance**

Run:

```bash
ssh -i ~/.ssh/distill_h100 wrk-0nkyr4av9l9d@ssh.deployments.targon.com \
    "tail -F /root/reliquary/validator.log" | grep -E "accepted prompt|rewards=|sealed|train_step"
```

Expected within ~5–10 min of uptime:
- Multiple `rejected ... rewards=[...]` lines with varied reward vectors (not all `[1,1,1,1,1,1,1,1]` like GSM8K).
- At least one `accepted prompt=...` line.
- Eventually `Window N sealed` followed by `train_step: ...` (not `empty batch, skipping`).

If after 30 min every submission is still `rewards=[1,1,1,1,1,1,1,1]`, the reward normalizer is probably over-strict — file an issue; do not patch live, cycle back to task 3 and extend the normalization rules with the observed failure cases.

- [ ] **Step 5: No commit** (ops step — nothing to commit).

---

## Self-Review

**1. Spec coverage.** The goal was (a) replace GSM8K with MATH so the zone filter sees mid-difficulty problems, (b) keep the Environment Protocol intact, (c) update the default + docs. Task 1–5 build the MATH env against the Protocol; Task 6 performs the atomic swap (loader + default + GSM8K deletion); Task 7 is the integration smoke; Task 8 updates docs; Task 9 deploys. Every requirement has at least one task.

**2. Placeholder scan.** No `TBD`, no `"add error handling"`, no `"similar to Task N"`. Every test body and implementation body is spelled out in full. The only two-line step that isn't a full code block is Task 8 step 2, because it's an in-place rewrite across five docs — documenting each line would balloon the plan. The rewrite rules list is prescriptive enough that two different implementers would produce the same edits.

**3. Type consistency.** `_last_boxed_only_string`, `_strip_boxed_wrapper`, `_normalize_answer`, `_compute_math_reward`, `MATHEnvironment` are all referenced with the exact same names in every task that uses them. The `Environment` Protocol's shape (`name`, `__len__`, `get_problem`, `compute_reward`) is preserved — the new class matches the old signature exactly, so no downstream caller needs changes.

**4. One gap flagged for the implementer.** The Hendrycks dataset's `solution` field almost always ends with `\boxed{...}`, but a handful of rows in the test split have the answer inside `\fbox{...}` or with the `\boxed` applied inside a `\begin{align*}` block; Task 5 step 3 handles `\fbox` but not `align*`. If the integration smoke (Task 7) surfaces rows with empty `ground_truth`, the implementer should add a fallback: strip `\begin{align*}...\end{align*}` scaffolding before re-running `_last_boxed_only_string`. This is called out here so the reviewer knows it's a known-bounded risk, not a missed case.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-switch-gsm8k-to-math.md`.

**Execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each task is self-contained and produces a commit.

2. **Inline Execution** — Execute tasks in this session with batch checkpoints.

Which approach?
