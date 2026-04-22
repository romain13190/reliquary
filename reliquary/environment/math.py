"""Hendrycks MATH environment for Reliquary verifiable inference.

Loads the full Hendrycks MATH dataset (12 500 problems) from the qwedsacf mirror
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


def _strip_boxed_wrapper(s: str) -> str:
    """If ``s`` starts with \\boxed{ or \\fbox{ and ends with a matching },
    return the inner content. Otherwise return ``s`` unchanged.
    """
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return s


# ---------------------------------------------------------------------------
# Answer normalization — conservative LaTeX simplification for comparison
# ---------------------------------------------------------------------------

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


class MATHEnvironment:
    """Environment backed by the full Hendrycks MATH set (12 500 problems).

    Ground truths are extracted once from the ``solution`` field by taking
    the content of the last \\boxed{...}; completions are scored with the
    same extraction against the completion text.

    Uses the ``qwedsacf/competition_math`` HF mirror — the original
    ``hendrycks/competition_math`` was removed from the Hub.
    """

    name: str = "math"

    # Class-level cache: populated on first instantiation so tests in the
    # same process share the one HF download.
    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        if MATHEnvironment._dataset_cache is None:
            import datasets as hf_datasets  # local import keeps module importable w/o datasets
            MATHEnvironment._dataset_cache = hf_datasets.load_dataset(
                "qwedsacf/competition_math", split="train"
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
