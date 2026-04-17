"""GSM8K environment for Reliquary verifiable inference.

Loads the openai/gsm8k dataset (train split) and provides:
- Deterministic (prompt, ground_truth) pairs indexed by integer.
- A reward function that extracts the numeric answer from a completion
  and compares it (float-wise, normalized) to the ground truth.
"""

from __future__ import annotations

import hashlib
import re
from typing import ClassVar, Optional


# ---------------------------------------------------------------------------
# Reward logic — extracted as a module-level function for easy unit testing
# without constructing a GSM8KEnvironment (which loads the dataset).
# ---------------------------------------------------------------------------

# Matches LaTeX \boxed{...} — captures the inner content.
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

# Matches numbers with optional thousands separators and optional decimal part.
# Handles negatives.  Captures the LAST occurrence via findall[-1].
# The thousands-separator branch requires at least one comma group so that
# plain integers like "1234" are matched whole by the second branch (greedy).
_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
)


def _normalize_number(text: str) -> str:
    """Strip whitespace and commas, then strip trailing decimal zeros."""
    text = text.strip().replace(",", "")
    # Remove trailing zeros after a decimal point (42.00 → 42, 3.50 → 3.5)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _extract_candidate(completion: str) -> Optional[str]:
    """Extract the best numeric candidate from a completion string.

    Priority:
        1. Last \\boxed{...} match (LaTeX answer boxes).
        2. Last bare numeric token.

    Returns the raw extracted string (un-normalized), or None if nothing found.
    """
    boxed_matches = _BOXED_RE.findall(completion)
    if boxed_matches:
        return boxed_matches[-1]

    number_matches = _NUMBER_RE.findall(completion)
    if number_matches:
        return number_matches[-1]

    return None


def _compute_gsm8k_reward(problem: dict, completion: str) -> float:
    """Compute reward for a GSM8K completion.

    Returns 1.0 if the extracted numeric answer matches the ground truth
    (after normalization and float comparison), 0.0 otherwise.
    Never raises.
    """
    try:
        ground_truth = str(problem.get("ground_truth", ""))
        candidate_raw = _extract_candidate(completion)
        if candidate_raw is None:
            return 0.0

        candidate_norm = _normalize_number(candidate_raw)
        gt_norm = _normalize_number(ground_truth)

        # Float comparison with tight tolerance.
        candidate_f = float(candidate_norm)
        gt_f = float(gt_norm)
        if abs(candidate_f - gt_f) < 1e-6:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# GSM8KEnvironment
# ---------------------------------------------------------------------------

class GSM8KEnvironment:
    """Environment backed by the openai/gsm8k train split.

    The dataset is loaded once and cached at the class level so that
    multiple instantiations in the same process are cheap.
    """

    name: str = "gsm8k"

    # Class-level cache: populated on first instantiation.
    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        if GSM8KEnvironment._dataset_cache is None:
            import datasets as hf_datasets  # local import keeps module importable w/o datasets
            GSM8KEnvironment._dataset_cache = hf_datasets.load_dataset(
                "openai/gsm8k", "main", split="train"
            )
        self._dataset = GSM8KEnvironment._dataset_cache

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        """Return problem at *index* (wraps modulo dataset length)."""
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        question: str = row["question"]
        # Ground truth is the part after "####" in the answer field.
        answer_field: str = row["answer"]
        gt_str = answer_field.split("####")[-1].strip()
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        return {
            "prompt": question,
            "ground_truth": gt_str,
            "id": problem_id,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        """Score a completion using GSM8K numeric reward."""
        return _compute_gsm8k_reward(problem, completion)
