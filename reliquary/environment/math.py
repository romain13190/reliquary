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
