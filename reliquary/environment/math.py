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
