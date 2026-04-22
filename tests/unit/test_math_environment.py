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
