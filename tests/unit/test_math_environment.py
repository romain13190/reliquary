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
