"""Tests for the _prefixes_distinct helper in reliquary.validator.batcher."""

from __future__ import annotations

import pytest

from reliquary.validator.batcher import _prefixes_distinct


class TestPrefixesDistinct:
    def test_all_distinct(self) -> None:
        """4 completions with pairwise-distinct prefixes after the prompt → True."""
        prompt_len = 3
        prefix_len = 4
        # Each completion has a unique run of tokens after the prompt.
        token_lists = [
            [10, 20, 30, 1, 2, 3, 4, 99, 99],   # prefix = [1,2,3,4]
            [10, 20, 30, 5, 6, 7, 8, 99, 99],   # prefix = [5,6,7,8]
            [10, 20, 30, 9, 10, 11, 12, 99, 99], # prefix = [9,10,11,12]
            [10, 20, 30, 13, 14, 15, 16, 99, 99],# prefix = [13,14,15,16]
        ]
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is True

    def test_two_share_prefix(self) -> None:
        """2 completions sharing the prefix → False."""
        prompt_len = 2
        prefix_len = 3
        token_lists = [
            [0, 0, 1, 2, 3],
            [0, 0, 1, 2, 3],   # duplicate of first
            [0, 0, 7, 8, 9],
            [0, 0, 4, 5, 6],
        ]
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is False

    def test_all_identical(self) -> None:
        """All 4 completions with the same tokens → False."""
        prompt_len = 1
        prefix_len = 4
        tokens = [99, 1, 2, 3, 4, 5, 6]
        token_lists = [tokens[:] for _ in range(4)]
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is False

    def test_one_too_short(self) -> None:
        """One completion shorter than prompt_length + prefix_len → False."""
        prompt_len = 3
        prefix_len = 4
        token_lists = [
            [10, 20, 30, 1, 2, 3, 4],   # exactly prompt+prefix — ok
            [10, 20, 30, 5, 6, 7, 8],
            [10, 20, 30, 9, 10, 11],    # only 3 tokens of prefix (needs 4) → too short
            [10, 20, 30, 13, 14, 15, 16],
        ]
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is False

    def test_prefix_len_one_distinct(self) -> None:
        """prefix_len=1 and each completion differs at position 0 after prompt → True."""
        prompt_len = 2
        prefix_len = 1
        token_lists = [
            [0, 0, 1, 99, 99],
            [0, 0, 2, 99, 99],
            [0, 0, 3, 99, 99],
            [0, 0, 4, 99, 99],
        ]
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is True

    def test_prefix_len_zero(self) -> None:
        """prefix_len=0 is a degenerate case: empty prefix tuples are all equal,
        but since we extract zero tokens, there is nothing to distinguish.
        The function returns True because the spec document says prefix_len=0
        is vacuously satisfied (no prefix constraint imposed).
        """
        prompt_len = 2
        prefix_len = 0
        token_lists = [
            [0, 0, 1, 2, 3],
            [0, 0, 4, 5, 6],
            [0, 0, 7, 8, 9],
            [0, 0, 10, 11, 12],
        ]
        # All empty-tuple prefixes: degenerate → True (vacuously distinct)
        assert _prefixes_distinct(token_lists, prompt_len, prefix_len) is True
