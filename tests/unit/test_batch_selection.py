"""select_batch: deterministic, cooldown-aware batch assembly."""

from dataclasses import dataclass

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
        _sub("second", prompt_idx=42, signed_round=1001),
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
    batch = select_batch(subs, b=8, current_window=150, cooldown_map=cd)
    assert [s.hotkey for s in batch] == ["hk"]


def test_tiebreak_deterministic_on_same_round():
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("alice", prompt_idx=1, signed_round=1000, merkle_root=b"\xff" * 32),
        _sub("bob", prompt_idx=2, signed_round=1000, merkle_root=b"\x01" * 32),
    ]
    batch_a = select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    batch_b = select_batch(list(reversed(subs)), b=8, current_window=100, cooldown_map=cd)
    assert [s.hotkey for s in batch_a] == [s.hotkey for s in batch_b]


def test_cooldown_not_mutated_by_select():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub("hk", prompt_idx=42, signed_round=1000)]
    select_batch(subs, b=8, current_window=100, cooldown_map=cd)
    assert cd.is_in_cooldown(42, 100) is False


def test_partial_fill_when_all_cooldown_blocked():
    cd = CooldownMap(cooldown_windows=50)
    for idx in (1, 2, 3):
        cd.record_batched(idx, window=100)
    subs = [_sub(f"hk{i}", prompt_idx=i, signed_round=1000) for i in (1, 2, 3)]
    batch = select_batch(subs, b=8, current_window=110, cooldown_map=cd)
    assert batch == []
