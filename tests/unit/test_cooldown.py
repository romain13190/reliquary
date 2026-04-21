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
