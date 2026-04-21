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


import json
import tempfile
from pathlib import Path


def test_persist_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cd.json"
    m1 = CooldownMap(cooldown_windows=50)
    m1.record_batched(prompt_idx=42, window=100)
    m1.record_batched(prompt_idx=7, window=105)
    m1.save(path)

    m2 = CooldownMap(cooldown_windows=50)
    m2.load(path)
    assert m2.is_in_cooldown(42, 110) is True
    assert m2.is_in_cooldown(7, 110) is True
    assert m2.is_in_cooldown(99, 110) is False


def test_load_missing_file_is_empty(tmp_path: Path):
    m = CooldownMap(cooldown_windows=50)
    m.load(tmp_path / "nonexistent.json")
    assert len(m) == 0


def test_load_malformed_file_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    m = CooldownMap(cooldown_windows=50)
    with pytest.raises(json.JSONDecodeError):
        m.load(path)


def test_rebuild_from_history_takes_most_recent():
    """If the same prompt appeared in multiple archived windows, keep the latest."""
    archived = [
        {"window_start": 100, "batch": [{"prompt_idx": 42}, {"prompt_idx": 7}]},
        {"window_start": 105, "batch": [{"prompt_idx": 42}, {"prompt_idx": 99}]},
        {"window_start": 110, "batch": [{"prompt_idx": 7}]},
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    # Prompt 42 last seen at 105
    assert m.is_in_cooldown(42, 120) is True
    assert m.is_in_cooldown(42, 155) is True
    assert m.is_in_cooldown(42, 156) is False
    # Prompt 7 last seen at 110
    assert m.is_in_cooldown(7, 159) is True
    assert m.is_in_cooldown(7, 160) is False
    # Prompt 99 last seen at 105
    assert m.is_in_cooldown(99, 154) is True


def test_rebuild_ignores_windows_older_than_cooldown():
    """Windows older than cooldown horizon are pointless to load."""
    archived = [
        {"window_start": 10, "batch": [{"prompt_idx": 42}]},   # way expired
        {"window_start": 105, "batch": [{"prompt_idx": 7}]},   # fresh
    ]
    m = CooldownMap(cooldown_windows=50)
    m.rebuild_from_history(archived, current_window=120)
    assert m.is_in_cooldown(42, 120) is False  # expired long ago
    assert m.is_in_cooldown(7, 120) is True
