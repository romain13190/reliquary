"""ValidatorState: simple JSON-backed counters for window_n + checkpoint_n."""

import os
from pathlib import Path

from reliquary.validator.state_persistence import ValidatorState


def test_default_counters_are_zero(tmp_path: Path):
    s = ValidatorState(path=str(tmp_path / "s.json"))
    assert s.window_n == 0
    assert s.checkpoint_n == 0


def test_save_and_load_roundtrip(tmp_path: Path):
    p = str(tmp_path / "s.json")
    s1 = ValidatorState(path=p)
    s1.window_n = 42
    s1.checkpoint_n = 7
    s1.save()

    s2 = ValidatorState(path=p)
    s2.load()
    assert s2.window_n == 42
    assert s2.checkpoint_n == 7


def test_load_missing_file_keeps_defaults(tmp_path: Path):
    s = ValidatorState(path=str(tmp_path / "missing.json"))
    s.load()
    assert s.window_n == 0
    assert s.checkpoint_n == 0


def test_atomic_save_no_leftover_tmp(tmp_path: Path):
    """Save uses tmp + rename so partial writes don't corrupt."""
    p = tmp_path / "s.json"
    s = ValidatorState(path=str(p))
    s.window_n = 1
    s.save()
    assert p.exists()
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".s.")]
    assert leftovers == []


def test_save_creates_parent_dir(tmp_path: Path):
    """Save creates intermediate directories if missing."""
    p = tmp_path / "deep" / "nested" / "state.json"
    s = ValidatorState(path=str(p))
    s.window_n = 5
    s.save()
    assert p.exists()
