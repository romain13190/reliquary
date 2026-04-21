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


def test_miner_scores_ema_roundtrip(tmp_path: Path):
    """miner_scores_ema dict survives save/load."""
    p = str(tmp_path / "s.json")
    s1 = ValidatorState(path=p)
    s1.window_n = 10
    s1.miner_scores_ema = {"alice": 0.25, "bob": 0.1}
    s1.save()

    s2 = ValidatorState(path=p)
    s2.load()
    assert abs(s2.miner_scores_ema["alice"] - 0.25) < 1e-9
    assert abs(s2.miner_scores_ema["bob"] - 0.1) < 1e-9


def test_default_miner_scores_ema_is_empty(tmp_path: Path):
    """Fresh ValidatorState starts with an empty EMA dict."""
    s = ValidatorState(path=str(tmp_path / "s.json"))
    assert s.miner_scores_ema == {}


def test_load_legacy_file_without_ema_key(tmp_path: Path):
    """Old state files without miner_scores_ema load cleanly (defaults to {})."""
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"window_n": 5, "checkpoint_n": 2}))

    s = ValidatorState(path=str(p))
    s.load()
    assert s.window_n == 5
    assert s.checkpoint_n == 2
    assert s.miner_scores_ema == {}
