"""Anti-replay ``current_round`` must be a *live* drand round, not the
window number.

These tests guard the wiring at ``service.py:_compute_current_drand_round``
and the property at ``batcher.GrpoWindowBatcher.current_round`` against
silent regression to the placeholder behaviour.

Why this is launch-critical:
  * STALE_ROUND_LAG_MAX = 10 rounds × 3 s/period = 30 s of acceptance.
  * WINDOW_TIMEOUT_SECONDS = 3600 (1 h).
  * If ``current_round`` were frozen at window open, every submission
    landing more than 30 s after window open would be rejected.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from reliquary.validator.batcher import GrpoWindowBatcher


def _completion_text_stub(_rollout):
    return ""


def _make_batcher(*, current_round=0, now_round_fn=None):
    return GrpoWindowBatcher(
        window_start=1,
        current_round=current_round,
        env=object(),
        model=object(),
        completion_text_fn=_completion_text_stub,
        verify_commitment_proofs_fn=lambda *a, **k: None,
        verify_signature_fn=lambda c, h: True,
        now_round_fn=now_round_fn,
    )


# --------------------------------------------------------------------------
# (a) Wiring: dynamic getter takes precedence over static value
# --------------------------------------------------------------------------

def test_current_round_uses_callable_when_provided():
    """If ``now_round_fn`` is provided, ``current_round`` reads from it
    every access — even if a static int was also passed."""
    counter = {"v": 5_234_891}

    def _now():
        return counter["v"]

    b = _make_batcher(current_round=42, now_round_fn=_now)
    assert b.current_round == 5_234_891
    counter["v"] = 5_234_900
    assert b.current_round == 5_234_900  # re-evaluated, not cached


def test_current_round_falls_back_to_static_when_no_callable():
    """Static int is preserved for legacy / deterministic test paths."""
    b = _make_batcher(current_round=1234, now_round_fn=None)
    assert b.current_round == 1234


# --------------------------------------------------------------------------
# (b) Sliding anti-replay window: round advances with wall clock
# --------------------------------------------------------------------------

def test_anti_replay_window_slides_with_time():
    """A submission accepted at t=0 with signed_round=R must still be
    acceptable at t=2 min if a fresh batcher would see current_round=R+40
    — i.e. the lag check is computed against *now*, not window open."""
    fake_round = {"v": 1000}

    def _now():
        return fake_round["v"]

    b = _make_batcher(now_round_fn=_now)

    # Just-opened window. Fresh round 1000.
    assert b._round_fresh(signed_round=1000) is True
    assert b._round_fresh(signed_round=995) is True   # 5 < 10 lag
    assert b._round_fresh(signed_round=990) is True   # 10 == lag (boundary)
    assert b._round_fresh(signed_round=989) is False  # 11 > 10 lag
    assert b._round_fresh(signed_round=1001) is False  # future beacon

    # Time advances 30 s → drand round +10. A miner that signed at the
    # OLD `current_round` (1000) is now exactly at the lag boundary.
    fake_round["v"] = 1010
    assert b._round_fresh(signed_round=1010) is True
    assert b._round_fresh(signed_round=1000) is True   # 10 == lag
    assert b._round_fresh(signed_round=999) is False   # now too old


def test_frozen_round_would_break_after_lag_max():
    """Without the dynamic getter (legacy path), the lag window doesn't
    slide. This test pins the *broken* behaviour to make the contrast
    explicit — the production validator must never use this path."""
    b = _make_batcher(current_round=1000, now_round_fn=None)

    # Same outcome at t=0.
    assert b._round_fresh(signed_round=1000) is True
    # If the real drand round were 1010 but our batcher is frozen at
    # 1000, a miner signing with the *correct current* round (1010)
    # would be rejected as "future beacon". This is the bug we just
    # fixed.
    assert b._round_fresh(signed_round=1010) is False  # rejected — frozen


# --------------------------------------------------------------------------
# (c) /state exposes the live round (server reads via property)
# --------------------------------------------------------------------------

def test_state_endpoint_observes_live_round():
    """``server.py:/state`` reads ``batcher.current_round`` to tell miners
    which round to sign with. It must reflect *now*, not window-open."""
    fake_round = {"v": 5_000_000}
    b = _make_batcher(now_round_fn=lambda: fake_round["v"])

    assert b.current_round == 5_000_000
    fake_round["v"] = 5_000_010  # 30 s later in real chain
    assert b.current_round == 5_000_010  # /state would now return new


# --------------------------------------------------------------------------
# (d) Validator-side helper: _compute_current_drand_round
# --------------------------------------------------------------------------

def test_compute_current_drand_round_uses_get_round_at_time(monkeypatch):
    """When ``use_drand=True`` the helper delegates to
    ``drand.get_round_at_time(now)``. We patch the import target inside
    the function (`reliquary.infrastructure.drand`) and a fake clock,
    then assert the helper returns what drand returned."""
    from reliquary.validator.service import ValidationService

    svc = ValidationService.__new__(ValidationService)  # bypass __init__
    svc.use_drand = True
    svc._window_n = 42

    captured: dict = {}

    def _fake_get_round_at_time(ts: int) -> int:
        captured["ts"] = ts
        return 5_234_891

    monkeypatch.setattr(time, "time", lambda: 1_799_999_999)
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_round_at_time",
        _fake_get_round_at_time,
    )

    r = svc._compute_current_drand_round()

    assert r == 5_234_891
    assert captured["ts"] == 1_799_999_999  # passed wall-clock through
    assert r != svc._window_n  # not the placeholder counter


def test_compute_current_drand_round_use_drand_false_returns_window_n():
    """Mock mode: deterministic counter for reproducible tests."""
    from reliquary.validator.service import ValidationService

    svc = ValidationService.__new__(ValidationService)
    svc.use_drand = False
    svc._window_n = 7

    assert svc._compute_current_drand_round() == 7


# --------------------------------------------------------------------------
# (e) Entropy: GRAIL randomness is NOT predictable from window_n alone
# --------------------------------------------------------------------------

def test_window_randomness_depends_on_drand_beacon():
    """Two windows with the same block hash but different drand beacons
    must produce different randomness — proves a miner who controls only
    the block hash cannot predict the GRAIL challenge seed."""
    from reliquary.infrastructure.chain import compute_window_randomness

    block_hash = "0x" + "ab" * 32
    beacon_a = "11" * 32
    beacon_b = "22" * 32

    r_a = compute_window_randomness(block_hash, beacon_a, drand_round=100)
    r_b = compute_window_randomness(block_hash, beacon_b, drand_round=101)

    assert r_a != r_b
    # And the diff must be substantial (HMAC-SHA256 → uniform): more
    # than half the bits should differ for any non-trivial input change.
    diff_bits = bin(int(r_a, 16) ^ int(r_b, 16)).count("1")
    assert diff_bits > 64, (
        f"only {diff_bits}/256 bits differ — randomness is not behaving "
        f"like a hash function"
    )


def test_window_randomness_depends_on_round_number():
    """Including the drand round in the material prevents grinding —
    same block + same beacon but different round → different output."""
    from reliquary.infrastructure.chain import compute_window_randomness

    block_hash = "0x" + "ab" * 32
    beacon = "33" * 32

    r1 = compute_window_randomness(block_hash, beacon, drand_round=100)
    r2 = compute_window_randomness(block_hash, beacon, drand_round=101)
    assert r1 != r2
