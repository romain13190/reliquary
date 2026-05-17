"""Parallel race in ``_http_get_json``.

The old implementation shuffled ``DRAND_URLS`` and tried relays
serially. When the first pick was slow/dead, callers ate the full
timeout (~5s) before any fallback fired. The new behavior races all
relays in parallel and returns as soon as one responds with a
parsable 200 JSON — wall-clock cost becomes bounded by the FASTEST
relay, not the SLOWEST.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from reliquary.infrastructure import drand as D


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else "ok"

    def json(self):
        if self._payload is None:
            raise ValueError("no payload")
        return self._payload


def _make_relay_get(latency_by_base: dict[str, float], payload_by_base: dict[str, dict | None]):
    """Build a Session.get replacement that sleeps per-base and returns the canned payload."""

    def _fake_get(url, timeout=None, headers=None):
        base = next((b for b in latency_by_base if url.startswith(b)), None)
        if base is None:
            raise RuntimeError(f"unexpected url in test: {url}")
        time.sleep(latency_by_base[base])
        payload = payload_by_base.get(base)
        if payload is None:
            # Simulate connection refused / timeout: raise like requests would.
            raise RuntimeError(f"simulated network error from {base}")
        return _FakeResp(200, payload)

    return _fake_get


def test_http_get_json_returns_fastest_relay():
    """All 5 relays fire in parallel; the one that returns first wins,
    regardless of shuffle order. Total wall-clock must be near the
    fastest relay's latency, not the sum."""
    relays = D.DRAND_URLS
    assert len(relays) >= 3, "test assumes >=3 relays configured"

    # Fastest relay is the third in the list — race must surface it.
    latencies = {r: 0.5 for r in relays}
    latencies[relays[2]] = 0.05
    payloads = {r: {"randomness": "aa" * 32, "round": 1, "from": r} for r in relays}

    with patch.object(D._SESSION, "get", side_effect=_make_relay_get(latencies, payloads)):
        t0 = time.monotonic()
        result = D._http_get_json(["/v2/chains/x/rounds/1"])
        elapsed = time.monotonic() - t0

    assert result is not None
    assert result["from"] == relays[2], f"expected winner {relays[2]}, got {result.get('from')}"
    assert elapsed < 0.30, (
        f"parallel race took {elapsed*1000:.0f}ms — should be near 50ms "
        f"(fastest relay), not 500ms (slowest)"
    )


def test_http_get_json_survives_majority_failure():
    """Four relays fail/timeout, one responds at 200ms. Result must
    arrive in ~200ms, NOT 4 × (connect_timeout + read_timeout)."""
    relays = D.DRAND_URLS
    assert len(relays) >= 3
    latencies = {r: 0.05 for r in relays}  # failures return fast in this stub
    latencies[relays[-1]] = 0.20  # the one slow-but-good relay
    payloads = {r: None for r in relays}  # all fail by default
    payloads[relays[-1]] = {"randomness": "bb" * 32, "round": 2, "from": relays[-1]}

    with patch.object(D._SESSION, "get", side_effect=_make_relay_get(latencies, payloads)):
        t0 = time.monotonic()
        result = D._http_get_json(["/v2/chains/x/rounds/2"])
        elapsed = time.monotonic() - t0

    assert result is not None
    assert result["from"] == relays[-1]
    assert elapsed < 0.50, (
        f"single-survivor race took {elapsed*1000:.0f}ms — should be ~200ms"
    )


def test_http_get_json_returns_none_when_all_fail():
    """Contract preserved: if every relay errors, return None (callers
    handle the fallback / raise themselves)."""
    relays = D.DRAND_URLS
    latencies = {r: 0.05 for r in relays}
    payloads = {r: None for r in relays}

    with patch.object(D._SESSION, "get", side_effect=_make_relay_get(latencies, payloads)):
        result = D._http_get_json(["/v2/chains/x/rounds/3"])

    assert result is None


def test_http_get_json_tries_all_paths_if_first_path_404s():
    """The path-list fallback (v2 → v1 → root_v1) must still work.
    Per-base we race relays, but if every relay returns non-200 for
    path[0], the function must fall through to path[1]."""
    relays = D.DRAND_URLS
    latencies = {r: 0.05 for r in relays}
    # All relays return None (treated as failure) for the v2 path, but
    # return a payload for the v1 fallback path. The fake_get below is
    # path-aware via the url-prefix dispatch.
    def _fake_get(url, timeout=None, headers=None):
        base = next((b for b in relays if url.startswith(b)), None)
        time.sleep(latencies[base])
        if "/v2/" in url:
            raise RuntimeError("v2 endpoint down")
        return _FakeResp(200, {"randomness": "cc" * 32, "round": 4, "path": "v1"})

    with patch.object(D._SESSION, "get", side_effect=_fake_get):
        result = D._http_get_json(["/v2/chains/x/rounds/4", "/x/public/4"])

    assert result is not None
    assert result["path"] == "v1"
