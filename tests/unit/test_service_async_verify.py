"""Tests for the validator's background drand-verify task.

Pins three properties:
  1. ``_set_window_randomness`` does NOT block on the bittensor_drand
     cross-check. The hot OPEN path stays sub-400ms even when the
     cross-check takes seconds. (Verified in Task 5.)
  2. If the cross-check fails after OPEN, the batcher's
     ``beacon_invalid`` flag flips to True. (Verified in Task 5.)
  3. ``_train_and_publish`` skips seal/train/archive when the flag
     is set — preserves the security gate end-to-end. (Verified in Task 5.)

This file initially holds the Task-4-level tests pinning the new
``_derive_randomness`` return signature.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_derive_randomness_returns_tuple_in_drand_mode(monkeypatch):
    """The hot-path refactor needs the raw beacon dict so the
    background verify task can cross-check the right (round, sig).
    _derive_randomness now returns (randomness_str, beacon_dict).
    """
    from reliquary.validator import service as svc_mod

    class _StubSvc:
        use_drand = True
        async def _derive_randomness(self, subtensor, target_window):
            return await svc_mod.ValidationService._derive_randomness(
                self, subtensor, target_window,
            )

    # Stub the drand calls so the test doesn't hit the network.
    fake_beacon = {
        "round": 999,
        "randomness": "aa" * 32,
        "signature": "bb" * 48,
    }
    fake_chain_info = {"genesis_time": 1692803367, "period": 3}

    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_beacon",
        lambda round_id, use_drand: fake_beacon,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain",
        lambda: fake_chain_info,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_current_drand_round",
        lambda *a, **kw: 999,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_window_randomness",
        lambda *a, **kw: "computed_randomness_hex",
    )

    result = await _StubSvc()._derive_randomness(subtensor=None, target_window=42)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2
    randomness, beacon = result
    assert randomness == "computed_randomness_hex"
    assert beacon == fake_beacon


@pytest.mark.asyncio
async def test_derive_randomness_returns_none_beacon_in_mock_mode(monkeypatch):
    """When use_drand is False, the legacy block-hash path returns
    (randomness, None) so the verify scheduler skips cleanly."""
    from reliquary.validator import service as svc_mod

    class _StubSvc:
        use_drand = False
        async def _derive_randomness(self, subtensor, target_window):
            return await svc_mod.ValidationService._derive_randomness(
                self, subtensor, target_window,
            )

    async def _fake_block_hash(subtensor, window_n):
        return "0xdeadbeef"

    monkeypatch.setattr(
        "reliquary.infrastructure.chain.get_block_hash",
        _fake_block_hash,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_window_randomness",
        lambda block_hash: "mock_randomness",
    )

    randomness, beacon = await _StubSvc()._derive_randomness(
        subtensor="stub", target_window=7,
    )
    assert randomness == "mock_randomness"
    assert beacon is None
