"""EMA miner scoring — convergence, decay, burn math."""

from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from reliquary.constants import B_BATCH, EMA_ALPHA


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_service():
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._miner_scores_ema = defaultdict(float)
    return svc


def _fake_batch(hotkey_counts: dict[str, int]):
    """Build a list of fake ValidSubmissions from a (hotkey → count) map."""
    batch = []
    for hk, count in hotkey_counts.items():
        for _ in range(count):
            sub = MagicMock()
            sub.hotkey = hk
            batch.append(sub)
    return batch


def test_ema_converges_to_expected_fraction():
    """Steady state: one miner winning 2/8 slots every window → EMA ≈ 0.25."""
    svc = _make_service()
    for _ in range(500):
        svc._update_ema(_fake_batch({"alice": 2}))
    assert abs(svc._miner_scores_ema["alice"] - 0.25) < 0.001


def test_inactive_miner_decays_to_zero():
    """Miner active 50 windows then stops — EMA drops below 1% after ~200 more windows."""
    svc = _make_service()
    # 50 windows active
    for _ in range(50):
        svc._update_ema(_fake_batch({"alice": 8}))  # full batch
    peak = svc._miner_scores_ema["alice"]
    assert peak > 0.5  # pretty active

    # Stop — 200 windows of inactivity (empty batches)
    for _ in range(200):
        svc._update_ema([])

    # EMA should have decayed substantially
    final = svc._miner_scores_ema.get("alice", 0.0)
    assert final < 0.01 * peak  # less than 1% of peak


def test_sum_equals_fill_rate_steady_state():
    """If batch consistently half-full, sum(EMAs) converges to ~0.5."""
    svc = _make_service()
    # Fill 4/8 slots every window — 4 different miners, 1 slot each
    for _ in range(500):
        svc._update_ema(_fake_batch({"a": 1, "b": 1, "c": 1, "d": 1}))
    total = sum(svc._miner_scores_ema.values())
    assert abs(total - 0.5) < 0.005


def test_full_batch_sum_converges_to_one():
    """Full batch every window → sum of EMAs → 1.0 (zero burn at steady state)."""
    svc = _make_service()
    for _ in range(500):
        # 8 different miners, 1 slot each
        batch_counts = {f"hk{i}": 1 for i in range(8)}
        svc._update_ema(_fake_batch(batch_counts))
    total = sum(svc._miner_scores_ema.values())
    assert abs(total - 1.0) < 0.005


def test_empty_batch_decays_all_emas():
    """Empty batch for one window reduces every EMA by (1 - α)."""
    svc = _make_service()
    # Seed some EMAs
    svc._miner_scores_ema["alice"] = 0.5
    svc._miner_scores_ema["bob"] = 0.3

    svc._update_ema([])  # empty batch

    alpha = EMA_ALPHA
    expected_alice = (1 - alpha) * 0.5
    expected_bob = (1 - alpha) * 0.3
    assert abs(svc._miner_scores_ema["alice"] - expected_alice) < 1e-9
    assert abs(svc._miner_scores_ema["bob"] - expected_bob) < 1e-9


def test_prune_keeps_dict_bounded():
    """Near-zero EMAs (below 1e-6) are pruned to keep the dict small."""
    svc = _make_service()
    # Lots of short-lived miners
    for i in range(50):
        svc._update_ema(_fake_batch({f"ghost_{i}": 1}))
    # Lots of decay
    for _ in range(1000):
        svc._update_ema([])
    # Should have pruned most
    assert len(svc._miner_scores_ema) < 10


@pytest.mark.asyncio
async def test_submit_does_not_clear_ema():
    """_submit_weights must NOT reset _miner_scores_ema."""
    from unittest.mock import AsyncMock
    svc = _make_service()
    svc._miner_scores_ema["alice"] = 0.2
    svc._miner_scores_ema["bob"] = 0.3

    # Mock chain methods so submit runs without real network
    import reliquary.validator.service as svc_mod
    svc_mod.chain.get_metagraph = AsyncMock(return_value=MagicMock(
        hotkeys=["alice", "bob"], uids=[1, 2],
    ))
    svc_mod.chain.set_weights = AsyncMock(return_value=True)

    await svc._submit_weights(MagicMock())

    assert svc._miner_scores_ema["alice"] == 0.2
    assert svc._miner_scores_ema["bob"] == 0.3


def test_ema_in_memory_only_no_persist():
    """EMA is in-memory only — no state file is written."""
    import os
    import tempfile
    svc = _make_service()
    svc._miner_scores_ema["alice"] = 0.5
    svc._update_ema(_fake_batch({"alice": 2}))
    # No state file should have been written anywhere
    assert not os.path.exists("reliquary/state/checkpoint.json")
