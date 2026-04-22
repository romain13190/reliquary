"""WeightOnlyValidator main loop: read archives, replay EMA, submit on chain tempo."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeWallet:
    class _Hk:
        ss58_address = "5FReader"
    hotkey = _Hk()


def _archive(window_start, batch):
    return {
        "window_start": window_start,
        "batch": [{"hotkey": hk, "prompt_idx": 0} for hk in batch],
    }


@pytest.mark.asyncio
async def test_replay_ema_deterministic():
    from reliquary.validator.weight_only import WeightOnlyValidator
    archives = [
        _archive(1, ["alice", "bob"]),
        _archive(2, ["alice"]),
        _archive(3, ["bob", "carol"]),
    ]
    ema = WeightOnlyValidator._replay_ema(archives)
    assert "alice" in ema
    assert "bob" in ema
    assert "carol" in ema
    # alice and bob each contributed across more windows than carol
    assert ema["alice"] > ema["carol"]
    assert ema["bob"] > ema["carol"]


@pytest.mark.asyncio
async def test_replay_ema_is_order_invariant_given_sorted_input():
    from reliquary.validator.weight_only import WeightOnlyValidator
    archives = [
        _archive(2, ["alice"]),
        _archive(1, ["bob"]),
        _archive(3, ["alice", "bob"]),
    ]
    ema_a = WeightOnlyValidator._replay_ema(archives)
    ema_b = WeightOnlyValidator._replay_ema(list(reversed(archives)))
    # Both inputs sort to [1,2,3] internally → identical output
    assert ema_a == ema_b


@pytest.mark.asyncio
async def test_replay_ema_empty_archives():
    from reliquary.validator.weight_only import WeightOnlyValidator
    ema = WeightOnlyValidator._replay_ema([])
    assert ema == {}


@pytest.mark.asyncio
async def test_submit_weights_maps_hotkeys_to_uids():
    """_submit_weights calls chain.set_weights with correct uid mapping."""
    from reliquary.validator.weight_only import WeightOnlyValidator

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)

    fake_meta = MagicMock()
    fake_meta.hotkeys = ["alice", "bob", "carol"]
    fake_meta.uids = [10, 20, 30]

    import reliquary.validator.weight_only as wov_mod
    original_get_meta = wov_mod.chain.get_metagraph
    original_set_weights = wov_mod.chain.set_weights
    wov_mod.chain.get_metagraph = AsyncMock(return_value=fake_meta)
    captured = {}

    async def _fake_set_weights(subtensor, wallet, netuid, uids, weights):
        captured["uids"] = uids
        captured["weights"] = weights
        return True

    wov_mod.chain.set_weights = _fake_set_weights

    try:
        await wov._submit_weights(
            MagicMock(),
            {"alice": 0.5, "bob": 0.3},
            burn_weight=0.2,
        )
    finally:
        wov_mod.chain.get_metagraph = original_get_meta
        wov_mod.chain.set_weights = original_set_weights

    from reliquary.constants import UID_BURN
    assert 10 in captured["uids"]   # alice → uid 10
    assert 20 in captured["uids"]   # bob → uid 20
    assert UID_BURN in captured["uids"]


@pytest.mark.asyncio
async def test_run_loop_submits_after_interval():
    """run() submits weights when enough blocks have passed."""
    import asyncio
    from reliquary.validator.weight_only import WeightOnlyValidator
    from reliquary.constants import WEIGHT_SUBMISSION_INTERVAL

    wov = WeightOnlyValidator(wallet=_FakeWallet(), netuid=81)
    wov._last_submit_block = 0

    call_count = 0

    import reliquary.validator.weight_only as wov_mod
    original_get_block = wov_mod.chain.get_current_block
    original_list_all = wov_mod.storage.list_all_window_keys
    original_list_recent = wov_mod.storage.list_recent_datasets
    original_submit = wov._submit_weights

    wov_mod.chain.get_current_block = AsyncMock(return_value=WEIGHT_SUBMISSION_INTERVAL + 1)
    wov_mod.storage.list_all_window_keys = AsyncMock(return_value=[1, 2, 3])
    wov_mod.storage.list_recent_datasets = AsyncMock(return_value=[
        _archive(1, ["alice"]),
        _archive(2, ["alice"]),
        _archive(3, ["bob"]),
    ])

    async def _fake_submit(subtensor, miner_weights, burn_weight):
        nonlocal call_count
        call_count += 1
        return True

    wov._submit_weights = _fake_submit

    try:
        # Run one iteration then cancel
        task = asyncio.create_task(wov.run(MagicMock()))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        wov_mod.chain.get_current_block = original_get_block
        wov_mod.storage.list_all_window_keys = original_list_all
        wov_mod.storage.list_recent_datasets = original_list_recent
        wov._submit_weights = original_submit

    assert call_count >= 1
