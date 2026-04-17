"""Unit tests for ValidationService orchestration logic."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import WINDOW_LENGTH
from reliquary.validator.service import ValidationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(**kwargs) -> ValidationService:
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5Validator..."
    model = MagicMock()
    tokenizer = MagicMock()
    env = MagicMock()
    env.name = "test_env"
    env.__len__ = MagicMock(return_value=100)
    env.get_problem = MagicMock(
        side_effect=lambda idx: {
            "id": f"prob_{idx:04x}",
            "prompt": f"Question {idx}",
            "ground_truth": "42",
        }
    )
    env.compute_reward = MagicMock(return_value=1.0)

    defaults = dict(
        wallet=wallet,
        model=model,
        tokenizer=tokenizer,
        env=env,
        netuid=1,
        use_drand=False,
        http_host="127.0.0.1",
        http_port=19999,
    )
    defaults.update(kwargs)
    return ValidationService(**defaults)


# ---------------------------------------------------------------------------
# 1. _compute_target_window
# ---------------------------------------------------------------------------


class TestComputeTargetWindow:
    """``target = (block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH``.

    Parametrised on the current ``WINDOW_LENGTH`` so the assertions stay
    correct whether the constant is 5 or 30.
    """

    def test_block_just_after_window_start(self) -> None:
        wl = WINDOW_LENGTH
        block = wl * 3 + 1
        assert ValidationService._compute_target_window(block) == wl * 2

    def test_block_one_before_next_window(self) -> None:
        wl = WINDOW_LENGTH
        block = wl * 4 - 1
        assert ValidationService._compute_target_window(block) == wl * 2

    def test_block_at_exact_window_boundary(self) -> None:
        wl = WINDOW_LENGTH
        block = wl * 4
        assert ValidationService._compute_target_window(block) == wl * 3


# ---------------------------------------------------------------------------
# 2. _run_window settles early when batcher is complete
# ---------------------------------------------------------------------------


class TestRunWindowSettlesEarly:
    @pytest.mark.asyncio
    async def test_settles_early_accumulates_scores_and_clears_batcher(self):
        svc = _make_service()

        # Patch chain.get_block_hash to return a fixed 64-char hex string
        randomness_hex = "ab" * 32  # 64 hex chars

        fake_batcher = MagicMock()
        fake_batcher.is_window_complete.return_value = True
        fake_batcher.get_miner_scores.return_value = {"miner_a": 5.0, "miner_b": 3.0}
        fake_batcher.get_archive_data.return_value = {"window_start": 60, "slots": []}

        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_block_hash", new=AsyncMock(return_value=randomness_hex)),
            patch("reliquary.validator.service.chain.compute_window_randomness", return_value=randomness_hex),
            patch("reliquary.validator.service.derive_window_prompts", return_value=[
                {"id": f"p{i}", "prompt": f"Q{i}", "ground_truth": "1"}
                for i in range(8)
            ]),
            patch("reliquary.validator.service.WindowBatcher", return_value=fake_batcher),
            patch("reliquary.validator.service.storage.upload_window_dataset", new=AsyncMock(return_value=True)),
        ):
            await svc._run_window(subtensor, 60)

        # Scores accumulated
        assert svc._miner_scores["miner_a"] == 5.0
        assert svc._miner_scores["miner_b"] == 3.0
        # Active batcher cleared
        assert svc.server.active_batcher is None


# ---------------------------------------------------------------------------
# 3. _run_window clears active batcher even on archive failure
# ---------------------------------------------------------------------------


class TestRunWindowClearsBatcherOnArchiveFailure:
    @pytest.mark.asyncio
    async def test_active_batcher_cleared_even_on_archive_exception(self):
        svc = _make_service()

        randomness_hex = "cd" * 32

        fake_batcher = MagicMock()
        fake_batcher.is_window_complete.return_value = True
        fake_batcher.get_miner_scores.return_value = {"miner_x": 2.0}
        fake_batcher.get_archive_data.return_value = {"window_start": 90, "slots": []}

        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_block_hash", new=AsyncMock(return_value=randomness_hex)),
            patch("reliquary.validator.service.chain.compute_window_randomness", return_value=randomness_hex),
            patch("reliquary.validator.service.derive_window_prompts", return_value=[
                {"id": f"p{i}", "prompt": f"Q{i}", "ground_truth": "1"}
                for i in range(8)
            ]),
            patch("reliquary.validator.service.WindowBatcher", return_value=fake_batcher),
            patch(
                "reliquary.validator.service.storage.upload_window_dataset",
                new=AsyncMock(side_effect=RuntimeError("S3 unavailable")),
            ),
        ):
            # Should NOT raise — exception is caught internally
            await svc._run_window(subtensor, 90)

        # Batcher must be cleared regardless of archive failure
        assert svc.server.active_batcher is None


# ---------------------------------------------------------------------------
# 4. _submit_weights sends only non-zero miners
# ---------------------------------------------------------------------------


class TestSubmitWeightsNonZeroOnly:
    @pytest.mark.asyncio
    async def test_only_non_zero_miners_in_uids(self):
        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 5.0, "b": 0.0, "c": 3.0})

        meta = MagicMock()
        meta.hotkeys = ["a", "b", "c", "d"]
        meta.uids = [1, 2, 3, 4]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("reliquary.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        set_weights_mock.assert_called_once()
        call_args = set_weights_mock.call_args
        submitted_uids = call_args.args[3]  # positional: subtensor, wallet, netuid, uids, weights
        assert 1 in submitted_uids  # a → uid 1
        assert 3 in submitted_uids  # c → uid 3
        assert 2 not in submitted_uids  # b → uid 2, excluded (weight 0)
        assert 4 not in submitted_uids  # d not in scores


# ---------------------------------------------------------------------------
# 5. _submit_weights skips set_weights when all scores are zero
# ---------------------------------------------------------------------------


class TestSubmitWeightsSkipsWhenAllZero:
    @pytest.mark.asyncio
    async def test_set_weights_not_called_when_all_zero(self):
        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 0.0})
        # No burn either — nothing to emit at all.
        svc._burn_accumulated = 0.0

        meta = MagicMock()
        meta.hotkeys = ["a"]
        meta.uids = [1]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("reliquary.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        set_weights_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 6. _submit_weights routes burn to UID_BURN
# ---------------------------------------------------------------------------


class TestSubmitWeightsBurnRouting:
    @pytest.mark.asyncio
    async def test_burn_weight_sent_to_uid_zero(self):
        """Burn share must land in the uids list paired with UID_BURN."""
        from reliquary.constants import UID_BURN

        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 100.0})
        svc._burn_accumulated = 100.0  # 50% of budget would burn with exp=1

        meta = MagicMock()
        meta.hotkeys = ["a"]
        meta.uids = [1]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("reliquary.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        set_weights_mock.assert_called_once()
        call_args = set_weights_mock.call_args
        submitted_uids = call_args.args[3]
        submitted_weights = call_args.args[4]
        assert UID_BURN in submitted_uids
        burn_idx = submitted_uids.index(UID_BURN)
        # burn_score 100 / (miner_linear 100 + burn 100) = 0.5
        assert submitted_weights[burn_idx] == pytest.approx(0.5)
        # Total (miners + burn) sums to 1.0
        assert sum(submitted_weights) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_full_burn_when_all_miners_zero(self):
        """All miner scores 0 but burn > 0 → entire emission to UID_BURN."""
        from reliquary.constants import UID_BURN

        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 0.0, "b": 0.0})
        svc._burn_accumulated = 500.0

        meta = MagicMock()
        meta.hotkeys = ["a", "b"]
        meta.uids = [1, 2]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("reliquary.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        submitted_uids = set_weights_mock.call_args.args[3]
        submitted_weights = set_weights_mock.call_args.args[4]
        assert submitted_uids == [UID_BURN]
        assert submitted_weights == [pytest.approx(1.0)]

    @pytest.mark.asyncio
    async def test_no_burn_entry_when_burn_zero(self):
        """Burn == 0 → UID_BURN absent from submission (cleaner on-chain diff)."""
        from reliquary.constants import UID_BURN

        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 10.0})
        svc._burn_accumulated = 0.0

        meta = MagicMock()
        meta.hotkeys = ["a"]
        meta.uids = [1]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("reliquary.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        submitted_uids = set_weights_mock.call_args.args[3]
        assert UID_BURN not in submitted_uids


# ---------------------------------------------------------------------------
# 7. _run_window calls finalize_due_slots and accumulates burn
# ---------------------------------------------------------------------------


class TestRunWindowFinalizesAndAccumulatesBurn:
    @pytest.mark.asyncio
    async def test_finalize_due_slots_called_and_burn_accumulated(self):
        """Verify the window loop polls finalize + accumulates get_burn_score."""
        svc = _make_service()

        fake_batcher = MagicMock()
        fake_batcher.is_window_complete = MagicMock(
            side_effect=[False, True]  # one poll, then complete
        )
        fake_batcher.finalize_due_slots = MagicMock(return_value=0)
        fake_batcher.get_miner_scores = MagicMock(return_value={"m": 20.0})
        fake_batcher.get_burn_score = MagicMock(return_value=7.5)
        fake_batcher.get_archive_data = MagicMock(return_value={"slots": []})

        subtensor = MagicMock()

        with (
            patch("reliquary.validator.service.chain.get_block_hash",
                  new=AsyncMock(return_value="deadbeef")),
            patch("reliquary.validator.service.derive_window_prompts", return_value=[
                {"id": f"p{i}", "prompt": f"q{i}", "ground_truth": "1"}
                for i in range(8)
            ]),
            patch("reliquary.validator.service.WindowBatcher", return_value=fake_batcher),
            patch("reliquary.validator.service.storage.upload_window_dataset",
                  new=AsyncMock(return_value=True)),
        ):
            await svc._run_window(subtensor, 60)

        # finalize_due_slots is called at least once in the polling loop + once
        # as the safety net after the deadline breaks out.
        assert fake_batcher.finalize_due_slots.call_count >= 2
        # Burn accumulated into the service's running total.
        assert svc._burn_accumulated == pytest.approx(7.5)
        assert svc._miner_scores["m"] == pytest.approx(20.0)
