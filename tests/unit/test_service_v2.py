"""End-to-end: service creates GrpoWindowBatcher per window, seals at window
close, computes weights v2-flavoured."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import B_BATCH
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.cooldown import CooldownMap


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


def test_service_creates_grpo_window_batcher():
    """The service's open_grpo_window() returns a GrpoWindowBatcher wired up
    with the shared CooldownMap."""
    from reliquary.validator.service import open_grpo_window

    shared_cooldown = CooldownMap(cooldown_windows=50)
    batcher = open_grpo_window(
        window_start=100,
        current_round=999,
        env=_FakeEnv(),
        model=None,
        cooldown_map=shared_cooldown,
        tokenizer=MagicMock(),
    )
    assert isinstance(batcher, GrpoWindowBatcher)
    assert batcher.window_start == 100
    assert batcher._cooldown is shared_cooldown


@pytest.mark.asyncio
async def test_rebuild_cooldown_from_history_populates_map():
    """ValidationService._rebuild_cooldown_from_history fetches archives
    from R2 and populates the cooldown map."""
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    from reliquary.validator.service import ValidationService

    # Build a fake service — minimal state to call the rebuild method.
    class FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
        hotkey = _Hk()

    svc = ValidationService(
        wallet=FakeWallet(),
        model=None,
        tokenizer=None,
        env=_FakeEnv(),
        netuid=99,
    )

    archives = [
        {"window_start": 100, "batch": [{"prompt_idx": 42}]},
        {"window_start": 101, "batch": [{"prompt_idx": 7}]},
    ]
    svc._window_n = 105  # authoritative counter

    with patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=archives),
    ):
        await svc._rebuild_cooldown_from_history()

    # Should now know about prompts 42 and 7.
    assert len(svc._cooldown_map) == 2


def test_service_update_ema_for_sealed_batch():
    """After seal_batch, service updates per-hotkey EMA from the batch."""
    from collections import defaultdict
    from reliquary.protocol.submission import RolloutSubmission
    from reliquary.validator.service import ValidationService
    from reliquary.constants import EMA_ALPHA

    class FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    rollouts = [
        RolloutSubmission(tokens=[1], reward=1.0, commit={"tokens": [1]})
        for _ in range(8)
    ]
    batch = [
        ValidSubmission(
            hotkey=f"hk{i}", prompt_idx=i, signed_round=100,
            merkle_root_bytes=b"\x00" * 32, sigma=0.5, rollouts=rollouts,
        )
        for i in range(5)
    ]

    from unittest.mock import MagicMock
    svc = ValidationService(
        wallet=FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=MagicMock(), netuid=99,
    )
    svc._miner_scores_ema = defaultdict(float)

    svc._update_ema(batch)

    # 5 miners each contributed 1/B_BATCH of the batch.
    for i in range(5):
        expected = EMA_ALPHA * (1.0 / B_BATCH)
        assert abs(svc._miner_scores_ema[f"hk{i}"] - expected) < 1e-9
