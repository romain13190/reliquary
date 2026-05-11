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


def test_open_window_does_not_expose_batcher_before_activation():
    """_open_window builds the batcher in a non-active state; only
    _activate_window registers it with the HTTP server.

    Regression for the prod cascade where finney WebSocket 503s during
    _set_window_randomness left the batcher exposed with randomness=""
    and every miner submission crashed in indices_from_root('').
    """
    from reliquary.validator.service import ValidationService, WindowState

    class FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
        hotkey = _Hk()

    svc = ValidationService(
        wallet=FakeWallet(),
        model=None,
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    svc.server = MagicMock()
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest.return_value = None

    svc._open_window()
    # Batcher exists internally but the server hasn't been told.
    assert svc._active_batcher is not None
    svc.server.set_active_batcher.assert_not_called()
    assert svc._current_window_state != WindowState.OPEN

    svc._activate_window()
    # Now the server is registered and the state has flipped to OPEN.
    svc.server.set_active_batcher.assert_called_once_with(svc._active_batcher)
    assert svc._current_window_state == WindowState.OPEN

