"""End-to-end: service creates GrpoWindowBatcher per window, seals at window
close, computes weights v2-flavoured."""

from dataclasses import dataclass
from unittest.mock import MagicMock

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


def test_service_compute_weights_for_sealed_batch():
    """After seal_batch, service computes flat 1/B weights."""
    from reliquary.protocol.submission import RolloutSubmission
    from reliquary.validator.service import compute_weights_for_window

    rollouts = [
        RolloutSubmission(tokens=[1], reward=1.0, commit={"tokens": [1]})
        for _ in range(8)
    ]
    batch = [
        ValidSubmission(
            hotkey=f"hk{i}", prompt_idx=i, signed_round=100,
            merkle_root_bytes=b"\x00" * 32, k=4, rollouts=rollouts,
        )
        for i in range(5)
    ]
    miner_weights, burn_weight = compute_weights_for_window(batch)
    assert len(miner_weights) == 5
    assert all(abs(w - 1.0 / B_BATCH) < 1e-9 for w in miner_weights.values())
    assert abs(burn_weight - 3.0 / B_BATCH) < 1e-9
