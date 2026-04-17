"""Tests for reliquary.miner.engine.MiningEngine (no real GPU required)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import MINER_BATCH_SIZE, PROMPTS_PER_WINDOW, GROUP_SIZE
from reliquary.protocol.submission import (
    CompletionSubmission,
    SlotState,
    SubmissionResponse,
    WindowStateResponse,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_wallet():
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5FakeHotkey" + "x" * 35
    return wallet


def _make_engine(validator_url_override=None):
    """Build a MiningEngine with all heavy deps mocked out."""
    from reliquary.miner.engine import MiningEngine
    from types import SimpleNamespace

    vllm_model = MagicMock()
    hf_model = MagicMock()
    hf_model.name_or_path = "fake-model"
    # Give the mock a real config so resolve_hidden_size succeeds
    hf_model.config = SimpleNamespace(hidden_size=64)
    tokenizer = MagicMock()
    # encode returns a flat list of 5 prompt tokens
    tokenizer.encode.return_value = [10, 20, 30, 40, 50]

    # vllm generate returns a tensor-like: outputs[0].tolist() = prompt + 4 new tokens
    # We make each call produce a slightly different result via side_effect
    import torch
    gen_tokens = [10, 20, 30, 40, 50, 1, 2, 3, 4]
    output_mock = MagicMock()
    output_mock.__getitem__ = lambda self, i: MagicMock(
        tolist=MagicMock(return_value=gen_tokens)
    )
    vllm_model.generate.return_value = output_mock
    vllm_model.device = "cpu"

    wallet = _make_wallet()

    class _FakeEnv:
        name = "fake"
        _size = 20

        def __len__(self):
            return self._size

        def get_problem(self, idx):
            return {
                "prompt": f"Q{idx % self._size}",
                "ground_truth": str(idx),
                "id": f"id-{idx % self._size:04d}",
            }

        def compute_reward(self, problem, completion):
            return 0.0

    engine = MiningEngine(
        vllm_model=vllm_model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        wallet=wallet,
        env=_FakeEnv(),
        validator_url_override=validator_url_override,
    )
    return engine


def _stub_completion_submission():
    return CompletionSubmission(
        tokens=[10, 20, 30, 40, 50, 1, 2, 3, 4],
        commit={
            "tokens": [10, 20, 30, 40, 50, 1, 2, 3, 4],
            "commitments": [],
            "proof_version": "v5",
            "model": {"name": "fake-model", "layer_index": -1},
            "signature": "aa" * 64,
            "beacon": {"randomness": "ab" * 32},
            "rollout": {
                "prompt_length": 5,
                "completion_length": 4,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": [-1.0, -1.0, -1.0, -1.0],
            },
        },
    )


def _accepted_response(slot_index=0):
    return SubmissionResponse(
        accepted=True,
        reason="ok",
        settled=False,
        slot_count=4,
    )


def _window_state_with_settled(settled_slots: set[int]) -> WindowStateResponse:
    return WindowStateResponse(
        window_start=1000,
        slot_states=[
            SlotState(
                slot_index=i,
                prompt_id=f"id-{i:04d}",
                count=0,
                settled=(i in settled_slots),
            )
            for i in range(PROMPTS_PER_WINDOW)
        ],
    )


# ---------------------------------------------------------------------------
# Patch helpers: block out all real randomness/chain/GPU paths
# ---------------------------------------------------------------------------

PATCH_CHAIN_HASH = "reliquary.miner.engine.chain.get_block_hash"
PATCH_CHAIN_RANDOMNESS = "reliquary.miner.engine.chain.compute_window_randomness"
PATCH_GET_WINDOW_STATE = "reliquary.miner.engine.get_window_state"
PATCH_SUBMIT_BATCH = "reliquary.miner.engine.submit_batch"
PATCH_TIME = "reliquary.miner.engine.time.monotonic"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_window_skips_settled_slots():
    """Slot 0 is settled — submit_batch must NOT be called for it but IS called for slot 1."""
    engine = _make_engine(validator_url_override="http://localhost:8888")

    # Stub _build_completion_submission so we don't need real GPU
    stub_cs = _stub_completion_submission()
    engine._build_completion_submission = MagicMock(return_value=stub_cs)

    # 4 completions with distinct prefixes
    diverse = [
        {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
         "prompt_length": 5,
         "completion_tokens": [i, 0, 0, 0]}
        for i in range(1, MINER_BATCH_SIZE + 1)
    ]
    engine._generate_targeted_batch = MagicMock(return_value=diverse)

    # Time never expires
    monotonic_vals = [0.0] * 100
    monotonic_iter = iter(monotonic_vals)

    with (
        patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc123")),
        patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
        patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(
            return_value=_window_state_with_settled({0})
        )),
        patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())) as mock_submit,
        patch(PATCH_TIME, side_effect=lambda: next(monotonic_iter)),
    ):
        results = await engine.mine_window(
            subtensor=MagicMock(),
            window_start=1000,
            use_drand=False,
        )

    # submit_batch called once for slot 1 only (slot 0 is settled)
    assert mock_submit.call_count == PROMPTS_PER_WINDOW - 1
    # All submitted requests must be for non-settled slots (1 through 7)
    submitted_slot_indices = [
        call.args[1].slot_index for call in mock_submit.call_args_list
    ]
    assert 0 not in submitted_slot_indices


@pytest.mark.asyncio
async def test_mine_window_stops_at_deadline():
    """Deadline reached after first slot — only 1 submission occurs."""
    engine = _make_engine(validator_url_override="http://localhost:8888")

    stub_cs = _stub_completion_submission()
    engine._build_completion_submission = MagicMock(return_value=stub_cs)
    diverse = [
        {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
         "prompt_length": 5,
         "completion_tokens": [i, 0, 0, 0]}
        for i in range(1, MINER_BATCH_SIZE + 1)
    ]
    engine._generate_targeted_batch = MagicMock(return_value=diverse)

    # First call: inside deadline (deadline computed from first call value)
    # mine_window logic:
    #   deadline = first_monotonic() + WINDOW_LENGTH * BLOCK_TIME_SECONDS - UPLOAD_BUFFER
    # = 0 + 30*12 - 30 = 330
    # For slot 0: checked at call #2 → 0 (still inside)
    # After slot 0 submit: checked at call #3 → 400 (past deadline)
    times = iter([0.0, 0.0, 400.0] + [400.0] * 50)

    with (
        patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc123")),
        patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
        patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(
            return_value=_window_state_with_settled(set())
        )),
        patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())) as mock_submit,
        patch(PATCH_TIME, side_effect=lambda: next(times)),
    ):
        await engine.mine_window(
            subtensor=MagicMock(),
            window_start=1000,
            use_drand=False,
        )

    assert mock_submit.call_count == 1


@pytest.mark.asyncio
async def test_mine_window_skips_slot_when_diverse_batch_fails():
    """When _generate_diverse_batch returns fewer than 4 completions, slot is skipped."""
    engine = _make_engine(validator_url_override="http://localhost:8888")

    stub_cs = _stub_completion_submission()
    engine._build_completion_submission = MagicMock(return_value=stub_cs)

    # Only 2 completions — not enough, slot must be skipped
    only_two = [
        {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
         "prompt_length": 5,
         "completion_tokens": [i, 0, 0, 0]}
        for i in range(1, 3)
    ]
    engine._generate_targeted_batch = MagicMock(return_value=only_two)

    monotonic_vals = iter([0.0] * 100)

    with (
        patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc123")),
        patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
        patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(
            return_value=_window_state_with_settled(set())
        )),
        patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())) as mock_submit,
        patch(PATCH_TIME, side_effect=lambda: next(monotonic_vals)),
    ):
        await engine.mine_window(
            subtensor=MagicMock(),
            window_start=1000,
            use_drand=False,
        )

    # No submissions because _generate_diverse_batch never returns 4
    assert mock_submit.call_count == 0


# ---------------------------------------------------------------------------
# Smart strategy tests: _choose_target_reward + rare-class targeting
# ---------------------------------------------------------------------------


from reliquary.miner.engine import MiningEngine


class TestChooseTargetReward:
    def test_empty_histogram_no_preference(self) -> None:
        assert MiningEngine._choose_target_reward({}) is None

    def test_balanced_no_preference(self) -> None:
        assert MiningEngine._choose_target_reward({"1.0": 10, "0.0": 10}) is None

    def test_imbalanced_picks_rare_class(self) -> None:
        # count_1=20, count_0=5 → 0.0 is rarer
        assert MiningEngine._choose_target_reward({"1.0": 20, "0.0": 5}) == 0.0
        # count_1=5, count_0=12 → 1.0 is rarer
        assert MiningEngine._choose_target_reward({"1.0": 5, "0.0": 12}) == 1.0

    def test_slot_at_group_size_returns_sentinel(self) -> None:
        # Total counts == GROUP_SIZE → sentinel so caller skips the slot.
        assert (
            MiningEngine._choose_target_reward(
                {"1.0": GROUP_SIZE // 2, "0.0": GROUP_SIZE // 2}
            )
            == "slot_full"
        )

    def test_rare_class_still_chosen_when_plenty_of_room(self) -> None:
        # Even if one class has most completions, as long as total < GROUP_SIZE
        # the miner should target the rare class.
        assert MiningEngine._choose_target_reward({"1.0": 20, "0.0": 8}) == 0.0


class TestGenerateTargetedBatch:
    def _build_engine_with_env_rewards(self, reward_seq):
        """Build an engine whose FakeEnv returns rewards from ``reward_seq`` in order."""
        engine = _make_engine(validator_url_override="http://localhost")
        rewards = iter(reward_seq)
        engine.env.compute_reward = MagicMock(side_effect=lambda p, c: next(rewards))

        # Rotate completion tokens so prefix dedup accepts them all.
        counter = {"n": 0}

        def fake_generate(input_tensor, **kwargs):
            counter["n"] += 1
            tokens = [10, 20, 30, 40, 50, counter["n"], 99, 99, 99, 99, 99, 99, 99]
            output = MagicMock()
            output.__getitem__ = lambda self, i: MagicMock(
                tolist=MagicMock(return_value=tokens)
            )
            return output

        engine.vllm_model.generate = MagicMock(side_effect=fake_generate)
        return engine

    def test_no_target_accepts_all(self) -> None:
        # env irrelevant when target is None
        engine = self._build_engine_with_env_rewards([0.0] * 10)
        problem = {"prompt": "Q", "ground_truth": "42", "id": "p"}
        batch = engine._generate_targeted_batch(problem, "ab" * 32, target_reward=None)
        assert len(batch) == MINER_BATCH_SIZE
        # Should NOT have called compute_reward when target is None
        engine.env.compute_reward.assert_not_called()

    def test_target_filters_out_wrong_reward(self) -> None:
        # Alternate 0.0 / 1.0 — target 1.0 keeps every other attempt.
        engine = self._build_engine_with_env_rewards([0.0, 1.0] * 20)
        problem = {"prompt": "Q", "ground_truth": "42", "id": "p"}
        batch = engine._generate_targeted_batch(problem, "ab" * 32, target_reward=1.0)
        assert len(batch) == MINER_BATCH_SIZE
        # Needed at least 8 attempts (4 rejected + 4 kept).
        assert engine.vllm_model.generate.call_count >= 8

    def test_gives_up_when_model_cant_produce_target(self) -> None:
        # Env always returns 0.0, target is 1.0 → no completion ever matches.
        engine = self._build_engine_with_env_rewards([0.0] * 100)
        problem = {"prompt": "Q", "ground_truth": "42", "id": "p"}
        batch = engine._generate_targeted_batch(problem, "ab" * 32, target_reward=1.0)
        assert batch == []
        # Cap is MINER_BATCH_SIZE * 10 = 40
        assert engine.vllm_model.generate.call_count == MINER_BATCH_SIZE * 10


class TestMinerUsesRewardsHistogram:
    @pytest.mark.asyncio
    async def test_target_reward_passed_to_generator_based_on_state(self) -> None:
        """When /state shows imbalance, miner calls _generate_targeted_batch with
        the rare class as target_reward."""
        engine = _make_engine(validator_url_override="http://localhost:8888")
        engine._build_completion_submission = MagicMock(
            return_value=_stub_completion_submission()
        )
        diverse = [
            {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
             "prompt_length": 5,
             "completion_tokens": [i, 0, 0, 0]}
            for i in range(1, MINER_BATCH_SIZE + 1)
        ]
        engine._generate_targeted_batch = MagicMock(return_value=diverse)

        # Slot 0 imbalanced: 20 corrects vs 5 wrongs (total 25 < GROUP_SIZE)
        # → miner should target 0.0 as rare class.
        state = WindowStateResponse(
            window_start=1000,
            slot_states=[
                SlotState(
                    slot_index=i,
                    prompt_id=f"id-{i:04d}",
                    count=25 if i == 0 else 0,
                    settled=False,
                    rewards={"1.0": 20, "0.0": 5} if i == 0 else {},
                )
                for i in range(PROMPTS_PER_WINDOW)
            ],
        )

        monotonic_vals = iter([0.0] * 100)
        with (
            patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc")),
            patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
            patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(return_value=state)),
            patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())),
            patch(PATCH_TIME, side_effect=lambda: next(monotonic_vals)),
        ):
            await engine.mine_window(
                subtensor=MagicMock(), window_start=1000, use_drand=False,
            )

        # First call is for slot 0 with target_reward=0.0 (rare class)
        first_call = engine._generate_targeted_batch.call_args_list[0]
        assert first_call.args[2] == 0.0  # target_reward kwarg

    @pytest.mark.asyncio
    async def test_full_slot_skipped(self) -> None:
        """Slot already at GROUP_SIZE → skipped, no generate, no submit."""
        engine = _make_engine(validator_url_override="http://localhost:8888")
        engine._generate_targeted_batch = MagicMock()  # must NOT be called for slot 0
        engine._build_completion_submission = MagicMock(
            return_value=_stub_completion_submission()
        )

        state = WindowStateResponse(
            window_start=1000,
            slot_states=[
                SlotState(
                    slot_index=i,
                    prompt_id=f"id-{i:04d}",
                    count=GROUP_SIZE if i == 0 else 0,
                    settled=False,
                    rewards=(
                        {"1.0": GROUP_SIZE // 2, "0.0": GROUP_SIZE // 2}
                        if i == 0 else {}
                    ),
                )
                for i in range(PROMPTS_PER_WINDOW)
            ],
        )

        # Any other slots still accept — let the engine iterate normally.
        engine._generate_targeted_batch.side_effect = lambda *args, **kw: [
            {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
             "prompt_length": 5,
             "completion_tokens": [i, 0, 0, 0]}
            for i in range(1, MINER_BATCH_SIZE + 1)
        ]

        monotonic_vals = iter([0.0] * 100)
        with (
            patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc")),
            patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
            patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(return_value=state)),
            patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())) as mock_submit,
            patch(PATCH_TIME, side_effect=lambda: next(monotonic_vals)),
        ):
            await engine.mine_window(
                subtensor=MagicMock(), window_start=1000, use_drand=False,
            )

        # 7 slots submitted (slot 0 skipped).
        assert mock_submit.call_count == PROMPTS_PER_WINDOW - 1
        submitted_slots = [c.args[1].slot_index for c in mock_submit.call_args_list]
        assert 0 not in submitted_slots


@pytest.mark.asyncio
async def test_validator_url_override_skips_metagraph_lookup():
    """When validator_url_override is set, subtensor is never touched."""
    engine = _make_engine(validator_url_override="http://override:9999")

    stub_cs = _stub_completion_submission()
    engine._build_completion_submission = MagicMock(return_value=stub_cs)
    diverse = [
        {"tokens": [10, 20, 30, 40, 50, i, 0, 0, 0],
         "prompt_length": 5,
         "completion_tokens": [i, 0, 0, 0]}
        for i in range(1, MINER_BATCH_SIZE + 1)
    ]
    engine._generate_targeted_batch = MagicMock(return_value=diverse)

    # Subtensor that explodes on any attribute access
    class _ExplodingSubtensor:
        def __getattr__(self, name):
            raise AssertionError(f"subtensor.{name} was unexpectedly accessed")

    monotonic_vals = iter([0.0] * 100)

    with (
        patch(PATCH_CHAIN_HASH, new=AsyncMock(return_value="abc123")),
        patch(PATCH_CHAIN_RANDOMNESS, return_value="aa" * 32),
        patch(PATCH_GET_WINDOW_STATE, new=AsyncMock(
            return_value=_window_state_with_settled(set())
        )),
        patch(PATCH_SUBMIT_BATCH, new=AsyncMock(return_value=_accepted_response())),
        patch(PATCH_TIME, side_effect=lambda: next(monotonic_vals)),
    ):
        # Must complete without touching _ExplodingSubtensor
        await engine.mine_window(
            subtensor=_ExplodingSubtensor(),
            window_start=1000,
            use_drand=False,
        )
    # If we reach here, the subtensor was never accessed — test passes
