"""Pydantic schemas for v2 GRPO market submissions."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
    WindowState,
)


def _valid_rollouts(k: int = 4):
    """k successes, (M - k) failures, all with well-formed GRAIL fields."""
    rollouts = []
    for i in range(M_ROLLOUTS):
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5],
                reward=1.0 if i < k else 0.0,
                commit={"proof_version": "v5", "tokens": [1, 2, 3, 4, 5]},
            )
        )
    return rollouts


def test_valid_request_parses():
    req = BatchSubmissionRequest(
        miner_hotkey="hk" * 24,
        prompt_idx=42,
        window_start=1000,
        signed_round=999_999,
        merkle_root="00" * 32,
        rollouts=_valid_rollouts(k=4),
        checkpoint_hash="sha256:test",
    )
    assert req.prompt_idx == 42
    assert len(req.rollouts) == M_ROLLOUTS


def test_wrong_rollout_count_rejected():
    with pytest.raises(ValidationError, match="rollouts"):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=42,
            window_start=1000,
            signed_round=999_999,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(k=4)[:7],  # 7 instead of M
        )


def test_negative_prompt_idx_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=-1,
            window_start=1000,
            signed_round=999_999,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(),
        )


def test_malformed_merkle_root_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=0,
            window_start=1000,
            signed_round=999_999,
            merkle_root="zz",
            rollouts=_valid_rollouts(),
        )


def test_all_reject_reasons_serialisable():
    for reason in RejectReason:
        resp = BatchSubmissionResponse(accepted=False, reason=reason)
        assert resp.model_dump()["reason"] == reason.value


def test_accepted_response():
    resp = BatchSubmissionResponse(accepted=True, reason=RejectReason.ACCEPTED)
    dumped = resp.model_dump()
    assert dumped["accepted"] is True
    assert dumped["reason"] == RejectReason.ACCEPTED.value


def test_grpo_batch_state_exposes_cooldown():
    state = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=100,
        anchor_block=1000,
        current_round=999,
        cooldown_prompts=[42, 7, 99],
        valid_submissions=12,
        checkpoint_n=0,
    )
    dumped = state.model_dump()
    assert set(dumped["cooldown_prompts"]) == {42, 7, 99}


def test_new_reject_reasons_exist():
    """Schema/Token/Termination validators emit dedicated reject codes."""
    assert RejectReason.BAD_SCHEMA.value == "bad_schema"
    assert RejectReason.BAD_TOKENS.value == "bad_tokens"
    assert RejectReason.BAD_TERMINATION.value == "bad_termination"
