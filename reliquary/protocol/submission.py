"""Pydantic v2 models for the miner→validator GRPO submission protocol.

The validator HTTP server (reliquary/validator/server.py) accepts these payloads
and the miner submitter (reliquary/miner/submitter.py) produces them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enum import Enum

from reliquary.constants import M_ROLLOUTS


# ---------------------------------------------------------------------------
# v2 GRPO Market schemas
# ---------------------------------------------------------------------------


class RejectReason(str, Enum):
    """Canonical reject codes emitted by the v2 validator.

    ``ACCEPTED`` is a sentinel used in success responses (``accepted=True``).
    All other values are mutually exclusive; only the first failure reason
    is reported per submission.
    """

    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    BAD_PROMPT_IDX = "bad_prompt_idx"
    STALE_ROUND = "stale_round"
    PROMPT_IN_COOLDOWN = "prompt_in_cooldown"
    GRAIL_FAIL = "grail_fail"
    REWARD_MISMATCH = "reward_mismatch"
    OUT_OF_ZONE = "out_of_zone"
    WRONG_ROLLOUT_COUNT = "wrong_rollout_count"
    WINDOW_MISMATCH = "window_mismatch"
    WINDOW_NOT_ACTIVE = "window_not_active"
    WRONG_CHECKPOINT = "wrong_checkpoint"


class WindowState(str, Enum):
    """Current phase of a batch-driven window (v2.1)."""

    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint published; transient — back to OPEN once next window opens


class RolloutSubmission(BaseModel):
    """A single rollout's payload: tokens, miner-claimed reward, GRAIL commit."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1)
    reward: float  # miner's local env.compute_reward value; validator re-checks
    commit: dict[str, Any]


class BatchSubmissionRequest(BaseModel):
    """v2 miner→validator payload: one group of M rollouts on one prompt."""

    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(..., min_length=1)
    prompt_idx: int = Field(..., ge=0)
    window_start: int = Field(..., ge=0)
    signed_round: int = Field(..., ge=0)
    merkle_root: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    rollouts: list[RolloutSubmission]

    @field_validator("rollouts")
    @classmethod
    def _rollout_count_is_M(cls, v):
        if len(v) != M_ROLLOUTS:
            raise ValueError(
                f"rollouts must have exactly {M_ROLLOUTS} entries, got {len(v)}"
            )
        return v


class BatchSubmissionResponse(BaseModel):
    """Validator verdict on a submission."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: RejectReason


class GrpoBatchState(BaseModel):
    """Live window state for miners polling ``/window/{n}/state`` (v2)."""

    model_config = ConfigDict(extra="forbid")

    window_start: int = Field(..., ge=0)
    current_round: int = Field(..., ge=0)
    cooldown_prompts: list[int] = Field(default_factory=list)
    valid_submissions: int = Field(..., ge=0)
