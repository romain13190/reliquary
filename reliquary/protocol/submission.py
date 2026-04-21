"""Pydantic v2 models for the miner→validator GRPO submission protocol.

The validator HTTP server (reliquary/validator/server.py) accepts these payloads
and the miner submitter (reliquary/miner/submitter.py) produces them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enum import Enum

from reliquary.constants import COMPLETIONS_PER_SUBMISSION, M_ROLLOUTS, PROMPTS_PER_WINDOW


class CompletionSubmission(BaseModel):
    """A single GRAIL-proven completion within a batch submission."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1)
    commit: dict[str, Any]


class SubmissionRequest(BaseModel):
    """A miner's batch of 1..COMPLETIONS_PER_SUBMISSION completions for one slot."""

    model_config = ConfigDict(extra="forbid")

    window_start: int = Field(..., ge=0)
    slot_index: int = Field(..., ge=0, le=PROMPTS_PER_WINDOW - 1)
    prompt_id: str = Field(..., min_length=1)
    miner_hotkey: str = Field(..., min_length=1)
    completions: list[CompletionSubmission]

    @field_validator("completions")
    @classmethod
    def _completion_count_within_plafond(
        cls, v: list[CompletionSubmission]
    ) -> list[CompletionSubmission]:
        if len(v) < 1 or len(v) > COMPLETIONS_PER_SUBMISSION:
            raise ValueError(
                f"completions must have between 1 and {COMPLETIONS_PER_SUBMISSION} "
                f"entries, got {len(v)}"
            )
        return v


class SubmissionResponse(BaseModel):
    """Validator's verdict on a submission."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str
    settled: bool = False
    slot_count: int = Field(..., ge=0)


class SlotState(BaseModel):
    """Per-slot snapshot exposed to miners polling window state."""

    model_config = ConfigDict(extra="forbid")

    slot_index: int = Field(..., ge=0, le=PROMPTS_PER_WINDOW - 1)
    prompt_id: str
    count: int = Field(..., ge=0)
    settled: bool
    # Histogram of accepted completion rewards in this slot, e.g.
    # {"1.0": 28, "0.0": 0}. Lets miners pick the rare class to maximise
    # their advantage-based score under the slot's final distribution.
    # Keys are stringified floats because JSON keys must be strings.
    rewards: dict[str, int] = Field(default_factory=dict)


class WindowStateResponse(BaseModel):
    """All slots for a window — used by miners to skip already-settled slots."""

    model_config = ConfigDict(extra="forbid")

    window_start: int = Field(..., ge=0)
    slot_states: list[SlotState]


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
