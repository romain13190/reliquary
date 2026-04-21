"""GrpoWindowBatcher — v2 orchestrator for the free-prompt GRPO market.

Replaces the slot-based ``WindowBatcher`` once Task 11 wires it in. Holds
a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from reliquary.constants import (
    BATCH_PROMPT_COOLDOWN_WINDOWS,
    B_BATCH,
    M_ROLLOUTS,
)
from reliquary.environment.base import Environment
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batch_selection import select_batch
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.verifier import (
    is_in_zone,
    rewards_to_k,
    verify_reward_claim,
)

logger = logging.getLogger(__name__)


# Maximum drand-round lag tolerated: a miner's ``signed_round`` may be up to
# this many rounds behind ``current_round`` to be accepted. Newer than
# current_round is always rejected (replay of future beacon).
STALE_ROUND_LAG_MAX = 10


@dataclass
class ValidSubmission:
    """A submission that passed all v2 verification checks."""

    hotkey: str
    prompt_idx: int
    signed_round: int
    merkle_root_bytes: bytes
    merkle_root: bytes = field(init=False)  # alias for select_batch Protocol
    k: int = 0
    rollouts: list[RolloutSubmission] = field(default_factory=list)
    arrived_at: float = 0.0

    def __post_init__(self):
        self.merkle_root = self.merkle_root_bytes


class GrpoWindowBatcher:
    """Accepts v2 submissions, runs the full verification pipeline, and
    exposes ``valid_submissions()`` + ``select_batch()`` at window close.
    """

    def __init__(
        self,
        window_start: int,
        current_round: int,
        env: Environment,
        model: Any,
        *,
        cooldown_map: CooldownMap | None = None,
        bootstrap: bool = False,
        completion_text_fn: Callable[[RolloutSubmission], str],
        verify_commitment_proofs_fn: Callable[..., tuple[bool, int, int]] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        verify_proof_version_fn: Callable[[dict], bool] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.window_start = window_start
        self.current_round = current_round
        self.env = env
        self.model = model
        self.bootstrap = bootstrap
        self._completion_text = completion_text_fn
        self._time_fn = time_fn or time.monotonic

        self._cooldown = cooldown_map or CooldownMap(
            cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS
        )

        if verify_commitment_proofs_fn is None:
            from reliquary.validator.verifier import verify_commitment_proofs
            verify_commitment_proofs_fn = verify_commitment_proofs
        if verify_signature_fn is None:
            from reliquary.validator.verifier import verify_signature
            verify_signature_fn = verify_signature
        if verify_proof_version_fn is None:
            from reliquary.validator.verifier import verify_proof_version
            verify_proof_version_fn = verify_proof_version

        self._verify_commitment = verify_commitment_proofs_fn
        self._verify_signature = verify_signature_fn
        self._verify_proof_version = verify_proof_version_fn

        self._lock = threading.Lock()
        self._valid: list[ValidSubmission] = []
        self.randomness: str = ""

    # ----------------------------- ingestion -----------------------------

    def accept_submission(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        """Run the full verification pipeline; append to ``_valid`` on success."""
        with self._lock:
            return self._accept_locked(request)

    def _accept_locked(
        self, request: BatchSubmissionRequest
    ) -> BatchSubmissionResponse:
        if request.window_start != self.window_start:
            return self._reject(RejectReason.WINDOW_MISMATCH)
        if request.prompt_idx >= len(self.env):
            return self._reject(RejectReason.BAD_PROMPT_IDX)
        if not self._round_fresh(request.signed_round):
            return self._reject(RejectReason.STALE_ROUND)
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return self._reject(RejectReason.PROMPT_IN_COOLDOWN)

        problem = self.env.get_problem(request.prompt_idx)
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            if not verify_reward_claim(self.env, problem, text, rollout.reward):
                return self._reject(RejectReason.REWARD_MISMATCH)

        k = rewards_to_k([r.reward for r in request.rollouts])
        if not is_in_zone(k, bootstrap=self.bootstrap):
            return self._reject(RejectReason.OUT_OF_ZONE)

        for rollout in request.rollouts:
            if not self._verify_proof_version(rollout.commit):
                return self._reject(RejectReason.GRAIL_FAIL)
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return self._reject(RejectReason.BAD_SIGNATURE)
            passed, _, _ = self._verify_commitment(
                rollout.commit, self.model, self.randomness
            )
            if not passed:
                return self._reject(RejectReason.GRAIL_FAIL)

        self._valid.append(
            ValidSubmission(
                hotkey=request.miner_hotkey,
                prompt_idx=request.prompt_idx,
                signed_round=request.signed_round,
                merkle_root_bytes=bytes.fromhex(request.merkle_root),
                k=k,
                rollouts=list(request.rollouts),
                arrived_at=self._time_fn(),
            )
        )
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    def _round_fresh(self, signed_round: int) -> bool:
        if signed_round > self.current_round:
            return False
        return (self.current_round - signed_round) <= STALE_ROUND_LAG_MAX

    @staticmethod
    def _reject(reason: RejectReason) -> BatchSubmissionResponse:
        return BatchSubmissionResponse(accepted=False, reason=reason)

    # ----------------------------- accessors -----------------------------

    def valid_submissions(self) -> list[ValidSubmission]:
        with self._lock:
            return list(self._valid)

    def seal_batch(self) -> list[ValidSubmission]:
        with self._lock:
            batch = select_batch(
                self._valid,
                b=B_BATCH,
                current_window=self.window_start,
                cooldown_map=self._cooldown,
            )
            for sub in batch:
                self._cooldown.record_batched(sub.prompt_idx, self.window_start)
            return batch

    def get_state(self) -> GrpoBatchState:
        with self._lock:
            return GrpoBatchState(
                window_start=self.window_start,
                current_round=self.current_round,
                cooldown_prompts=sorted(
                    self._cooldown.current_cooldown_set(self.window_start)
                ),
                valid_submissions=len(self._valid),
            )
