"""GrpoWindowBatcher — orchestrator for the free-prompt GRPO market.

Holds a flat list of validated submissions per window + a reference to the
validator's shared ``CooldownMap``.
"""

from __future__ import annotations

import asyncio
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
    WindowState,
)
from reliquary.validator.batch_selection import select_batch
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.verifier import (
    evaluate_token_distribution,
    is_in_zone,
    rewards_std,
    verify_logprobs_claim,
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
    sigma: float = 0.0
    rollouts: list[RolloutSubmission] = field(default_factory=list)
    completion_texts: list[str] = field(default_factory=list)
    arrived_at: float = 0.0
    # Filter telemetry (worst-case across this submission's rollouts).
    # Captured for post-hoc threshold calibration without re-running tests.
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
    # Miner-claimed checkpoint hash at submit time — useful for post-hoc
    # forensic analysis of who lied about their checkpoint.
    claimed_checkpoint_hash: str = ""

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
        verify_commitment_proofs_fn: Callable[..., Any] | None = None,
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
        # Reference for per-submission response_time. Set at construction so
        # ``arrived_at - window_opened_at`` is the seconds the miner took
        # from window-open to accepted submission.
        self.window_opened_at: float = self._time_fn()

        self._cooldown = (
            cooldown_map if cooldown_map is not None
            else CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
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
        # Accumulated reject reasons this window (RejectReason.value → count).
        # Persisted in the R2 archive so miners can see which filter is
        # rejecting the most submissions in any given round.
        self.reject_counts: dict[str, int] = {}

        # v2.1: seal_event fires the moment the B-th distinct non-cooldown
        # valid submission lands. Service awaits this to close the window.
        # Stored as threading.Event for sync-safe set(); the asyncio.Event
        # is created lazily on first access to bind to the current loop.
        self._seal_flag: threading.Event = threading.Event()
        self._seal_event: asyncio.Event | None = None
        # v2.1: checkpoint hash miners must match. Empty string disables
        # the gate (test convenience / pre-first-publish).
        self.current_checkpoint_hash: str = ""

    @property
    def seal_event(self) -> asyncio.Event:
        """Lazy asyncio.Event bound to whichever loop accesses it first."""
        if self._seal_event is None:
            self._seal_event = asyncio.Event()
            if self._seal_flag.is_set():
                self._seal_event.set()
        return self._seal_event

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
        # v2.1: checkpoint hash gate. Empty string = gate disabled
        # (pre-first-publish or test convenience).
        if self.current_checkpoint_hash and request.checkpoint_hash != self.current_checkpoint_hash:
            return self._reject(RejectReason.WRONG_CHECKPOINT)
        if request.prompt_idx >= len(self.env):
            return self._reject(RejectReason.BAD_PROMPT_IDX)
        if not self._round_fresh(request.signed_round):
            return self._reject(RejectReason.STALE_ROUND)
        if self._cooldown.is_in_cooldown(request.prompt_idx, self.window_start):
            return self._reject(RejectReason.PROMPT_IN_COOLDOWN)

        problem = self.env.get_problem(request.prompt_idx)
        completion_texts = []
        for rollout in request.rollouts:
            text = self._completion_text(rollout)
            completion_texts.append(text)
            if not verify_reward_claim(self.env, problem, text, rollout.reward):
                return self._reject(RejectReason.REWARD_MISMATCH)

        sigma = rewards_std([r.reward for r in request.rollouts])
        if not is_in_zone(sigma, bootstrap=self.bootstrap):
            return self._reject(RejectReason.OUT_OF_ZONE)

        # Per-submission worst-case filter telemetry (across all rollouts).
        sketch_diff_max = 0
        lp_dev_max: float | None = None
        dist_q10_min: float | None = None

        for rollout in request.rollouts:
            if not self._verify_proof_version(rollout.commit):
                return self._reject(RejectReason.GRAIL_FAIL)
            if not self._verify_signature(rollout.commit, request.miner_hotkey):
                return self._reject(RejectReason.BAD_SIGNATURE)
            proof = self._verify_commitment(
                rollout.commit, self.model, self.randomness
            )
            if proof.sketch_diff_max > sketch_diff_max:
                sketch_diff_max = proof.sketch_diff_max
            if not proof.all_passed:
                return self._reject(RejectReason.GRAIL_FAIL)

            # Behavioural checks (use cached logits from the GRAIL forward pass).
            # Skip gracefully if the logits tensor is empty (legacy stubs in tests).
            if proof.logits.numel() == 0:
                continue

            rollout_dict = rollout.commit.get("rollout", {}) or {}
            prompt_len = int(rollout_dict.get("prompt_length", 0))
            completion_len = int(rollout_dict.get("completion_length", 0))
            claimed_lp = rollout_dict.get("token_logprobs", []) or []

            lp_ok, lp_dev = verify_logprobs_claim(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                claimed_logprobs=claimed_lp,
                logits=proof.logits,
                challenge_randomness=self.randomness,
            )
            if lp_dev is not None and lp_dev != float("inf"):
                if lp_dev_max is None or lp_dev > lp_dev_max:
                    lp_dev_max = float(lp_dev)
            if not lp_ok:
                logger.info(
                    "reject reason=logprob_mismatch hotkey=%s median_dev=%.4f",
                    request.miner_hotkey, lp_dev,
                )
                return self._reject(RejectReason.LOGPROB_MISMATCH)

            from reliquary.constants import T_PROTO
            dist_ok, dist_metrics = evaluate_token_distribution(
                tokens=rollout.commit["tokens"],
                prompt_length=prompt_len,
                completion_length=completion_len,
                logits=proof.logits,
                temperature=T_PROTO,
            )
            if dist_metrics and "q10" in dist_metrics:
                q10 = float(dist_metrics["q10"])
                if dist_q10_min is None or q10 < dist_q10_min:
                    dist_q10_min = q10
            if dist_ok is False:
                logger.info(
                    "reject reason=distribution_suspicious hotkey=%s %s",
                    request.miner_hotkey, dist_metrics,
                )
                return self._reject(RejectReason.DISTRIBUTION_SUSPICIOUS)

        self._valid.append(
            ValidSubmission(
                hotkey=request.miner_hotkey,
                prompt_idx=request.prompt_idx,
                signed_round=request.signed_round,
                merkle_root_bytes=bytes.fromhex(request.merkle_root),
                sigma=sigma,
                rollouts=list(request.rollouts),
                completion_texts=completion_texts,
                arrived_at=self._time_fn(),
                sketch_diff_max=sketch_diff_max,
                lp_dev_max=lp_dev_max,
                dist_q10_min=dist_q10_min,
                claimed_checkpoint_hash=request.checkpoint_hash,
            )
        )

        # v2.1: fire seal_event when B distinct non-cooldown prompts have been accepted.
        distinct_eligible = len({
            s.prompt_idx for s in self._valid
            if not self._cooldown.is_in_cooldown(s.prompt_idx, self.window_start)
        })
        if distinct_eligible >= B_BATCH and not self._seal_flag.is_set():
            self._seal_flag.set()
            if self._seal_event is not None:
                self._seal_event.set()

        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED
        )

    def _round_fresh(self, signed_round: int) -> bool:
        if signed_round > self.current_round:
            return False
        return (self.current_round - signed_round) <= STALE_ROUND_LAG_MAX

    def _reject(self, reason: RejectReason) -> BatchSubmissionResponse:
        self.reject_counts[reason.value] = self.reject_counts.get(reason.value, 0) + 1
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
                state=WindowState.OPEN,
                window_n=self.window_start,
                anchor_block=self.window_start,
                current_round=self.current_round,
                cooldown_prompts=sorted(
                    self._cooldown.current_cooldown_set(self.window_start)
                ),
                valid_submissions=len(self._valid),
                checkpoint_n=0,
            )
