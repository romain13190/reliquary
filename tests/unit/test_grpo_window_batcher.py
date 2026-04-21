"""GrpoWindowBatcher: accepts submissions, enforces verification pipeline,
exposes select_batch at window close."""

from dataclasses import dataclass
from typing import Any

import pytest

from reliquary.constants import B_BATCH, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.batcher_v2 import GrpoWindowBatcher


class FakeEnv:
    name = "fake"
    def __len__(self):
        return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "a", "id": f"pid-{idx}"}
    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


def _always_true_grail(commit, model, randomness):
    return True, 1, 1


def _always_false_grail(commit, model, randomness):
    return False, 0, 1


def _always_true_sig(commit, hotkey):
    return True


def _always_true_proof_version(commit):
    return True


def _request(
    prompt_idx=42, signed_round=1000, window_start=500,
    rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for i, r in enumerate(rewards):
        text = "CORRECT" if r > 0.5 else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5],
                reward=r,
                commit={
                    "proof_version": "v5",
                    "tokens": [1, 2, 3, 4, 5],
                    "completion_text_for_test": text,
                },
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        signed_round=signed_round,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )


def _make_batcher(**overrides) -> GrpoWindowBatcher:
    kwargs = dict(
        window_start=500,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        verify_proof_version_fn=_always_true_proof_version,
        completion_text_fn=lambda rollout: rollout.commit.get(
            "completion_text_for_test", ""
        ),
    )
    kwargs.update(overrides)
    return GrpoWindowBatcher(**kwargs)


def test_constructor_sets_window_and_round():
    b = _make_batcher()
    assert b.window_start == 500
    assert b.current_round == 1000


def test_reject_window_mismatch():
    b = _make_batcher()
    req = _request(window_start=999)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WINDOW_MISMATCH


def test_reject_stale_round():
    b = _make_batcher(current_round=1000)
    req = _request(signed_round=500)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.STALE_ROUND


def test_accept_in_zone_submission():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED
    assert len(b.valid_submissions()) == 1


def test_reject_out_of_zone_all_fail():
    b = _make_batcher()
    req = _request(rewards=[0.0] * 8)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE
    assert len(b.valid_submissions()) == 0


def test_reject_out_of_zone_all_pass():
    b = _make_batcher()
    req = _request(rewards=[1.0] * 8)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE


def test_reject_grail_fail():
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    req = _request()
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.GRAIL_FAIL


def test_reject_reward_mismatch():
    b = _make_batcher()
    rollouts = []
    for i in range(M_ROLLOUTS):
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < 4 else 0.0,
                commit={
                    "proof_version": "v5",
                    "tokens": [1, 2, 3],
                    "completion_text_for_test": "wrong",
                },
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=500,
        signed_round=1000,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.REWARD_MISMATCH
