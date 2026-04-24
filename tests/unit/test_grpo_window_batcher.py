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
from reliquary.validator.batcher import GrpoWindowBatcher


class FakeEnv:
    name = "fake"
    def __len__(self):
        return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "a", "id": f"pid-{idx}"}
    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


def _always_true_grail(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _always_false_grail(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=False, passed=0, checked=1, logits=torch.empty(0))


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
        checkpoint_hash="sha256:test",
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
        checkpoint_hash="sha256:test",
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.REWARD_MISMATCH


# --- seal_batch + cooldown lifecycle ---

def test_seal_batch_empty_pool_returns_empty():
    b = _make_batcher()
    assert b.seal_batch() == []


def test_seal_batch_fifo_across_many_submissions():
    b = _make_batcher(current_round=1010)
    # Jumbled rounds within the fresh range [1000, 1010] (LAG_MAX=10); duplicates
    # allowed since that window is only 11 values wide but B_BATCH may exceed 11.
    round_nums = [1010 - (i * 3) % 11 for i in range(B_BATCH)]
    for i, round_num in enumerate(round_nums):
        req = _request(
            prompt_idx=i, signed_round=round_num, hotkey=f"hk{i}",
        )
        resp = b.accept_submission(req)
        assert resp.accepted, f"unexpected reject for {i}: {resp.reason}"
    batch = b.seal_batch()
    assert len(batch) == B_BATCH
    rounds = [s.signed_round for s in batch]
    assert rounds == sorted(rounds)


def test_seal_batch_cooldown_recorded():
    b = _make_batcher()
    req = _request(prompt_idx=42, signed_round=1000)
    b.accept_submission(req)
    batch = b.seal_batch()
    assert len(batch) == 1
    assert b._cooldown.is_in_cooldown(42, b.window_start + 1) is True


def test_sealed_batch_respects_cooldown_from_previous_window():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    b = _make_batcher(window_start=120, cooldown_map=cd)
    req = _request(prompt_idx=42, signed_round=1000, window_start=120)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN


def test_state_endpoint_exposes_cooldown():
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    cd = CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
    cd.record_batched(prompt_idx=42, window=100)
    cd.record_batched(prompt_idx=7, window=105)
    b = _make_batcher(window_start=110, cooldown_map=cd)
    state = b.get_state()
    assert set(state.cooldown_prompts) == {42, 7}
    assert state.valid_submissions == 0


def test_distinct_prompts_in_batch_only():
    b = _make_batcher(current_round=1002)
    b.accept_submission(_request(prompt_idx=42, signed_round=1000, hotkey="alice"))
    b.accept_submission(_request(prompt_idx=42, signed_round=1001, hotkey="bob"))
    b.accept_submission(_request(prompt_idx=7, signed_round=1002, hotkey="carol"))
    batch = b.seal_batch()
    assert len(batch) == 2
    hotkeys = {s.hotkey for s in batch}
    assert hotkeys == {"alice", "carol"}


# --- v2.1 seal_event + checkpoint_hash gating ---

import asyncio

import pytest


def _request_v21(prompt_idx=42, signed_round=1000, window_start=500,
                 rewards=None, hotkey="hk", checkpoint_hash="sha256:abc"):
    """v2.1 request: includes the required checkpoint_hash field."""
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for i, r in enumerate(rewards):
        text = "CORRECT" if r > 0.5 else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5], reward=r,
                commit={
                    "proof_version": "v5", "tokens": [1, 2, 3, 4, 5],
                    "completion_text_for_test": text,
                },
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=window_start, signed_round=signed_round,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
    )


def test_reject_wrong_checkpoint():
    """Submission with checkpoint_hash != batcher's current is rejected."""
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:current"
    req = _request_v21(checkpoint_hash="sha256:stale")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_CHECKPOINT


def test_accept_matching_checkpoint():
    b = _make_batcher()
    b.current_checkpoint_hash = "sha256:current"
    req = _request_v21(checkpoint_hash="sha256:current")
    resp = b.accept_submission(req)
    assert resp.accepted is True


def test_empty_checkpoint_hash_disables_gate():
    """When batcher.current_checkpoint_hash is "", any hash is accepted
    (test convenience — simulates pre-first-publish)."""
    b = _make_batcher()
    b.current_checkpoint_hash = ""
    req = _request_v21(checkpoint_hash="anything")
    resp = b.accept_submission(req)
    assert resp.accepted is True


@pytest.mark.asyncio
async def test_seal_event_set_when_b_valid_distinct_landed():
    """seal_event fires when the B-th valid distinct-prompt non-cooldown
    submission is accepted."""
    b = _make_batcher(current_round=2000)
    b.current_checkpoint_hash = "sha256:hash"
    assert not b.seal_event.is_set()
    # Round must stay in [current_round - LAG_MAX, current_round] = [1990, 2000];
    # wrap so B_BATCH submissions all land in that range.
    for i in range(B_BATCH):
        req = _request_v21(
            prompt_idx=i, signed_round=1990 + (i % 11), hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    # Give the event loop a tick to ensure the event is visible.
    await asyncio.wait_for(b.seal_event.wait(), timeout=0.1)
    assert b.seal_event.is_set()


def test_seal_event_not_set_with_only_duplicate_prompts():
    """Two submissions on same prompt → only first counts → seal_event not set."""
    b = _make_batcher(current_round=2000)
    b.current_checkpoint_hash = "sha256:hash"
    for i in range(2):
        req = _request_v21(
            prompt_idx=42, signed_round=1993 + i, hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    # Only 1 distinct prompt → not enough for seal
    assert not b.seal_event.is_set()


def test_seal_event_not_set_with_fewer_than_b():
    """Fewer than B valid submissions → no seal."""
    b = _make_batcher(current_round=2000)
    b.current_checkpoint_hash = "sha256:hash"
    for i in range(B_BATCH - 1):
        req = _request_v21(
            prompt_idx=i, signed_round=1993 + i, hotkey=f"hk{i}",
            checkpoint_hash="sha256:hash",
        )
        b.accept_submission(req)
    assert not b.seal_event.is_set()
