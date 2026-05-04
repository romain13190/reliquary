"""GrpoWindowBatcher: accepts submissions, enforces verification pipeline,
exposes select_batch at window close."""

from dataclasses import dataclass
from typing import Any

import pytest

from reliquary.constants import B_BATCH, CHALLENGE_K, M_ROLLOUTS
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


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate.

    Default produces a ``CHALLENGE_K + 4`` token sequence: 4 prompt tokens,
    ``CHALLENGE_K`` completion tokens (the minimum the proof needs).
    """
    if tokens is None:
        tokens = list(range(CHALLENGE_K + prompt_length))
    seq_len = len(tokens)
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v5",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def _request(
    prompt_idx=42, signed_round=1000, window_start=500,
    rewards=None, hotkey="hk",
) -> BatchSubmissionRequest:
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for r in rewards:
        commit = _make_commit(success=r > 0.5, total_reward=r)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=r,
                commit=commit,
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
    class _DefaultFakeTokenizer:
        eos_token_id = 99

    kwargs = dict(
        window_start=500,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        tokenizer=_DefaultFakeTokenizer(),
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        verify_proof_version_fn=_always_true_proof_version,
        completion_text_fn=lambda rollout: (
            "CORRECT" if rollout.reward > 0.5 else "wrong"
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
    # Override completion_text_fn to always return "wrong", creating a reward mismatch
    # when claim is 1.0
    b = _make_batcher(completion_text_fn=lambda rollout: "wrong")
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=False, total_reward=0.0)
        # Claim high reward for first 4, but completion_text_fn will return "wrong"
        # which computes to 0.0 reward, triggering REWARD_MISMATCH
        claimed_reward = 1.0 if i < 4 else 0.0
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=claimed_reward,
                commit=commit,
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
    for r in rewards:
        commit = _make_commit(success=r > 0.5, total_reward=r)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"], reward=r,
                commit=commit,
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


# ---------------------------------------------------------------------------
# Prompt-binding (canonical_prompt_tokens_fn)
# ---------------------------------------------------------------------------
#
# A miner can pass every other check while having generated under a modified
# prompt (CoT prefix, alternate chat template, few-shot examples) by:
#   1. Running their forward pass on prompt_modified
#   2. Sending the resulting completions + GRAIL sketch to the validator
#   3. Claiming the canonical prompt_idx
# GRAIL alone won't catch this because the validator re-runs forward on the
# *miner-supplied tokens* — both produce the same sketch.
#
# canonical_prompt_tokens_fn closes the gap: the validator computes the
# canonical prompt tokens for the claimed prompt_idx from its own env +
# tokenizer, and rejects any submission whose tokens[:prompt_length] diverges.


def _request_with_prompt_tokens(
    *, prompt_idx: int, prompt_tokens: list[int],
    completion_tokens: list[int] | None = None,
    rewards: list[float] | None = None, hotkey: str = "hk",
):
    """Like ``_request`` but sets ``commit['rollout']['prompt_length']`` and
    builds ``commit['tokens']`` = prompt + completion explicitly so the
    validator's prompt-binding check has something to inspect."""
    if completion_tokens is None:
        completion_tokens = [99]
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for r in rewards:
        text = "CORRECT" if r > 0.5 else "wrong"
        full_tokens = list(prompt_tokens) + list(completion_tokens)
        rollouts.append(
            RolloutSubmission(
                tokens=full_tokens, reward=r,
                commit={
                    "proof_version": "v5",
                    "tokens": full_tokens,
                    "rollout": {
                        "prompt_length": len(prompt_tokens),
                        "completion_length": len(completion_tokens),
                    },
                    "completion_text_for_test": text,
                },
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=500, signed_round=1000,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash="",  # gate disabled for these tests
    )


def test_prompt_mismatch_rejected_when_canonical_differs():
    """Miner runs forward pass on a CoT-prefixed prompt but claims the
    canonical prompt_idx → validator detects the prompt_tokens don't match
    its env's canonical version → PROMPT_MISMATCH before any GRAIL compute."""
    canonical = [10, 11, 12]            # what the env says prompt 42 is
    miner_used = [99, 10, 11, 12]       # CoT prefix + canonical question

    b = _make_batcher(
        canonical_prompt_tokens_fn=lambda idx: canonical if idx == 42 else [],
    )
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=miner_used, completion_tokens=[200, 201],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_MISMATCH


def test_prompt_match_accepted_when_canonical_equals():
    """Honest miner: prompt_tokens match the env's canonical version → check
    is a no-op, submission proceeds through the rest of the pipeline."""
    canonical = [10, 11, 12]
    b = _make_batcher(
        canonical_prompt_tokens_fn=lambda idx: canonical if idx == 42 else [],
    )
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=canonical, completion_tokens=[200, 201],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


def test_no_canonical_fn_disables_check():
    """When ``canonical_prompt_tokens_fn`` is None (test stubs), the binding
    check is skipped — preserves backward compatibility for existing tests
    that don't carry a real tokenizer."""
    b = _make_batcher()  # no canonical_prompt_tokens_fn passed
    # Use an arbitrary prompt_tokens; without a canonical, nothing to compare.
    req = _request_with_prompt_tokens(
        prompt_idx=42, prompt_tokens=[7, 8, 9],
    )
    resp = b.accept_submission(req)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED


# ---------------------------------------------------------------------------
# FIFO short-circuit (SUPERSEDED)
# ---------------------------------------------------------------------------
#
# select_batch picks the submission with the smallest signed_round per
# prompt_idx. Once we've accepted a submission with signed_round=R for
# prompt 42, any future submission for prompt 42 with signed_round >= R
# can't beat it — reject early to skip the (~3s GPU) GRAIL forward pass.
# A submission with signed_round < R legitimately races ahead and falls
# through to the full pipeline.


def test_superseded_rejects_higher_signed_round_same_prompt():
    """Second arrival for the same prompt with a *later* signed_round can't
    beat the incumbent → reject before reward/GRAIL compute."""
    b = _make_batcher()
    first = _request_v21(
        prompt_idx=42, signed_round=995, hotkey="A", checkpoint_hash="",
    )
    second = _request_v21(
        prompt_idx=42, signed_round=998, hotkey="B", checkpoint_hash="",
    )
    r1 = b.accept_submission(first)
    r2 = b.accept_submission(second)
    assert r1.accepted is True
    assert r2.accepted is False
    assert r2.reason == RejectReason.SUPERSEDED
    # Only the incumbent is in _valid — the SUPERSEDED reject didn't pollute it.
    assert len(b._valid) == 1
    assert b._valid[0].hotkey == "A"


def test_superseded_rejects_equal_signed_round_same_prompt():
    """Same signed_round can't strictly beat the incumbent (tiebreak is
    deterministic but still a tie at the round level) → reject early.
    This also handles the same-miner-spamming-same-prompt DoS case."""
    b = _make_batcher()
    first = _request_v21(
        prompt_idx=42, signed_round=1000, hotkey="A", checkpoint_hash="",
    )
    second = _request_v21(
        prompt_idx=42, signed_round=1000, hotkey="A", checkpoint_hash="",
    )
    assert b.accept_submission(first).accepted is True
    r2 = b.accept_submission(second)
    assert r2.accepted is False
    assert r2.reason == RejectReason.SUPERSEDED


def test_lower_signed_round_falls_through_and_wins_at_seal():
    """A late-arriving submission with strictly smaller signed_round
    legitimately races ahead — it must run through the full pipeline,
    not be rejected. select_batch will then pick it over the incumbent."""
    b = _make_batcher()
    incumbent = _request_v21(
        prompt_idx=42, signed_round=998, hotkey="B", checkpoint_hash="",
    )
    challenger = _request_v21(
        prompt_idx=42, signed_round=995, hotkey="A", checkpoint_hash="",
    )
    assert b.accept_submission(incumbent).accepted is True
    r2 = b.accept_submission(challenger)
    assert r2.accepted is True
    assert r2.reason == RejectReason.ACCEPTED
    # Both are in _valid; select_batch resolves by signed_round.
    assert len(b._valid) == 2
    from reliquary.validator.batch_selection import select_batch
    batch = select_batch(
        b._valid, b=B_BATCH, current_window=b.window_start, cooldown_map=b._cooldown,
    )
    assert len(batch) == 1
    assert batch[0].hotkey == "A"  # smaller signed_round won


def test_different_prompts_tracked_independently():
    """SUPERSEDED is per-prompt — accepting prompt 42 must not block prompt 43."""
    b = _make_batcher()
    r_a = b.accept_submission(_request_v21(
        prompt_idx=42, signed_round=1000, hotkey="A", checkpoint_hash="",
    ))
    r_b = b.accept_submission(_request_v21(
        prompt_idx=43, signed_round=1000, hotkey="A", checkpoint_hash="",
    ))
    assert r_a.accepted is True
    assert r_b.accepted is True
    assert len(b._valid) == 2
    assert b._best_round_per_prompt == {42: 1000, 43: 1000}


def test_supersede_only_records_after_full_success():
    """If a submission gets rejected mid-pipeline (e.g. GRAIL fail), it must
    NOT update _best_round_per_prompt — otherwise an honest later submission
    with a higher signed_round would be wrongly rejected as SUPERSEDED."""
    # Use an always-failing GRAIL so the submission is rejected post-cheap-checks.
    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    first_fails = _request_v21(
        prompt_idx=42, signed_round=1000, hotkey="A", checkpoint_hash="",
    )
    r1 = b.accept_submission(first_fails)
    assert r1.accepted is False
    assert r1.reason == RejectReason.GRAIL_FAIL
    # No incumbent recorded for prompt 42.
    assert 42 not in b._best_round_per_prompt
    # A subsequent honest submission for prompt 42 with a later signed_round
    # must NOT be wrongly rejected as SUPERSEDED.
    b._verify_commitment = _always_true_grail
    r2 = b.accept_submission(_request_v21(
        prompt_idx=42, signed_round=998, hotkey="B", checkpoint_hash="",
    ))
    assert r2.accepted is True


def test_constructor_accepts_tokenizer():
    """Tokenizer must be passable to the batcher (used by TerminationValidator)."""
    class FakeTokenizer:
        eos_token_id = 99

    fake_tok = FakeTokenizer()
    b = _make_batcher(tokenizer=fake_tok)
    assert b.tokenizer is fake_tok
