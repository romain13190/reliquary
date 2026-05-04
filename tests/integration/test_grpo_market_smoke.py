"""Smoke test: submission → batch → cooldown → weights over 3 windows.

No on-chain dependencies — all verifiers are stubbed. The goal is to
prove the pieces fit together end-to-end.
"""

import hashlib

import pytest

from reliquary.constants import B_BATCH, CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    RejectReason,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from collections import defaultdict


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"q{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "WIN" in c else 0.0


class _ModelStub:
    """Minimal model stub so verify_tokens can resolve vocab/length limits."""
    class config:
        vocab_size = 10_000
        max_position_embeddings = 4096


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate."""
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


def _rollouts(k):
    out = []
    for i in range(M_ROLLOUTS):
        reward = 1.0 if i < k else 0.0
        commit = _make_commit(success=reward > 0.5, total_reward=reward)
        out.append(RolloutSubmission(
            tokens=commit["tokens"],
            reward=reward,
            commit=commit,
        ))
    return out


def _merkle(n: int) -> str:
    return hashlib.sha256(str(n).encode()).hexdigest()


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _make_batcher(window, cooldown):
    return GrpoWindowBatcher(
        window_start=window,
        current_round=window * 10 + 100,  # plenty of headroom for signed_round
        env=FakeEnv(),
        model=_ModelStub(),
        cooldown_map=cooldown,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "WIN" if r.reward > 0.5 else "",
    )


def test_two_windows_with_cooldown():
    """Window 0: B_BATCH+2 miners submit on distinct prompts, B_BATCH win batch.
    Window 1: same prompts → all rejected for cooldown.
    Window 4 (after cooldown=3): prompt 0 eligible again."""
    cooldown = CooldownMap(cooldown_windows=3)  # small for test speed
    n_submissions = B_BATCH + 2

    # Window 0: current_round = 0 * 10 + 100 = 100; signed_round must be in [90, 100]
    b0 = _make_batcher(window=0, cooldown=cooldown)
    for i in range(n_submissions):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}",
            prompt_idx=i,
            window_start=0,
            signed_round=95,  # within LAG_MAX=10 of current_round=100
            merkle_root=_merkle(i),
            rollouts=_rollouts(k=4),
            checkpoint_hash="sha256:test",
        )
        resp = b0.accept_submission(req)
        assert resp.accepted, f"unexpected reject for hk{i}: {resp.reason}"
    batch0 = b0.seal_batch()
    assert len(batch0) == B_BATCH
    # select_batch sorts by tiebreak hash; any B_BATCH of the submitted prompts may win
    batched_prompts = {s.prompt_idx for s in batch0}
    assert len(batched_prompts) == B_BATCH
    assert batched_prompts.issubset(set(range(n_submissions)))

    # Window 1: re-submit the exact batched prompts → all rejected for cooldown
    b1 = _make_batcher(window=1, cooldown=cooldown)
    for prompt_idx in batched_prompts:
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{prompt_idx}",
            prompt_idx=prompt_idx,
            window_start=1,
            signed_round=105,  # within LAG_MAX=10 of current_round=110
            merkle_root=_merkle(100 + prompt_idx),
            rollouts=_rollouts(k=4),
            checkpoint_hash="sha256:test",
        )
        resp = b1.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN

    # Window 4 (after cooldown=3 expires): re-use any batched prompt
    # current_round = 4 * 10 + 100 = 140; signed_round must be in [130, 140]
    reuse_prompt = next(iter(sorted(batched_prompts)))
    b4 = _make_batcher(window=4, cooldown=cooldown)
    req = BatchSubmissionRequest(
        miner_hotkey=f"hk{reuse_prompt}",
        prompt_idx=reuse_prompt,
        window_start=4,
        signed_round=135,  # within LAG_MAX=10 of current_round=140
        merkle_root=_merkle(500),
        rollouts=_rollouts(k=4),
        checkpoint_hash="sha256:test",
    )
    resp = b4.accept_submission(req)
    assert resp.accepted is True, f"expected eligibility after cooldown, got {resp.reason}"


def _run_ema_windows(hotkey_counts_per_window: list[dict[str, int]]) -> defaultdict:
    """Simulate _update_ema over multiple windows; return final EMA dict."""
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService

    class _W:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    svc = ValidationService(
        wallet=_W(), model=MagicMock(), tokenizer=MagicMock(),
        env=MagicMock(), netuid=99,
    )
    svc._miner_scores_ema = defaultdict(float)

    for counts in hotkey_counts_per_window:
        batch = []
        for hk, n in counts.items():
            for _ in range(n):
                sub = MagicMock()
                sub.hotkey = hk
                batch.append(sub)
        svc._update_ema(batch)

    return svc._miner_scores_ema


def test_weights_for_full_batch():
    """Full batch every window — after convergence, EMA sum → 1.0, burn ≈ 0."""
    counts = {f"hk{i}": 1 for i in range(B_BATCH)}
    ema = _run_ema_windows([counts] * 500)
    total = sum(ema.values())
    burn = max(0.0, 1.0 - total)
    assert abs(total + burn - 1.0) < 1e-9
    assert abs(burn) < 0.005
    assert len(ema) == B_BATCH


def test_weights_for_partial_batch_burns_rest():
    """Partial batch (filled/B_BATCH) every window — burn → (B_BATCH - filled) / B_BATCH."""
    filled = 5
    counts = {f"hk{i}": 1 for i in range(filled)}
    ema = _run_ema_windows([counts] * 500)
    total = sum(ema.values())
    burn = max(0.0, 1.0 - total)
    assert abs(burn - (B_BATCH - filled) / B_BATCH) < 0.005


def test_out_of_zone_rejected_end_to_end():
    """A submission with k=0 (all fail) is rejected at the zone filter."""
    cooldown = CooldownMap(cooldown_windows=50)
    b = _make_batcher(window=0, cooldown=cooldown)
    req = BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=0,
        signed_round=95,  # within LAG_MAX=10 of current_round=100
        merkle_root=_merkle(42),
        rollouts=_rollouts(k=0),  # all losses
        checkpoint_hash="sha256:test",
    )
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.OUT_OF_ZONE
