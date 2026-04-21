"""Smoke test: submission → batch → cooldown → weights over 3 windows.

No on-chain dependencies — all verifiers are stubbed. The goal is to
prove the pieces fit together end-to-end.
"""

import hashlib

import pytest

from reliquary.constants import B_BATCH, M_ROLLOUTS
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


def _rollouts(k):
    out = []
    for i in range(M_ROLLOUTS):
        text = "WIN" if i < k else "lose"
        out.append(RolloutSubmission(
            tokens=[1, 2, 3],
            reward=1.0 if i < k else 0.0,
            commit={"tokens": [1, 2, 3], "proof_version": "v5",
                    "completion_text_for_test": text},
        ))
    return out


def _merkle(n: int) -> str:
    return hashlib.sha256(str(n).encode()).hexdigest()


def _make_batcher(window, cooldown):
    return GrpoWindowBatcher(
        window_start=window,
        current_round=window * 10 + 100,  # plenty of headroom for signed_round
        env=FakeEnv(),
        model=None,
        cooldown_map=cooldown,
        verify_commitment_proofs_fn=lambda c, m, r: (True, 1, 1),
        verify_signature_fn=lambda c, h: True,
        verify_proof_version_fn=lambda c: True,
        completion_text_fn=lambda r: r.commit.get("completion_text_for_test", ""),
    )


def test_two_windows_with_cooldown():
    """Window 0: 10 miners submit on distinct prompts, 8 win batch.
    Window 1: same prompts → all rejected for cooldown.
    Window 4 (after cooldown=3): prompt 0 eligible again."""
    cooldown = CooldownMap(cooldown_windows=3)  # small for test speed

    # Window 0: current_round = 0 * 10 + 100 = 100; signed_round must be in [90, 100]
    b0 = _make_batcher(window=0, cooldown=cooldown)
    for i in range(10):
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
    # select_batch sorts by tiebreak hash; any 8 of the 10 prompts may win
    batched_prompts = {s.prompt_idx for s in batch0}
    assert len(batched_prompts) == B_BATCH
    assert batched_prompts.issubset(set(range(10)))

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
    from reliquary.validator.state_persistence import ValidatorState
    import tempfile, os

    class _W:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    with tempfile.TemporaryDirectory() as tmp:
        svc = ValidationService(
            wallet=_W(), model=MagicMock(), tokenizer=MagicMock(),
            env=MagicMock(), netuid=99,
        )
        svc._state = ValidatorState(os.path.join(tmp, "s.json"))
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
    """Partial batch (5/8) every window — after convergence, burn ≈ 3/8."""
    counts = {f"hk{i}": 1 for i in range(5)}
    ema = _run_ema_windows([counts] * 500)
    total = sum(ema.values())
    burn = max(0.0, 1.0 - total)
    assert abs(burn - 3.0 / B_BATCH) < 0.005


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
