"""Validator HTTP server — v2 GRPO market endpoints."""

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    RejectReason,
)
from reliquary.validator.batcher_v2 import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


def _batcher(window_start=500, cooldown_map=None):
    return GrpoWindowBatcher(
        window_start=window_start,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=lambda c, m, r: (True, 1, 1),
        verify_signature_fn=lambda c, h: True,
        verify_proof_version_fn=lambda c: True,
        completion_text_fn=lambda r: r.commit.get("completion_text_for_test", ""),
    )


def _request(prompt_idx=42, window_start=500, signed_round=1000, k=4):
    rollouts = []
    for i in range(M_ROLLOUTS):
        text = "CORRECT" if i < k else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < k else 0.0,
                commit={"tokens": [1, 2, 3], "proof_version": "v5", "completion_text_for_test": text},
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=prompt_idx,
        window_start=window_start,
        signed_round=signed_round,
        merkle_root="00" * 32,
        rollouts=rollouts,
    )


def test_submit_returns_queued_on_active_window():
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == RejectReason.ACCEPTED.value


def test_submit_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 503


def test_submit_409_on_window_mismatch():
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request(window_start=999).model_dump(mode="json"))
    assert resp.status_code == 409


def test_state_endpoint_returns_grpo_batch_state():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    client = TestClient(server.app)
    resp = client.get("/window/500/state")
    assert resp.status_code == 200
    state = GrpoBatchState(**resp.json())
    assert state.window_start == 500
    assert 42 in state.cooldown_prompts


def test_state_endpoint_404_on_wrong_window():
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    client = TestClient(server.app)
    resp = client.get("/window/999/state")
    assert resp.status_code == 404
