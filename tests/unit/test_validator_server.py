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
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _batcher(window_start=500, cooldown_map=None):
    batcher = GrpoWindowBatcher(
        window_start=window_start,
        current_round=1000,
        env=FakeEnv(),
        model=None,
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        verify_proof_version_fn=lambda c: True,
        completion_text_fn=lambda r: r.commit.get("completion_text_for_test", ""),
    )
    batcher.current_checkpoint_hash = "sha256:test"
    return batcher


def _request(prompt_idx=42, window_start=500, signed_round=1000, k=4, checkpoint_hash="sha256:test"):
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
        checkpoint_hash=checkpoint_hash,
    )


def test_submit_returns_queued_on_active_window():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == RejectReason.ACCEPTED.value


def test_submit_503_when_no_active_batcher():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 503


def test_submit_409_on_window_mismatch():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request(window_start=999).model_dump(mode="json"))
    assert resp.status_code == 409


def test_state_endpoint_returns_grpo_batch_state():
    from reliquary.protocol.submission import WindowState
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    state = GrpoBatchState(**resp.json())
    assert state.window_n == 500
    assert 42 in state.cooldown_prompts


def test_state_endpoint_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "no_active_window"


# --- v2.1: state-aware endpoints ---

def test_submit_rejects_when_state_not_open():
    """When state != OPEN, /submit returns a non-accepted response."""
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.TRAINING)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "window_not_active"


def test_submit_accepted_when_state_open():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True


def test_state_endpoint_returns_window_state_enum():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "open"
    assert body["window_n"] == 500
    assert body["checkpoint_n"] == 0  # no checkpoint published yet
    assert body["checkpoint_repo_id"] is None
    assert body["checkpoint_revision"] is None


def test_state_endpoint_exposes_checkpoint_when_set():
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=7,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_007",
        signature="ed25519:sig",
    ))
    client = TestClient(server.app)
    resp = client.get("/state")
    body = resp.json()
    assert body["checkpoint_n"] == 7
    assert body["checkpoint_repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["checkpoint_revision"] == "rev_sha_007"


def test_checkpoint_endpoint_404_when_none_published():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 404


def test_checkpoint_endpoint_returns_manifest_when_set():
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=42,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_042",
        signature="ed25519:sig_42",
    ))
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_n"] == 42
    assert body["repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["revision"] == "rev_sha_042"
    assert body["signature"] == "ed25519:sig_42"
