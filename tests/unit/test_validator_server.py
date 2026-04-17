"""Tests for the validator FastAPI server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import (
    COMPLETIONS_PER_SUBMISSION,
    PROMPTS_PER_WINDOW,
)
from reliquary.protocol.submission import SubmissionResponse, WindowStateResponse, SlotState
from reliquary.validator.server import ValidatorServer


def _payload() -> dict:
    return {
        "window_start": 1000,
        "slot_index": 0,
        "prompt_id": "abc123def4567890",
        "miner_hotkey": "5HotkeyTest" + "x" * 35,
        "completions": [
            {"tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 + i], "commit": {"v": 1}}
            for i in range(COMPLETIONS_PER_SUBMISSION)
        ],
    }


@pytest.fixture
def server() -> ValidatorServer:
    return ValidatorServer(host="127.0.0.1", port=0)


@pytest.fixture
def client(server: ValidatorServer) -> TestClient:
    return TestClient(server.app)


def test_health_when_idle_reports_no_active_window(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "active_window": None}


def test_health_with_active_batcher_reports_window(
    server: ValidatorServer, client: TestClient
) -> None:
    fake_batcher = MagicMock()
    fake_batcher.window_start = 1000
    server.set_active_batcher(fake_batcher)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["active_window"] == 1000


def test_submit_503_when_no_active_window(client: TestClient) -> None:
    r = client.post("/submit", json=_payload())
    assert r.status_code == 503
    assert r.json()["detail"] == "no_active_window"


def test_submit_409_on_window_mismatch(
    server: ValidatorServer, client: TestClient
) -> None:
    fake_batcher = MagicMock()
    fake_batcher.window_start = 999  # different
    server.set_active_batcher(fake_batcher)
    r = client.post("/submit", json=_payload())
    assert r.status_code == 409
    assert r.json()["detail"] == "window_mismatch"


def test_submit_delegates_to_batcher(
    server: ValidatorServer, client: TestClient
) -> None:
    fake_batcher = MagicMock()
    fake_batcher.window_start = 1000
    fake_batcher.accept_submission.return_value = SubmissionResponse(
        accepted=True, reason="ok", settled=False, slot_count=4
    )
    server.set_active_batcher(fake_batcher)
    r = client.post("/submit", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["slot_count"] == 4
    assert fake_batcher.accept_submission.called


def test_window_state_404_when_no_batcher(client: TestClient) -> None:
    r = client.get("/window/1000/state")
    assert r.status_code == 404


def test_window_state_404_when_window_mismatch(
    server: ValidatorServer, client: TestClient
) -> None:
    fake_batcher = MagicMock()
    fake_batcher.window_start = 1000
    server.set_active_batcher(fake_batcher)
    r = client.get("/window/2000/state")
    assert r.status_code == 404


def test_window_state_returns_batcher_snapshot(
    server: ValidatorServer, client: TestClient
) -> None:
    fake_batcher = MagicMock()
    fake_batcher.window_start = 1000
    snap = WindowStateResponse(
        window_start=1000,
        slot_states=[
            SlotState(slot_index=i, prompt_id=f"p{i}", count=i, settled=False)
            for i in range(PROMPTS_PER_WINDOW)
        ],
    )
    fake_batcher.get_window_state.return_value = snap
    server.set_active_batcher(fake_batcher)
    r = client.get("/window/1000/state")
    assert r.status_code == 200
    assert r.json()["window_start"] == 1000
    assert len(r.json()["slot_states"]) == PROMPTS_PER_WINDOW


def test_submit_validates_payload_shape(client: TestClient) -> None:
    bad = _payload()
    bad["completions"] = []  # empty list rejected by pydantic min_length
    fake_batcher = MagicMock()
    fake_batcher.window_start = 1000
    # Even with active batcher, pydantic should reject before reaching it.
    r = client.post("/submit", json=bad)
    assert r.status_code == 422
