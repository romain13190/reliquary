"""Tests for the miner HTTP submitter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from reliquary.constants import COMPLETIONS_PER_SUBMISSION, VALIDATOR_HTTP_PORT
from reliquary.miner.submitter import (
    NoValidatorFoundError,
    SubmissionError,
    discover_validator_url,
    get_window_state,
    submit_batch,
)
from reliquary.protocol.submission import (
    SlotState,
    SubmissionRequest,
    SubmissionResponse,
    WindowStateResponse,
)


def _request() -> SubmissionRequest:
    return SubmissionRequest(
        window_start=1000,
        slot_index=0,
        prompt_id="abc1234567890def",
        miner_hotkey="5HotkeyTest" + "x" * 35,
        completions=[
            {"tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 + i], "commit": {"v": 1}}
            for i in range(COMPLETIONS_PER_SUBMISSION)
        ],
    )


# --------------------- discover_validator_url ---------------------


def test_discover_picks_first_permitted_with_routable_axon() -> None:
    meta = SimpleNamespace(
        validator_permit=[False, True, True],
        axons=[
            SimpleNamespace(ip="1.1.1.1", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=9000),
            SimpleNamespace(ip="3.3.3.3", port=9001),
        ],
    )
    assert discover_validator_url(meta) == "http://2.2.2.2:9000"


def test_discover_skips_unset_axon_ip() -> None:
    meta = SimpleNamespace(
        validator_permit=[True, True],
        axons=[
            SimpleNamespace(ip="0.0.0.0", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=8888),
        ],
    )
    assert discover_validator_url(meta) == "http://2.2.2.2:8888"


def test_discover_falls_back_to_default_port_when_axon_port_zero() -> None:
    meta = SimpleNamespace(
        validator_permit=[True],
        axons=[SimpleNamespace(ip="1.1.1.1", port=0)],
    )
    assert discover_validator_url(meta) == f"http://1.1.1.1:{VALIDATOR_HTTP_PORT}"


def test_discover_raises_when_no_permitted() -> None:
    meta = SimpleNamespace(
        validator_permit=[False, False],
        axons=[
            SimpleNamespace(ip="1.1.1.1", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=8888),
        ],
    )
    with pytest.raises(NoValidatorFoundError):
        discover_validator_url(meta)


def test_discover_raises_when_metagraph_malformed() -> None:
    with pytest.raises(NoValidatorFoundError):
        discover_validator_url(SimpleNamespace())


# --------------------- submit_batch ---------------------


@pytest.mark.asyncio
async def test_submit_batch_success() -> None:
    response = SubmissionResponse(
        accepted=True, reason="ok", settled=False, slot_count=4
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response.model_dump())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        got = await submit_batch("http://val", _request(), client=client)
    assert got == response


@pytest.mark.asyncio
async def test_submit_batch_4xx_returns_non_accepted_response() -> None:
    # 4xx responses (e.g. 422 validation, 409 window mismatch) are deterministic
    # rejections — they're surfaced as a non-accepted response with the detail
    # baked into `reason` so callers can log without exception handling.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "window_mismatch"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        got = await submit_batch("http://val", _request(), client=client)
    assert got.accepted is False
    assert "window_mismatch" in got.reason


@pytest.mark.asyncio
async def test_submit_batch_retries_on_5xx_then_succeeds() -> None:
    calls = {"n": 0}
    response = SubmissionResponse(
        accepted=True, reason="ok", settled=True, slot_count=32
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"detail": "internal"})
        return httpx.Response(200, json=response.model_dump())

    transport = httpx.MockTransport(handler)
    with patch("reliquary.miner.submitter.asyncio.sleep") as fake_sleep:
        fake_sleep.return_value = None
        async with httpx.AsyncClient(transport=transport) as client:
            got = await submit_batch("http://val", _request(), client=client)
    assert got.accepted is True
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_submit_batch_raises_after_all_5xx_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    with patch("reliquary.miner.submitter.asyncio.sleep", return_value=None):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(SubmissionError):
                await submit_batch("http://val", _request(), client=client)


@pytest.mark.asyncio
async def test_submit_batch_retries_on_transport_error() -> None:
    calls = {"n": 0}
    response = SubmissionResponse(
        accepted=True, reason="ok", settled=False, slot_count=4
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("conn refused")
        return httpx.Response(200, json=response.model_dump())

    transport = httpx.MockTransport(handler)
    with patch("reliquary.miner.submitter.asyncio.sleep", return_value=None):
        async with httpx.AsyncClient(transport=transport) as client:
            got = await submit_batch("http://val", _request(), client=client)
    assert got.accepted is True
    assert calls["n"] == 2


# --------------------- get_window_state ---------------------


@pytest.mark.asyncio
async def test_get_window_state_success() -> None:
    state = WindowStateResponse(
        window_start=1000,
        slot_states=[
            SlotState(slot_index=i, prompt_id=f"p{i}", count=i, settled=False)
            for i in range(8)
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/window/1000/state"
        return httpx.Response(200, json=state.model_dump())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        got = await get_window_state("http://val", 1000, client=client)
    assert got == state


@pytest.mark.asyncio
async def test_get_window_state_404_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "window_not_active"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SubmissionError):
            await get_window_state("http://val", 1000, client=client)
