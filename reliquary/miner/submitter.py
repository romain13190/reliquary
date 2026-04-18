"""HTTP client used by miners to push GRPO submissions to the validator.

V1 assumption: a single validator. Discovery returns the first axon advertised
by a hotkey holding `validator_permit`. Multi-validator routing is intentionally
out of scope here — see the GRPO refactor plan.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    SubmissionRequest,
    SubmissionResponse,
    WindowStateResponse,
)

logger = logging.getLogger(__name__)

# Retry configuration: 3 attempts, exponential backoff 1s / 2s / 4s.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
# Default timeout is generous: the validator may need several seconds to verify
# a submission even in the async-queue path (the queue can back up under load).
# Miners running against slow links (Targon port-forward etc.) benefit further.
_DEFAULT_TIMEOUT = 60.0


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted."""


def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Return the HTTP URL of the first validator advertised on the metagraph.

    Picks the first uid with validator_permit=True and an axon IP that isn't
    the unset placeholder. Multi-validator coordination is out of scope; this
    deliberately picks ONE validator.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        # Use the validator's own port if it's set; fall back to the protocol default.
        axon_port = getattr(axon, "port", None) or port
        return f"http://{ip}:{axon_port}"
    raise NoValidatorFoundError("no validator with permit and routable axon")


async def submit_batch(
    url: str,
    request: SubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SubmissionResponse:
    """POST a submission to the validator with retry + backoff.

    Raises `SubmissionError` if all attempts fail. A 4xx response is treated
    as a final answer (the batch is malformed; retrying won't help) and is
    parsed into a SubmissionResponse so the caller can act on it.
    """
    payload = request.model_dump(mode="json")
    return await _post_with_retry(
        f"{url}/submit",
        payload,
        SubmissionResponse,
        client=client,
        timeout=timeout,
    )


async def get_window_state(
    url: str,
    window_start: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> WindowStateResponse:
    """GET the validator's current view of a window's slot fill state."""
    return await _get_with_retry(
        f"{url}/window/{window_start}/state",
        WindowStateResponse,
        client=client,
        timeout=timeout,
    )


async def _post_with_retry(
    full_url: str,
    json_payload: dict,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.post(full_url, json=json_payload, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "submit attempt %d to %s failed: %r (type=%s)",
                    attempt, full_url, e, type(e).__name__,
                )
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            # 4xx means the request is malformed or the validator rejected it
            # for a deterministic reason — retrying is pointless. Parse and return.
            if 400 <= resp.status_code < 500:
                # 422 (validation) and 503/409 don't carry a SubmissionResponse —
                # surface them as a non-accepted response with the detail string.
                detail = _safe_detail(resp)
                if response_model is SubmissionResponse:
                    return SubmissionResponse(
                        accepted=False,
                        reason=f"http_{resp.status_code}:{detail}",
                        settled=False,
                        slot_count=0,
                    )
                raise SubmissionError(f"HTTP {resp.status_code}: {detail}")
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def _get_with_retry(
    full_url: str,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.get(full_url, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if resp.status_code == 404:
                # Window not active — caller's job to handle.
                raise SubmissionError(f"window not active at {full_url}")
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


def _safe_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]
