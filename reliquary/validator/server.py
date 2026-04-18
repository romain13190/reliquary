"""FastAPI server: receives miner submissions, exposes window state.

The server holds a reference to the active `WindowBatcher`, which is rotated
by the validator service at the start of each window. /submit drops requests
on an asyncio queue and returns immediately; a worker drains the queue and
runs `accept_submission` in a worker thread so heavy GRAIL verifs don't block
HTTP responses (otherwise large bodies from real miners would spend the whole
miner budget waiting for the verifier to return).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    SubmissionRequest,
    SubmissionResponse,
    WindowStateResponse,
)
from reliquary.validator.batcher import WindowBatcher

logger = logging.getLogger(__name__)


class _Health(BaseModel):
    status: str
    active_window: int | None


class ValidatorServer:
    """Lifecycle wrapper around the FastAPI app + uvicorn server."""

    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self.active_batcher: WindowBatcher | None = None
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        # Queue is created eagerly so the /submit handler can enqueue even
        # when the server is running under a TestClient (where start()/the
        # worker are never invoked). Late-binding to the running loop is fine
        # on Python 3.10+ since asyncio.Queue is loop-agnostic at init.
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None

    def set_active_batcher(self, batcher: WindowBatcher | None) -> None:
        self.active_batcher = batcher

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Reliquary Validator", version="1.0")

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return _Health(
                status="ok",
                active_window=(
                    self.active_batcher.window_start if self.active_batcher else None
                ),
            )

        @app.post("/submit", response_model=SubmissionResponse)
        async def submit(request: SubmissionRequest) -> SubmissionResponse:
            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            if request.window_start != batcher.window_start:
                raise HTTPException(status_code=409, detail="window_mismatch")
            await self._submit_queue.put((request, batcher))
            return SubmissionResponse(
                accepted=True,
                reason="queued",
                settled=False,
                slot_count=batcher.slots[request.slot_index].count,
            )

        @app.get(
            "/window/{window_start}/state", response_model=WindowStateResponse
        )
        async def window_state(window_start: int) -> WindowStateResponse:
            batcher = self.active_batcher
            if batcher is None or batcher.window_start != window_start:
                raise HTTPException(status_code=404, detail="window_not_active")
            return batcher.get_window_state()

        return app

    async def _submit_worker(self) -> None:
        """Drain the submit queue and verify each request off the event loop."""
        while True:
            try:
                request, batcher = await self._submit_queue.get()
            except asyncio.CancelledError:
                return
            try:
                response = await asyncio.to_thread(
                    batcher.accept_submission, request
                )
                if response.accepted:
                    logger.info(
                        "accepted slot %d from %s (count=%d)",
                        request.slot_index,
                        request.miner_hotkey[:12],
                        response.slot_count,
                    )
                else:
                    logger.warning(
                        "rejected slot %d from %s: %s",
                        request.slot_index,
                        request.miner_hotkey[:12],
                        response.reason,
                    )
            except Exception:
                logger.exception(
                    "submission worker failed on slot %d", request.slot_index
                )

    async def start(self) -> None:
        """Start uvicorn + submit worker in the background; idempotent."""
        if self._task is not None:
            return
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(self._submit_worker())
        # Yield once so uvicorn can bind before the caller proceeds.
        await asyncio.sleep(0)
        logger.info("Validator HTTP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            self._server = None
