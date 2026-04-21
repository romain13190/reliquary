"""FastAPI server: receives v2 GRPO market submissions, exposes window state.

/submit drops requests on an asyncio queue (worker thread drains it off the
event loop so GRAIL verification doesn't block HTTP responses). Under
TestClient (no worker running), /submit runs synchronously so tests see
the real verdict.
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
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)
from reliquary.validator.batcher import GrpoWindowBatcher

logger = logging.getLogger(__name__)


class _Health(BaseModel):
    status: str
    active_window: int | None


class ValidatorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self.active_batcher: GrpoWindowBatcher | None = None
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        self.active_batcher = batcher

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Reliquary Validator", version="2.0")

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return _Health(
                status="ok",
                active_window=(
                    self.active_batcher.window_start if self.active_batcher else None
                ),
            )

        @app.post("/submit", response_model=BatchSubmissionResponse)
        async def submit(request: BatchSubmissionRequest) -> BatchSubmissionResponse:
            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            if request.window_start != batcher.window_start:
                raise HTTPException(status_code=409, detail="window_mismatch")

            # Under TestClient (no worker running) we run synchronously so tests
            # see the real accept verdict; under uvicorn we enqueue for the
            # worker and return a provisional ACCEPTED. The worker's real
            # verdict surfaces in logs.
            if self._worker_task is None:
                return batcher.accept_submission(request)

            await self._submit_queue.put((request, batcher))
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED
            )

        @app.get(
            "/window/{window_start}/state", response_model=GrpoBatchState
        )
        async def window_state(window_start: int) -> GrpoBatchState:
            batcher = self.active_batcher
            if batcher is None or batcher.window_start != window_start:
                raise HTTPException(status_code=404, detail="window_not_active")
            return batcher.get_state()

        return app

    async def _submit_worker(self) -> None:
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
                        "accepted prompt=%d hotkey=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                    )
                else:
                    logger.warning(
                        "rejected prompt=%d hotkey=%s reason=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                        response.reason.value,
                    )
            except Exception:
                logger.exception(
                    "submission worker failed on prompt %d", request.prompt_idx
                )

    async def start(self) -> None:
        if self._task is not None:
            return
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(self._submit_worker())
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
