"""Validator-side wandb telemetry.

Opt-in: requires ``WANDB_API_KEY`` env var. Fails soft on any exception
and on missing ``wandb`` package. Never propagates to the validator main
loop. Run id is deterministic (``{hotkey[:8]}-{version}``) so restarts
resume the same wandb run; bump ``WANDB_TRAINING_VERSION`` (or set
``RELIQUARY_WANDB_VERSION``) to start a new run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from reliquary.constants import WANDB_PROJECT, WANDB_TRAINING_VERSION

logger = logging.getLogger(__name__)

_run: Any = None
_enabled: bool = False
_log_warned: bool = False  # suppress repeated log-path warnings


def is_active() -> bool:
    return _enabled


def init(hotkey_ss58: str, config: dict) -> None:
    """Initialise wandb if ``WANDB_API_KEY`` is set. No-op otherwise.

    Fail-soft: any exception (ImportError, network, auth) disables
    telemetry for the rest of the session and logs a warning.
    """
    global _run, _enabled
    if not os.getenv("WANDB_API_KEY"):
        logger.info("wandb: WANDB_API_KEY not set, telemetry disabled")
        return
    # Active paths added in later tasks.


def log_training_step(metrics: dict, step: int | None) -> None:
    """Forward a metrics dict to wandb.log. No-op if disabled. Fail-soft."""
    if not _enabled:
        return
    # Active path added in later task.


def finish() -> None:
    """Close the wandb run. No-op if disabled. Fail-soft."""
    if not _enabled:
        return
    # Active path added in later task.


def _reset_for_tests() -> None:
    """Test-only helper: reset module state so each test starts fresh."""
    global _run, _enabled, _log_warned
    _run = None
    _enabled = False
    _log_warned = False
