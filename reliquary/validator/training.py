"""GRPO training step — STUB for v2.1.

This stub validates the orchestration path (seal → train → publish →
pull). The real GRPO loss + optimizer update plugs in here in a
follow-up PR with no protocol change.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def train_step(model: Any, batch: list) -> Any:
    """Run one GRPO step on *batch*. STUB: returns the model unchanged.

    The real implementation will:
      1. Compute per-rollout advantages within each group
      2. Forward pass + log-prob extraction
      3. Compute clipped PPO/GRPO loss + KL term
      4. Backward + optimizer step
      5. Return the updated model (or same reference, mutated in-place)

    For v2.1 we only need the orchestration to work end-to-end; the
    weights stay frozen so miners and validator continue to share the
    same effective model. The checkpoint_n counter still bumps so the
    manifest signals progress.

    Args:
        model: the model object (torch.nn.Module in production).
        batch: list of ValidSubmission entries from the sealed window.

    Returns:
        The same model object (stub). Real impl returns the updated
        model.
    """
    logger.info(
        "train_step (stub) called with batch of %d submissions — "
        "weights not modified",
        len(batch),
    )
    return model
