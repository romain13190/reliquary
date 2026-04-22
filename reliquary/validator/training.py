"""GRPO training step for Reliquary v2.1.

Single-step-per-window GRPO implementation: group-relative advantages
computed from the rewards in each ValidSubmission, PPO-clipped surrogate
loss, KL penalty against a frozen reference model (the validator's
starting checkpoint). Linear warmup + cosine LR schedule.

Uses miner-provided token log-probs (from the GRAIL commit) as π_old —
saves one forward pass per rollout.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from reliquary.validator import telemetry
from reliquary.constants import (
    GRAD_CLIP_NORM, KL_BETA, LEARNING_RATE, LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS, PPO_CLIP_EPSILON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-global state — persists across train_step calls for the same model
# ---------------------------------------------------------------------------

_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
_ref_model: Optional[Any] = None
_model_id: Optional[int] = None


def _lazy_init(model) -> bool:
    """Create optimizer, scheduler, and reference model snapshot on first
    call for a given model object. No-op on subsequent calls with the same
    model. If the caller swaps to a different model instance, we rebuild.

    Returns True on success, False if the model is not a usable nn.Module
    (e.g. a MagicMock in tests or a non-torch placeholder).
    """
    global _optimizer, _scheduler, _ref_model, _model_id
    if _model_id == id(model):
        return True

    try:
        params = list(model.parameters())
    except (AttributeError, TypeError):
        logger.warning("_lazy_init: model has no .parameters(); skipping init")
        return False
    if not params:
        logger.warning("_lazy_init: model.parameters() is empty; skipping init")
        return False

    _optimizer = torch.optim.AdamW(
        params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    def _lr_lambda(step: int) -> float:
        if step < LR_WARMUP_WINDOWS:
            return (step + 1) / LR_WARMUP_WINDOWS
        progress = (step - LR_WARMUP_WINDOWS) / max(
            1, LR_COSINE_MAX_WINDOWS - LR_WARMUP_WINDOWS
        )
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    _scheduler = torch.optim.lr_scheduler.LambdaLR(_optimizer, _lr_lambda)

    # Reference model: deep-copy, eval mode, frozen. Used for KL.
    _ref_model = copy.deepcopy(model)
    _ref_model.eval()
    for p in _ref_model.parameters():
        p.requires_grad = False

    _model_id = id(model)
    logger.info("Training state initialised (optimizer, scheduler, ref_model)")
    return True


def reset_training_state() -> None:
    """Clear the module-level singletons. Used by tests to start fresh.

    Production code should never call this — it throws away optimiser
    momentum and the reference model snapshot.
    """
    global _optimizer, _scheduler, _ref_model, _model_id
    _optimizer = None
    _scheduler = None
    _ref_model = None
    _model_id = None


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a model)
# ---------------------------------------------------------------------------

def _compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative normalized advantages.

    mean = mean(rewards); std = pop-std(rewards); return (r - mean) / std.
    Degenerate group (std == 0) → all zeros (no signal, group will be skipped).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


# ---------------------------------------------------------------------------
# Per-rollout loss (forward-pass heavy — uses the model)
# ---------------------------------------------------------------------------

def _rollout_loss(
    model,
    ref_model,
    rollout,
    advantage: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (ppo_loss, kl_term) for one rollout.

    Both scalars, averaged over completion tokens. Forward passes run in
    bf16 autocast; softmax / log-softmax cast back to fp32 for numerical
    stability.

    π_old comes from the miner's GRAIL commit (rollout.commit["rollout"]
    ["token_logprobs"]) — saves an extra forward pass.
    """
    tokens_list = rollout.tokens
    prompt_length = rollout.commit.get("rollout", {}).get("prompt_length", 0)
    old_logprobs_list = rollout.commit.get("rollout", {}).get("token_logprobs", [])

    if prompt_length <= 0 or not old_logprobs_list:
        raise ValueError("rollout missing prompt_length or token_logprobs")

    tokens = torch.tensor([tokens_list], device=device)  # [1, T]

    # Current model forward pass (with grad)
    dtype_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
        if device.type in ("cuda", "cpu") else torch.autocast(device_type="cpu", enabled=False)
    with dtype_ctx:
        logits = model(tokens).logits[0]  # [T, vocab]

    # log π_new(token_{t+1} | context_<=t) — cast to fp32 for stability
    log_probs_all = F.log_softmax(logits[:-1].float(), dim=-1)
    next_tokens = tokens[0, 1:]  # [T-1]
    new_logprobs = log_probs_all.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

    # Slice to completion tokens only: logits[prompt_length-1] predicts
    # tokens[prompt_length] (first completion token).
    new_logprobs_c = new_logprobs[prompt_length - 1:]

    # Reference model forward pass (no grad)
    with torch.no_grad():
        with dtype_ctx:
            ref_logits = ref_model(tokens).logits[0]
        ref_log_probs_all = F.log_softmax(ref_logits[:-1].float(), dim=-1)
        ref_logprobs = ref_log_probs_all.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
    ref_logprobs_c = ref_logprobs[prompt_length - 1:]

    # π_old from miner (same completion slice)
    old_logprobs = torch.tensor(
        old_logprobs_list, device=device, dtype=new_logprobs_c.dtype,
    )
    if len(old_logprobs) != len(new_logprobs_c):
        raise ValueError(
            f"log-prob length mismatch: miner reported {len(old_logprobs)}, "
            f"model predicts {len(new_logprobs_c)} completion tokens"
        )

    # PPO clipped surrogate
    log_ratio = new_logprobs_c - old_logprobs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantage
    ppo_loss = -torch.min(surr1, surr2).mean()

    # KL(π_new || π_ref) — Schulman's k3 estimator:
    #   kl ≈ exp(ref - new) - 1 - (ref - new)
    # Unbiased, low-variance, always ≥ 0.
    kl_log_ratio = ref_logprobs_c - new_logprobs_c
    kl = (torch.exp(kl_log_ratio) - 1 - kl_log_ratio).mean()

    return ppo_loss, kl


# ---------------------------------------------------------------------------
# Main entry point — one GRPO step per call
# ---------------------------------------------------------------------------

def train_step(model, batch: list, window_index: int | None = None) -> Any:
    """Run one GRPO step on *batch* (list of ValidSubmission).

    If *window_index* is provided, it is used as the wandb step when
    telemetry is enabled — aligning the x-axis of wandb charts with the
    subnet window index. Safe to omit in tests.

    Each ValidSubmission has M rollouts on the same prompt. Within-group
    advantages are computed from rewards; PPO-clipped surrogate is
    summed across all rollouts; KL penalty against the frozen reference
    is added. One optimiser step per train_step call.

    Returns the (same) model — in-place mutations happen via the
    optimiser.
    """
    if not batch:
        logger.info("train_step: empty batch, skipping")
        return model

    if not _lazy_init(model):
        logger.info("train_step: model not initializable (non-torch?), skipping")
        return model
    assert _optimizer is not None and _scheduler is not None and _ref_model is not None

    model.train()
    device = next(model.parameters()).device

    _optimizer.zero_grad()

    n_total_rollouts = sum(len(g.rollouts) for g in batch)
    total_ppo = 0.0
    total_kl = 0.0
    n_processed = 0
    n_skipped = 0

    for group in batch:
        rewards = [r.reward for r in group.rollouts]
        advantages = _compute_advantages(rewards)
        if all(a == 0.0 for a in advantages):
            n_skipped += 1
            logger.debug("skipping degenerate group on prompt_idx=%d", group.prompt_idx)
            continue

        for rollout, adv in zip(group.rollouts, advantages):
            try:
                ppo_loss, kl = _rollout_loss(
                    model=model, ref_model=_ref_model,
                    rollout=rollout, advantage=adv, device=device,
                )
            except ValueError as e:
                logger.warning("rollout skipped: %s", e)
                continue
            loss = (ppo_loss + KL_BETA * kl) / n_total_rollouts
            loss.backward()
            total_ppo += ppo_loss.item()
            total_kl += kl.item()
            n_processed += 1

    if n_processed == 0:
        logger.info("train_step: no valid rollouts processed")
        return model

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    _optimizer.step()
    _scheduler.step()
    lr = _scheduler.get_last_lr()[0]

    logger.info(
        "train_step: lr=%.2e ppo=%.4f kl=%.4f grad_norm=%.3f rollouts=%d/%d",
        lr, total_ppo / n_processed, total_kl / n_processed,
        float(grad_norm), n_processed, n_total_rollouts,
    )

    # Emit structured metrics to wandb (no-op if telemetry disabled).
    all_rewards = [r.reward for g in batch for r in g.rollouts]
    n_rewards = len(all_rewards)
    reward_mean = sum(all_rewards) / n_rewards
    reward_var = sum((r - reward_mean) ** 2 for r in all_rewards) / n_rewards
    reward_std = reward_var ** 0.5
    n_groups = len(batch)
    metrics = {
        "train/lr": lr,
        "train/ppo_loss": total_ppo / n_processed,
        "train/kl": total_kl / n_processed,
        "train/grad_norm": float(grad_norm),
        "train/rollouts_processed": n_processed,
        "train/rollouts_total": n_total_rollouts,
        "train/valid_rollout_ratio": n_processed / n_total_rollouts,
        "rewards/mean": reward_mean,
        "rewards/std": reward_std,
        "rewards/min": min(all_rewards),
        "rewards/max": max(all_rewards),
        "batch/n_groups": n_groups,
        "batch/n_degenerate_groups": n_skipped,
        "batch/degenerate_ratio": n_skipped / n_groups,
    }
    telemetry.log_training_step(metrics, step=window_index)

    return model
