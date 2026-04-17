"""Single-layer forward pass utility.

Provides a memory-efficient forward pass that extracts hidden states from
exactly one transformer layer plus logits, without materializing all
intermediate hidden states.  Used by both the miner (proof generation) and
the validator (proof verification) to guarantee identical numerical results.

Both miner and validator must produce bit-identical hidden states for the
same (model, tokens, attention_mask) triple. Using a single shared
implementation with explicit use_cache=False eliminates divergence.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def forward_single_layer(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass returning hidden states at *one* layer plus logits.

    When layer_index == -1 (last hidden state), calls the base model
    directly — avoids output_hidden_states=True which stores all
    intermediate layers.

    Args:
        model: HuggingFace CausalLM (e.g. AutoModelForCausalLM).
        input_ids: [batch, seq_len] token ids.
        attention_mask: [batch, seq_len] mask (1 = real, 0 = pad).
        layer_index: Which hidden layer to return. -1 for last.

    Returns:
        (hidden_states, logits) where hidden_states has shape
        [batch, seq_len, hidden_dim] and logits has shape
        [batch, seq_len, vocab_size], both on the model device.
    """
    base_model_prefix = getattr(model, "base_model_prefix", "")
    base = getattr(model, base_model_prefix, None) if base_model_prefix else None
    lm_head = getattr(model, "lm_head", None)

    if layer_index == -1 and base is not None and lm_head is not None:
        base_out = base(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        h = base_out.last_hidden_state
        logits = lm_head(h)
        logger.debug(
            "forward_single_layer: efficient path | batch=%d seq_len=%d",
            h.shape[0],
            h.shape[1],
        )
        return h, logits

    logger.warning(
        "forward_single_layer: falling back to output_hidden_states=True | "
        "layer_index=%d base_model_prefix=%s has_base=%s has_lm_head=%s",
        layer_index,
        base_model_prefix or "N/A",
        base is not None,
        lm_head is not None,
    )
    outs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outs.hidden_states[layer_index], outs.logits
