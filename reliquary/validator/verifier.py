"""GRAIL proof verification — primitives used by the WindowBatcher.

The orchestration (which prompt, which slot, which miner) lives in
`reliquary.validator.batcher`. This module only exposes the per-commit
checks that touch the model or the signature scheme.
"""

import logging
from typing import Any

from reliquary.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    MAX_TOKENS_PER_ROLLOUT,
)

logger = logging.getLogger(__name__)


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    from reliquary.protocol.signatures import verify_commit_signature

    return verify_commit_signature(commit, hotkey)


def verify_proof_version(commit: dict) -> bool:
    """Hard check: proof version must match protocol."""
    return commit.get("proof_version") == GRAIL_PROOF_VERSION


def verify_commitment_proofs(
    commit: dict,
    model: Any,
    window_randomness: str,
) -> tuple[bool, int, int]:
    """Hard check: verify GRAIL sketch commitments against model forward pass.

    Returns:
        (all_passed, passed_count, checked_count)
    """
    import torch

    from reliquary.protocol.crypto import indices_from_root
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.hf_compat import resolve_hidden_size

    tokens = commit["tokens"]
    commitments = commit["commitments"]

    # SECURITY: Miner must provide exactly one commitment per token.
    # Otherwise they can omit commitments for positions they can't forge,
    # and the verifier would silently skip those challenges.
    seq_len = len(tokens)
    if len(commitments) != seq_len:
        logger.warning(
            "Commitment count mismatch: %d commitments for %d tokens",
            len(commitments), seq_len,
        )
        return False, 0, 0

    # SECURITY: Reject sequences that would cause GPU OOM.
    if seq_len > MAX_TOKENS_PER_ROLLOUT:
        logger.warning(
            "Token sequence too long: %d tokens (max %d)",
            seq_len, MAX_TOKENS_PER_ROLLOUT,
        )
        return False, 0, 0

    # SECURITY: Always use the validator's independently-computed randomness.
    # Never trust the miner's claimed beacon — a miner who controls the
    # randomness can predict which positions are challenged and only forge those.
    randomness = window_randomness

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    expected_challenges = min(CHALLENGE_K, seq_len)
    challenge_indices = indices_from_root(
        tokens, randomness, seq_len, expected_challenges
    )

    input_ids = torch.tensor([tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        hidden_states, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)

    hidden_states = hidden_states[0]  # Remove batch dim: [seq_len, hidden_dim]

    passed = 0
    checked = 0
    for idx in challenge_indices:
        if idx >= seq_len:
            continue
        checked += 1
        miner_commit = commitments[idx]
        validator_hidden = hidden_states[idx]
        valid, _ = verifier.verify_commitment(
            validator_hidden, miner_commit, r_vec, seq_len, idx
        )
        if valid:
            passed += 1

    # SECURITY: All expected challenge positions must be checked and pass.
    # A miner cannot benefit from having fewer positions verified.
    all_passed = passed == checked and checked >= expected_challenges
    return all_passed, passed, checked
