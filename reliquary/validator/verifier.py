"""GRAIL proof verification — primitives used by GrpoWindowBatcher.

The orchestration lives in `reliquary.validator.batcher`. This module only
exposes the per-commit checks that touch the model or the signature scheme.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch

from reliquary.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    MAX_TOKENS_PER_ROLLOUT,
)

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    """Return value of verify_commitment_proofs.

    The ``logits`` field is the cached per-token logits tensor from the
    validator's forward pass at LAYER_INDEX. Downstream behavioural
    validators (logprob / distribution) read it to avoid re-running the
    forward pass.
    """

    all_passed: bool
    passed: int
    checked: int
    logits: torch.Tensor  # shape: [seq_len, vocab_size], on CPU


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
) -> ProofResult:
    """Hard check: verify GRAIL sketch commitments against model forward pass.

    Returns a ``ProofResult`` whose ``logits`` field carries the full-vocab
    logits tensor from the validator's forward pass. Downstream behavioural
    validators (logprob / distribution) read it without re-forwarding.
    """
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
        return ProofResult(
            all_passed=False, passed=0, checked=0,
            logits=torch.empty(0),
        )

    # SECURITY: Reject sequences that would cause GPU OOM.
    if seq_len > MAX_TOKENS_PER_ROLLOUT:
        logger.warning(
            "Token sequence too long: %d tokens (max %d)",
            seq_len, MAX_TOKENS_PER_ROLLOUT,
        )
        return ProofResult(
            all_passed=False, passed=0, checked=0,
            logits=torch.empty(0),
        )

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
        hidden_states, logits_batch = forward_single_layer(
            model, input_ids, None, LAYER_INDEX
        )

    hidden_states = hidden_states[0]  # Remove batch dim: [seq_len, hidden_dim]
    logits = logits_batch[0].detach().to("cpu")

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
    return ProofResult(
        all_passed=all_passed, passed=passed, checked=checked, logits=logits,
    )


def verify_reward_claim(
    env: Any,
    problem: dict,
    completion_text: str,
    claimed: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Re-compute the env's reward on *completion_text* and compare to *claimed*.

    Miners declare the reward of each completion in their submission (saves
    validator compute when they can pre-filter out-of-zone) but the validator
    re-runs ``env.compute_reward`` to check honesty. A mismatch means the
    miner lied about reward, warranting rejection.

    Returns True iff |env_reward - claimed| <= tolerance. The small tolerance
    absorbs float64 formatting round-trip (JSON serialisation) noise.
    """
    try:
        actual = env.compute_reward(problem, completion_text)
    except Exception:
        return False
    return abs(float(actual) - float(claimed)) <= tolerance


def rewards_std(rewards: list[float]) -> float:
    """Population standard deviation of a rollout group's rewards.

    Returns 0.0 for empty or single-element lists (degenerate — no
    information). The population formula (divide by n, not n-1) is
    used because we want the std of THIS specific sample, not an
    estimator of the underlying distribution's std.
    """
    n = len(rewards)
    if n < 2:
        return 0.0
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    return variance ** 0.5


def is_in_zone(sigma: float, *, bootstrap: bool = False) -> bool:
    """True iff *sigma* exceeds the minimum threshold for training signal.

    Steady state: σ ≥ SIGMA_MIN (0.43).
    Bootstrap: σ ≥ BOOTSTRAP_SIGMA_MIN (0.33).

    A group with σ below this is dropped because its rollouts cluster
    too tightly for the normalised advantage (r - μ) / σ to carry a
    usable gradient signal.
    """
    from reliquary.constants import BOOTSTRAP_SIGMA_MIN, SIGMA_MIN

    if sigma < 1e-8:
        return False   # degenerate
    return sigma >= (BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN)
