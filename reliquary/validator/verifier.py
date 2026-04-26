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

    ``sketch_diff_max`` is the worst per-position |miner_sketch -
    validator_sketch| across the K challenge positions (mod PRIME_Q),
    surfaced for post-hoc threshold calibration even when the proof
    passed the current tolerance.
    """

    all_passed: bool
    passed: int
    checked: int
    logits: torch.Tensor  # shape: [seq_len, vocab_size], on CPU
    sketch_diff_max: int = 0


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
    sketch_diff_max = 0
    for idx in challenge_indices:
        if idx >= seq_len:
            continue
        checked += 1
        miner_commit = commitments[idx]
        validator_hidden = hidden_states[idx]
        valid, diag = verifier.verify_commitment(
            validator_hidden, miner_commit, r_vec, seq_len, idx
        )
        sketch_diff = int((diag or {}).get("sketch_diff", 0))
        if sketch_diff > sketch_diff_max:
            sketch_diff_max = sketch_diff
        if valid:
            passed += 1

    # SECURITY: All expected challenge positions must be checked and pass.
    # A miner cannot benefit from having fewer positions verified.
    all_passed = passed == checked and checked >= expected_challenges
    return ProofResult(
        all_passed=all_passed, passed=passed, checked=checked, logits=logits,
        sketch_diff_max=sketch_diff_max,
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


def verify_logprobs_claim(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    claimed_logprobs: list[float],
    logits: torch.Tensor,
    challenge_randomness: str,
) -> tuple[bool, float]:
    """Hard check: validate miner-claimed per-token logprobs against the
    validator's re-computation on K=CHALLENGE_K challenged positions.

    For each challenge position ``i``, compute the validator's logprob of
    ``tokens[i]`` from ``log_softmax(logits[i - 1])``, then
    ``dev_i = exp(|model_lp - miner_lp|) - 1``. The **median** deviation
    across the K positions is compared against ``LOGPROB_IS_EPS``.

    Median (not mean) is robust to the bf16 outliers honest miners see
    on cross-GPU runs. Threshold 0.10 was calibrated at 0 % FP on ~430k
    honest trials in the original GRAIL repo.

    ``claimed_logprobs`` accepts two layouts:
    - Full-sequence (length == len(tokens)): prompt positions are
      ignored, completion positions read directly by absolute index.
    - Completion-only (length == completion_length): position-j entry
      corresponds to absolute index ``prompt_length + j``.

    Returns ``(is_valid, median_dev)``. ``median_dev`` is ``inf`` when
    the check cannot be executed (completion too short, malformed
    payload, out-of-range index).
    """
    import math
    from statistics import median

    from reliquary.constants import CHALLENGE_K, LOGPROB_IS_EPS
    from reliquary.protocol.crypto import indices_from_root_in_range

    if completion_length < CHALLENGE_K:
        return False, float("inf")

    if len(claimed_logprobs) == len(tokens):
        def miner_lp_at(abs_idx: int) -> float:
            return float(claimed_logprobs[abs_idx])
    elif len(claimed_logprobs) == completion_length:
        def miner_lp_at(abs_idx: int) -> float:
            return float(claimed_logprobs[abs_idx - prompt_length])
    else:
        return False, float("inf")

    challenge_idxs = indices_from_root_in_range(
        tokens,
        challenge_randomness,
        prompt_length,
        prompt_length + completion_length,
        CHALLENGE_K,
    )
    if len(challenge_idxs) != CHALLENGE_K:
        return False, float("inf")

    devs: list[float] = []
    for abs_idx in challenge_idxs:
        pos = abs_idx - 1
        if pos < 0 or pos >= logits.size(0):
            return False, float("inf")
        dist = torch.log_softmax(logits[pos].float(), dim=-1)
        model_lp = float(dist[tokens[abs_idx]].item())
        devs.append(math.exp(abs(model_lp - miner_lp_at(abs_idx))) - 1.0)

    median_dev = float(median(devs))
    return median_dev <= LOGPROB_IS_EPS, median_dev


def evaluate_token_distribution(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    logits: torch.Tensor,
    temperature: float,
) -> tuple[bool | None, dict]:
    """Soft check: detect suspicious chosen-token probability distributions.

    For each completion step ``t``, apply ``temperature`` to
    ``logits[t - 1]``, softmax, and read the probability the validator's
    model would have assigned to the token the miner actually emitted
    at position ``t``. Collect those probabilities and compute summary
    stats.

    Returns ``(is_valid, metrics)``:
      - ``True``   — distribution is consistent with sampling from the
                     validator's model at ``temperature``
      - ``False``  — suspicious (median or q10 collapsed below threshold
                     → miner likely sampled from a different model)
      - ``None``   — insufficient steps (< SAMPLING_MIN_STEPS) — caller
                     defaults to accept (not enough signal)

    ``metrics`` carries ``mean``, ``median``, ``q10``, ``low_frac``,
    ``high_frac`` regardless of the decision (empty dict only when
    there's insufficient data).
    """
    import numpy as np

    from reliquary.constants import (
        SAMPLING_HIGH_P,
        SAMPLING_LOW_P,
        SAMPLING_LOW_Q10_MAX,
        SAMPLING_MEDIAN_LOW_MAX,
        SAMPLING_MIN_STEPS,
    )

    if completion_length < SAMPLING_MIN_STEPS:
        return None, {}

    probs: list[float] = []
    for t in range(prompt_length, prompt_length + completion_length):
        if t == 0 or t - 1 >= logits.size(0) or t >= len(tokens):
            continue
        step = (logits[t - 1].float() / float(temperature)).softmax(dim=-1)
        probs.append(float(step[tokens[t]].item()))

    if len(probs) < SAMPLING_MIN_STEPS:
        return None, {}

    x = np.asarray(probs, dtype=np.float64)
    metrics = {
        "mean":      float(x.mean()),
        "median":    float(np.median(x)),
        "q10":       float(np.quantile(x, 0.10)),
        "low_frac":  float((x <= SAMPLING_LOW_P).mean()),
        "high_frac": float((x >= SAMPLING_HIGH_P).mean()),
    }

    suspicious = (
        metrics["median"] < SAMPLING_MEDIAN_LOW_MAX
        or metrics["q10"] < SAMPLING_LOW_Q10_MAX
    )
    return (not suspicious), metrics
