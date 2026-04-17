"""GRAIL Verifier for GPU/Framework-Agnostic Proof.

Key innovations:
1. Top-K selection: Focus on important activations (stable)
2. Logarithmic bucketing: Coarse quantization reduces sensitivity
3. Sketch verification: Random linear projection for cryptographic binding

Security: ~10^-167 forgery probability across K=32 challenged positions.
"""

from __future__ import annotations

import logging
import math

import torch

from reliquary.constants import (
    PRIME_Q,
    PROOF_COEFF_RANGE,
    PROOF_NUM_BUCKETS,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
    PROOF_TOPK,
)

logger = logging.getLogger(__name__)


def log_magnitude_bucket(value: float, num_buckets: int = PROOF_NUM_BUCKETS) -> int:
    """Map activation to logarithmic magnitude bucket with sign preservation.

    Args:
        value: Activation value to bucket
        num_buckets: Number of buckets per sign (default: 8)

    Returns:
        Signed bucket index in [-num_buckets+1, 0, num_buckets-1]
    """
    if math.isnan(value):
        logger.warning(
            "NaN value encountered in hidden state. Treating as zero bucket."
        )
        return 0

    if math.isinf(value):
        logger.warning(
            "Infinity value encountered in hidden state. Clamping to maximum bucket."
        )
        return num_buckets - 1 if value > 0 else -(num_buckets - 1)

    abs_val = abs(value)

    if abs_val < 1e-6:
        return 0

    log_val = math.log2(abs_val + 1.0)
    scale_factor = num_buckets / 10.0
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))

    return bucket if value >= 0 else -bucket


def log_magnitude_bucket_vectorized(
    values: torch.Tensor,
    num_buckets: int = PROOF_NUM_BUCKETS,
) -> torch.Tensor:
    """Vectorized log-magnitude bucketing, bit-identical to the scalar version.

    Uses float64 arithmetic to match the scalar path.
    """
    abs_vals = values.abs().to(torch.float64)
    scale_factor = num_buckets / 10.0

    log_vals = torch.log2(abs_vals + 1.0)
    raw_buckets = (log_vals * scale_factor).to(torch.int64)
    raw_buckets = torch.clamp(raw_buckets, min=0, max=num_buckets - 1)

    sign_positive = values >= 0
    buckets = torch.where(sign_positive, raw_buckets, -raw_buckets)

    # Edge cases (applied in priority order)
    zero = torch.zeros_like(buckets)
    deadzone_mask = abs_vals < 1e-6
    buckets = torch.where(deadzone_mask, zero, buckets)

    nan_mask = torch.isnan(values)
    buckets = torch.where(nan_mask, zero, buckets)

    inf_mask = torch.isinf(values)
    if inf_mask.any():
        pos_inf = torch.tensor(num_buckets - 1, dtype=torch.int64, device=values.device)
        neg_inf = torch.tensor(-(num_buckets - 1), dtype=torch.int64, device=values.device)
        inf_buckets = torch.where(values > 0, pos_inf, neg_inf)
        buckets = torch.where(inf_mask, inf_buckets, buckets)

    return buckets


def adaptive_sketch_tolerance(position: int, sequence_length: int) -> int:
    """Compute position-dependent sketch tolerance.

    tolerance = base + growth * sqrt(position)
    """
    return int(PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH * math.sqrt(position))


class GRAILVerifier:
    """Sketch-based verifier for framework-agnostic hidden state proofs."""

    def __init__(
        self,
        hidden_dim: int,
        topk: int = PROOF_TOPK,
        num_buckets: int = PROOF_NUM_BUCKETS,
        r_coeff_range: int = PROOF_COEFF_RANGE,
    ):
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.num_buckets = num_buckets
        self.r_coeff_range = r_coeff_range

    def generate_r_vec(self, randomness_hex: str) -> torch.Tensor:
        """Generate small bounded coefficient vector from randomness.

        Returns:
            Tensor of shape [topk] with int8 coefficients in [-R, R]
        """
        from reliquary.protocol.crypto import RNG_LABEL, prf

        clean_hex = randomness_hex.strip().replace("0x", "").replace("0X", "")
        if len(clean_hex) % 2 != 0:
            clean_hex = "0" + clean_hex

        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=2 * self.topk,
        )

        import numpy as np

        int16_vals = np.frombuffer(raw, dtype=">i2")[: self.topk]
        coeffs = (np.abs(int16_vals) % (2 * self.r_coeff_range + 1)) - self.r_coeff_range

        return torch.from_numpy(coeffs.astype(np.int8))

    def create_commitment(self, hidden_state: torch.Tensor, r_vec: torch.Tensor) -> dict:
        """Create commitment for a single token position."""
        abs_hidden = torch.abs(hidden_state)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices = topk_result.indices

        indices, _ = torch.sort(indices)
        values = hidden_state[indices]

        buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in values],
            dtype=torch.int8,
        )

        sketch = torch.dot(buckets.to(torch.int32), r_vec.to(torch.int32))
        sketch_val = int(sketch.item()) % PRIME_Q

        return {"sketch": sketch_val}

    def create_commitments_batch(self, h_layer: torch.Tensor, r_vec: torch.Tensor) -> list[dict]:
        """Create commitments for all positions at once (vectorized).

        Produces bit-identical results to calling create_commitment() in a loop.
        """
        seq_len = h_layer.size(0)

        abs_h = h_layer.abs()
        _, topk_indices = torch.topk(abs_h, k=self.topk, dim=1)
        del abs_h

        topk_indices, _ = torch.sort(topk_indices, dim=1)
        signed_values = torch.gather(h_layer, dim=1, index=topk_indices)

        buckets = log_magnitude_bucket_vectorized(signed_values, self.num_buckets)
        del signed_values

        buckets_f = buckets.to(torch.float32)
        r_vec_f = r_vec.to(torch.float32).to(buckets_f.device)
        sketches = (buckets_f @ r_vec_f).to(torch.int64)
        del buckets, buckets_f

        sketches_list = sketches.tolist()
        sketch_vals = [s % PRIME_Q for s in sketches_list]

        return [{"sketch": sketch_vals[pos]} for pos in range(seq_len)]

    def verify_commitment(
        self,
        validator_hidden: torch.Tensor,
        miner_commitment: dict,
        r_vec: torch.Tensor,
        sequence_length: int,
        position: int,
    ) -> tuple[bool, dict]:
        """Verify commitment using sketch check."""
        tolerance = adaptive_sketch_tolerance(position, sequence_length)

        abs_hidden = torch.abs(validator_hidden)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices, _ = torch.sort(topk_result.indices)
        validator_values = validator_hidden[indices]

        validator_buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in validator_values],
            dtype=torch.int8,
        )

        validator_sketch = torch.dot(validator_buckets.to(torch.int32), r_vec.to(torch.int32))
        validator_sketch_val = int(validator_sketch.item()) % PRIME_Q

        miner_sketch_val = miner_commitment["sketch"]
        sketch_diff = abs(validator_sketch_val - miner_sketch_val)
        mod_diff = min(sketch_diff, PRIME_Q - sketch_diff)
        is_valid = mod_diff <= tolerance

        diagnostics = {
            "sketch_diff": mod_diff,
            "sketch_valid": is_valid,
            "sketch_tolerance": tolerance,
            "overall_valid": is_valid,
            "validator_sketch": validator_sketch_val,
            "miner_sketch": miner_sketch_val,
            "position": position,
        }

        if not is_valid:
            sample_vals = validator_values[:5].tolist() if len(validator_values) >= 5 else validator_values.tolist()
            sample_buckets = validator_buckets[:5].tolist() if len(validator_buckets) >= 5 else validator_buckets.tolist()
            logger.warning(
                "[verify_commitment] SKETCH MISMATCH: position=%d | "
                "validator_sketch=%d | miner_sketch=%d | diff=%d | tolerance=%d | "
                "sample_values=%s | sample_buckets=%s | hidden_norm=%.4f",
                position, validator_sketch_val, miner_sketch_val, mod_diff, tolerance,
                [f"{v:.4f}" for v in sample_vals], sample_buckets,
                float(validator_hidden.norm().item()),
            )

        return is_valid, diagnostics
