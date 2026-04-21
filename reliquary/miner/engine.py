"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    BLOCK_TIME_SECONDS,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    The reference miner uses uniform-random selection with rejection
    sampling against the cooldown set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    rng = rng or _random
    n = len(env)
    if len(cooldown_prompts) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex."""
    import hashlib

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(repr(r.tokens).encode())
        h.update(repr(r.reward).encode())
        h.update(repr(r.commit).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = int(
            os.environ.get("RELIQUARY_MAX_NEW_TOKENS", MAX_NEW_TOKENS_PROTOCOL_CAP)
        ),
        validator_url_override: str | None = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int,
        use_drand: bool = True,
    ) -> list:
        """v2: pick prompts freely, generate M rollouts each, submit batch.

        Returns the list of BatchSubmissionResponse objects collected.
        """
        import httpx
        import random

        # Lazy imports — submitter v2 helpers are defined in Task 13.
        from reliquary.miner.submitter import (
            SubmissionError,
            discover_validator_url,
            get_window_state_v2,
            submit_batch_v2,
        )

        randomness = await self._compute_randomness(subtensor, window_start, use_drand)

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        deadline = (
            time.monotonic()
            + WINDOW_LENGTH * BLOCK_TIME_SECONDS
            - UPLOAD_BUFFER
        )
        logger.info(
            "Mining v2 window %d — %.0fs budget, validator %s",
            window_start,
            WINDOW_LENGTH * BLOCK_TIME_SECONDS - UPLOAD_BUFFER,
            url,
        )

        rng = random.Random()
        results = []

        async with httpx.AsyncClient(timeout=30) as client:
            while time.monotonic() < deadline:
                try:
                    state = await get_window_state_v2(url, window_start, client=client)
                except SubmissionError as exc:
                    logger.debug("state fetch failed: %s", exc)
                    continue

                cooldown_set = set(state.cooldown_prompts)
                signed_round = state.current_round

                try:
                    prompt_idx = pick_prompt_idx(self.env, cooldown_set, rng=rng)
                except RuntimeError:
                    logger.info("env fully in cooldown; stopping")
                    break

                problem = self.env.get_problem(prompt_idx)

                generations = self._generate_m_rollouts(problem, randomness)
                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d / %d rollouts for prompt %d; skipping",
                        len(generations), M_ROLLOUTS, prompt_idx,
                    )
                    continue

                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness)
                    for gen in generations
                ]

                merkle_root = _compute_merkle_root(rollout_submissions)

                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=window_start,
                    signed_round=signed_round,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                )
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    logger.info(
                        "submitted prompt %d — accepted=%s reason=%s",
                        prompt_idx, resp.accepted, resp.reason.value,
                    )
                    results.append(resp)
                except SubmissionError as exc:
                    logger.error("submit failed prompt %d: %s", prompt_idx, exc)

        return results

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO. No cherry-picking."""
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        rollouts = []
        for _ in range(M_ROLLOUTS):
            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens],
                    device=getattr(self.vllm_model, "device", "cpu"),
                )
                outputs = self.vllm_model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=T_PROTO,
                )
            all_tokens = outputs[0].tolist()
            rollouts.append({
                "tokens": all_tokens,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from block hash (+ optional drand beacon)."""
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        return chain.compute_window_randomness(block_hash)

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # HF forward pass on proof GPU
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # Token log-probs from HF (bit-identical with validator)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Sign
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }
