"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol: deterministic prompts derived from beacon randomness → 4
prefix-distinct completions per slot → HTTP submission to validator.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from reliquary.constants import (
    BLOCK_TIME_SECONDS,
    DIVERSITY_PREFIX_LEN,
    GROUP_SIZE,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MINER_BATCH_SIZE,
    PROMPTS_PER_WINDOW,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
from reliquary.miner.prompts import derive_window_prompts
from reliquary.miner.submitter import (
    SubmissionError,
    discover_validator_url,
    get_window_state,
    submit_batch,
)
from reliquary.protocol.submission import CompletionSubmission, SubmissionRequest

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


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
        """Iterate over 8 deterministic prompts and submit 4 completions per slot.

        Returns the list of SubmissionResponse objects collected during the
        window.
        """
        import httpx

        # 1. Compute window randomness
        randomness = await self._compute_randomness(subtensor, window_start, use_drand)

        # 2. Derive 8 deterministic prompts
        problems = derive_window_prompts(self.env, randomness, PROMPTS_PER_WINDOW)

        # 3. Resolve validator URL
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # 4. Deadline: end of window minus upload buffer
        deadline = (
            time.monotonic()
            + WINDOW_LENGTH * BLOCK_TIME_SECONDS
            - UPLOAD_BUFFER
        )
        logger.info(
            "Mining window %d — %.0fs budget, validator %s",
            window_start,
            WINDOW_LENGTH * BLOCK_TIME_SECONDS - UPLOAD_BUFFER,
            url,
        )

        results = []

        # 5. Shared HTTP client for all submissions this window
        async with httpx.AsyncClient(timeout=30) as client:
            for slot_index, problem in enumerate(problems):
                # 6a. Check deadline before each slot
                if time.monotonic() >= deadline:
                    logger.info(
                        "deadline reached, stopping at slot %d", slot_index
                    )
                    break

                # 6b. Fetch window state to decide strategy:
                #     - skip if slot is settled or both quotas full
                #     - pick target reward class (rare) to maximise advantage
                target_reward: float | None = None
                try:
                    state = await get_window_state(url, window_start, client=client)
                    slot_state = next(
                        (s for s in state.slot_states if s.slot_index == slot_index),
                        None,
                    )
                    if slot_state is not None:
                        if slot_state.settled:
                            logger.debug("slot %d already settled, skipping", slot_index)
                            continue
                        target_reward = self._choose_target_reward(slot_state.rewards)
                        if target_reward == "slot_full":
                            logger.debug(
                                "slot %d: full, skipping", slot_index,
                            )
                            continue
                except SubmissionError as exc:
                    # Window not active yet or racing the validator — attempt anyway
                    # with no targeting (target_reward stays None).
                    logger.debug(
                        "get_window_state for slot %d failed (%s); submitting untargeted",
                        slot_index, exc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Unexpected error fetching window state for slot %d: %s",
                        slot_index, exc,
                    )
                    continue

                # 6c. Generate MINER_BATCH_SIZE prefix-distinct completions,
                # targeting the picked reward class if any (sample-and-filter).
                diverse = self._generate_targeted_batch(
                    problem, randomness, target_reward,
                )
                if len(diverse) < MINER_BATCH_SIZE:
                    logger.warning(
                        "slot %d: only got %d completions after max attempts "
                        "(target_reward=%s, need %d) — skipping",
                        slot_index, len(diverse), target_reward,
                        MINER_BATCH_SIZE,
                    )
                    continue

                # 6d. Build CompletionSubmission objects
                completions = [
                    self._build_completion_submission(gen, randomness)
                    for gen in diverse
                ]

                # 6e. Build and send SubmissionRequest
                request = SubmissionRequest(
                    window_start=window_start,
                    slot_index=slot_index,
                    prompt_id=problem["id"],
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    completions=completions,
                )
                try:
                    response = await submit_batch(url, request, client=client)
                    logger.info(
                        "slot %d: accepted=%s reason=%r settled=%s slot_count=%d",
                        slot_index,
                        response.accepted,
                        response.reason,
                        response.settled,
                        response.slot_count,
                    )
                    results.append(response)
                except SubmissionError as exc:
                    logger.error("slot %d: submission failed: %s", slot_index, exc)

        return results

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

    @staticmethod
    def _choose_target_reward(rewards_hist: dict[str, int]):
        """Pick the reward class the miner should target for this slot.

        Strategy under the market-free settlement (no per-class quota):
          * If the slot has already reached GROUP_SIZE → return the
            sentinel ``"slot_full"`` so the caller skips the slot.
          * Otherwise target the RARE class (smaller count) to maximise
            |advantage| at settlement. Ties → None (no preference).

        Returns:
            1.0, 0.0, None, or the sentinel ``"slot_full"``.
        """
        count_1 = rewards_hist.get("1.0", 0)
        count_0 = rewards_hist.get("0.0", 0)

        if count_1 + count_0 >= GROUP_SIZE:
            return "slot_full"
        if count_1 < count_0:
            return 1.0
        if count_0 < count_1:
            return 0.0
        return None  # balanced / empty — no preference

    def _generate_targeted_batch(
        self,
        problem: dict,
        randomness: str,
        target_reward: float | None,
    ) -> list[dict]:
        """Generate up to MINER_BATCH_SIZE prefix-distinct completions,
        optionally filtered to match ``target_reward``.

        Uses the env locally to score each candidate (deterministic — same
        formula as the validator). Rejects candidates whose first
        ``DIVERSITY_PREFIX_LEN`` tokens collide with an earlier accepted
        candidate. Caps attempts to avoid unbounded loops when the model
        can't produce the target class for this prompt.
        """
        import torch

        # Generous attempt cap: 10× the target batch size. Enough for typical
        # rejection sampling without blowing up when the model is near-
        # deterministic on this prompt (in which case we give up and let
        # the caller skip the slot).
        max_attempts = MINER_BATCH_SIZE * 10

        prompt_tokens: list[int] = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        seen_prefixes: set[tuple] = set()
        completions: list[dict] = []

        for _ in range(max_attempts):
            if len(completions) >= MINER_BATCH_SIZE:
                break

            with torch.no_grad():
                input_tensor = torch.tensor(
                    [prompt_tokens], device=getattr(self.vllm_model, "device", "cpu")
                )
                outputs = self.vllm_model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                )
            all_tokens: list[int] = outputs[0].tolist()
            completion_tokens = all_tokens[prompt_length:]

            # Filter by target reward if requested.
            if target_reward is not None:
                completion_text = self.tokenizer.decode(completion_tokens)
                reward = self.env.compute_reward(problem, completion_text)
                if reward != target_reward:
                    continue

            # Prefix dedup (intra-batch — the validator also enforces cross-miner).
            prefix = tuple(completion_tokens[:DIVERSITY_PREFIX_LEN])
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            completions.append(
                {
                    "tokens": all_tokens,
                    "prompt_length": prompt_length,
                    "completion_tokens": completion_tokens,
                }
            )

        return completions

    def _build_completion_submission(
        self, generation: dict, randomness: str
    ) -> CompletionSubmission:
        """Construct a GRAIL-proven CompletionSubmission from a generation dict.

        Reproduces the proof construction from the original _generate_and_prove:
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

        commit = {
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

        return CompletionSubmission(tokens=all_tokens, commit=commit)
