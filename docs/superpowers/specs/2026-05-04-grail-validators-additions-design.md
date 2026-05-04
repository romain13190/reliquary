# GRAIL validators additions — design

**Date:** 2026-05-04
**Status:** approved (awaiting plan)
**Author:** brainstorm session

## Context

Catalyst's validator pipeline (`reliquary/validator/batcher.py:_accept_locked`) currently runs a per-rollout sequence of checks: prompt binding, proof_version, signature, GRAIL sketch verification, logprob median IS-deviation, and a basic distribution heuristic. Compared to upstream GRAIL's pipeline (`grail/grail/validation/pipeline.py`), Catalyst is missing three structural / behavioural checks that close real attack surfaces or make miner failure modes legible. This spec adds them.

A fourth grail check — applying HuggingFace logits processors (`RepetitionPenaltyLogitsProcessor`, `TemperatureLogitsWarper`) inside the distribution validator — was evaluated and deferred. Justification in the YAGNI section below.

## Goals

- Catch malformed commits before any GPU work, with a clear miner-facing reject reason.
- Catch wrong-tokenizer / wrong-vocab token submissions before they crash the validator's forward pass.
- Catch artificially-truncated rollouts (rollouts that don't end with EOS) — these carry the strongest forgery incentive in RL settings, since a miner can stop generation right before a low-quality continuation would tank the reward.
- Maintain the existing fail-fast / hard-severity model: no soft-accumulation framework introduced.

## Non-goals

- Introducing the soft / accumulated-failure mechanism that grail uses for `RewardValidator` and `DistributionValidator`. Catalyst stays fail-fast.
- Reworking the distribution validator. Items 4 (HF logits processors) is deferred — see YAGNI.
- Changing the protocol-level sampling constants (`T_PROTO`, `TOP_P_PROTO`, `TOP_K_PROTO`). They stay fixed network-wide.
- Designing the miner-side over-generate buffer required to absorb max-tokens-truncated rollouts under the strict TerminationValidator. That is a separate spec.
- Adding miner advantage transport in the submission protocol (would enable an `_validate_grpo_groups`-style consistency check, but is unrelated to the model-identity guarantees here).

## Severity decision

All three new validators are **hard** (immediate rejection of the whole `BatchSubmissionRequest`). Catalyst's batcher is entirely fail-fast today; introducing soft accumulation is a non-trivial refactor of `_accept_locked` and out of scope. Grail's calibrated thresholds (`MIN_EOS_PROBABILITY = 0.02` etc.) are already conservative enough to keep honest false-positive rates near zero.

## Final per-rollout pipeline

In `batcher._accept_locked`, inside `for rollout in request.rollouts:`:

```
1. CommitModel.model_validate(rollout.commit)         → BAD_SCHEMA       [NEW]
2. verify_tokens(commit.tokens, model.config)         → BAD_TOKENS       [NEW]
3. canonical_prompt_tokens binding                    → PROMPT_MISMATCH  [exists]
4. verify_signature                                   → BAD_SIGNATURE    [exists]
5. verify_commitment_proofs → ProofResult.logits      → GRAIL_FAIL       [exists]
6. verify_termination(commit, tokenizer, logits)      → BAD_TERMINATION  [NEW]
7. verify_logprobs_claim                              → LOGPROB_MISMATCH [exists]
8. evaluate_token_distribution                        → DISTRIBUTION_SUSPICIOUS [exists]
```

Cheap checks first, GPU forward in the middle, post-forward behavioural checks last (they reuse `proof.logits` so cost is amortised).

## Item 1 — TokenValidator

### What

Reject any rollout whose `commit["tokens"]` contains an out-of-vocabulary id, a negative id, a non-int element, an empty list, or a sequence longer than the model's max context length.

### How

The function `verify_tokens(tokens, model_config) -> bool` already exists in `reliquary/protocol/tokens.py:34` with exactly the right logic (vocab bounds + max-length + non-empty). It is currently dead code — no caller. Wire it into `batcher._accept_locked` as step 2 of the per-rollout loop.

```python
if not verify_tokens(rollout.commit["tokens"], self.model.config):
    return self._reject(RejectReason.BAD_TOKENS)
```

`verify_tokens` is called *after* SchemaValidator (which guarantees `tokens` is a non-empty `list[int]`) but *before* signature / GRAIL — order matters because a wrong-vocab token would crash `nn.Embedding` if it reached the forward pass.

### Reject reason

New enum value `BAD_TOKENS` in `RejectReason` (`reliquary/protocol/submission.py`). Granular telemetry: `reject_counts["bad_tokens"]` separates "miner with wrong tokenizer" from "miner that forges signatures" (`bad_signature`) — two different problems, two different fixes.

### Cleanup

Remove the redundant `seq_len > MAX_TOKENS_PER_ROLLOUT` check at `verifier.py:91-99`. It is fully covered by `_validate_sequence_length` inside `verify_tokens`. Keeping both means two places define the seq-length contract.

### Tests

Three new unit tests on the batcher:
- token id ≥ vocab_size → `BAD_TOKENS`
- token id < 0 → `BAD_TOKENS`
- empty token list → `BAD_TOKENS`

## Item 2 — SchemaValidator (Pydantic)

### What

Validate the structure of `rollout.commit` (currently `dict[str, Any]`) against a Pydantic v2 model. Catches malformed submissions before any GPU work, gives miners actionable error messages, and lets us delete several ad-hoc checks scattered across the codebase.

### How

Add three new Pydantic models to `reliquary/protocol/submission.py` (same logical layer as `BatchSubmissionRequest`, no new file):

```python
class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    layer_index: int

class BeaconInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    randomness: str = Field(..., pattern=r"^[0-9a-fA-F]+$")

class RolloutMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_length: int = Field(..., ge=0)
    completion_length: int = Field(..., gt=0, le=MAX_NEW_TOKENS_PROTOCOL_CAP)
    success: bool
    total_reward: float
    advantage: float
    token_logprobs: list[float]

class CommitModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens: list[int] = Field(..., min_length=CHALLENGE_K)  # >= 32
    commitments: list[dict]
    proof_version: Literal["v5"]                              # ties to GRAIL_PROOF_VERSION
    model: ModelInfo
    signature: str = Field(..., pattern=r"^[0-9a-fA-F]+$")
    beacon: BeaconInfo
    rollout: RolloutMetadata

    @field_validator("commitments")
    @classmethod
    def _commitments_len_matches_tokens(cls, v, info):
        tokens = info.data.get("tokens", [])
        if len(v) != len(tokens):
            raise ValueError(
                f"commitments length {len(v)} must equal tokens length {len(tokens)}"
            )
        return v

    @field_validator("rollout")
    @classmethod
    def _lengths_consistent(cls, v, info):
        tokens = info.data.get("tokens", [])
        if v.prompt_length + v.completion_length != len(tokens):
            raise ValueError(
                f"prompt_length({v.prompt_length}) + completion_length({v.completion_length}) "
                f"must equal len(tokens)={len(tokens)}"
            )
        if len(v.token_logprobs) != len(tokens):
            raise ValueError(
                f"token_logprobs length {len(v.token_logprobs)} must equal tokens length {len(tokens)}"
            )
        return v
```

### Invocation strategy

**Explicit validation at the top of the per-rollout loop**, not by changing `RolloutSubmission.commit` to `CommitModel`. Rationale:

If we made `RolloutSubmission.commit: CommitModel`, FastAPI would reject malformed payloads with a 422 *before* the batcher sees them. We would lose the line in `reject_counts` (R2 telemetry archive) and the miner would receive a generic HTTP error instead of `BatchSubmissionResponse(accepted=False, reason=BAD_SCHEMA)`. Keeping the explicit `CommitModel.model_validate(rollout.commit)` call inside `_accept_locked` preserves the existing reject-counts flow and gives miners the same response shape as every other failure.

```python
try:
    CommitModel.model_validate(rollout.commit)
except ValidationError:
    return self._reject(RejectReason.BAD_SCHEMA)
```

### `extra="forbid"`

Consistent with the outer envelope (`BatchSubmissionRequest`, `RolloutSubmission` both use `extra="forbid"`). Any unknown key in `commit` is rejected. This prevents protocol drift — a miner that ships extra fields gets rejected immediately, forcing a clean protocol bump when fields change.

### Reject reason

New enum value `BAD_SCHEMA` in `RejectReason`.

### Cleanup

- Remove `verify_proof_version()` in `verifier.py:53-55`. Covered by `Literal["v5"]` on the Pydantic model.
- Remove the `len(commitments) != seq_len` check in `verifier.py:80-88`. Covered by `_commitments_len_matches_tokens` field validator.
- Remove the `verify_proof_version_fn` injection in `GrpoWindowBatcher.__init__` (lines 92, 133-135, 139, 256-257) — the function it called no longer exists.

### Tests

Five new unit tests on `CommitModel.model_validate`:
- missing required key → `ValidationError` (e.g. `tokens` absent)
- `proof_version != "v5"` → `ValidationError`
- `len(commitments) != len(tokens)` → `ValidationError`
- `prompt_length + completion_length != len(tokens)` → `ValidationError`
- unknown extra key → `ValidationError` (forbid)

Plus one batcher integration test: malformed commit returns `BAD_SCHEMA`.

## Item 3 — TerminationValidator (strict EOS-only)

### What

Reject any rollout whose token sequence does not end with the tokenizer's EOS token, with that EOS having a model-predicted probability above `MIN_EOS_PROBABILITY` at the second-to-last position.

### Why strict (no max-tokens fallback)

In RL environments where reward depends on the model parsing/producing a final answer in a specific format (boxed math answer, code block, JSON output), a rollout that hits `max_new_tokens` without producing the final answer scores 0 — it is "wasted" generation. There is no legitimate reason for a healthy rollout to hit the hard cap; the model should sample EOS naturally when its reasoning is complete.

The threat model: a miner generates, then truncates the sequence at a position where the partial output looks correct (e.g., right after `\boxed{42}` but before whatever incorrect continuation the model would have produced) and submits the truncated version. Strict EOS-only termination closes this — the miner cannot pick where to stop, the model has to sample EOS at the end.

`MAX_NEW_TOKENS_PROTOCOL_CAP = 8192` stays in place as a hard OOM/DoS guard on the validator's forward pass, but is no longer accepted as a valid termination reason.

### How

New function in `reliquary/validator/verifier.py`:

```python
def verify_termination(commit: dict, tokenizer: Any, logits: torch.Tensor) -> bool:
    """Hard check: rollout must end with EOS token at p(EOS) >= MIN_EOS_PROBABILITY.

    Reuses the per-token logits already cached from verify_commitment_proofs,
    so this is O(vocab) on a single position — no extra forward pass.
    """
    tokens = commit["tokens"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return False
    if tokens[-1] != eos_id:
        return False
    # logits[-2] is the distribution that produced tokens[-1] (the EOS).
    p_eos = float(torch.softmax(logits[-2].float(), dim=-1)[eos_id].item())
    return p_eos >= MIN_EOS_PROBABILITY
```

Called as step 6 of the per-rollout loop in `batcher._accept_locked`, after `verify_commitment_proofs` returns the cached logits:

```python
proof = self._verify_commitment(rollout.commit, self.model, self.randomness)
if not proof.all_passed:
    return self._reject(RejectReason.GRAIL_FAIL)

if proof.logits.numel() > 0:  # skip stub backward-compat
    if not verify_termination(rollout.commit, self.tokenizer, proof.logits):
        return self._reject(RejectReason.BAD_TERMINATION)
```

The batcher does not currently hold a tokenizer reference — needs to be injected via `__init__` (the validator service already has one for canonical prompt token derivation).

### Constants

New in `reliquary/constants.py`:

```python
# Minimum probability the model must have assigned to EOS at the position
# that produced it. Below this threshold, the rollout is presumed to be
# artificially truncated. Calibrated by upstream grail at 0% honest FP.
MIN_EOS_PROBABILITY = 0.02
```

### Miner-side change

Remove the `RELIQUARY_MAX_NEW_TOKENS` env-var override in `reliquary/miner/engine.py:153`:

```python
# before
max_new_tokens: int = int(
    os.environ.get("RELIQUARY_MAX_NEW_TOKENS", MAX_NEW_TOKENS_PROTOCOL_CAP)
),
# after
max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
```

The env-var override has no production use case — a miner that lowers their cap below the protocol value will see truncated rollouts get rejected as `BAD_TERMINATION`. The cap stays at `MAX_NEW_TOKENS_PROTOCOL_CAP = 8192` purely as an OOM guard.

### Reject reason

New enum value `BAD_TERMINATION` in `RejectReason`.

### Tests

Three new unit tests:
- last token is not EOS → `BAD_TERMINATION`
- last token is EOS but `p(EOS) < MIN_EOS_PROBABILITY` (synthetic logits) → `BAD_TERMINATION`
- last token is EOS and `p(EOS) >= MIN_EOS_PROBABILITY` → accept

## Item 4 — HF logits processors in distribution check (DEFERRED)

### Decision: do not add now

Catalyst's `evaluate_token_distribution` currently does plain `softmax(logits / T_PROTO)`. Grail's `DistributionValidator` additionally applies `RepetitionPenaltyLogitsProcessor` and `TemperatureLogitsWarper` from HuggingFace's sampling pipeline before reading the chosen-token probability.

Under Catalyst's current protocol constants (`reliquary/constants.py:180-184`):

```
T_PROTO = 0.9
TOP_P_PROTO = 1.0    # no top-p truncation
TOP_K_PROTO = 0      # no top-k truncation
# (no REP_PENALTY_PROTO defined → implicitly 1.0)
```

The miner's effective sampling pipeline is "temperature 0.9 + full distribution". Catalyst's current `softmax(logits / T_PROTO)` produces results mathematically identical to what the miner sampled from. Importing `RepetitionPenaltyLogitsProcessor` to apply `penalty=1.0` is gratuitous complexity — it computes the same value we already compute.

### Wake-up triggers (track as a GitHub issue)

Three protocol changes would force this decision to be revisited:

1. **`T_PROTO` becomes per-checkpoint variable.** The validator would need to read the temperature from checkpoint metadata or from the commit, and `evaluate_token_distribution` would need to take it as a parameter.
2. **`REP_PENALTY_PROTO != 1.0`.** This is the actual reason grail's design exists. The validator must apply `RepetitionPenaltyLogitsProcessor` because the miner's chosen-token distribution is no longer just a temperature scaling.
3. **`TOP_P_PROTO < 1.0` or `TOP_K_PROTO > 0`.** ⚠️ The grail design *cannot* be reused here. Top-k / top-p set logits to `-inf`, and the cutoff boundary drifts between miner prefill and validator decode in bf16. Grail explicitly skipped these processors for that reason. Catalyst would need a custom distribution-validation strategy (e.g., per-position ratio normalization `r_t = P_raw(chosen) / E_p_raw[t]`).

These triggers are not silent: changing any of them would cause `evaluate_token_distribution` reject rates to spike on honest miners, making the regression obvious on the validator dashboard.

## Files affected

### New code

- `reliquary/protocol/submission.py` — add `CommitModel`, `ModelInfo`, `BeaconInfo`, `RolloutMetadata`, three new `RejectReason` values
- `reliquary/validator/verifier.py` — add `verify_termination()`
- `reliquary/validator/batcher.py` — wire SchemaValidator, TokenValidator, TerminationValidator into `_accept_locked`; inject tokenizer in `__init__`
- `reliquary/constants.py` — add `MIN_EOS_PROBABILITY = 0.02`

### Removed code

- `reliquary/validator/verifier.py` — `verify_proof_version()`, `seq_len > MAX_TOKENS_PER_ROLLOUT` check inside `verify_commitment_proofs`, `len(commitments) != seq_len` check inside `verify_commitment_proofs`
- `reliquary/miner/engine.py` — `RELIQUARY_MAX_NEW_TOKENS` env-var override (replaced by hardcoded `MAX_NEW_TOKENS_PROTOCOL_CAP`)
- `reliquary/validator/batcher.py` — `verify_proof_version_fn` injection parameter and field

### New tests

Approximately 12 new unit tests (3 for tokens, 6 for schema, 3 for termination), plus update of any existing batcher tests whose stub commits no longer pass the new schema gate.

## Out of scope (separate specs)

- Miner-side over-generate buffer to absorb max-tokens-truncated rollouts. With strict TerminationValidator, a single max-tokens hit in M rollouts rejects the whole submission. The miner needs to over-generate (e.g. M+2 rollouts, drop non-EOS, keep first M EOS-terminated) to maintain economic viability. Sketch: bump `M_ROLLOUTS` worth of generation to `M_ROLLOUTS + buffer` in `_generate_rollouts`, filter post-generation, fall back to single re-generation if still under M.
- Soft-accumulation severity framework (would let one bad rollout out of M not kill the whole submission).
- Miner-shipped `advantage` for grail's `_validate_grpo_groups`-style consistency check.

## Open questions

None at this point — all design decisions confirmed in brainstorm.
