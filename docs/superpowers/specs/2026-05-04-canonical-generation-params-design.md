# Canonical generation params + grail-style DistributionValidator — design

**Date:** 2026-05-04
**Status:** approved (awaiting plan)
**Implementation order:** ships AFTER `2026-05-04-grail-validators-additions-design.md`

## Context

The GRAIL protocol can cryptographically prove "the miner ran this model on these tokens" but cannot cryptographically prove "the miner sampled these tokens at temperature T". Sampling-recipe enforcement is inherently statistical: the validator's `DistributionValidator` reconstructs the chosen-token distribution at the canonical sampling params and rejects rollouts whose statistics deviate too far from honest sampling.

Catalyst today reads sampling params (`T_PROTO`, `TOP_P_PROTO`, `TOP_K_PROTO`) from hardcoded constants in `reliquary/constants.py`. This works because there is exactly one set of values network-wide, but it has two problems:

1. **No source of truth attached to the checkpoint.** If sampling params ever need to change per-checkpoint version, both miners and validators must coordinate a code release. Today this is impossible to do safely.
2. **Repetition penalty is implicit.** Catalyst has no `REP_PENALTY_PROTO` constant; the miner's `vllm.generate()` call doesn't pass a `repetition_penalty` argument, so vLLM defaults to 1.0 (no-op). The validator's `evaluate_token_distribution` doesn't model rep_penalty either. The day someone wants to bump the value, both sides need surgery in lockstep.

Upstream grail solves both by attaching `generation_params` to the published checkpoint metadata. Both sides read from the same canonical source. This spec replicates that pattern in Catalyst.

## Goals

- A single canonical source for sampling params (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`), attached to the checkpoint at publish time.
- Miner reads these params and uses them in `vllm.generate()`. No more hardcoded constants in the miner generation call.
- Validator's `evaluate_token_distribution` reads the same params and applies `RepetitionPenaltyLogitsProcessor` + `TemperatureLogitsWarper` (grail's drift-safe processor subset) before computing chosen-token probability.
- Future protocol bumps (e.g. enabling rep_penalty=1.1) require zero code changes outside `constants.py` — the new value is published in the next checkpoint and both sides pick it up automatically.

## Non-goals

- Adding the `min_low` and `q10_low_initial` distribution rules from grail (4a/4b in the brainstorm). Tracked in backlog for a separate pass once we observe miner behaviour.
- Soft-accumulation severity framework (a single suspicious rollout still rejects the whole submission). Out of scope.
- Crypto-hard sampling-recipe enforcement (e.g. miner ships `token_logprobs` post-sampling-recipe so any deviation is detected by `LogprobValidator`). Tracked separately as a deeper protocol change.
- Changing the `top_p` / `top_k` handling in the validator. Even when activated by the protocol, these processors are explicitly NOT applied in the distribution check — they set logits to `-inf` and the cutoff boundary drifts between miner prefill and validator decode in bf16, causing false positives on honest miners. This matches grail's design.

## Severity

Unchanged. The `DistributionValidator` keeps emitting `DISTRIBUTION_SUSPICIOUS` and stays a hard fail-fast check (consistent with item 1 brainstorm decision in the previous spec).

## Mechanism

### Storage location

`generation_params.json` lives inside the HuggingFace snapshot folder, alongside `model.safetensors`, `config.json`, and tokenizer files. Schema:

```json
{
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "max_tokens": 8192
}
```

### Why this location

- **Integrity is free.** HuggingFace's revision SHA covers every byte in the snapshot. The validator's existing signature `(checkpoint_n || revision)` already binds the file indirectly — tampering with `generation_params.json` changes the revision and invalidates the signature.
- **No protocol change.** `ManifestEntry` (the on-chain commit miners see via `/state`) is unchanged. No new field, no new signature scheme.
- **Symmetric access.** Miner and validator both call `snapshot_download(repo_id, revision)` and get a `local_path`. Reading one extra file is `json.load(open(local_path / "generation_params.json"))`.

### Rejected alternatives

- **`ManifestEntry.generation_params: dict`** — requires extending the on-chain commit message and the `/state` payload. Invasive for marginal benefit (the integrity is already there via revision SHA).
- **HF model card YAML metadata** — rigid format with no slot for sampling params; would have to be fetched separately from `snapshot_download`.

## Constants

New in `reliquary/constants.py`:

```python
# Repetition penalty applied during sampling. 1.0 = no-op.
# When changed, both miner generation and validator distribution check
# pick up the new value via generation_params.json on the next published
# checkpoint — no code coordination required.
REP_PENALTY_PROTO = 1.0
```

Existing constants `T_PROTO`, `TOP_P_PROTO`, `TOP_K_PROTO` are unchanged in value, but the validator's publish path becomes the only place that reads them — runtime code (miner generation, validator distribution check) now reads from `generation_params.json` instead. `MAX_NEW_TOKENS_PROTOCOL_CAP` keeps two roles: written into `gen_params["max_tokens"]` at publish time, and also used as the upper bound of `CommitModel.rollout.completion_length` (from the previous spec) so a miner cannot lie above the protocol cap.

## Validator side

### Publish path (`reliquary/validator/checkpoint.py`)

`_default_save_hf_format` writes `generation_params.json` into the snapshot directory before upload:

```python
def _default_save_hf_format(model, tokenizer, path):
    import json
    from reliquary.constants import (
        T_PROTO, TOP_P_PROTO, TOP_K_PROTO,
        REP_PENALTY_PROTO, MAX_NEW_TOKENS_PROTOCOL_CAP,
    )
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    with open(path / "generation_params.json", "w") as f:
        json.dump({
            "temperature":        T_PROTO,
            "top_p":              TOP_P_PROTO,
            "top_k":              TOP_K_PROTO,
            "repetition_penalty": REP_PENALTY_PROTO,
            "max_tokens":         MAX_NEW_TOKENS_PROTOCOL_CAP,
        }, f, sort_keys=True, indent=2)
```

### Validator model load (`reliquary/validator/service.py`)

The validator's `_default_load_model` (line ~103) reads the file and attaches it to the model object:

```python
def _default_load_model(local_path: str) -> Any:
    import json
    from pathlib import Path
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(local_path, ...)
    with open(Path(local_path) / "generation_params.json") as f:
        model.gen_params = json.load(f)
    return model
```

If the file is missing or malformed, the load fails with an explicit `FileNotFoundError` / `JSONDecodeError`. No graceful degradation — see "Strict fallback" below.

### Distribution validator (`reliquary/validator/verifier.py`)

`evaluate_token_distribution` is rewritten to apply HF logits processors before computing chosen-token probability. New signature accepts `gen_params` and `model_device`:

```python
def evaluate_token_distribution(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    logits: torch.Tensor,
    gen_params: dict,
    model_device: torch.device,
) -> tuple[bool | None, dict]:
    """Soft check: chosen-token distribution consistent with sampling at canonical params.
    
    Applies the drift-safe subset of HF's sample-mode pipeline (rep_penalty + temperature)
    before reading the chosen-token probability. top_p / top_k are deliberately NOT
    applied — they create -inf logits whose cutoff boundary drifts between miner
    prefill and validator decode in bf16, causing false positives on honest miners.
    """
    from transformers import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
    )
    from reliquary.constants import (
        SAMPLING_HIGH_P, SAMPLING_LOW_P,
        SAMPLING_LOW_Q10_MAX, SAMPLING_MEDIAN_LOW_MAX,
        SAMPLING_MIN_STEPS,
    )
    
    if completion_length < SAMPLING_MIN_STEPS:
        return None, {}
    
    rep = float(gen_params["repetition_penalty"])
    temperature = float(gen_params["temperature"])
    
    processors_list = []
    if rep != 1.0:
        processors_list.append(RepetitionPenaltyLogitsProcessor(rep))
    if temperature != 1.0:
        processors_list.append(TemperatureLogitsWarper(temperature))
    processors = LogitsProcessorList(processors_list)
    
    full_input_ids = torch.tensor(
        tokens, dtype=torch.long, device=model_device
    ).unsqueeze(0)
    
    probs: list[float] = []
    for t in range(prompt_length, prompt_length + completion_length):
        if t == 0 or t - 1 >= logits.size(0) or t >= len(tokens):
            continue
        step_logits = processors(
            full_input_ids[:, :t],
            logits[t - 1].to(device=model_device, dtype=torch.float32).unsqueeze(0).clone(),
        )
        step_probs = torch.softmax(step_logits, dim=-1).squeeze(0)
        probs.append(float(step_probs[tokens[t]].item()))
    
    if len(probs) < SAMPLING_MIN_STEPS:
        return None, {}
    
    # ... existing metric computation (mean, median, q10, low_frac, high_frac)
    # Suspicion rules unchanged in this spec — only median_low + q10_low.
    # min_low and q10_low_initial deferred to a later spec.
```

### Batcher (`reliquary/validator/batcher.py`)

`_accept_locked` already calls `evaluate_token_distribution` at line 297. The call needs three changes:

1. Pass `gen_params=self.model.gen_params` instead of the hardcoded `temperature=T_PROTO`
2. Pass `model_device=next(self.model.parameters()).device`
3. Drop the `from reliquary.constants import T_PROTO` import inside the function

The `GrpoWindowBatcher` doesn't need a new constructor parameter — it already holds `self.model`, and `gen_params` lives on it after the load.

## Miner side

### Checkpoint load (`reliquary/miner/engine.py`)

`_load_checkpoint` (line ~292) reads `generation_params.json` and attaches it to both `hf_model` and `vllm_model`. Both attribute names use `gen_params` for symmetry with the validator:

```python
def _load_checkpoint(self, local_path: str):
    import json
    from pathlib import Path
    from transformers import AutoModelForCausalLM
    
    if getattr(self, "_loaded_checkpoint_path", None) == local_path:
        return
    
    new_hf = AutoModelForCausalLM.from_pretrained(local_path, ...)
    new_gen = AutoModelForCausalLM.from_pretrained(local_path, ...)
    
    with open(Path(local_path) / "generation_params.json") as f:
        gen_params = json.load(f)
    new_hf.gen_params = gen_params
    new_gen.gen_params = gen_params
    
    self.hf_model = new_hf
    self.vllm_model = new_gen
    self._loaded_checkpoint_path = local_path
```

### Generation call (`reliquary/miner/engine.py:386-395`)

The hardcoded `T_PROTO`, `TOP_P_PROTO`, `TOP_K_PROTO` constants in the `vllm_model.generate()` call are replaced with reads from `self.vllm_model.gen_params`:

```python
gp = self.vllm_model.gen_params
outputs = self.vllm_model.generate(
    input_tensor,
    max_new_tokens=gp["max_tokens"],     # was self.max_new_tokens
    do_sample=True,
    temperature=gp["temperature"],        # was T_PROTO
    top_p=gp["top_p"],                    # was TOP_P_PROTO
    top_k=gp["top_k"],                    # was TOP_K_PROTO
    repetition_penalty=gp["repetition_penalty"],  # NEW (was implicitly 1.0)
    pad_token_id=self.tokenizer.pad_token_id,
)
```

The `max_new_tokens=MAX_NEW_TOKENS_PROTOCOL_CAP` change from the previous spec stays — but the constant is now read indirectly via `gen_params["max_tokens"]`. The previous spec's removal of the `RELIQUARY_MAX_NEW_TOKENS` env-var override stands.

## Strict fallback

If `generation_params.json` is missing from a checkpoint snapshot:

- **Validator load:** `FileNotFoundError` propagates → service refuses to start with that checkpoint. The validator stays on its previous checkpoint (or fails to start cold). Operator alert.
- **Miner load:** `FileNotFoundError` propagates → mining loop fails on the next checkpoint rotation. Operator alert.

No silent fallback to constants. The deployment order is:
1. Deploy validator with new publish code
2. Wait for next checkpoint publish (file is now present in HF)
3. Deploy miner with new read code

If miners deploy before step 2 they will fail to load against the previous checkpoint — but since the previous spec already coordinates a release for the Schema/Token/Termination changes, the timing is the operator's normal release management problem, not a code design problem.

## Files affected

### Modified

- `reliquary/constants.py` — add `REP_PENALTY_PROTO = 1.0`
- `reliquary/validator/checkpoint.py` — write `generation_params.json` in `_default_save_hf_format`
- `reliquary/validator/service.py` — read `generation_params.json` in `_default_load_model`, attach to `model.gen_params`
- `reliquary/validator/verifier.py` — rewrite `evaluate_token_distribution` signature + body to read `gen_params` and apply HF processors
- `reliquary/validator/batcher.py` — update `evaluate_token_distribution` call site (drop hardcoded `temperature=T_PROTO`, pass `gen_params=self.model.gen_params`, pass `model_device`)
- `reliquary/miner/engine.py` — read `generation_params.json` in `_load_checkpoint`, replace hardcoded sampling constants in `vllm_model.generate()` call

### New tests

- Validator publish: `_default_save_hf_format` writes `generation_params.json` with the canonical constants
- Validator load: missing `generation_params.json` raises `FileNotFoundError`
- Validator load: malformed JSON raises `JSONDecodeError`
- Validator load: model object has `model.gen_params` populated correctly
- Distribution validator: `rep_penalty != 1.0` activates `RepetitionPenaltyLogitsProcessor` (mock processor, verify it's called)
- Distribution validator: `temperature != 1.0` activates `TemperatureLogitsWarper`
- Distribution validator: under canonical params and honest sampling, distribution check passes
- Distribution validator: under deviated sampling (e.g. simulated T=2.0 tokens), median_low triggers
- Miner: `_load_checkpoint` populates `vllm_model.gen_params` from the file
- Miner: generation call uses `gen_params` values (mock vLLM, verify call kwargs)

## Out of scope (separate specs)

- **`min_low` and `q10_low_initial` distribution rules** (4a/4b from the brainstorm). Tracked for a follow-up spec once we observe production miner behaviour and decide whether the basic median + q10 detection is sufficient.
- **Soft-accumulation severity framework** that would let one suspicious rollout out of M survive instead of killing the whole submission.
- **Crypto-hard sampling-recipe enforcement** — making the miner ship `token_logprobs` post-sampling-recipe so `LogprobValidator` catches any temperature deviation directly. Deeper protocol change.
- **Per-position ratio normalization** for the distribution check (`r_t = P_raw(chosen) / E_p_raw[t]`) — the proper fix for top_p/top_k drift mentioned in grail's TODO at `~/.claude/plans/federated-sparking-flute.md`. Only relevant if Catalyst ever sets `TOP_P_PROTO < 1.0` or `TOP_K_PROTO > 0`.

## Open questions

None at this point — all decisions confirmed in brainstorm.
