# Reliquary

Bittensor subnet 81 (`netuid 81`, network `finney`).
A decentralized verifiable-inference market that generates on-policy GRPO
rollouts with cryptographic proofs.

Every completion produced by a miner carries a **GRAIL proof** — a compact
sketch commitment that lets validators re-run the forward pass and confirm
the generation came from the announced model. The resulting dataset is
usable as training data without trusting any individual miner.

---

## How it works (v2 GRPO market)

Each window (~60 s, 1 training step):

1. Miners pick a `prompt_idx` from the env (GSM8K, ~7473 problems) —
   avoiding the validator's published cooldown set.
2. Each miner generates 8 rollouts at `T_PROTO` (≈ 0.9), computes local
   rewards, builds GRAIL commits, and POSTs a `BatchSubmissionRequest`
   to the validator.
3. Validator verifies in order: signature → prompt_idx in range → round
   freshness → prompt not in cooldown → reward claims match
   `env.compute_reward` → GRAIL sketch matches model forward-pass
   → `k ∈ [2, 6]` (approvable zone).
4. At window close, validator selects the first `B = 8` in-zone
   submissions (FIFO by signed drand round, distinct prompts) for the
   training batch. Sealed batch becomes the GRPO step input; each
   batched prompt enters a 50-window cooldown (or 10 during bootstrap).
5. Weights: each batch member earns `1/B` of window emission; unused
   slots burn to `UID_BURN`.

Miners who submit outside the zone, on cooled-down prompts, or too slow
to make the batch earn **zero**. The speed + distinct-prompt
competition removes any incentive for cherry-picking (cherry-picking
takes extra compute → later `signed_round` → displaced from the batch).

Full design rationale: `docs/superpowers/specs/2026-04-20-grpo-market-design.md`.

---

## Flow at a glance

```
                   drand beacon
                        │
                        ▼
            Miner picks prompt_idx freely
            (avoids cooldown set from /window/{n}/state)
                        │
   ┌────────────────────┴────────────────────┐
   │                                         │
   ▼                                         ▼
Miner (2 GPUs)                         Validator (1 GPU)
─ Pick prompt_idx ∉ cooldown           ─ Serve HTTP /submit
─ Generate M=8 rollouts at T_PROTO     ─ Full verification pipeline:
─ Compute local rewards                    • signature
─ Build GRAIL proof (HF forward            • prompt_idx range
  + sketch commits)                        • round freshness
─ POST BatchSubmissionRequest ────▶        • cooldown check
                                           • reward match
                                           • GRAIL sketch match
                                           • zone check k ∈ [2,6]
                               ─ Batch: first B=8 distinct-prompt
                                 in-zone submissions (FIFO)
                               ─ Seal batch → GRPO training step
                               ─ Update cooldown map
                               ─ Upload dataset to R2
                               ─ Every WEIGHT_SUBMISSION_INTERVAL →
                                 set_weights on-chain (flat 1/B)
```

---

## Roadmap

### v1 — Verifiable inference *(superseded)*

Miners produced verified rollouts on a frozen reference checkpoint pinned
by the subnet. 8 validator-derived prompts per window, 32 completions per
slot, advantage-based scoring.

### v2 — On-policy training inside the subnet *(current)*

The subnet trains the reference checkpoint itself from the rollouts miners
produce, then rotates the checkpoint on-chain. Miners freely pick prompts,
compete on speed and novelty, and earn flat `1/B` for each batch slot they
fill.

**What shipped:**
- Free prompt selection — miners pick any `prompt_idx`, no validator-derived
  slots.
- Zone filter `k ∈ [2, 6]` — requires at least 2 correct and 2 incorrect
  rollouts per submission; degenerate sets are rejected.
- `CooldownMap` — batched prompts enter a 50-window cooldown (10 during
  bootstrap) so the training corpus stays diverse.
- Flat `1/B` weight per batch member; unused slots burn to `UID_BURN`.
- Bootstrap mode — first `BOOTSTRAP_WINDOWS` windows use wider zone [1, 7]
  and a shorter 10-window cooldown to fill the first batches faster.
- GRPO training step runs on the sealed batch each window; checkpoint
  rotates on-chain after each step.

### v3 — Open inference-market API

External users — researchers, AI startups doing RL — bring **their own
environment** (prompts + reward function) and the subnet produces
rollouts on demand at a published price. No GPU operations on the user
side, cryptographic receipts of every generation, and a decentralized
fallback when no single provider is willing or able.

This generalises Reliquary from "verifiable RL data factory for one
pinned problem (GSM8K)" to "verifiable RL data factory for any problem
the user can specify". It's the natural endpoint of v1 → v2: once the
subnet trains itself and proves its outputs, the same infrastructure
serves external workloads.

**Open design questions:**
- API surface: REST, gRPC, or Bittensor-native dendrite.
- Economic model: per-rollout TAO pricing, credits, or subscription.
- Env contract: how users ship arbitrary reward code safely to miners
  (sandboxing, resource limits, trust model).
- Isolation between the subnet's own v2 training jobs and external v3
  jobs running on the same miner GPUs.

---

## Running it

- **Miner** (2 GPUs) → [`docs/mining.md`](docs/mining.md)
- **Validator** (1 GPU + R2) → [`docs/validating.md`](docs/validating.md)

---

## Repository layout

```
reliquary/
  cli/              reliquary mine / reliquary validate
  protocol/         GRAIL proof construction, signatures, submission schemas
  shared/           forward_single_layer — the bit-identical forward pass
  infrastructure/   drand client, Bittensor chain, R2 storage
  environment/      prompt sources (gsm8k)
  miner/            MiningEngine — generate + prove + submit
  validator/        GrpoWindowBatcher, HTTP server, cooldown, weights
tests/unit/         covers all of the above
docs/               operator tutorials
```

---

## Status

Version `0.2.0` · Python ≥ 3.11 · CUDA 12 · MIT licence.

`GRAIL` in this repo refers specifically to the **cryptographic proof
system** used to bind completions to the reference model. The rest of
the project, the subnet, and the CLI are named Reliquary.
