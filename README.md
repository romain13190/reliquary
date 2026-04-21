# Reliquary

Bittensor subnet 81 (`netuid 81`, network `finney`).
A decentralized verifiable-inference market that generates on-policy GRPO
rollouts with cryptographic proofs.

Every completion produced by a miner carries a **GRAIL proof** — a compact
sketch commitment that lets validators re-run the forward pass and confirm
the generation came from the announced model. The resulting dataset is
usable as training data without trusting any individual miner.

---

## How it works (v2.1 batch-driven training)

Each window is one training step. The cadence is event-driven, not
time-based: a window seals the instant 8 valid distinct-prompt
non-cooldown submissions land.

1. Validator opens a window. `/window/state` exposes `state = OPEN`
   and the current `checkpoint_n`, `checkpoint_url`, `checkpoint_hash`.
2. Miners poll `/window/state`. If `checkpoint_n` advanced, they
   download `checkpoint_url` and load the new weights.
3. Miner picks a `prompt_idx` (skipping cooldown prompts), generates 8
   rollouts at protocol-fixed `T_PROTO=0.9`, computes local rewards,
   builds GRAIL commits, and POSTs a `BatchSubmissionRequest` — the
   payload includes `checkpoint_hash` so stale-checkpoint submissions
   fail fast.
4. Validator verifies: checkpoint_hash → window match → prompt_idx bounds
   → round freshness → cooldown → reward claims → zone (`k ∈ [2, 6]`) →
   GRAIL sketch. Any failure returns a `RejectReason`.
5. The instant B=8 valid distinct-prompt submissions land, the batcher
   fires `seal_event`. Validator transitions state to `TRAINING` and
   runs one GRPO step (stub in v2.1; real loss plugs into
   `validator/training.py`).
6. Validator transitions state to `PUBLISHING`, writes the new
   checkpoint, uploads to R2, signs `(checkpoint_n || file_hash)` with
   ed25519, updates the `/checkpoint` manifest.
7. State → `READY`. Next window opens. Batched prompts enter a 50-window
   cooldown.
8. Every `ROLLING_WINDOWS` windows (default 72), weights are submitted
   on-chain: each batch member of the interval earns `1/B` of its window
   share, unused slots burn to `UID_BURN`.

Safety net: if a window collects fewer than B submissions,
`WINDOW_TIMEOUT_SECONDS` (default 600 s) fires and the validator seals
the partial batch — unused slots still burn.

Full design rationale:
`docs/superpowers/specs/2026-04-21-batch-driven-windows-design.md`.

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

### v2.1 — Batch-driven windows *(current)*

The subnet trains the reference checkpoint itself from the rollouts miners
produce, then rotates the checkpoint on-chain. Miners freely pick prompts,
compete on speed and novelty, and earn flat `1/B` for each batch slot they
fill. Windows are event-driven — a window seals the instant B=8 valid
distinct-prompt submissions land, with a 600 s timeout safety net.

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
- Batch-driven seal: `seal_event` fires on the B-th distinct valid
  submission, not on a timer. `WINDOW_TIMEOUT_SECONDS = 600` is the
  safety net for low-traffic conditions.
- `checkpoint_hash` in every `BatchSubmissionRequest` — stale-checkpoint
  submissions are rejected immediately with `WRONG_CHECKPOINT`.
- Checkpoint published to R2 under
  `reliquary/checkpoints/{validator_hotkey}/{n}.safetensors`, signed by
  the validator hotkey; `/checkpoint` endpoint returns the manifest.
- Validator state (`window_n`, `checkpoint_n`) persisted to
  `reliquary/state/checkpoint.json` — survives restarts.
- GRPO training step stub in `reliquary/validator/training.py`; real loss
  plugs in with no protocol change.

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
