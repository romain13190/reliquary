# Reliquary

Bittensor subnet 81 (`netuid 81`, network `finney`).
A decentralized verifiable-inference market that generates on-policy GRPO
rollouts with cryptographic proofs.

Every completion produced by a miner carries a **GRAIL proof** — a compact
sketch commitment that lets validators re-run the forward pass and confirm
the generation came from the announced model. The resulting dataset is
usable as training data without trusting any individual miner.

---

## What Reliquary does today

Every ~60 s window the network produces up to **256 verified rollouts**
(8 prompts × 32 completions/slot) from a pinned reference checkpoint.
The cycle:

1. **Beacon** → miners and validators derive 8 identical prompts from
   `drand ⊕ block_hash`.
2. **Rollout market** → miners compete per slot. Each validator exposes
   a live reward histogram at `/window/{n}/state`; miners read it and
   target the under-represented reward class to maximise their payout.
   Each batch is atomically verified (GRAIL proof + diversity +
   signature) or rejected as a whole.
3. **Settlement** → the slot freezes at 32 accepted completions or 60 s,
   whichever comes first. No quota is enforced during collection —
   balance is a market outcome, not a rule.
4. **Publication** → the full dataset (prompts + completions + rewards)
   is gzipped and uploaded to R2 at `reliquary/dataset/window-{n}.json.gz`.
5. **Weights** → every 72 windows (~72 min) the validator aggregates
   per-miner scores (superlinear exponent 4 for sybil resistance) and
   routes the unclaimed budget to `UID_BURN` on chain.

The deliverable **today** is the stream of verified GRPO-ready datasets.
A downstream trainer — any party, any model — can consume it.

---

## The mechanism

Reliquary does not pay miners for *work*. It pays them for **signal** —
the GRPO-usable information each completion brings to its slot. The
scoring rule applied after a slot freezes is:

```
score(c) = |c.reward − mean_slot| / std_slot
```

Three consequences fall out of that formula, and together they are the
core of the design:

**Self-balancing without a quota.** A slot that already has 28 corrects
and 4 wrongs pays ~5× more per wrong than per correct. Miners read
`/state` live and pivot toward whatever class is scarce. The 50/50
equilibrium emerges from the incentive; it is never enforced by a cap.

**Collective burn as a coordination device.** A degenerate slot
(`std = 0`, e.g. {32, 0}) pays *zero* to everyone — even the miners who
submitted valid completions. The emission share that a balanced slot
would have produced is explicitly routed to `UID_BURN`. A miner cannot
monopolise a class without destroying their own payout, which turns
diversification into a Nash equilibrium rather than a good deed.

**Fixed budget, not re-normalised.** The notional budget per window is
`PROMPTS_PER_WINDOW × GROUP_SIZE = 256` units of |advantage|. What
miners fail to claim *burns*; it is not redistributed to the miners who
did show up. The economic weight of each completion stays stable across
windows regardless of how many miners compete, and lazy windows
genuinely shrink supply instead of inflating the remaining participants.

**Signal, not tokens.** The |z-score| formulation is the information
content of a completion in a GRPO update. The R2 dataset a downstream
trainer consumes is already filtered and weighted by learning value —
the market prices exactly what the trainer would have priced anyway.

---

## Flow at a glance

```
                   drand beacon + block hash
                              │
                              ▼
                8 deterministic prompts per window
                              │
   ┌──────────────────────────┴──────────────────────────┐
   │                                                     │
   ▼                                                     ▼
Miner (2 GPUs)                                    Validator (1 GPU)
─ Same 8 prompts                                  ─ Same 8 prompts
─ Read /state, pick rare                          ─ Serve HTTP /submit
  reward class per slot                           ─ For each batch:
─ Generate N completions                              • verify GRAIL proof
─ Build GRAIL proof (HF                               • check prefix-distinct
  forward + commitments)                              • append to slot
─ POST /submit  ─────────────────────▶            ─ Auto-finalize at 32
                                                    or timeout 60 s
                                                  ─ Score by |advantage|
                                                  ─ Upload dataset to R2
                                                  ─ 72 windows →
                                                    set_weights on-chain
```

Per-slot cap: `GROUP_SIZE = 32`. No per-class quota — the slot is a free
market. Advantage scoring steers miners toward balance post-hoc.

---

## Roadmap

### v1 — Verifiable inference *(current)*

Miners produce verified rollouts on a **frozen reference checkpoint**
pinned by the subnet. No training happens inside Reliquary yet — the
published R2 dataset is the product. External consumers (research labs,
RL teams) pull the stream.

**Status:** stabilising. Shipped in the current code:
- Free-market slot settlement — no per-class quota, no hotkey dedup, only
  capacity + cross-miner prefix dedup.
- Advantage-based scoring over the full accepted set. Rare class pays
  proportionally; degenerate slots burn.
- Miner strategic targeting via live histogram on `/window/{n}/state`.
- R2 archive in GRPO-ready format.

**Remaining v1 gaps:**
- `model_name` in GRAIL commits is signed but not compared against a
  canonical reference. Low-impact while the checkpoint is frozen; must be
  closed before v2 opens.
- Offline-precompute of GSM8K answers is a theoretical attack on the
  fixed dataset. Self-resolves in v2 when the checkpoint rotates.

### v2 — On-policy training inside the subnet

The subnet **trains the reference checkpoint itself** from the rollouts
miners produce, then rotates the checkpoint on-chain. This turns
Reliquary into a real closed-loop, on-policy GRPO trainer — and the
training step itself is performed by the subnet's participants, not
outsourced to a third party.


**Open design questions** (to be resolved before v2 ships):
- **Who runs the training step, and how is consensus on the next
  checkpoint reached?** Candidates: every validator trains independently
  and the metagraph-weighted majority of checkpoint hashes wins; a
  rotating election based on stake or score; a deterministic split of
  GRPO gradients across participants who each compute a piece.
- **Cadence.** 1 step per window (≈ 4-minute checkpoint rotation) vs.
  accumulate *k* windows per step. Affects how fast miners must resync
  and how much on-policy drift is tolerable.
- **Model announcement protocol.** Today miners and validators
  coordinate checkpoints off-chain (CLI flag convention). v2 needs a
  formal channel — on-chain Bittensor subnet commitments, or a
  validator-signed `/current_model` HTTP endpoint — so that miners
  auto-rotate the moment training publishes a new checkpoint and the
  GRAIL sketch check starts rejecting stale-model rollouts.
- **Training-cost economics.** Whether validators running the training
  step earn an additional slice of emissions, or whether training is a
  precondition for `validator_permit`.

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
  environment/      prompt sources (gsm8k for v1)
  miner/            MiningEngine — generate + prove + submit
  validator/        WindowBatcher, HTTP server, scoring, weights
tests/unit/         277 tests covering all of the above
docs/               operator tutorials
```

---

## Status

Version `0.1.0` · Python ≥ 3.11 · CUDA 12 · MIT licence.

`GRAIL` in this repo refers specifically to the **cryptographic proof
system** used to bind completions to the reference model. The rest of
the project, the subnet, and the CLI are named Reliquary.
