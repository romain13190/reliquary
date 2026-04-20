# GRPO Market — Reliquary v2 Incentive Redesign

Date: 2026-04-20
Status: Design, pending user review
Scope: Replaces v1's slot-based advantage scoring with a free-prompt market
geared for in-subnet GRPO training.

## Problem

v1 incentivises per-completion advantage (rare-reward-class pays more within
a fixed slot). That was fine for "verifiable inference as dataset", but it
biases the rollout distribution:

- Miners strategically submit rare classes instead of honest samples.
- The collected dataset no longer matches π(·|prompt); empirical variance
  inside each slot is inflated by selection.
- For v2 (training inside the subnet), GRPO needs rollouts that **are**
  honest samples from the current policy, otherwise the gradient estimator
  is biased.

The v1 scoring and the v2 training goal are in direct conflict. v2 needs
a new incentive mechanism that:

1. Rewards miners for **surfacing apprenable prompts** (curriculum
   discovery), not for filtering outputs.
2. Produces an **unbiased** sample per submission (i.i.d. from π).
3. Keeps cherry-picking economically irrational **without** requiring
   deterministic re-execution (which GRAIL v1 was explicitly designed to
   avoid via tolerant sketches).

## Approach

### Core shift

Miners compete on **which prompt_idx** to submit, and they earn
participation-tier payment for each crypto-valid submission plus a
batch-inclusion bonus for the first 8 in the "apprenable zone".

- Miner picks `prompt_idx` freely from the env's task pool.
- Miner generates `M = 8` rollouts at protocol-fixed `T`, `top_p`, `top_k`.
- Miner submits all 8 completions + rewards + GRAIL sketches, signed with
  the most recent drand round they saw.
- Validator re-computes rewards locally (must match miner's claims),
  verifies GRAIL per completion, filters on `k ∈ [2, 6]`, then admits the
  first 8 valid submissions (FIFO by signed round, distinct prompts) into
  the training batch.
- Payment is **flat within tier**: all batch members split BATCH_POOL
  equally; all participation members split PARTICIPATION_POOL equally.

### Why this works

**No cherry-pick incentive in payment.** Flat payment within tier means a
miner who cherry-picks to push `k` from 3 to 4 earns **the same** as one
who honestly hits `k=2`. The only economic lever is "are you in the
batch?" — answered by speed, not by `k` quality.

**Speed beats cherry-picking by construction.** Cherry-picking requires
generating more than 8 rollouts. More rollouts = later `signed_round` =
later FIFO position = displaced from the batch by faster honest miners.
Missing the batch = drop from BATCH_POOL share to PARTICIPATION_POOL
share (≈ 2–3× loss). The speed penalty exceeds any cherry-pick gain.

**GRAIL guarantees model authenticity.** The existing v1 proof system
(teacher-forcing + tolerant sketch) prevents a miner from fabricating
completions. This is unchanged.

**Cross-GPU determinism is not required.** We do NOT seed-bind the
rollouts or require bit-exact re-execution. GRAIL's existing tolerance
handles hardware heterogeneity — same flow as v1.

### Parameters (protocol-fixed)

| Constant | Value | Role |
|---|---|---|
| `M` | 8 | rollouts per submission |
| `T_proto` | 0.9 (tunable, non-zero) | sampling temperature — fixed so miners can't inflate diversity |
| `top_p_proto`, `top_k_proto`, `MAX_NEW_TOKENS_PROTOCOL_CAP` | fixed | same reason |
| `ZONE_K_MIN`, `ZONE_K_MAX` | 2, 6 | apprenable zone; `k ∉ [2,6]` → rejected |
| `B` | 8 | batch size for training step |
| `BATCH_POOL_FRACTION` | 0.7 | share of window emission to batch members |
| `PARTICIPATION_POOL_FRACTION` | 0.3 | share of window emission to in-zone non-batch members |

## Protocol flow

### 1. Miner submission

```
1. prompt_idx = miner.pick_prompt()                    # free choice in [0, len(env))
2. problem = env.get_problem(prompt_idx)
3. completions = [generate(problem.prompt, T_proto) for _ in range(M)]
4. rewards = [env.compute_reward(problem, c) for c in completions]
5. grail_sketches = [grail.commit(model, problem.prompt, c) for c in completions]
6. signed_round = latest_drand_round_observed()
7. signature = sign(hotkey, (prompt_idx, merkle_root(leaves), signed_round))
8. POST /submit { prompt_idx, completions, rewards, grail_sketches, signed_round, signature }
```

### 2. Validator verification (per submission)

```
def verify(submission):
    if not signature_valid(submission): return REJECT
    if not prompt_idx_valid(submission.prompt_idx): return REJECT
    problem = env.get_problem(submission.prompt_idx)

    for (c, claimed_reward, sketch) in submission.triples:
        if not grail.verify(c, problem.prompt, sketch, model): return REJECT
        if env.compute_reward(problem, c) != claimed_reward: return REJECT

    k = sum(submission.rewards)
    if not (ZONE_K_MIN <= k <= ZONE_K_MAX): return OUT_OF_ZONE   # still rejected
    return VALID
```

Three outcomes:

- `REJECT` (crypto or reward mismatch) → no payment
- `OUT_OF_ZONE` (`k ∉ [2,6]`) → no payment
- `VALID` → eligible for batch selection + payment

### 3. Batch selection (end of window)

```
def select_batch(valid_submissions):
    # Deterministic cross-validator order: round, then tie-break hash.
    ordered = sorted(
        valid_submissions,
        key=lambda s: (s.signed_round, hash(s.hotkey, s.prompt_idx, s.merkle_root)),
    )
    batch = []
    seen_prompts = set()
    for s in ordered:
        if len(batch) == B: break
        if s.prompt_idx in seen_prompts: continue
        batch.append(s)
        seen_prompts.add(s.prompt_idx)
    return batch
```

Any valid submission not selected into the batch (because the prompt was
already taken, or because the batch filled first) still earns
participation payment.

### 4. Payment (weight computation)

```
valid = [s for s in all_submissions if s.status == VALID]
batch = select_batch(valid)
participation = [s for s in valid if s not in batch]

# Flat within tier.
for s in batch:
    reward[s.hotkey] += BATCH_POOL_FRACTION / len(batch)
for s in participation:
    reward[s.hotkey] += PARTICIPATION_POOL_FRACTION / len(participation)
# Unused fractions (empty batch, empty participation) go to UID_BURN.
```

### 5. GRPO training step

Trainer pulls the batch from storage, runs one GRPO update. For each group
of 8 rollouts:

```
mean_g = k_g / M                          # p̂ in [0.25, 0.75]
std_g = sqrt(mean_g * (1 - mean_g))       # ∈ [0.43, 0.50]
for j in range(M):
    advantage[j] = (reward[j] - mean_g) / std_g
```

Then standard GRPO/PPO loss with clip + KL regularization.

## Anti-cherry-pick analysis

### Within-batch payment is flat

A cherry-picker in the batch earns exactly the same share as an honest
miner in the batch. No marginal incentive to fabricate `k=4`.

### Cherry-picking displaces from the batch

To cherry-pick, a miner generates more than 8 rollouts (typically 2–3× the
budget to reliably hit a target `k`). That extra compute:

- Delays the submission → later `signed_round` → later FIFO position.
- When the batch fills first (8 fast honest submissions with distinct
  prompts), the cherry-picker drops to participation → loss factor
  `BATCH_POOL / PARTICIPATION_POOL ≈ 2.3×` (with the default 0.7/0.3 split).

### Cherry-picking doesn't help passing the zone gate

The filter is `k ∈ [2, 6]`. Under honest sampling with `true p ∈ [0.2, 0.8]`,
the probability of `k ∈ [2, 6]` is ≥ 95%. Cherry-picking can only help at
the extreme tails (`p < 0.2` or `p > 0.8`), which are exactly the prompts
we don't want to train on. Miners have no reason to force such prompts
into the zone.

### Attack surface summary

| Attack | Neutralised by |
|---|---|
| Fabricated completions | GRAIL v1 (teacher-forced sketch) |
| Fabricated rewards | Validator re-computes via `env.compute_reward` |
| Sample filtering (cherry-pick within prompt) | Flat payment + speed penalty |
| Temperature inflation | `T_proto` fixed at protocol level |
| Submit same prompt from many hotkeys | Batch diversity (one prompt per batch slot) |
| Replay old window's submission | `signed_round` must be in current window |
| Collusion between miners on same prompt | All but first (by round) fall to participation |

Residual risk: a miner who is **both** very fast **and** willing to
spend extra compute might still break even on cherry-picking in borderline
economic regimes. The gap between honest and cherry-pick reward is small
enough that compute cost dominates in nearly all realistic cases, but
this is an economic argument, not a cryptographic proof. Monitoring
(see below) catches aggregate deviation.

## Changes vs v1

### Protocol-level breaking changes

- `SubmissionRequest` shape: adds `signed_round`, drops `slot_index` and
  `prompt_id` (prompt is now identified by `prompt_idx` chosen by miner).
- Rewards are submitted by miner and re-verified by validator, not just
  computed by validator post-hoc.
- Cross-miner prefix-dedup (`DIVERSITY_PREFIX_LEN`) removed — diversity is
  now enforced at the prompt level (one prompt per batch slot), not at
  the token prefix level.

### Files touched

| File | Change |
|---|---|
| `reliquary/constants.py` | Add `ZONE_K_MIN/MAX`, `T_PROTO`, `BATCH_POOL_FRACTION`, `PARTICIPATION_POOL_FRACTION`, `B`. Deprecate `GROUP_SIZE`, `PROMPTS_PER_WINDOW`, `DIVERSITY_PREFIX_LEN`. |
| `reliquary/protocol/submission.py` | New `SubmissionRequest` with `prompt_idx`, `signed_round`, `rewards`, `merkle_root`. Remove slot fields. |
| `reliquary/miner/engine.py` | Pick `prompt_idx` freely (strategy = operator's choice; reference impl: random-in-range). Generate `M=8` at `T_PROTO`. Compute local rewards. Sign with latest observed round. |
| `reliquary/miner/submitter.py` | Send new request shape to `/submit`. |
| `reliquary/validator/batcher.py` | Rewrite: no more slots; instead a flat list of valid submissions per window. New `select_batch()`. New `get_miner_scores()` returning flat per-tier allocations. |
| `reliquary/validator/verifier.py` | Add `verify_reward_claims` (re-run `env.compute_reward` on each submitted completion). GRAIL path unchanged. |
| `reliquary/validator/weights.py` | Simplify: flat within tier, no superlinear on individual scores since scoring is binary in/out of zone. UID_BURN still absorbs unused pool fractions. |
| `reliquary/validator/server.py` | `/submit` accepts new shape. `/window/{n}/state` returns list of submissions instead of slot histograms. |
| `reliquary/environment/base.py` | Add `get_problem(idx)` contract (already exists for GSM8K); document that index must be a stable integer in `[0, len(env))`. |

### Tests

New tests under `tests/unit/`:

- `test_batch_selection.py`
  - FIFO order respected on distinct prompts
  - Same-prompt collisions → only first (by round) wins the batch
  - Batch caps at `B`
  - Empty valid pool → empty batch (all emission burns)
  - Tie-breaking determinism (same input → same output across runs)
- `test_zone_filter.py`
  - `k ∈ [2, 6]` passes, all other values rejected
  - `k=0` and `k=M` rejected specifically (no participation payment)
- `test_reward_verification.py`
  - Miner claim matches validator re-computation → accept
  - Miner claim mismatches → reject with `reward_mismatch`
- `test_flat_payment.py`
  - All batch members receive identical payment
  - All participation members receive identical payment
  - Pool fractions sum to 1.0 minus UID_BURN leak
- `test_anti_cherry_pick_economics.py` (property-based)
  - For a grid of `(true_p, compute_cost_ratio)`, honest strategy dominates
    cherry-pick strategy in expected payment.

Delete or rewrite:

- `tests/unit/test_batcher.py` (slot-based tests → obsolete)
- `tests/unit/test_diversity.py` (prefix-dedup → obsolete; keep only if
  we decide to retain prefix-level dedup as a sanity check)

## Bootstrap phase

New subnets / new model checkpoints start with few miners and a model
whose success distribution may concentrate outside `[0.2, 0.8]`. To avoid
a dead cold start:

- `BOOTSTRAP_WINDOWS = 100` — first 100 windows apply relaxed thresholds
- `BOOTSTRAP_K_RANGE = (1, 7)` — wider zone during bootstrap
- `BOOTSTRAP_FLOOR_EMISSION = 0.01` — every crypto-valid submission earns
  at least a floor, even outside zone, during bootstrap
- `BOOTSTRAP_M = 4` — smaller groups for easier hit rate (reverts to 8 at
  window 100)
- Curated env subset in `env_version=bootstrap` — prompts pre-screened to
  fall in `[0.2, 0.8]` for the initial checkpoint

These knobs live in constants; changes require coordinated deployment.

## Assumptions and validation plan

This design rests on assumptions that must be validated empirically
**before** mainnet.

### Assumption 1 — GRAIL v1's tolerant sketch catches fabricated completions with adequate soundness

v1 already runs in production. New risk: in the new protocol, the miner
also claims `reward`, which the validator re-computes. If GRAIL sketches
ever false-negative on an honest completion, the whole submission is lost.
Conversely if GRAIL ever false-positives on a fabricated one, we pay a
fake submission.

**Validation**: run the existing GRAIL test suite + add a red-team
exercise targeting the new payload shape (rewards + merkle root over all
M). Target false-negative rate < 1%, false-positive rate < 0.1%.

### Assumption 2 — Cherry-picking is economically dominated by speed

The argument is quantitative: factor-2.3× BATCH_POOL / PARTICIPATION_POOL
advantage, combined with factor-2–3× extra compute for cherry-picking,
means the honest strategy dominates for typical `(compute_cost, reward)`
ratios.

**Validation**: run a testnet simulation with 50 miners, half honest and
half cherry-picking at varying intensities. Measure earnings per compute
unit; honest should dominate across the regime grid.

### Assumption 3 — `signed_round` consensus is robust

All validators must agree on the order of submissions (by `signed_round`).
drand rounds are public, but miner-claimed rounds must be within a
verifiable window (not arbitrarily future).

**Validation**: multi-validator integration test. Same submission set,
different validator nodes → identical batch.

### Assumption 4 — Env has enough prompts in zone for the current checkpoint

If the model is very weak (everything `k=0`) or very strong
(everything `k=M`), the zone is empty → empty batches → no training
signal → model doesn't improve → vicious cycle.

**Validation**: offline sweep on GSM8K with the initial checkpoint.
Expect ≥ 30% of prompts to land in `[0.2, 0.8]`. If lower, curate
bootstrap subset.

### Assumption 5 — Training converges under this data distribution

Even with honest samples, GRPO may fail to converge if the curriculum
distribution is too peaky (all miners submit the same few "juicy"
prompts) or the policy drifts too fast between windows.

**Validation**: testnet training run over 1000+ steps, held-out eval
pass@1 on a stable eval split. Track entropy, mode collapse, KL to
reference.

## Out of scope (explicit non-goals)

- **Seed-binding via beacon**: considered and rejected. The cross-GPU
  determinism cost outweighs the security benefit, because the economic
  deterrence (flat within tier + speed pressure) already closes the
  cherry-pick loop.
- **Score-weighted payment within tier**: considered and rejected. Would
  reintroduce marginal cherry-pick incentive. The 5–10% gradient-quality
  loss from flat payment is an acceptable tradeoff.
- **Continuous-reward environments**: GSM8K is binary. When continuous
  rewards arrive (partial credit, MATH), the zone filter generalises to
  `ZONE_VAR_MIN ≤ Var(rewards) ≤ ZONE_VAR_MAX`, and flat payment is
  unaffected.
- **Decentralised training (multi-validator GRPO step)**: this spec
  assumes a single logical trainer per window (any validator can take
  the role, but only one update is consensus-selected). Full
  decentralised training is v2.1.
- **Explicit anti-sybil cap per hotkey**: the existing Bittensor
  registration cost + the batch diversity constraint bound sybil damage.
  If monitoring shows sybil abuse, a per-hotkey submission cap can be
  added without protocol change.

## Migration from v1

Single branch. Because this is a protocol-breaking change, v1 and v2
cannot co-exist on the same netuid; migration is a cutover at a
pre-announced block height.

1. Merge the spec + implementation branch.
2. Tag release.
3. Announce cutover block.
4. All miners and validators upgrade simultaneously at cutover.
5. First window of v2 runs in bootstrap mode.

## Open questions for reviewer

1. Default `T_proto` value: 0.9 vs 1.0 vs 0.7. Will be determined by
   training team based on target sampling distribution for the chosen
   base model.
2. Should `BATCH_POOL_FRACTION` be fixed at 0.7, or tunable by governance
   / sudo-key for rebalancing?
3. Do we want a per-hotkey submission cap from day one (centralisation
   hedge), or ship lean and add only if monitoring shows abuse?
4. `DIVERSITY_PREFIX_LEN` from v1 — keep as a sanity check (rejects
   obvious copy-paste across miners) or drop entirely? Prompt-level
   diversity in the batch probably subsumes it.

These are surface-level tuning choices; the structural design does not
depend on the answers.
