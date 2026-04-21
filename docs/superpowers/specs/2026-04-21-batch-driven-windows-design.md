# Batch-Driven Windows + Synchronous Training — Reliquary v2.1 Design

Date: 2026-04-21
Branch: `feat/batch-driven-windows`
Status: Design pending user review
Scope: Replaces the current time-based window loop with a batch-driven one.
A window closes the instant 8 valid distinct-prompt non-cooldown
submissions land; the validator immediately runs a GRPO step, publishes
the new checkpoint, and miners pull it before the next window opens.
Builds on the v2 GRPO market spec (`2026-04-20-grpo-market-design.md`)
and extends the existing checkpoint signaling design
(`2026-04-19-checkpoint-signaling-design.md`).

## Problem

The v2 GRPO market introduced free-prompt submission, zone filter, FIFO
selection, and flat 1/B payment. But the **window mechanics inherited
from v1** still apply: a window is `WINDOW_LENGTH = 5` blocks ≈ 60 s,
the validator collects whatever arrives during that fixed interval,
seals at the deadline, then a separate cadence (every 72 windows ≈ 1 h
12 min) submits weights on-chain. Training is decoupled from windowing.

This is wasteful when miners are fast (5 valid submissions might land
in 8 s — the validator still waits 52 s for nothing) and structurally
prevents the tight per-batch training loop the v2 vision needs:
generate → train → next checkpoint → generate again.

The v2 roadmap calls for **synchronous training behind rollouts**
(see `project_subnet_roadmap.md` memory). This spec is the missing
piece: drive window cadence from batch-completion events, run the GRPO
step inline, distribute the new checkpoint, gate miner submissions on
checkpoint freshness.

## Approved design (8 decisions)

| # | Question | Choice |
|---|---|---|
| Q1 | Window timeout when miners are slow | Long timeout (e.g. 10 min) → seal partial batch (k<8) → unused slots burn |
| Q2 | Cooldown unit | Number of windows (= number of training steps), not wall-clock time |
| Q3 | Window identifier | Monotonic counter `window_n`, with `anchor_block` recorded for debug |
| Q4a | Checkpoint trust | Hash exposed via `/checkpoint`, signed ed25519 by validator |
| Q4b | Multi-validator checkpointing | Single-trainer stopgap; multi-validator decentralisation deferred to v2.2 |
| Q4c | Miner checkpoint enforcement | Miner includes `checkpoint_hash` in submission; validator rejects on mismatch |
| Q5 | Wait-state exposure | `state ∈ {OPEN, TRAINING, PUBLISHING, READY}` exposed via `/window/state` |
| Q6 | Drand round in submission | Kept — used as deterministic tiebreak for cross-validator consensus |

## State machine

A window is one full lap of this state machine, driven by the validator:

```
            ┌─────────────────┐  8 valid distinct + non-cooldown      ┌──────────────────┐
            │                 │  submissions accepted                  │                  │
            │      OPEN       │ ──────────────────────────────────────▶│     TRAINING     │
            │                 │                                        │                  │
            │ accept /submit  │                                        │ run GRPO step    │
            │ expose state    │                                        │ produce          │
            │                 │  OR timeout (WINDOW_TIMEOUT_SECONDS)   │ checkpoint_n+1   │
            └─────────────────┘ ──────────────────────────────────────▶└────────┬─────────┘
                                seal partial batch (may be empty)               │
                                                                                ▼
                                                                        ┌──────────────────┐
            ┌─────────────────┐  miner downloads → loads → polls       │                  │
            │                 │  /window/state until READY              │   PUBLISHING     │
            │      READY      │ ◀──────────────────────────────────────│                  │
            │                 │  validator finishes upload to R2 +      │ upload weights to│
            │ accept /submit  │  signs new manifest                     │ R2, sign, set    │
            │ on checkpoint_n │                                         │ /checkpoint      │
            │                 │                                         │ endpoint         │
            └────────┬────────┘                                         └──────────────────┘
                     │
                     │ next 8 land
                     ▼
                  back to OPEN of window n+1
```

Concretely the validator's main loop becomes:

```python
window_n = recover_or_init()
while True:
    open_window(window_n)
    seal_event = await wait_for(8 valid OR window_timeout)
    train_step(seal_event.batch)              # state = TRAINING
    publish_checkpoint(window_n + 1)          # state = PUBLISHING
    set_window_state(READY)
    window_n += 1
    if window_n % ROLLING_WINDOWS == 0:
        await submit_weights_on_chain()
```

There is no `asyncio.sleep(WINDOW_LENGTH * BLOCK_TIME_SECONDS)` anymore;
the cadence is fully event-driven.

## Components

### 1. Window state model

New enum + extended state response in `reliquary/protocol/submission.py`:

```python
class WindowState(str, Enum):
    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint available, /submit will open as soon as next window seals back to OPEN

class GrpoBatchState(BaseModel):
    state: WindowState
    window_n: int                    # monotonic counter
    anchor_block: int                # block at which OPEN started (debug)
    current_round: int               # latest drand round seen by validator
    cooldown_prompts: list[int]
    valid_submissions: int           # only meaningful when state == OPEN
    checkpoint_n: int                # currently published checkpoint
    checkpoint_url: str | None       # null until first PUBLISHING completes
    checkpoint_hash: str | None      # ed25519 signature of (checkpoint_n, file_hash)
```

### 2. Submission gating

`BatchSubmissionRequest` adds a required field:

```python
class BatchSubmissionRequest(BaseModel):
    ...
    checkpoint_hash: str = Field(..., min_length=1)
```

Validator's first cheap-check (before GRAIL):

```python
if request.checkpoint_hash != self.current_checkpoint_hash:
    return REJECT(RejectReason.WRONG_CHECKPOINT)
```

A new `RejectReason.WRONG_CHECKPOINT` value.

### 3. Window timeout & seal

A new constant:

```python
WINDOW_TIMEOUT_SECONDS = 600   # 10 minutes — generous safety net, not the cadence
```

When the timeout fires before `B` valid submissions land:

```python
batch = batcher.seal_batch()    # may have 0 < len < B entries
empty_slots = B - len(batch)    # contributes to UID_BURN as before
```

If `len(batch) == 0`, the window still progresses (no training step
happens, no new checkpoint is published, just bump `window_n` and reopen
on the same checkpoint).

### 4. Cooldown — same semantics, new clock

Cooldown is still expressed in **windows** (= training steps):
`BATCH_PROMPT_COOLDOWN_WINDOWS = 50` means a batched prompt is
ineligible for the next 50 *training steps*, regardless of how long
each takes. `CooldownMap` is unchanged; only the `current_window` value
passed to it changes from "block-derived" to "monotonic counter".

The R2 archive still uses `window_start: int` as the key; for v2.1 this
is the `window_n` counter. The cooldown rebuild from R2 history works
identically.

### 5. Checkpoint signaling extension

Builds on `2026-04-19-checkpoint-signaling-design.md`. The existing
`/checkpoint` endpoint returns a manifest. We extend it:

```
GET /checkpoint
→ 200 {
    "checkpoint_n": 42,
    "repo": "aivolutionedge/reliquary-sn",
    "revision": "...",
    "file_url": "https://r2.../reliquary/checkpoints/42.safetensors",
    "file_hash": "sha256:...",
    "signature": "ed25519:..."   # validator signs (checkpoint_n || file_hash)
}
```

The validator publishes a new entry **after each successful training
step** by:
1. Saving the new model weights to a local temp file
2. Computing `file_hash = sha256(file_bytes)`
3. Uploading to R2 at `reliquary/checkpoints/{validator_hotkey}/{n}.safetensors`
4. Signing `(n || file_hash)` with the wallet hotkey
5. Atomically updating the in-memory manifest served by `/checkpoint`

Miners trust the validator's signature today (single-trainer model).
Multi-validator consensus on checkpoint hash is v2.2.

### 6. Miner mine_window loop

```python
while True:
    state = await get_window_state_v2(url)
    if state.state != WindowState.OPEN:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        continue

    # Pull new checkpoint if needed.
    if state.checkpoint_n > local_checkpoint_n:
        await download_and_load(state.checkpoint_url, state.checkpoint_hash)
        local_checkpoint_n = state.checkpoint_n
        local_checkpoint_hash = state.checkpoint_hash

    prompt_idx = pick_prompt_idx(env, set(state.cooldown_prompts))
    rollouts = generate_M_rollouts(prompt_idx)
    request = BatchSubmissionRequest(
        ...,
        checkpoint_hash=local_checkpoint_hash,
        signed_round=state.current_round,
    )
    await submit_batch_v2(url, request)
```

Miner stops submitting between TRAINING and READY because `state !=
OPEN`. Once READY → next batch's OPEN, miner sees new `checkpoint_n` →
downloads → resumes.

### 7. Backward-compat with v2.0 windowing

This is **breaking** for the validator main loop. Miners that don't
include `checkpoint_hash` get rejected. v2.0 and v2.1 cannot coexist on
the same netuid. Cutover requires coordinated upgrade like the v1→v2
cutover did.

## Anti-cheat properties preserved

- **Cherry-picking**: still gated by FIFO + zero-payment-outside-batch
  + flat 1/B → unchanged from v2.0.
- **Stale checkpoint**: a miner running on an old checkpoint gets
  WRONG_CHECKPOINT rejection, before GRAIL → cheap reject, no damage.
- **Replay across windows**: each submission carries `checkpoint_hash`
  pinned to the current window's checkpoint → can't replay a v2.0
  rollout under v2.1.
- **Validator cheating on checkpoint**: miner verifies ed25519
  signature on `(checkpoint_n || file_hash)` → can detect a tampered
  manifest. Single-validator trust still required for the model
  contents themselves; multi-validator hash consensus is v2.2.

## What changes vs v2.0

### Files modified

| File | Change |
|---|---|
| `reliquary/constants.py` | Add `WINDOW_TIMEOUT_SECONDS`. Drop `WINDOW_LENGTH`-driven scheduling (still used for the on-chain weight cadence). |
| `reliquary/protocol/submission.py` | Add `WindowState` enum, extend `GrpoBatchState`, add `checkpoint_hash` to `BatchSubmissionRequest`, add `RejectReason.WRONG_CHECKPOINT`. |
| `reliquary/validator/batcher.py` | `GrpoWindowBatcher` exposes a `seal_event: asyncio.Event` set when 8 valid land. Add `current_checkpoint_hash` field; `_accept_locked` rejects on mismatch first. |
| `reliquary/validator/service.py` | Replace time-based loop with state-machine loop. Add `_train_step()` and `_publish_checkpoint()` methods. Manage `window_n` counter. |
| `reliquary/validator/server.py` | `/window/state` returns extended `GrpoBatchState`; `/submit` rejects when `state != OPEN`. |
| `reliquary/miner/engine.py` | mine_window becomes a poll-loop on `state.state`; download+load checkpoint when `checkpoint_n` increments. |
| `reliquary/miner/submitter.py` | Add `checkpoint_hash` to outbound payload. |
| `reliquary/validator/checkpoint.py` (new) | Encapsulate checkpoint produce/upload/sign/manifest logic. |
| `reliquary/validator/training.py` (new) | Encapsulate GRPO step. Initial impl: stub that just bumps `checkpoint_n` without actually training (real training is plugin point). |

### Tests added

- `test_window_state_machine.py` — OPEN→TRAINING→PUBLISHING→READY transitions
- `test_seal_on_8_valid.py` — batcher fires `seal_event` exactly when B distinct-non-cooldown valid submissions accepted
- `test_window_timeout_partial_seal.py` — timeout fires before 8 → partial batch sealed
- `test_checkpoint_hash_gating.py` — submission with stale checkpoint_hash → WRONG_CHECKPOINT
- `test_checkpoint_publish.py` — produce → upload → sign → manifest update
- `test_miner_checkpoint_pull.py` — miner detects new checkpoint_n, downloads, switches model
- `test_window_counter_persistence.py` — `window_n` survives validator restart (rebuild from R2 history like cooldown)

## Open questions for reviewer

1. **Training implementation**: this spec stubs the actual GRPO update.
   The first iteration just bumps `checkpoint_n` without modifying
   weights, to validate the orchestration. Real training comes in a
   follow-up. Acceptable?
2. **Multi-validator divergence**: with single-trainer-per-window, the
   network has 1 source of truth per window. If multiple validators
   run, they fork the model. Do we accept this for v2.1 (each validator
   serves its own miners) or do we add lightweight election (v2.2
   work)?
3. **`WINDOW_TIMEOUT_SECONDS = 600`**: is 10 min the right value? Too
   short = bursty traffic stalls early. Too long = a dead miner field
   blocks training for 10 min. Worth A/B on testnet.
4. **`checkpoint_n` storage**: persisted where? Local JSON on validator
   + rebuild from R2 manifest history at startup, mirroring the
   cooldown approach.

## Out of scope

- **Real GRPO training implementation** — this spec wires the
  orchestration; the actual `train_step(batch) → new_weights` is a
  separate work item, plugged into `validator/training.py`.
- **Multi-validator checkpoint consensus** — v2.2.
- **Decentralised model storage** (instead of R2) — v2.2 / v3.
- **Backward-compat with v2.0** — single coordinated cutover.
