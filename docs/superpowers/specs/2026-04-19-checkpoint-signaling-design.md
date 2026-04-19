# Checkpoint Signaling — Design

Date: 2026-04-19
Branch: `main`
Status: Design pending approval

## Problem

Today both miner and validator take `--checkpoint` as a required CLI arg and
load the model **once at startup** (`reliquary/cli/main.py:29, 132`;
`reliquary/validator/service.py:49`). There is no mechanism to rotate the
model without a coordinated restart. The v2 roadmap adds a trainer that
produces new checkpoints behind rollouts; miners need a way to learn which
checkpoint is currently active so they generate with the same weights the
validator will use to verify.

Secondary motivation: when v2 trainer lands, an inconsistent miner/validator
pair causes **every GRAIL verification to fail** (hidden states diverge) —
miners get zero score during the mismatch window.

## Approved approach

**Validator publishes the active checkpoint over HTTP; miners poll it once
per window; rotation is pre-announced via an `activation_block` so both sides
swap atomically.**

Source of truth is a local JSON manifest file on the validator host
(operator writes it; v1 assumes trainer ≈ operator). The manifest is read by
the validator and re-exposed to miners via a new `/checkpoint` endpoint on
the existing `VALIDATOR_HTTP_PORT`. Miners discover the validator the same
way they do today (`discover_validator_url`, `reliquary/miner/submitter.py:41`).

### Manifest format

File path: `--checkpoint-manifest <path>` CLI arg on the `validate` command.
If omitted, validator falls back to the existing `--checkpoint` arg and
serves a static manifest with `activation_block=0, next=null`.

```json
{
  "current": {
    "repo": "aivolutionedge/reliquary-sn",
    "revision": "abc123def456"
  },
  "next": {
    "repo": "aivolutionedge/reliquary-sn",
    "revision": "def456abc789",
    "activation_block": 12345
  }
}
```

- `current` is always set. `repo` is an HF Hub model id (or a local path for
  dev); `revision` is an HF commit SHA pinning reproducibility. For local
  paths, `revision` is `null` (passed through to HF as `revision=None`).
- `next` is optional. `activation_block` **must** be a multiple of
  `WINDOW_LENGTH` so rotation aligns with window boundaries. The resolver
  accepts a `next` whose `activation_block` is already in the past; it
  simply resolves to `next`. Operator is responsible for eventually
  promoting it to `current` to keep the file tidy.

**Effective checkpoint selection logic** (identical on miner and validator):

```
effective = next if (next is not None and current_block >= next.activation_block) else current
```

This lets the trainer leave `next` populated even after activation without
requiring a second manifest write — the resolver handles the transition.
Operator is expected to promote `next → current` and clear `next` before the
*following* rotation to keep the file clean.

### `/checkpoint` endpoint

New HTTP endpoint on the validator:

```
GET /checkpoint
→ 200 {
    "current": { "repo": str, "revision": str },
    "next": null | { "repo": str, "revision": str, "activation_block": int }
  }
```

The validator reads the manifest file **on every request** (cheap — small
JSON, cached by the OS page cache). No watchdog, no invalidation logic.

### Miner flow

Once per window (at the top of `mine_window`, before generation):

1. `GET /checkpoint` on the discovered validator URL.
2. Compute `effective = resolve(manifest, current_block)`.
3. If the `(repo, revision)` tuple differs from what is currently loaded →
   reload **tokenizer + both models** (`vllm_model` and `hf_model`) via
   `AutoTokenizer.from_pretrained(repo, revision=revision)` and
   `AutoModelForCausalLM.from_pretrained(repo, revision=revision, ...)`.
   Before loading, `del` the old models and call
   `torch.cuda.empty_cache()` so VRAM is actually freed (Python GC alone
   won't release it promptly). The engine caches the currently-loaded
   `(repo, revision)` pair to keep the equality check a pure string compare.
4. Regardless: if `next` is set and not yet activated → pre-download in
   background via `huggingface_hub.snapshot_download(repo, revision=revision)`
   so the reload at activation is cache-warm and fast. Pre-download runs
   in a `asyncio.to_thread` task; only one pre-download runs at a time.

On network/HTTP failure reading `/checkpoint`: **continue with currently
loaded model**. The miner doesn't hard-fail because a stale endpoint is
worse than a stale model — the validator will also be stale if the endpoint
is down, so consistency is preserved.

### Validator flow

Same resolution logic as the miner, but drives itself from the local
manifest file (no HTTP self-call):

1. Background task polls `current_block` every `POLL_INTERVAL_SECONDS`.
2. Reads manifest, computes `effective`.
3. If `effective != currently_loaded` → swap `self.model`.
4. Pre-download `next` when seen (same HF snapshot primitive).

Model swap is atomic at the Python level: load the new model into CPU / a
fresh GPU slot, then reassign `self.model = new_model` and `self.tokenizer =
new_tokenizer` under an `asyncio.Lock` so no rotation races with a new
`WindowBatcher` construction. The old model is `del`'d and
`torch.cuda.empty_cache()` is called. `WindowBatcher` instances created
after the swap see the new model; any in-flight verification uses the
reference it already captured, which is acceptable because verification is
bounded to a single window and rotation happens at window boundaries
(guaranteed by `activation_block % WINDOW_LENGTH == 0`).

### Transition semantics

- Operator: push new revision to HF Hub, then update manifest with
  `next = {repo, revision, activation_block}` where `activation_block` is at
  least `current_block + CEIL(download_time_seconds / BLOCK_TIME_SECONDS) +
  safety_margin`, rounded up to a `WINDOW_LENGTH` multiple. Safety margin
  defaults to one window (5 blocks).
- Both miner and validator see `next` at their next poll, start pre-download.
- At `activation_block`, both sides resolve to `next` and reload. Because
  `snapshot_download` has already populated the HF cache, reload is fast
  (seconds, not minutes).
- If a miner starts fresh mid-transition (before activation): loads `current`,
  pre-downloads `next`, swaps at activation. Handled by the resolver.
- If a miner starts fresh post-activation: resolver returns `next` directly,
  miner loads `next.revision` on first window.

### Out of scope for v1 (explicitly)

- Multiple trainers writing the manifest.
- Decentralized trust (on-chain commit, validator consensus).
- Rollback of a bad checkpoint — operator does it manually by writing a
  fresh manifest with `current = known_good, next = null`.
- Graceful handling of a miner stuck on an old revision past activation:
  the validator just rejects their submissions (hidden states mismatch), and
  the miner recovers on the next window's poll.

## Files to change

- `reliquary/protocol/submission.py` — add `CheckpointManifest`, `CheckpointEntry`,
  `ActiveCheckpointEntry` pydantic models (reused by server and miner client).
- `reliquary/validator/server.py` — add `GET /checkpoint` handler wired to a
  manifest-reader callable injected by `ValidationService`.
- `reliquary/validator/service.py` — load manifest, expose it to the server,
  add the model-rotation loop.
- `reliquary/cli/main.py` — add `--checkpoint-manifest` to the `validate`
  command; plumb into `ValidationService`.
- `reliquary/miner/submitter.py` — add `get_checkpoint(url) -> CheckpointManifest`.
- `reliquary/miner/engine.py` — call `/checkpoint` at the top of
  `mine_window`, reload models when effective revision changes, trigger
  background pre-download for `next`.
- `reliquary/constants.py` — add `CHECKPOINT_PREDOWNLOAD_SAFETY_BLOCKS` (5).
- Tests: unit tests for the resolver, integration test for `/checkpoint`
  round-trip, miner reload scenarios (fresh, in-transition, post-activation).

## Testing

- Resolver pure-function unit tests: current only, next-before-activation,
  next-at-activation, next-after-activation.
- FastAPI TestClient for `/checkpoint` with temp manifest file.
- Miner engine test with a mock validator server: verify model reload is
  called exactly when revision changes, not when it matches.
- Manual end-to-end on a local two-process setup (miner + validator on the
  same box, HF local paths) to confirm no regression on the existing
  submission flow.
