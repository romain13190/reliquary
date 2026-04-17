# GRPO Refactor — Notes

Date: 2026-04-16
Branch: `feat/grpo-batching`

## What changed

GRAIL shifted from "miners pick random prompts → upload throwaway rollouts to
S3 → validators sample, re-verify, count uniques → weight ∝ unique^4" to
"validator owns the prompts → miners submit synchronously over HTTP →
validator verifies on the fly → weight ∝ (sum of rewards)^4 → settled
{prompt, 32 completions, 32 rewards} bundles archived to S3 as a GRPO
training dataset".

Per window (30 blocks ≈ 6 min):

1. Both roles derive 8 prompts deterministically:
   `idx = int(SHA256(beacon_randomness || slot_idx)[:8]) % len(env)`.
2. Miners generate **4 prefix-distinct completions** per slot
   (first `DIVERSITY_PREFIX_LEN=8` generated tokens must be pairwise
   distinct), build GRAIL commitments + signature, POST `/submit`.
3. The validator's `WindowBatcher` verifies atomically: window/slot match,
   no duplicate `(hotkey, slot)`, prompt prefix matches the slot's
   validator-derived prompt, GRAIL proofs pass, then computes the reward
   via the env. Either all 4 in the batch land or none do.
4. A slot settles at `GROUP_SIZE=32` accepted completions (8 distinct
   miners × 4 completions). When all 8 slots settle, the window completes.
5. Per-window scores accumulate into `_miner_scores: dict[str, float]`.
   Every `ROLLING_WINDOWS=12` windows the validator computes
   `weights = compute_weights(scores)` (superlinear x^4) and pushes to chain.
6. The settled dataset is gzipped and uploaded to
   `grail/dataset/window-{n}.json.gz` for downstream training.

## How to run

Install deps (already in `pyproject.toml`):
```
fastapi>=0.110, uvicorn[standard]>=0.27
```

Validator:
```
.venv/bin/python -m grail.cli.main validate \
    --checkpoint <path> \
    --network finney \
    --netuid 81 \
    --wallet-name default --hotkey default \
    --environment gsm8k \
    --http-host 0.0.0.0 --http-port 8888
```

Miner:
```
.venv/bin/python -m grail.cli.main mine \
    --checkpoint <path> \
    --network finney \
    --netuid 81 \
    --wallet-name default --hotkey default \
    --environment gsm8k
    # optional: --validator-url http://127.0.0.1:8888 for local testing,
    # bypassing metagraph discovery
```

The miner discovers the validator from the metagraph (first hotkey with
`validator_permit=True` and a routable axon).

## Intentionally NOT implemented

**Multi-validator coordination.** V1 assumes a single validator. With more
than one validator, the design needs leader election OR per-validator
sub-windows (a miner only submits to one validator per window) OR a gossip
protocol so settled bundles can be cross-checked. None of that is in scope
for this refactor — `discover_validator_url` deliberately picks the FIRST
permitted validator and stops there. The `# TODO: multi-validator coordination`
marker is the right hook for the future iteration.

Other intentional gaps (carried over from the original spec):
- No persistence of in-flight batcher state across validator restarts.
  A restart mid-window loses the window's progress; miners get connection
  errors and recover at the next window boundary. Adding state persistence
  would re-introduce most of the old "polling for files" complexity.
- `upload_window_rollouts` / `download_window_rollouts` / `load_used_indices`
  / `save_used_indices` / `save_window_results` are still in
  `grail/infrastructure/storage.py` but unused. They'll be removed in a
  follow-up cleanup once we're confident no operator script depends on the
  old keyspace.
- The deprecated constants section in `grail/constants.py`
  (`MINER_SAMPLE_*`, `ROLLOUT_SAMPLE_*`, `VERIFICATION_BATCH_SIZE`,
  `BATCH_FAILURE_THRESHOLD`, `FAILURE_LOOKBACK_WINDOWS`,
  `USED_INDICES_MAX_AGE_WINDOWS`, `MAX_ROLLOUTS_PER_FILE`, `DATASET_NAME`,
  `DATASET_SPLIT`) is intentionally importable for the same reason — same
  follow-up will delete them.

## Test status

`.venv/bin/python -m pytest tests/unit/ -q` → 236 passed.

## Notable design decisions

- Prompt verification moved out of `verifier.py` and into `WindowBatcher`.
  Reason: in the new flow the validator owns the prompt (derived from the
  beacon), so the "miner declares dataset_index" attack surface is gone.
  The batcher tokenises `slot.problem["prompt"]` and checks the first
  `prompt_length` tokens of the submission match — there's no
  `dataset=None` codepath to worry about.
- Atomic batch accept/reject with hotkey only consumed on success: a miner
  whose batch is rejected (e.g. for a flaky GRAIL proof) can retry within
  the same slot, so transient failures don't burn the per-slot submission.
- `ValidatorServer.set_active_batcher(None)` between windows on purpose:
  POST `/submit` outside an active window returns 503, which the miner's
  client retries-then-gives-up correctly.
- `httpx` reuses a single `AsyncClient` across the 8 slot iterations in
  `mine_window` — TCP connection reuse matters when the validator is
  remote.
