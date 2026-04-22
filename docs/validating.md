# Running a Reliquary Validator

Operational guide for running a validator on Bittensor subnet 81. For conceptual background see [docs/concepts.md](concepts.md). For a full bootstrap walkthrough see [docs/deployment.md](deployment.md).

## Validator modes

There are two validator modes:

- **Trainer mode** (`reliquary validate --train`, the default) — runs the full state machine: accepts miner submissions, trains the model, publishes checkpoints to HF, archives rollout datasets to R2.
- **Weight-only mode** (`reliquary validate --no-train`) — no GPU, no HTTP server, no HF writes. Reads R2 archives, replays the EMA, and submits weights on-chain every `WEIGHT_SUBMISSION_INTERVAL = 360` blocks. Any number of weight-only validators reading the same R2 bucket arrive at identical weights automatically (deterministic replay).

The rest of this guide focuses on trainer mode. See [docs/deployment.md](deployment.md#7-running-a-weight-only-validator) for weight-only mode setup.

## What a validator does (v2.1)

Windows are event-driven: a window seals the instant `B_BATCH = 16` valid distinct-prompt submissions land. There is no fixed per-window timer — only the `WINDOW_TIMEOUT_SECONDS = 600` safety net fires if fewer than `B_BATCH` submissions arrive in time.

The validator runs a four-phase state machine per window:

1. **`OPEN`** — `_open_window()` creates a new `GrpoWindowBatcher`, increments `window_n`, and begins accepting submissions at `/submit`. The `/state` endpoint returns `state = "open"` plus `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`.

2. **`TRAINING`** — the instant `seal_event` fires (B-th distinct valid submission received, or timeout), the service transitions to `TRAINING`. Submissions are rejected with `WINDOW_NOT_ACTIVE`. `train_step()` computes group-relative advantages, runs PPO-clipped surrogate + KL loss, and applies one AdamW step. EMA scores are updated for all miners seen this window.

3. **`PUBLISHING`** — after training, the model is saved locally and, every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` windows, pushed to HF Hub. The validator signs `checkpoint_n || revision` with ed25519 and installs the manifest. The window rollout dataset is archived to R2. Between publish windows, the validator trains in-memory but does not push to HF.

4. **`READY`** — manifest is live; `checkpoint_n` increments only on a successful HF publish. The service immediately transitions back to `OPEN` for the next window.

Validator state is stateless on disk. At startup, `window_n` is derived from the maximum R2 archive window; `checkpoint_n` from HF commit history; EMA scores are replayed from the last ~216 R2 archives. No local state file is required — a restart on a new machine loses no data.

Every `WEIGHT_SUBMISSION_INTERVAL = 360` blocks (`ROLLING_WINDOWS = 72` windows at 5 blocks/window), the validator submits EMA-based weights on-chain. See [docs/concepts.md](concepts.md#ema-scoring) for the EMA calculation. Unused slots burn to `UID_BURN = 0`.

Only one validator is active per window. Miners auto-discover it via `validator_permit=True` + a routable axon.

### Verification pipeline (`/submit`)

Submissions are checked in this exact order:

1. **Window state** — rejected with `WINDOW_NOT_ACTIVE` if state is not `OPEN`.
2. **Window match** — `window_start` in the request must equal the current `window_n` (`WINDOW_MISMATCH` if not).
3. **Checkpoint hash** — `checkpoint_hash` must match the active window's `current_checkpoint_hash` (`WRONG_CHECKPOINT` if not). An empty batcher hash disables this gate (pre-first-publish convenience).
4. **`prompt_idx` range** — must be `0 ≤ prompt_idx < len(env)` (`BAD_PROMPT_IDX` if not).
5. **Round freshness** — `signed_round` must be within `STALE_ROUND_LAG_MAX = 10` rounds of `current_round`, and not from the future (`STALE_ROUND` if not).
6. **Cooldown check** — `prompt_idx` must not be in the active `CooldownMap` (`PROMPT_IN_COOLDOWN` if it is).
7. **Reward match** — validator recomputes `env.compute_reward` for each rollout; claim must match exactly (`REWARD_MISMATCH` if not).
8. **Zone filter** — population std σ of the group's rewards must satisfy `σ ≥ 0.43` (or `σ ≥ 0.33` during bootstrap; `OUT_OF_ZONE` if not).
9. **Proof version** — `commit["proof_version"]` must match `GRAIL_PROOF_VERSION` (`GRAIL_FAIL` if not).
10. **Signature** — GRAIL commit signature verified against `miner_hotkey` (`BAD_SIGNATURE` if not).
11. **GRAIL sketch** — validator re-runs the bit-identical forward pass and compares sketch commitments (`GRAIL_FAIL` if they diverge).

Submissions that pass all checks are appended to the window's valid pool. When the pool reaches 8 distinct-prompt valid submissions, `seal_event` fires automatically.

### Reject reason reference

| Reason | Cause |
|---|---|
| `WINDOW_NOT_ACTIVE` | State is `TRAINING`, `PUBLISHING`, or `READY` |
| `WINDOW_MISMATCH` | `window_start` does not match current `window_n` |
| `WRONG_CHECKPOINT` | `checkpoint_hash` is not the active HF revision |
| `BAD_PROMPT_IDX` | `prompt_idx >= len(env)` |
| `STALE_ROUND` | `signed_round` is more than 10 behind current or is from the future |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` is in the 50-window cooldown set |
| `REWARD_MISMATCH` | Miner's claimed reward does not match `env.compute_reward` |
| `OUT_OF_ZONE` | σ < 0.43 (steady) or σ < 0.33 (bootstrap) |
| `WRONG_ROLLOUT_COUNT` | Submission does not have exactly `M_ROLLOUTS = 8` rollouts |
| `BAD_SIGNATURE` | GRAIL commit signature verification failed |
| `GRAIL_FAIL` | Proof version mismatch or sketch divergence |
| `ACCEPTED` | All checks passed |

### CooldownMap

A long-lived `CooldownMap` tracks which prompts entered the training batch. At startup, it is rebuilt from the last `BATCH_PROMPT_COOLDOWN_WINDOWS = 50` R2 archive entries — providing a consistent cooldown state even after a hard restart without needing a separate local file.

- **Normal mode**: batched `prompt_idx` values enter a 50-window cooldown.
- **Bootstrap mode**: first `BOOTSTRAP_WINDOWS = 100` windows use a lower σ threshold (0.33 vs 0.43) to fill early batches faster.

### `/state` endpoint

`GET /state` returns a `GrpoBatchState` JSON object:

```json
{
  "state": "open",
  "window_n": 42,
  "anchor_block": 42,
  "current_round": 42,
  "cooldown_prompts": [42, 1701, 3388],
  "valid_submissions": 3,
  "checkpoint_n": 7,
  "checkpoint_repo_id": "your-org/reliquary-sn",
  "checkpoint_revision": "abc123def456..."
}
```

Notes:
- `anchor_block` and `current_round` are currently set to `window_n` (placeholder; real chain block and drand round wiring is v2.2).
- `checkpoint_repo_id` and `checkpoint_revision` are `null` before the first HF publish.
- Returns 503 if no active batcher (transient between windows or during publish).

Miners must read `cooldown_prompts` before picking `prompt_idx` and must copy `checkpoint_revision` as `checkpoint_hash` into every `BatchSubmissionRequest`.

### `/checkpoint` endpoint

`GET /checkpoint` returns the current checkpoint manifest signed by the validator hotkey:

```json
{
  "checkpoint_n": 7,
  "repo_id": "your-org/reliquary-sn",
  "revision": "abc123def456...",
  "signature": "ed25519:deadbeef..."
}
```

The signature covers `checkpoint_n || revision` encoded as `f"{checkpoint_n}|{revision}".encode()`. Miners and external consumers can verify authenticity using the validator's on-chain hotkey.

### R2 archive format

One gzipped JSON file per window at `reliquary/dataset/window-<N>.json.gz` (flat path, no hotkey prefix). The `validator_hotkey` field inside the JSON body records provenance:

```json
{
  "window_start": 42,
  "validator_hotkey": "5xxx...",
  "randomness": "...",
  "environment": "math",
  "batch": [
    {
      "hotkey": "5xxx...",
      "prompt_idx": 4821,
      "signed_round": 42,
      "sigma": 0.500,
      "prompt": "Natalia sold...",
      "ground_truth": "72",
      "rollouts": [
        {"tokens": [...], "completion_text": "...", "reward": 1.0},
        {"tokens": [...], "completion_text": "...", "reward": 0.0}
      ]
    }
  ]
}
```

Each rollout entry has `tokens`, `completion_text`, and `reward`. The top-level `sigma` field holds the reward std for that submission. This is the GRPO-ready rollout dataset consumed by external consumers reading from R2.

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPU | 1× A100 40 GB or better (model + reference model + gradients + activations in bfloat16) |
| CUDA | 12.x with `flash-attn`-compatible drivers |
| RAM | 64 GB minimum |
| Disk | 100 GB (staging checkpoints, local state, R2 upload queue) |
| Network | Public inbound TCP on your chosen port + outbound HTTPS to HF Hub, R2, and Bittensor chain |
| Bittensor wallet | Registered on netuid 81, stake above `validator_permit` threshold |
| HF Hub token | Write access to your model repo (`HF_TOKEN` env var) |
| R2 / S3 bucket | Writable, for window dataset archives |

## Install

```bash
git clone <repo-url> reliquary
cd reliquary
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
reliquary --help
```

## Configure your wallet

```bash
btcli wallet new-coldkey --wallet.name my_validator
btcli wallet new-hotkey  --wallet.name my_validator --wallet.hotkey default
btcli subnet register    --wallet.name my_validator --wallet.hotkey default --netuid 81
btcli stake add          --wallet.name my_validator --amount <TAO> --netuid 81
```

After staking, wait for `validator_permit` to flip to `True` — miners won't route to you until then.

## Set environment variables

```bash
export HF_TOKEN=hf_xxx

export R2_ACCOUNT_ID=<cloudflare-account-id>
export R2_BUCKET_ID=reliquary
export R2_ACCESS_KEY_ID=<key-id>
export R2_SECRET_ACCESS_KEY=<secret>
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
```

## Launch

```bash
reliquary validate \
    --network finney \
    --netuid 81 \
    --wallet-name my_validator \
    --hotkey default \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo-id your-org/reliquary-sn \
    --http-host 0.0.0.0 \
    --http-port 8888 \
    --external-ip <your-public-ip> \
    --external-port 8888 \
    --log-level INFO
```

CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint` | `Qwen/Qwen3-4B-Instruct-2507` | HF repo id or local path for initial model load |
| `--hf-repo-id` | `aivolutionedge/reliquary-sn` | HF repo to push trained checkpoints to |
| `--environment` | `math` | Must match what miners are running |
| `--http-host` | `0.0.0.0` | Bind address |
| `--http-port` | `8888` | Listen port; must be reachable by miners |
| `--external-ip` | *(empty)* | Public IP to advertise on-chain; leave empty if using `--validator-url` on miners |
| `--external-port` | `0` (→ http-port) | Public port to advertise on-chain |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Keep on for mainnet |

## What you should see

```
... | Starting Reliquary validator (network=finney, netuid=81, env=math, http=0.0.0.0:8888)
... | Loading model from Qwen/Qwen3-4B-Instruct-2507...
... | Validator HTTP server listening on 0.0.0.0:8888
... | Rebuilt cooldown from 0 archive windows (current=0, map size=0)
... | Validator started (v2.1): env=math, netuid=81, http=0.0.0.0:8888
... | accepted prompt=4821 hotkey=5xxx...
... | Window 1 sealed (B valid received)
... | State → TRAINING (window=1)
... | train_step: lr=5.00e-07 ppo=0.1234 kl=0.0012 grad_norm=0.432 rollouts=64/64
... | Published checkpoint 1 to your-org/reliquary-sn@abc123def...
... | State → READY; next window opening
```

Every 72 windows:

```
... | Submitting weights: 14 miners (ema_total=0.8750), burn=0.1250
...   5F3sa77: 0.093750
...   5H1dF9P: 0.062500
```

Sanity checks:

```bash
curl http://localhost:8888/health
# → {"status":"ok","active_window":42}

curl http://localhost:8888/state
# → {"state":"open","window_n":42,"checkpoint_n":7,"checkpoint_repo_id":"your-org/reliquary-sn","checkpoint_revision":"abc123...","cooldown_prompts":[...],"valid_submissions":3,...}

curl http://localhost:8888/checkpoint
# → {"checkpoint_n":7,"repo_id":"your-org/reliquary-sn","revision":"abc123...","signature":"ed25519:..."}
```

## Monitoring

```bash
# GPU busy during training, lighter between.
nvidia-smi

# Tail validator logs.
tail -f ~/validator.log

# Check R2 archive uploads (flat layout, no hotkey prefix).
aws s3 ls s3://reliquary/dataset/ \
    --endpoint-url https://<account>.r2.cloudflarestorage.com | tail -5
```

## Troubleshooting

- **No submissions arriving**: miners cannot reach you. Check that `--external-ip` and `--external-port` match your real public address, firewall allows inbound TCP on the port, and `validator_permit` is `True` in the metagraph.
- **HF publish failing**: check `HF_TOKEN` is set and has write access to the repo. Test with `huggingface-cli whoami` and `huggingface-cli upload <repo-id> <file> <path>`.
- **R2 upload failed**: check credentials, bucket name, and endpoint URL. Test with `aws s3 cp <file> s3://reliquary/test --endpoint-url <url>`.
- **High `GRAIL_FAIL` rate**: miners are running a different checkpoint or `attn_implementation`. Confirm everyone is on the same HF revision shown in `/state`.
- **High `WRONG_CHECKPOINT` rate**: expected for 1–2 submissions immediately after a publish. Sustained high rates indicate miners are not polling `/state` frequently enough.
- **Weights not submitted**: `set_weights` requires `validator_permit` and a non-trivial emission window. Check logs for `set_weights` attempts; failures are logged with the chain error string.
- **Batches consistently under-full**: check per-rejection-reason counts in the logs. During bootstrap the wider zone and shorter cooldown mitigate this. If no batch seals before 600 s, the partial batch seals automatically.
- **Window stuck in `TRAINING` / `PUBLISHING`**: check GPU availability (`nvidia-smi`) and HF/R2 connectivity. The state machine does not advance until the step completes.
- **CooldownMap empty after restart with no R2**: if R2 is unavailable at startup the map starts empty. Miners may temporarily get `PROMPT_IN_COOLDOWN` gaps once the map repopulates over subsequent windows. Similarly, `window_n` and EMA start at 0 if R2 is unreachable; weight submissions will be sparse until state rebuilds.
- **Optimizer instability after restart**: expected for `LR_WARMUP_WINDOWS = 10` windows. AdamW momentum and the scheduler step count are not persisted; they reset on every restart.
