# Running a Reliquary Validator

This is the minimal guide to start a validator on Bittensor subnet 81.

## What a validator does (v2.1)

Windows are event-driven: a window seals the instant B=8 valid distinct-prompt
submissions land. There is no fixed per-window timer — only the
`WINDOW_TIMEOUT_SECONDS = 600` safety net fires if fewer than B submissions
arrive in time.

The validator runs a state machine with four phases per window:

1. **`OPEN`** — `_open_window()` creates a new `GrpoWindowBatcher`, sets
   `ValidationService._state.window_n`, and begins accepting submissions at
   `/submit`. The `/window/state` endpoint returns `state = "open"` plus
   `checkpoint_n`, `checkpoint_url`, `checkpoint_hash`, and `cooldown_prompts`.
2. **`TRAINING`** — the instant `seal_event` fires (B-th distinct valid
   submission received, or timeout), the service transitions to `TRAINING`.
   Submissions are rejected with `WINDOW_NOT_ACTIVE`. `_train_and_publish()`
   calls `train_step()` (stub in v2.1; real GRPO loss plugs in here with
   no protocol change — see `reliquary/validator/training.py`).
3. **`PUBLISHING`** — after training, the new checkpoint is written locally,
   uploaded to R2 at
   `reliquary/checkpoints/{validator_hotkey}/{checkpoint_n}.safetensors`,
   signed with ed25519 (`sign(checkpoint_n || file_hash)`), and the
   `/checkpoint` manifest is updated.
4. **`READY`** — manifest is live; `checkpoint_n` increments. The service
   immediately transitions back to `OPEN` for the next window.

Validator state (`window_n`, `checkpoint_n`) is persisted to
`reliquary/state/checkpoint.json` and survives restarts without loss.

Every `WEIGHT_SUBMISSION_INTERVAL` blocks, the validator aggregates batch
hotkeys across the rolling `ROLLING_WINDOWS = 72` window interval (flat
`1/B` per miner slot) and sets on-chain weights. Unused slots burn to
`UID_BURN`.

**Only one validator is active per window.** Miners auto-discover it from
the metagraph via `validator_permit=True` + a routable axon.

### Verification pipeline (`/submit`)

Submissions are accepted or rejected in this exact order:

1. **Window state** — rejected with `WINDOW_NOT_ACTIVE` if the current
   state is not `OPEN` (e.g. validator is in `TRAINING` or `PUBLISHING`).
2. **Window match** — `window_start` in the request must equal the current
   `window_n` (`WINDOW_MISMATCH` if not).
3. **Checkpoint hash** — `checkpoint_hash` must match the active window's
   hash (`WRONG_CHECKPOINT` if not). An empty batcher hash disables this
   gate (pre-first-publish convenience).
4. **`prompt_idx` range** — must be `0 ≤ prompt_idx < len(env)`.
5. **Round freshness** — `signed_round` must be within `STALE_ROUND_LAG_MAX`
   rounds of `current_round` (not stale, not from the future).
6. **Cooldown check** — `prompt_idx` must not appear in the active
   `CooldownMap` (`PROMPT_IN_COOLDOWN` if it does).
7. **Reward match** — validator recomputes `env.compute_reward` for each
   rollout; claim must match exactly (`REWARD_MISMATCH` if not).
8. **Zone filter** — population std σ of the group's rewards must satisfy
   `σ ≥ 0.43` (`OUT_OF_ZONE` if not). During bootstrap: `σ ≥ 0.33`.
   For binary GSM8K rewards this is equivalent to 2–6 correct out of 8.
9. **GRAIL sketch** — validator re-runs the bit-identical forward pass and
   compares sketch commitments (`GRAIL_FAIL` if they diverge).

Submissions that pass all checks are appended to the window's valid pool.
When the pool reaches B=8 distinct-prompt valid submissions, `seal_event`
fires automatically.

### CooldownMap

A long-lived `CooldownMap` is persisted to a local JSON file and survives
validator restarts. At startup, it is rebuilt from the last
`BATCH_PROMPT_COOLDOWN_WINDOWS` R2 archive entries so the cooldown state
is consistent even after a hard restart.

- **Normal mode**: batched `prompt_idx` values enter a 50-window cooldown.
- **Bootstrap mode**: first `BOOTSTRAP_WINDOWS` windows use a 10-window
  cooldown and lower σ threshold (0.33 vs 0.43) to fill the initial
  batches faster while miner population and env coverage are thin.

### `/window/state` endpoint

Returns a `GrpoBatchState` JSON object:

```json
{
  "state": "open",
  "window_n": 42,
  "anchor_block": 1234567,
  "current_round": 9182736,
  "cooldown_prompts": [42, 1701, 3388, ...],
  "valid_submissions": 3,
  "checkpoint_n": 7,
  "checkpoint_url": "https://<r2>/.../7.safetensors",
  "checkpoint_hash": "sha256:abc123..."
}
```

Miners must read `cooldown_prompts` before picking their `prompt_idx` and
must copy `checkpoint_hash` verbatim into every `BatchSubmissionRequest`.
When `checkpoint_n` advances between polls, miners must download
`checkpoint_url` and reload their weights before submitting.

### `/checkpoint` endpoint

Returns the current checkpoint manifest signed by the validator hotkey:

```json
{
  "checkpoint_n": 7,
  "url": "https://<r2>/.../7.safetensors",
  "file_hash": "sha256:abc123...",
  "signature": "<ed25519-sig-hex>",
  "validator_hotkey": "5xxx..."
}
```

The signature covers `checkpoint_n || file_hash` (big-endian uint64 +
UTF-8 hash string). Miners and external consumers can verify authenticity
using the validator's on-chain hotkey.

### R2 archive format

One gzipped JSON file per window at `reliquary/dataset/window-<N>.json.gz`:

```json
{
  "window_start": 1234567,
  "environment": "gsm8k",
  "batch": [
    {
      "prompt_idx": 4821,
      "prompt": "Natalia sold…",
      "miner_hotkey": "5xxx…",
      "rollouts": [
        {"tokens": [...], "completion_text": "…", "reward": 1.0},
        ...
      ]
    },
    ...
  ]
}
```

This is the GRPO-ready rollout dataset consumed by the training step and
any external consumers reading from the bucket.

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPU | **1× NVIDIA GPU**, ≥ 24 GB VRAM (runs the reference model forward pass and training step) |
| CUDA | 12.x with `flash-attn` compatible drivers |
| RAM | 32 GB minimum |
| Disk | 50 GB (model weights + working datasets before upload) |
| Network | Public inbound TCP on your chosen HTTP port + outbound HTTPS to drand/R2 |
| Bittensor wallet | Created, registered on netuid 81, with enough stake to hold `validator_permit` |
| R2 / S3 bucket | Writable, for window dataset uploads |

## Install

```bash
git clone <repo-url> reliquary
cd reliquary
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
reliquary --help     # confirm CLI installed
```

## Configure your wallet and get a validator_permit

You need a bittensor hotkey already registered on netuid 81. To hold
`validator_permit`, your stake must be above the subnet's validator
threshold — check `btcli subnet metagraph --netuid 81` to see current
validator counts and stake requirements.

```bash
btcli wallet new-coldkey --wallet.name my_validator
btcli wallet new-hotkey  --wallet.name my_validator --wallet.hotkey default
btcli subnet register    --wallet.name my_validator --wallet.hotkey default --netuid 81
btcli stake add          --wallet.name my_validator --amount <TAO> --netuid 81
```

After staking, wait for `validator_permit` to flip to `True` on your UID —
miners won't send you traffic until then.

## Advertise your HTTP endpoint on-chain

Miners discover you via your axon's advertised IP/port. Set them before
starting:

```bash
btcli axon serve \
    --wallet.name my_validator \
    --wallet.hotkey default \
    --netuid 81 \
    --axon.ip <your-public-ip> \
    --axon.port 8888
```

Replace `<your-public-ip>` with the address that miners will reach. If you
run behind NAT, port-forward accordingly.

## R2 / S3 credentials

The validator uploads per-window datasets to a Cloudflare R2 bucket (or any
S3-compatible endpoint). Set:

```bash
export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_ACCESS_KEY_ID="<r2-access-key>"
export R2_SECRET_ACCESS_KEY="<r2-secret-key>"
export R2_BUCKET_ID="reliquary"                      # defaults to "reliquary"
# Optional — defaults to https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
# export R2_ENDPOINT_URL="https://..."
# export R2_REGION="us-east-1"                       # default
```

Make sure the bucket exists and the key can `PutObject`/`GetObject` on keys
prefixed `reliquary/`.

## Fetch the reference checkpoint

You must run the same model as miners — GRAIL's correctness depends on
bit-identical forward passes. Ask in the subnet's coordination channel for
the pinned checkpoint, then:

```bash
huggingface-cli download <org/model-id> --local-dir ./checkpoints/current
```

## Launch

```bash
reliquary validate \
    --network finney \
    --netuid 81 \
    --wallet-name my_validator \
    --hotkey default \
    --checkpoint ./checkpoints/current \
    --http-host 0.0.0.0 \
    --http-port 8888 \
    --log-level INFO
```

CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--environment` | `gsm8k` | Must match what miners are running. |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Keep on for mainnet. |
| `--http-host` | `0.0.0.0` | Bind address. Use `127.0.0.1` only for local testing. |
| `--http-port` | `8888` | Must match what you advertised via `btcli axon serve`. |

## What you should see

```
... | Starting Reliquary validator (network=finney, netuid=81, env=gsm8k, http=0.0.0.0:8888)
... | Loading model from ./checkpoints/current...
... | Validator HTTP server listening on 0.0.0.0:8888
... | Validator started: env=gsm8k, netuid=81, http=0.0.0.0:8888, rolling_windows=72
... | Window 42 open — checkpoint_n=7 cooldown_size=142
... | Accepted submission: hotkey=5xxx… prompt_idx=4821 sigma=0.500 round=9182736
... | seal_event fired — 8 distinct valid submissions received (window=42)
... | State → TRAINING (window=42)
... | train_step (stub) called with batch of 8 submissions — weights not modified
... | State → PUBLISHING (window=42, checkpoint_n=8)
... | Checkpoint 8 uploaded to R2 — sha256:abc123…
... | State → READY; next window opening
```

Every 72 windows you'll also see the weight submission log:

```
... | Submitting weights: 14 miners + burn to UID 0
...   5F3sa77: 0.125000  (1/8 × 4 windows)
...   5H1dF9P: 0.125000  (1/8 × 4 windows)
...   (top weighted miners)
```

Sanity checks:

```bash
curl http://localhost:8888/health
# → {"status":"ok","window_n":42,"checkpoint_n":7}

curl http://localhost:8888/window/state
# → {"state":"open","window_n":42,"checkpoint_n":7,"checkpoint_url":"...","checkpoint_hash":"sha256:...","cooldown_prompts":[...],"valid_submissions":3,...}

curl http://localhost:8888/checkpoint
# → {"checkpoint_n":7,"url":"...","file_hash":"sha256:...","signature":"...","validator_hotkey":"5xxx..."}
```

## Monitoring

```bash
# GPU busy during windows (generation + training), idle between.
nvidia-smi

# Tail validator logs.
tail -f ~/validator.log

# Check the most recent uploaded window file.
aws s3 ls s3://reliquary/dataset/ --endpoint-url https://<account>.r2.cloudflarestorage.com | tail -5
```

## Troubleshooting

- **No submissions arriving**: miners can't reach you. Check `btcli axon serve`
  output matches your real public IP + port, firewall allows inbound TCP, and
  `validator_permit` is `True`.
- **`R2 upload failed`**: credentials, bucket name, or endpoint URL wrong.
  Test with a plain `aws s3 cp` using the same env vars before blaming the
  validator.
- **High `GRAIL_FAIL` / `REWARD_MISMATCH` rates**: miners are running a
  different checkpoint than you. Confirm with the subnet coordinator that
  everyone is pinned to the same model hash.
- **Weights not submitted**: `set_weights` requires `validator_permit` AND a
  non-trivial emission window. Check logs for `set_weights` attempts; failures
  are logged with the chain error string.
- **Batches consistently under-full**: either not enough miners are online, or
  too many submissions are landing `OUT_OF_ZONE` or `PROMPT_IN_COOLDOWN`.
  Check per-rejection-reason counters in the logs. During the first
  `BOOTSTRAP_WINDOWS`, the wider zone and shorter cooldown should mitigate
  this automatically. If no batch seals before `WINDOW_TIMEOUT_SECONDS = 600`,
  the partial batch seals automatically — unused slots burn.
- **Window stuck in `TRAINING` / `PUBLISHING`**: the train or upload step is
  taking longer than expected. Check GPU availability (`nvidia-smi`) and R2
  connectivity. The state machine does not advance until the step completes.
- **CooldownMap diverged after restart**: the validator rebuilds the cooldown
  map from R2 on startup. If R2 is unavailable at startup, the map starts
  empty — miners may temporarily get `PROMPT_IN_COOLDOWN` once it repopulates
  over subsequent windows.
- **State file missing / corrupted**: if `reliquary/state/checkpoint.json` is
  absent, the validator starts from `window_n=0, checkpoint_n=0`. This is safe
  for a fresh deployment. After recovery, `checkpoint_n` will not match the
  last published checkpoint — miners will see a `WRONG_CHECKPOINT` gap until
  the next checkpoint publishes.
- **High `WRONG_CHECKPOINT` rate**: miners are lagging behind checkpoint
  updates. This is normal for 1–2 submissions right after a publish. Sustained
  high rates indicate miners are not polling `/window/state` frequently enough.
