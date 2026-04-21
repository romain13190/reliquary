# Running a Reliquary Validator

This is the minimal guide to start a validator on Bittensor subnet 81.

## What a validator does (v2)

Each window (≈ 60 s, 1 GRPO training step), the validator:

1. Runs a `GrpoWindowBatcher` that accepts miner submissions and tracks a
   shared `CooldownMap` of recently-batched prompts.
2. Exposes an HTTP endpoint (`/submit`, `/window/{n}/state`, `/health`) on
   port 8888.
3. Processes each `BatchSubmissionRequest` through the full verification
   pipeline (see below) and appends valid submissions to the in-flight pool.
4. At window close, calls `seal_batch()` — selects the first `B = 8`
   distinct-prompt, in-zone submissions (FIFO by `signed_round`), records
   each batched `prompt_idx` into the cooldown map, and archives the sealed
   batch to R2.
5. Runs the GRPO training step on the sealed batch; checkpoint is rotated
   on-chain after each step.
6. Every `WEIGHT_SUBMISSION_INTERVAL` blocks (≈ 72 windows / ~1 h),
   aggregates batch hotkeys across the rolling interval (flat `1/B` per
   miner slot) and sets on-chain weights. Unused slots burn to `UID_BURN`.

**Only one validator is active per window.** Miners auto-discover it from
the metagraph via `validator_permit=True` + a routable axon.

### Verification pipeline (`/submit`)

Submissions are accepted or rejected in this exact order:

1. **Signature** — ECDSA signature over the submission hash must verify
   against the submitting hotkey.
2. **`prompt_idx` range** — must be `0 ≤ prompt_idx < len(env)`.
3. **Round freshness** — `signed_round` must be within the window's
   acceptable drand range (not stale, not from the future).
4. **Cooldown check** — `prompt_idx` must not appear in the active
   `CooldownMap` (`PROMPT_IN_COOLDOWN` if it does).
5. **Reward match** — validator recomputes `env.compute_reward` for each
   rollout; claim must match exactly (`REWARD_MISMATCH` if not).
6. **GRAIL sketch** — validator re-runs the bit-identical forward pass and
   compares sketch commitments (`GRAIL_FAIL` if they diverge).
7. **Zone filter** — `k` (number of correct rollouts) must satisfy
   `k ∈ [2, 6]` (`OUT_OF_ZONE` if not). During bootstrap: `k ∈ [1, 7]`.

Submissions that pass all checks are appended to the window's valid pool.

### CooldownMap

A long-lived `CooldownMap` is persisted to a local JSON file and survives
validator restarts. At startup, it is rebuilt from the last
`BATCH_PROMPT_COOLDOWN_WINDOWS` R2 archive entries so the cooldown state
is consistent even after a hard restart.

- **Normal mode**: batched `prompt_idx` values enter a 50-window cooldown.
- **Bootstrap mode**: first `BOOTSTRAP_WINDOWS` windows use a 10-window
  cooldown and wider zone `[1, 7]` to fill the initial batches faster.

### `/window/{n}/state` endpoint

Returns a `GrpoBatchState` JSON object:

```json
{
  "window_start": 1234567,
  "cooldown_prompts": [42, 1701, 3388, ...],
  "valid_submissions": 3,
  "batch_sealed": false
}
```

Miners must read `cooldown_prompts` before picking their `prompt_idx` to
avoid a `PROMPT_IN_COOLDOWN` rejection.

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
... | Window 1234567 open — cooldown_size=142
... | Accepted submission: hotkey=5xxx… prompt_idx=4821 k=4 round=9182736
... | Window 1234567 sealed: batch_size=8 burn_slots=0
... | Uploaded GRPO dataset for window 1234567 (8 prompts, 1347212 bytes)
```

Every 72 windows (≈ 72 min) you'll also see the weight submission log:

```
... | Submitting weights: 14 miners + burn to UID 0
...   5F3sa77: 0.125000  (1/8 × 4 windows)
...   5H1dF9P: 0.125000  (1/8 × 4 windows)
...   (top weighted miners)
```

Sanity checks:

```bash
curl http://localhost:8888/health
# → {"status":"ok","active_window":1234567}

curl http://localhost:8888/window/1234567/state
# → {"window_start":1234567,"cooldown_prompts":[...],"valid_submissions":3,"batch_sealed":false}
```

`active_window` is `null` between windows (briefly, while the next one
spins up). `/window/{n}/state` returns `404 window_not_active` for any `n`
that isn't the currently open window.

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
  this automatically.
- **CooldownMap diverged after restart**: the validator rebuilds the cooldown
  map from R2 on startup. If R2 is unavailable at startup, the map starts
  empty — miners may temporarily get `PROMPT_IN_COOLDOWN` once it repopulates
  over subsequent windows.
