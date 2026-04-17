# Running a Reliquary Validator

This is the minimal guide to start a validator on Bittensor subnet 81.

## What a validator does

Each window (≈ 60 s, 5 blocks), the validator:

1. Derives the same 8 prompts as the miners from drand + block hash.
2. Exposes an HTTP endpoint (`/submit`, `/window/{n}/state`, `/health`) on port 8888.
3. Accepts miner submissions, verifies their GRAIL proofs against its own forward pass, and admits them to the slot if valid.
4. At window end, scores every admitted completion by its advantage relative to the slot, accumulates per-miner scores, and uploads the full dataset to R2.
5. Every `WEIGHT_SUBMISSION_INTERVAL` blocks (≈ 72 windows / ~1 h), aggregates the cumulative scores and sets on-chain weights.

**Only one validator is active per window.** Miners auto-discover it from the metagraph via `validator_permit=True` + a routable axon.

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPU | **1× NVIDIA GPU**, ≥ 24 GB VRAM (runs the reference model forward pass) |
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

You need a bittensor hotkey already registered on netuid 81. To hold `validator_permit`, your stake must be above the subnet's validator threshold — check `btcli subnet metagraph --netuid 81` to see current validator counts and stake requirements.

```bash
btcli wallet new-coldkey --wallet.name my_validator
btcli wallet new-hotkey  --wallet.name my_validator --wallet.hotkey default
btcli subnet register    --wallet.name my_validator --wallet.hotkey default --netuid 81
btcli stake add          --wallet.name my_validator --amount <TAO> --netuid 81
```

After staking, wait for `validator_permit` to flip to `True` on your UID — miners won't send you traffic until then.

## Advertise your HTTP endpoint on-chain

Miners discover you via your axon's advertised IP/port. Set them before starting:

```bash
btcli axon serve \
    --wallet.name my_validator \
    --wallet.hotkey default \
    --netuid 81 \
    --axon.ip <your-public-ip> \
    --axon.port 8888
```

Replace `<your-public-ip>` with the address that miners will reach. If you run behind NAT, port-forward accordingly.

## R2 / S3 credentials

The validator uploads per-window datasets (prompts + accepted completions + rewards) to a Cloudflare R2 bucket (or any S3-compatible endpoint). Set:

```bash
export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_ACCESS_KEY_ID="<r2-access-key>"
export R2_SECRET_ACCESS_KEY="<r2-secret-key>"
export R2_BUCKET_ID="reliquary"                      # defaults to "reliquary"
# Optional — defaults to https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
# export R2_ENDPOINT_URL="https://..."
# export R2_REGION="us-east-1"                       # default
```

Make sure the bucket exists and the key can `PutObject`/`GetObject` on keys prefixed `reliquary/`.

## Fetch the reference checkpoint

You must run the same model as miners — GRAIL's correctness depends on bit-identical forward passes. Ask in the subnet's coordination channel for the pinned checkpoint, then:

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
... | Slot 3 auto-finalized at 32/32 accepted
... | Window 1234567 settled (all slots finalized)
... | Window 1234567 scored: 14 miners (total 203.84), burn 52.16
... | Uploaded GRPO dataset for window 1234567 (8 slots, 1347212 bytes)
```

Every 72 windows (≈ 72 min) you'll also see the weight submission log:

```
... | Submitting weights: 14 miners + 0.1891 burn to UID 0 (miner_total=14672.51, burn_score=3421.04)
...   5F3sa77: 0.083421
...   5H1dF9P: 0.071304
...   (top-10 weighted miners)
```

Sanity checks:

```bash
curl http://localhost:8888/health
# → {"status":"ok","active_window":1234567}

curl http://localhost:8888/window/1234567/state
# → {"window_start":1234567,"slot_states":[...]}
```

`active_window` is `null` between windows (briefly, while the next one spins up). `/window/{n}/state` returns `404 window_not_active` for any `n` that isn't the currently open window.

## Every window produces

One gzipped JSON file in R2 at `reliquary/dataset/window-<N>.json.gz`:

```json
{
  "window_start": 1234567,
  "randomness": "…",
  "environment": "gsm8k",
  "slots": [
    {
      "slot_index": 0,
      "prompt_id": "…",
      "prompt": "Natalia sold…",
      "ground_truth": "72",
      "settled": true,
      "completions": [
        {"miner_hotkey": "5xxx…", "tokens": [...], "completion_text": "…", "reward": 1.0},
        …
      ]
    },
    …
  ]
}
```

This is the raw GRPO-ready rollout dataset. Consumers (training, analysis) read from this bucket.

## Monitoring

```bash
# GPU busy during windows, idle between.
nvidia-smi

# Tail validator logs.
tail -f ~/validator.log

# Check the most recent uploaded window file.
aws s3 ls s3://reliquary/dataset/ --endpoint-url https://<account>.r2.cloudflarestorage.com | tail -5
```

## Troubleshooting

- **No submissions arriving**: miners can't reach you. Check `btcli axon serve` output matches your real public IP + port, firewall allows inbound TCP, and `validator_permit` is `True`.
- **`R2 upload failed`**: credentials, bucket name, or endpoint URL wrong. Test with a plain `aws s3 cp` using the same env vars before blaming the validator.
- **High `duplicate_prefix` / `invalid_proof` rates**: miners are running a different checkpoint than you. Confirm with the subnet coordinator that everyone is pinned to the same model hash.
- **Weights not submitted**: `set_weights` requires `validator_permit` AND a non-trivial emission window. Check logs for `set_weights` attempts; failures are logged with the chain error string.
- **Window times out with few completions**: either not enough miners are online, or they can't reach your HTTP endpoint. Both show up as `len(accepted_completions) < GROUP_SIZE` at slot finalization.
