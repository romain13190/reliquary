# Deployment

End-to-end guide for launching a Reliquary validator and connecting miners. Assumes Bittensor-level familiarity; explicit about Reliquary-specific configuration.

## 1. Bittensor prerequisites

- Coldkey and hotkey created locally (`btcli wallet new-coldkey`, `btcli wallet new-hotkey`).
- Hotkey registered on netuid 81 (`btcli subnet register --netuid 81`).
- For validators: stake above the subnet's `validator_permit` threshold. Check `btcli subnet metagraph --netuid 81` for current thresholds.
- Miners only need registration; no minimum stake.

## 2. Hugging Face setup (validator only)

The validator pushes every checkpoint revision to a public HF repo. Miners pull from the same repo using the revision exposed in `/state`.

### Create the model repo

```bash
pip install huggingface-hub
huggingface-cli login            # saves token to ~/.cache/huggingface/token
huggingface-cli repo create your-org/reliquary-sn --type model
```

Seed it with the base model so `snapshot_download` succeeds on the first miner pull:

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./base-model
huggingface-cli upload your-org/reliquary-sn ./base-model . \
    --commit-message "seed from Qwen/Qwen3-4B-Instruct-2507"
```

### Set the token on the validator host

```bash
export HF_TOKEN=hf_xxx   # must have write access to your-org/reliquary-sn
```

The validator reads `HF_TOKEN` from the environment at startup. The default target repo is `aivolutionedge/reliquary-sn`; override with `--hf-repo-id`.

## 3. R2 (or S3) setup

The validator archives per-window rollout datasets to R2 at `reliquary/dataset/window-<N>.json.gz` (flat layout — no hotkey prefix). The `validator_hotkey` is stored inside the archive JSON body for provenance. The validator reads these archives at startup to derive `window_n`, replay the EMA, and reconstruct the prompt cooldown map.

Environment variables required on the validator host:

```bash
export R2_ACCOUNT_ID=<cloudflare-account-id>
export R2_BUCKET_ID=reliquary
export R2_ACCESS_KEY_ID=<key-id>
export R2_SECRET_ACCESS_KEY=<secret>
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
```

S3-compatible; AWS S3 works with the equivalent `AWS_*` environment variables and an appropriate endpoint.

Ensure the bucket exists and the key has `PutObject` and `GetObject` permissions on the `reliquary/` prefix.

## 4. Hardware requirements

### Validator

| Component | Requirement |
|---|---|
| GPU | 1× A100 40 GB or better (model + ref_model + gradients + activations in bfloat16) |
| RAM | 64 GB minimum |
| Disk | 100 GB (staging checkpoints, state file, dataset archives before upload) |
| Network | Stable HTTPS outbound (HF, R2, Bittensor chain), inbound TCP on your chosen HTTP port |

### Miner

| Component | Requirement |
|---|---|
| GPU | 1× or 2× NVIDIA GPU with 24 GB+ VRAM each. With two GPUs: GPU 0 handles generation, GPU 1 handles the GRAIL proof forward pass. With one GPU: both run on GPU 0. |
| RAM | 32 GB |
| Disk | 50 GB (model cache) |
| Network | Stable HTTPS outbound (HF, validator) |

Two GPUs are the reference configuration. One GPU works if VRAM is sufficient for two model copies (two copies of Qwen3-4B-Instruct = ~16 GB bfloat16).

## 5. Install

```bash
git clone <repo-url> reliquary
cd reliquary
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify:

```bash
reliquary --help
```

You should see `mine` and `validate` subcommands.

## 6. Launch the validator

```bash
reliquary validate \
    --wallet-name my_validator \
    --hotkey default \
    --netuid 81 \
    --network finney \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo-id your-org/reliquary-sn \
    --http-host 0.0.0.0 \
    --http-port 8888 \
    --external-ip <public-ip-of-this-host> \
    --external-port 8888
```

`--external-ip` / `--external-port` control what the validator publishes on-chain so miners can discover it from the metagraph. If omitted, miners must pass `--validator-url http://<ip>:<port>` explicitly.

`--checkpoint` is the base model loaded at startup. On subsequent windows the in-memory model is trained in-place and published to HF; the `--checkpoint` arg is only used for the initial load.

The first window seals, trains, and (because no checkpoint exists yet) immediately publishes checkpoint_n=1 to your HF repo. Subsequent publishes happen every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` windows.

Full CLI flag reference:

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint` | `Qwen/Qwen3-4B-Instruct-2507` | HF repo id or local path for the initial model load |
| `--hf-repo-id` | `aivolutionedge/reliquary-sn` | HF repo to push trained checkpoints to |
| `--environment` | `gsm8k` | Reward environment. Must match miners. |
| `--http-host` | `0.0.0.0` | Bind address |
| `--http-port` | `8888` | Listen port |
| `--external-ip` | *(empty)* | Public IP advertised on-chain via axon |
| `--external-port` | `0` (→ http-port) | Public port advertised on-chain |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Disable only for offline testing |

## 7. Running a weight-only validator

A weight-only validator reads archives from R2 and submits weights on-chain. It does not train, does not require a GPU, and does not write to HF. Operators who want to participate in consensus without the compute cost of training run this mode.

Environment variables needed: `R2_*` (read access) and bittensor wallet setup. No `HF_TOKEN` required.

```bash
reliquary validate --no-train \
    --wallet-name my_validator \
    --hotkey default \
    --netuid 81
```

The node polls the chain; every `WEIGHT_SUBMISSION_INTERVAL = 360` blocks it reads the last ~216 windows' archives from R2, replays the EMA, and sets weights on-chain. Multiple weight-only validators reading the same R2 bucket submit identical weights (deterministic EMA replay).

**Note (v2.1):** There is a single trainer writing to R2. Multiple trainers in the same bucket would collide on archive keys. Multi-trainer consensus is v2.2 work.

## 8. Launch a miner

```bash
reliquary mine \
    --wallet-name my_miner \
    --hotkey default \
    --netuid 81 \
    --network finney \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
    --environment gsm8k
```

`--checkpoint` is the fallback if the validator has no published checkpoint yet. Once the validator publishes, the miner auto-detects the latest HF revision via `/state` and downloads it.

If auto-discovery fails (no validator with `validator_permit` on the metagraph, or validator did not set `--external-ip`):

```bash
reliquary mine ... --validator-url http://<validator-ip>:8888
```

Full CLI flag reference:

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint` | *(required)* | Fallback model if validator has no checkpoint yet |
| `--environment` | `gsm8k` | Must match the validator |
| `--validator-url` | *(auto-discovered)* | Override for local testing or when metagraph discovery fails |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Disable only for offline testing |

## 9. First-run checklist

Verify each of these before considering the deployment healthy:

- [ ] Validator logs `Validator started (v2.1)`
- [ ] Validator logs `Rebuilt cooldown from 0 archive windows` (0 on first run)
- [ ] Validator logs `Published checkpoint 1 to your-org/reliquary-sn@<SHA>` after the first window seals
- [ ] `curl http://<validator>:8888/health` returns `{"status":"ok",...}`
- [ ] `curl http://<validator>:8888/state` returns a 200 with `"state":"open"` and `checkpoint_n > 0`
- [ ] `curl http://<validator>:8888/checkpoint` returns a 200 with `repo_id`, `revision`, and `signature`
- [ ] Miner logs `Validator at http://... is on checkpoint N`
- [ ] Miner logs `submitted window=N prompt=X accepted=True` at least once

## 10. Operational concerns

**HF repo bloat.** There is no automatic checkpoint GC. Each publish creates a new HF commit; old revisions accumulate. Plan periodic manual cleanup or a cron that deletes commits older than N days.

**R2 bucket growth.** Every window archives roughly 1 MB compressed. Set a lifecycle rule in your R2 bucket to expire objects under `reliquary/dataset/` older than your retention target.

**Validator restarts.** The optimizer (AdamW momentum) and LR scheduler step count are module-level singletons — they are not persisted across restarts. A restart resets warmup, causing a brief training instability. Minimize unnecessary restarts. `window_n` is derived at startup from the maximum R2 archive window present; `checkpoint_n` from HF commit history; the EMA is replayed from the last ~216 R2 archives. No local state file required — loss of disk means no data loss.

**Drand round (placeholder).** The `current_round` field in `/state` is currently a window counter, not a real drand beacon round. Miner and validator stay mutually consistent but external auditability via drand is limited until v2.2.

**Serving the axon.** The `--external-ip` / `--external-port` flags cause the validator to call `bt.Axon.serve` at startup. If the hotkey is not registered or stake is too low for `validator_permit`, this call will fail with a logged error. Miners then cannot discover the validator from the metagraph and need `--validator-url`.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 503 from `/submit` with `no_active_window` | Validator is in `TRAINING` or `PUBLISHING` | Miner should poll `/state` and retry when `state=open` |
| 409 from `/submit` with `window_mismatch` | Miner's `window_start` is stale | Refresh `/state` and use the current `window_n` |
| `WRONG_CHECKPOINT` | Miner's `checkpoint_hash` does not match active revision | Next `/state` poll updates the revision; miner will re-download |
| `OUT_OF_ZONE` | Rollout rewards are too uniform | Pick a different prompt |
| `REWARD_MISMATCH` | Miner's `env.compute_reward` disagrees with validator | Confirm env version alignment; both must run the same environment |
| `GRAIL_FAIL` | Sketch mismatch | Check that miner and validator use the same checkpoint, same `attn_implementation` (`flash_attention_2`), and same CUDA/torch build |
| Validator stuck in `TRAINING` or `PUBLISHING` | Slow GPU or HF upload | Check `nvidia-smi` for GPU contention; check HF Hub connectivity |
| `serve_axon failed` | Hotkey not registered or insufficient stake | Verify registration and stake with `btcli subnet metagraph --netuid 81` |
| Miners not discovering validator | `--external-ip` not set or axon not published | Set `--external-ip`, or have miners use `--validator-url` |
