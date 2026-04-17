# Running a Reliquary Miner

This is the minimal guide to start a miner on Bittensor subnet 81.

## What a miner does

Every ~60 seconds (one window = 5 blocks), every miner on the subnet:

1. Derives the same 8 prompts as everyone else from the drand beacon + block hash.
2. Generates rollouts (completions) on those prompts with the reference model.
3. Attaches a GRAIL cryptographic proof to each rollout.
4. Submits them to the active validator over HTTP.

The validator verifies the proofs and scores miners by how much useful GRPO training signal each rollout contributes (advantage-based scoring). Miners earn emissions proportional to their cumulative advantage across windows.

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPUs | **2× NVIDIA GPUs**, each ≥ 24 GB VRAM (one for generation, one for proof construction) |
| CUDA | 12.x with `flash-attn` compatible drivers |
| RAM | 32 GB minimum |
| Disk | 50 GB (model weights + caches) |
| Network | Stable outbound HTTPS to drand relays + HTTP to active validator |
| Bittensor wallet | Created and registered on netuid 81 |

Both GPUs load the same checkpoint — GPU 0 runs generation, GPU 1 runs the bit-identical HuggingFace forward pass used to construct the GRAIL proof. They must agree down to the tensor level, so avoid mixing GPU models where possible.

## Install

```bash
git clone <repo-url> reliquary
cd reliquary
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify the CLI is reachable:

```bash
reliquary --help
```

You should see `mine` and `validate` subcommands.

## Register your hotkey on the subnet

This is a Bittensor-level step, not Reliquary-specific. If you haven't already:

```bash
btcli wallet new-coldkey --wallet.name my_miner
btcli wallet new-hotkey  --wallet.name my_miner --wallet.hotkey default
btcli subnet register    --wallet.name my_miner --wallet.hotkey default --netuid 81
```

Check that your hotkey shows up on the metagraph with a valid UID before going further.

## Fetch the reference checkpoint

The miner and the active validator must run the exact same checkpoint — this is enforced indirectly by GRAIL proof verification (the validator re-runs your forward pass and compares sketches). Ask in the subnet's coordination channel for the current pinned checkpoint hash, then:

```bash
huggingface-cli download <org/model-id> --local-dir ./checkpoints/current
```

Any path that `AutoModelForCausalLM.from_pretrained(...)` accepts works here — local directory or HF hub id.

## Launch

```bash
reliquary mine \
    --network finney \
    --netuid 81 \
    --wallet-name my_miner \
    --hotkey default \
    --checkpoint ./checkpoints/current \
    --log-level INFO
```

Additional flags:

| Flag | Default | When to use it |
|---|---|---|
| `--environment` | `gsm8k` | Pinned by the protocol; don't change unless the subnet announces a migration. |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Turn off only for offline testing. Mainnet always uses drand. |
| `--validator-url` | *(auto-discovered)* | Override for local testing, e.g. `http://127.0.0.1:8888`. In production leave empty; the miner discovers the active validator from the metagraph. |

Environment variables the miner honors:

| Variable | Default | Purpose |
|---|---|---|
| `DRAND_CHAIN` | `quicknet` | Only override if drand announces a chain rotation. |

No R2/S3 credentials are needed on the miner — only the validator uploads the window dataset.

## What you should see

On a healthy startup:

```
... | Starting Reliquary miner (network=finney, netuid=81, env=gsm8k)
... | Loading models from ./checkpoints/current...
... | Miner ready. Entering main loop.
... | Mining window 1234567 — 30s budget, validator http://x.x.x.x:8888
... | slot 0: accepted=True  reason='ok' settled=False slot_count=4
... | slot 1: accepted=True  reason='ok' settled=False slot_count=4
...
```

If submissions are rejected, the `reason` string tells you why. The ones you'll see most often:

- `slot_full` — the slot already reached 32 completions; try another slot or window.
- `duplicate_prefix` — another miner already landed the same first 8 tokens; retry with a different sample.
- `prompt_mismatch` — your derived prompt doesn't match the validator's. Almost always means you're on the wrong `--environment` or your block hash fetch is out of sync.

## Monitoring & stopping

The miner loop runs until killed. It sleeps 6 s between windows and 12 s on errors. No state is kept between windows, so restarting is safe — you'll resume mining the next window.

Common operational checks:

```bash
# Both GPUs should be near full utilization during a window.
nvidia-smi

# Verify the miner is polling the chain cleanly.
grep -E "Mining window|error" ~/miner.log | tail -50
```

## Troubleshooting

- **`no validator with permit and routable axon`**: no active validator has advertised an HTTP endpoint on the metagraph yet. Wait or use `--validator-url` to point at a known one.
- **CUDA out of memory**: both models need VRAM; your checkpoint may be too large for your GPU. Use a smaller model or swap to GPUs with more VRAM.
- **`token_mismatch` / `invalid_proof`**: your local compute diverged from the validator's expected result. Most often caused by a different `attn_implementation` build or a different CUDA/torch version. Re-install on a clean env.
- **Nothing accepted for several windows**: check your checkpoint hash matches the subnet-announced one. GRAIL rejects proofs from the wrong model silently (via sketch mismatch).
