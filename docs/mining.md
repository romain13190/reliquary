# Running a Reliquary Miner

This is the minimal guide to start a miner on Bittensor subnet 81.

## What a miner does (v2.1)

Windows are event-driven, not time-based. A window seals the instant B=8
valid distinct-prompt submissions land; there is no fixed 60-second cadence.
Every miner on the subnet runs a continuous poll-submit loop:

1. **Polls `/window/state`** — calls `GET /window/state` continuously. The
   response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`,
   `checkpoint_url`, `checkpoint_hash`, and `cooldown_prompts`.
   - If `state != "open"`, the validator is in `TRAINING` or `PUBLISHING` —
     sleep briefly (1–2 s) and re-poll. Do not submit while the window is
     not open.
   - If `checkpoint_n` advanced since the last poll, download `checkpoint_url`
     and load the new weights before generating any rollouts.
2. **Picks a prompt** — selects a `prompt_idx` freely from the GSM8K env
   (~7473 problems), avoiding any index in `cooldown_prompts`. The window
   identifier is `state.window_n` — there is no block-based window start.
3. **Generates rollouts** — runs M=8 completions at `T_PROTO` (≈ 0.9),
   `top_p=TOP_P_PROTO`, `top_k=TOP_K_PROTO` on the current checkpoint.
4. **Computes local rewards** — calls `env.compute_reward` on each rollout.
   The validator recomputes rewards independently; claims that don't match
   are rejected (`REWARD_MISMATCH`). Only deterministic envs (like GSM8K)
   are used, so results are always reproducible.
5. **Builds GRAIL sketches** — runs the bit-identical HuggingFace forward
   pass and generates sketch commitments that bind the completions to the
   model (unchanged from v1).
6. **Submits** — POSTs a `BatchSubmissionRequest` to the validator containing:
   `prompt_idx`, 8 rollouts, local rewards, GRAIL commits, `merkle_root`,
   `signed_round` (the most recent drand round the miner observed), and
   `checkpoint_hash` (copied from the last `/window/state` response).

The validator processes submissions in real time. Only the first `B = 8`
accepted submissions with distinct `prompt_idx` values that pass the zone
filter form the training batch for that window.

### Prompt selection strategy

The reference miner strategy (`pick_prompt_idx` in
`reliquary/miner/engine.py`) is uniform-random sampling with
rejection-sampling against the cooldown set. Before generating rollouts,
fetch the window state:

```
GET /window/state  →  GrpoBatchState
```

- Read `cooldown_prompts` and pick any `prompt_idx ∉ cooldown_prompts`.
- Read `checkpoint_hash` and include it verbatim in your submission.
- Read `window_n` and use it as the authoritative window identifier —
  do not use block numbers for window timing.

There is no other constraint on which prompt to pick — the whole GSM8K
index is open.

### Zone filter

The validator counts how many of your 8 rollouts have `reward = 1`
(correct). Call that `k`. Submissions with `k ∈ [2, 6]` pass the zone
filter; submissions with `k < 2` or `k > 6` are rejected with
`OUT_OF_ZONE`. During bootstrap (first `BOOTSTRAP_WINDOWS` windows) the
zone is wider: `k ∈ [1, 7]`.

This means you cannot cherry-pick an easy prompt to get 8/8 corrects, nor
fail on a hard prompt with 0/8. Both extremes are worthless for GRPO
training and earn nothing.

### Rejection reasons

| Reason | Meaning | Action |
|---|---|---|
| `WRONG_CHECKPOINT` | `checkpoint_hash` in your submission doesn't match the active window | Re-poll `/window/state`, update to the latest `checkpoint_hash`, and retry |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` is in the active cooldown set | Retry with a different `prompt_idx` |
| `STALE_ROUND` | `signed_round` is too old or from the future | Ensure your drand client is synced |
| `OUT_OF_ZONE` | `k` (correct count) outside `[2, 6]` | Choose a different prompt; this one is too easy or too hard for you right now |
| `REWARD_MISMATCH` | Your local rewards don't match `env.compute_reward` | Check checkpoint and env version alignment |
| `GRAIL_FAIL` | Sketch doesn't match the validator's forward pass | Wrong checkpoint or CUDA environment mismatch |

`WRONG_CHECKPOINT` is the most common transient rejection. It happens in the
brief window between when the validator publishes a new checkpoint and when
your miner polls the update. Re-poll `/window/state`, pick up the new
`checkpoint_hash` and `checkpoint_url`, download the weights if `checkpoint_n`
advanced, and resubmit.

### Payment model

Only the `B = 8` submissions that make the sealed batch earn anything. Each
batch member earns a flat `1/B` of the window's emission. All other
submissions earn zero. The two levers are:

1. **Speed** — earlier `signed_round` → higher priority in FIFO selection.
   Extra computation (cherry-picking easy prompts) costs time → pushes
   `signed_round` later → higher risk of being displaced.
2. **Novel prompt** — if your `prompt_idx` duplicates one already in the
   batch, your submission is dropped even if it's otherwise valid. Spreading
   across the prompt space is always optimal.

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

Both GPUs load the same checkpoint — GPU 0 runs generation, GPU 1 runs the
bit-identical HuggingFace forward pass used to construct the GRAIL proof.
They must agree down to the tensor level, so avoid mixing GPU models where
possible.

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

Check that your hotkey shows up on the metagraph with a valid UID before
going further.

## Fetch the reference checkpoint

The miner and the active validator must run the exact same checkpoint — this
is enforced indirectly by GRAIL proof verification (the validator re-runs
your forward pass and compares sketches). Ask in the subnet's coordination
channel for the current pinned checkpoint hash, then:

```bash
huggingface-cli download <org/model-id> --local-dir ./checkpoints/current
```

Any path that `AutoModelForCausalLM.from_pretrained(...)` accepts works
here — local directory or HF hub id.

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

No R2/S3 credentials are needed on the miner — only the validator uploads
the window dataset.

## What you should see

On a healthy startup:

```
... | Starting Reliquary miner (network=finney, netuid=81, env=gsm8k)
... | Loading models from ./checkpoints/current...
... | Miner ready. Entering main loop.
... | window_n=42 state=open checkpoint_n=7 — polling validator http://x.x.x.x:8888
... | checkpoint_n advanced to 7 — downloading new weights
... | prompt_idx=4821 checkpoint_hash=sha256:abc… accepted=True reason='ok'
... | prompt_idx=4821 batch_position=3/8
```

If submissions are rejected, the `reason` string tells you why (see the
rejection table above). On `PROMPT_IN_COOLDOWN`, pick a new `prompt_idx`
and resubmit. On `WRONG_CHECKPOINT`, re-poll `/window/state` to get the
latest `checkpoint_hash` before retrying.

## Monitoring & stopping

The miner loop runs until killed. Between windows (when `/window/state`
returns `state != "open"`) it sleeps briefly (1–2 s) and re-polls. On
errors it backs off for up to 12 s. No state is kept between windows, so
restarting is safe — you'll resume mining the next open window.

Common operational checks:

```bash
# Both GPUs should be near full utilization during a window.
nvidia-smi

# Verify the miner is polling the chain cleanly.
grep -E "Mining window|error" ~/miner.log | tail -50
```

## Troubleshooting

- **`no validator with permit and routable axon`**: no active validator has
  advertised an HTTP endpoint on the metagraph yet. Wait or use
  `--validator-url` to point at a known one.
- **CUDA out of memory**: both models need VRAM; your checkpoint may be too
  large for your GPU. Use a smaller model or swap to GPUs with more VRAM.
- **`GRAIL_FAIL` / `REWARD_MISMATCH`**: your local compute diverged from the
  validator's expected result. Most often caused by a different
  `attn_implementation` build, CUDA/torch version mismatch, or wrong
  checkpoint. Re-install on a clean env and confirm the checkpoint hash.
- **All submissions land `OUT_OF_ZONE`**: the prompts you're selecting are
  too easy (k=8) or too hard (k=0) for the current checkpoint. The miner
  engine samples uniformly across the full index; if you've overridden
  prompt selection, broaden the range.
- **Persistent `WRONG_CHECKPOINT`**: your miner is not picking up the
  latest `checkpoint_hash` from `/window/state`. Ensure the poll loop
  updates the hash before each submission. If the validator has just
  published checkpoint N+1, submissions carrying the old hash are rejected
  until you re-poll.
- **Nothing accepted for several windows**: check your checkpoint hash
  matches the one advertised by `/window/state`. GRAIL rejects proofs from
  the wrong model via sketch mismatch; `WRONG_CHECKPOINT` rejects before
  even reaching GRAIL.
