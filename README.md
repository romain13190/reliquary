# Reliquary

Decentralized GRPO training for large language models on Bittensor subnet 81.

Reliquary is a coordination protocol that turns a set of independent GPU operators into a single distributed RLHF pipeline. Miners generate cryptographically-proven rollouts; the validator aggregates them into a GRPO training batch, updates a live LLM checkpoint, and publishes the result to Hugging Face — all without trusting any single participant.

## What it does

Each training window is one GRPO step. The cadence is event-driven: a window seals the instant eight valid, distinct-prompt rollout groups land. Miners race to submit; the first eight in (by `signed_round`) win the batch. The validator runs a PPO-clipped surrogate loss with a KL penalty against the frozen reference, then pushes the updated weights to a public HF repo. The whole cycle repeats immediately.

The network produces three artefacts: a continuously-trained model (published to HF every ten windows), a per-window rollout dataset (archived to R2), and a signed checkpoint manifest (served from `/checkpoint`) that lets anyone verify the chain of custody from a base model through every training step. The audit trail is cryptographic — each rollout carries a GRAIL sketch that lets the validator re-run the forward pass and confirm the generation came from the announced checkpoint.

Validators hold stake and run the training loop. Miners hold hotkeys, run GPU inference, and earn emission proportional to their share of batch slots over a rolling 72-window scoring interval. Downstream consumers — researchers, fine-tuning pipelines — pull the published HF checkpoint or the R2 rollout dataset directly.

## Quickstart

- To mine: see [docs/mining.md](docs/mining.md)
- To validate: see [docs/validating.md](docs/validating.md)
- To understand the mechanism: see [docs/concepts.md](docs/concepts.md)
- To deploy end-to-end: see [docs/deployment.md](docs/deployment.md)
- For GRPO loss internals: see [docs/training.md](docs/training.md)

## Architecture at a glance

```
┌─────────────┐    HTTP    ┌─────────────┐   HF push   ┌──────────────┐
│   Miners    │ ─────────▶ │  Validator  │ ──────────▶ │   HF Hub     │
│  (N nodes)  │ ◀───────── │  (1 node)   │             │ (model repo) │
└─────────────┘ /submit    └──────┬──────┘             └──────┬───────┘
     ▲         /state             │                            │
     │         /checkpoint        │ weights                    │ pull
     │                            ▼                            │
     │                   ┌──────────────┐                      │
     │                   │  Bittensor   │                      │
     │                   │  chain       │                      │
     │                   │  (set_weights│                      │
     │                   │   every 360  │                      │
     │                   │   blocks)    │                      │
     │                   └──────────────┘                      │
     │                                                         │
     │                   ┌──────────────┐                      │
     └───────────────────│     R2       │◀────── archive ──────┘
                         │ (rollouts +  │         (per window)
                         │  dataset)    │
                         └──────────────┘
```

Miners submit rollout groups to `/submit` and poll `/state` for checkpoint updates. The validator trains, publishes to HF, and broadcasts weights on-chain every `WEIGHT_SUBMISSION_INTERVAL = 360` blocks. Miners pull new weights via `/state` → HF `snapshot_download`. R2 stores the per-window rollout archive; the validator reads it at startup to rebuild the prompt cooldown map.

## Status

- **v1** — verifiable-inference dataset production (shipped, deprecated)
- **v2** — GRPO market with in-subnet training (shipped)
- **v2.1** — batch-driven windows, HF checkpoint distribution, EMA scoring (current)
- **v2.2** — multi-validator consensus + real drand-backed `signed_round` (planned)

## License

MIT — see `LICENSE`.
