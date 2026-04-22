# Running a Reliquary Validator via Docker

Two supported deployments:

1. **Trainer validator** — subnet owner only. GPU, HF write, full training loop. Manual restart.
2. **Weight-only validator** — anyone with a registered hotkey. CPU only, reads R2, submits EMA-based weights on-chain. Auto-updated by Watchtower.

## Prerequisites (both)

- Docker 24+ with Compose plugin.
- A Bittensor wallet registered on netuid 81 (coldkey stays offline — only the hotkey reaches this box).
- R2 credentials (bucket `reliquary`, the read-only keys are enough for weight-only).

## Trainer deployment (GPU host)

Additional prerequisites:

- NVIDIA driver + CUDA 12.8 on the host.
- NVIDIA Container Toolkit installed and configured.
- HF_TOKEN with write access to your checkpoint repo.
- 150 GB disk for staging checkpoints + HF cache.

```bash
git clone https://github.com/romain13190/reliquary.git
cd reliquary/docker
cp .env.example.trainer .env
# Edit .env — fill in BT_WALLET_NAME, BT_HOTKEY, RELIQUARY_HF_REPO_ID, HF_TOKEN, R2 creds.
export BT_WALLETS_DIR=/home/you/.bittensor/wallets
docker compose -f docker-compose.trainer.yml up -d
docker logs -f reliquary-trainer
```

To resume from an existing checkpoint after a restart, set `RELIQUARY_RESUME_FROM` in `.env` to a HF commit SHA (`sha:<40-hex>`) or a local checkpoint directory (`path:/root/reliquary/state/checkpoints/ckpt_<N>`).

**Security:** the wallet mount is readonly and contains only the hotkey + `coldkeypub.txt`. Never mount your coldkey into the container.

## Weight-only deployment (any Linux host)

```bash
git clone https://github.com/romain13190/reliquary.git
cd reliquary/docker
cp .env.example.weight-only .env
# Edit .env — fill in BT_WALLET_NAME, BT_HOTKEY, R2 creds.
export BT_WALLETS_DIR=/home/you/.bittensor/wallets
docker compose -f docker-compose.weight-only.yml up -d
```

Watchtower polls GHCR every 5 minutes and, when a new `:latest` tag appears, pulls it and restarts the validator container. No operator action required.

To verify Watchtower is alive:

```bash
docker logs watchtower | tail -20
```

You should see periodic `Checking containers for updated images` lines.

## Security notes on the wallet mount

```yaml
volumes:
  - ${BT_WALLETS_DIR}:/root/.bittensor/wallets:ro
```

- `:ro` (read-only) — the container cannot write to the wallet directory.
- The wallet directory should contain **only** `coldkeypub.txt` + `hotkeys/<your-hotkey>`. Not the coldkey private file.
- A good layout is to have a separate `operator-wallet/` folder with just the hotkey copied in and point `BT_WALLETS_DIR` at that, keeping the main `~/.bittensor/wallets` (with the coldkey) elsewhere.
