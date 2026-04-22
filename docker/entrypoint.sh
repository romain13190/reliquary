#!/bin/bash
# Entrypoint for the Reliquary validator image.
#
# Reads environment variables to build the `reliquary validate` argv.
# Required: BT_WALLET_NAME, BT_HOTKEY.
# Mode is controlled by RELIQUARY_TRAIN=1 (trainer) or !=1 (weight-only).
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required (the wallet dir name under ~/.bittensor/wallets)}"
: "${BT_HOTKEY:?BT_HOTKEY is required (the hotkey file name under wallets/<name>/hotkeys/)}"

args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
)

if [[ "${RELIQUARY_TRAIN:-0}" == "1" ]]; then
  : "${RELIQUARY_HF_REPO_ID:?RELIQUARY_HF_REPO_ID required in trainer mode (target HF repo for checkpoints)}"
  args+=(
    --train
    --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}"
    --hf-repo-id   "${RELIQUARY_HF_REPO_ID}"
    --http-host    "${RELIQUARY_HTTP_HOST:-0.0.0.0}"
    --http-port    "${RELIQUARY_HTTP_PORT:-8080}"
  )
  [[ -n "${RELIQUARY_EXTERNAL_IP:-}" ]]     && args+=(--external-ip   "${RELIQUARY_EXTERNAL_IP}")
  [[ -n "${RELIQUARY_EXTERNAL_PORT:-}" ]]   && args+=(--external-port "${RELIQUARY_EXTERNAL_PORT}")
  [[ -n "${RELIQUARY_RESUME_FROM:-}" ]]     && args+=(--resume-from   "${RELIQUARY_RESUME_FROM}")
else
  args+=(--no-train)
fi

echo "Launching: reliquary validate ${args[*]}"
exec reliquary validate "${args[@]}"
