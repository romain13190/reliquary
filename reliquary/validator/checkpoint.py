"""CheckpointStore: save → upload to HuggingFace → sign → manifest entry.

Single-validator (v2.1) implementation. The validator owns the
checkpoint lifecycle for the whole netuid; multi-validator consensus
on checkpoint hash is a v2.2 concern.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """A published checkpoint entry."""

    checkpoint_n: int
    repo_id: str          # HF repo, e.g. "aivolutionedge/reliquary-sn"
    revision: str         # HF commit SHA (serves as strong content hash)
    signature: str        # "ed25519:<hex>" — wallet signs (checkpoint_n || revision)


class _WalletLike(Protocol):
    """Minimal wallet shape — tests inject a stub, prod injects bittensor wallet."""

    class hotkey:
        ss58_address: str
        @staticmethod
        def sign(data: bytes) -> bytes: ...


class CheckpointStore:
    """Owns the in-memory current manifest + the publish lifecycle.

    Production wiring (defaults):
      * ``save_weights_fn`` → ``torch.save(model.state_dict(), path)``
      * ``upload_fn`` → HuggingFace Hub upload_file (lazy import)
    Tests inject both as mocks to avoid torch + HF deps.
    """

    def __init__(
        self,
        validator_hotkey: str,
        wallet: _WalletLike,
        repo_id: str,
        staging_dir_path: str,
        *,
        hf_token: str | None = None,
        upload_fn: Callable[..., Awaitable[str]] | None = None,
        save_weights_fn: Callable[[Any, Path], None] | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.wallet = wallet
        self.repo_id = repo_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.staging_dir = Path(staging_dir_path)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._upload = upload_fn or _default_upload
        self._save_weights = save_weights_fn or _default_save_weights
        self._current: ManifestEntry | None = None

    def current_manifest(self) -> ManifestEntry | None:
        return self._current

    async def publish(self, checkpoint_n: int, model: Any) -> ManifestEntry:
        """Save locally → upload to HF → sign (n || revision) → install manifest."""
        # 1. Save locally
        path = self.staging_dir / f"{checkpoint_n}.safetensors"
        self._save_weights(model, path)

        # 2. Upload to HF — returns the new commit revision (SHA)
        revision = await self._upload(
            local_path=str(path),
            repo_id=self.repo_id,
            path_in_repo="model.safetensors",
            commit_message=f"checkpoint {checkpoint_n}",
        )

        # 3. Sign (n || revision) — strong cross-validator proof
        sig_payload = f"{checkpoint_n}|{revision}".encode()
        sig_bytes = self.wallet.hotkey.sign(sig_payload)
        signature = "ed25519:" + sig_bytes.hex()

        # 4. Install manifest
        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            repo_id=self.repo_id,
            revision=revision,
            signature=signature,
        )
        self._current = entry
        logger.info(
            "Published checkpoint %d to %s@%s",
            checkpoint_n, self.repo_id, revision[:12],
        )
        return entry


# ---- production defaults (lazy-imported so tests don't drag torch/HF in) ----

async def _default_upload(
    local_path: str,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    """Upload via huggingface_hub.HfApi.upload_file.

    Returns the commit revision SHA (strong hash of the repo state).
    Runs in a thread — HfApi is sync.
    """
    import asyncio
    from huggingface_hub import HfApi

    def _sync_upload():
        api = HfApi()
        commit_info = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        # CommitInfo.oid holds the commit SHA
        return commit_info.oid

    return await asyncio.to_thread(_sync_upload)


def _default_save_weights(model: Any, path: Path) -> None:
    """Default save: torch.save the state_dict."""
    import torch
    torch.save(model.state_dict(), path)
