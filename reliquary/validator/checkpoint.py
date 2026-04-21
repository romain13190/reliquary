"""CheckpointStore: produce → hash → sign → upload → manifest entry.

Single-validator (v2.1) implementation. The validator owns the
checkpoint lifecycle for the whole netuid; multi-validator consensus
on checkpoint hash is a v2.2 concern.
"""

from __future__ import annotations

import hashlib
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
    file_url: str
    file_hash: str       # "sha256:<hex>"
    signature: str       # "ed25519:<hex>" — wallet signs (n || file_hash)


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
      * ``upload_fn`` → ``storage.upload_checkpoint_file`` (added in Task 4)
    Tests inject both as mocks to avoid torch + R2 deps.
    """

    def __init__(
        self,
        validator_hotkey: str,
        wallet: _WalletLike,
        staging_dir_path: str,
        *,
        upload_fn: Callable[[str, str], Awaitable[str]] | None = None,
        save_weights_fn: Callable[[Any, Path], None] | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.wallet = wallet
        self.staging_dir = Path(staging_dir_path)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._upload = upload_fn or _default_upload
        self._save_weights = save_weights_fn or _default_save_weights
        self._current: ManifestEntry | None = None

    def current_manifest(self) -> ManifestEntry | None:
        return self._current

    async def publish(self, checkpoint_n: int, model: Any) -> ManifestEntry:
        """Save → hash → sign → upload → install in manifest."""
        path = self.staging_dir / f"{checkpoint_n}.safetensors"
        self._save_weights(model, path)
        file_hash = self._sha256_file(path)
        sig_payload = f"{checkpoint_n}|{file_hash}".encode()
        sig_bytes = self.wallet.hotkey.sign(sig_payload)
        signature = "ed25519:" + sig_bytes.hex()

        key = f"reliquary/checkpoints/{self.validator_hotkey}/{checkpoint_n}.safetensors"
        url = await self._upload(str(path), key)

        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            file_url=url,
            file_hash=file_hash,
            signature=signature,
        )
        self._current = entry
        logger.info(
            "Published checkpoint %d (hash=%s, url=%s)",
            checkpoint_n, file_hash[:16], url,
        )
        return entry

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return "sha256:" + h.hexdigest()


# ---- production defaults (lazy-imported so tests don't drag torch/R2 in) ----

async def _default_upload(local_path: str, key: str) -> str:
    """Default upload using the R2 helper from Task 4."""
    from reliquary.infrastructure import storage
    return await storage.upload_checkpoint_file(local_path, key)


def _default_save_weights(model: Any, path: Path) -> None:
    """Default save: torch.save the state_dict."""
    import torch
    torch.save(model.state_dict(), path)
