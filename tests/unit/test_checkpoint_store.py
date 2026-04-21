"""CheckpointStore: produce → hash → sign → upload to R2 → in-memory manifest."""

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.validator.checkpoint import CheckpointStore, ManifestEntry


class FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        def sign(self, data):
            return b"signature_for_" + data[:16]
    hotkey = _Hk()


def _save_weights_stub(model, path):
    """Pretend to save model weights — write deterministic bytes."""
    path.write_bytes(b"weights_for_" + str(id(model)).encode())


def test_initial_manifest_is_none(tmp_path):
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
    )
    assert store.current_manifest() is None


@pytest.mark.asyncio
async def test_publish_writes_uploads_signs_and_serves(tmp_path):
    fake_upload = AsyncMock(return_value="https://r2.example/cp/1.safetensors")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    model = MagicMock(name="mock_model")
    entry = await store.publish(checkpoint_n=1, model=model)
    assert isinstance(entry, ManifestEntry)
    assert entry.checkpoint_n == 1
    assert entry.file_url == "https://r2.example/cp/1.safetensors"
    assert entry.file_hash.startswith("sha256:")
    assert entry.signature.startswith("ed25519:")
    fake_upload.assert_awaited_once()
    assert store.current_manifest() is entry


@pytest.mark.asyncio
async def test_publish_increments_overrides_previous(tmp_path):
    fake_upload = AsyncMock(side_effect=[
        "https://r2.example/cp/1.safetensors",
        "https://r2.example/cp/2.safetensors",
    ])
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    await store.publish(checkpoint_n=1, model=MagicMock())
    e2 = await store.publish(checkpoint_n=2, model=MagicMock())
    assert store.current_manifest() is e2
    assert store.current_manifest().checkpoint_n == 2


@pytest.mark.asyncio
async def test_file_hash_deterministic_for_same_bytes(tmp_path):
    """Same model bytes → same file_hash."""

    async def fake_upload(local_path, key):
        return f"https://r2.example/{key}"

    def deterministic_save(model, path):
        path.write_bytes(b"identical_bytes")

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=deterministic_save,
    )
    e1 = await store.publish(1, model=object())
    e2 = await store.publish(2, model=object())
    assert e1.file_hash == e2.file_hash  # same bytes → same hash
    assert e1.checkpoint_n != e2.checkpoint_n


@pytest.mark.asyncio
async def test_signature_includes_n_and_hash(tmp_path):
    """Signature payload = checkpoint_n || file_hash so tampering is detectable."""
    async def fake_upload(local_path, key):
        return f"https://r2.example/{key}"

    captured = {}

    class CapturingWallet:
        class _Hk:
            ss58_address = "5FHk"
            def sign(self, data):
                captured["signed"] = data
                return b"fake_sig"
        hotkey = _Hk()

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=CapturingWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    entry = await store.publish(42, model=object())
    assert b"42" in captured["signed"]
    assert entry.file_hash.encode() in captured["signed"]
