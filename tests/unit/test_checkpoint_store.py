"""CheckpointStore: save → upload to HuggingFace → sign → in-memory manifest."""

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
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
    )
    assert store.current_manifest() is None


@pytest.mark.asyncio
async def test_publish_writes_uploads_signs_and_serves(tmp_path):
    fake_upload = AsyncMock(return_value="abc123def456")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    model = MagicMock(name="mock_model")
    entry = await store.publish(checkpoint_n=1, model=model)
    assert isinstance(entry, ManifestEntry)
    assert entry.checkpoint_n == 1
    assert entry.repo_id == "aivolutionedge/reliquary-sn"
    assert entry.revision == "abc123def456"
    assert entry.signature.startswith("ed25519:")
    fake_upload.assert_awaited_once()
    assert store.current_manifest() is entry


@pytest.mark.asyncio
async def test_publish_increments_overrides_previous(tmp_path):
    fake_upload = AsyncMock(side_effect=[
        "rev_sha_001",
        "rev_sha_002",
    ])
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    await store.publish(checkpoint_n=1, model=MagicMock())
    e2 = await store.publish(checkpoint_n=2, model=MagicMock())
    assert store.current_manifest() is e2
    assert store.current_manifest().checkpoint_n == 2
    assert store.current_manifest().revision == "rev_sha_002"


@pytest.mark.asyncio
async def test_upload_fn_receives_correct_kwargs(tmp_path):
    """upload_fn is called with (local_path, repo_id, path_in_repo, commit_message)."""
    captured = {}

    async def capturing_upload(local_path, repo_id, path_in_repo, commit_message):
        captured["local_path"] = local_path
        captured["repo_id"] = repo_id
        captured["path_in_repo"] = path_in_repo
        captured["commit_message"] = commit_message
        return "captured_revision_sha"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=capturing_upload,
        save_weights_fn=_save_weights_stub,
    )
    entry = await store.publish(5, model=object())
    assert captured["repo_id"] == "aivolutionedge/reliquary-sn"
    assert captured["path_in_repo"] == "model.safetensors"
    assert "5" in captured["commit_message"]
    assert entry.revision == "captured_revision_sha"


@pytest.mark.asyncio
async def test_signature_includes_n_and_revision(tmp_path):
    """Signature payload = checkpoint_n || revision so tampering is detectable."""
    captured = {}

    class CapturingWallet:
        class _Hk:
            ss58_address = "5FHk"
            def sign(self, data):
                captured["signed"] = data
                return b"fake_sig"
        hotkey = _Hk()

    async def fake_upload(local_path, repo_id, path_in_repo, commit_message):
        return "revision_sha_42"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=CapturingWallet(),
        repo_id="aivolutionedge/reliquary-sn",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    entry = await store.publish(42, model=object())
    assert b"42" in captured["signed"]
    assert b"revision_sha_42" in captured["signed"]
    assert entry.signature == "ed25519:" + b"fake_sig".hex()


@pytest.mark.asyncio
async def test_repo_id_stored_in_manifest(tmp_path):
    """ManifestEntry carries the repo_id so miners can do from_pretrained(repo_id, revision)."""
    async def fake_upload(local_path, repo_id, path_in_repo, commit_message):
        return "some_rev"

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        repo_id="myorg/my-model",
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    entry = await store.publish(3, model=object())
    assert entry.repo_id == "myorg/my-model"
    assert entry.revision == "some_rev"
