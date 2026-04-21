"""ValidationService state machine: OPEN → TRAINING → PUBLISHING → READY."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.protocol.submission import WindowState


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0

    @property
    def name(self): return "fake"


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_service(tmp_path):
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(),
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    # Point persistence at a temp location so tests don't clobber real state.
    svc._state_path = str(tmp_path / "s.json")
    from reliquary.validator.state_persistence import ValidatorState
    svc._state = ValidatorState(svc._state_path)
    return svc


def test_service_initial_state_is_ready(tmp_path):
    svc = _make_service(tmp_path)
    assert svc._current_window_state == WindowState.READY


def test_open_window_sets_state_to_open(tmp_path):
    svc = _make_service(tmp_path)
    svc._open_window()
    assert svc._current_window_state == WindowState.OPEN
    assert svc._active_batcher is not None


def test_open_window_increments_window_n(tmp_path):
    svc = _make_service(tmp_path)
    initial = svc._state.window_n
    svc._open_window()
    assert svc._state.window_n == initial + 1


def test_set_state_transitions(tmp_path):
    svc = _make_service(tmp_path)
    for state in (WindowState.OPEN, WindowState.TRAINING,
                  WindowState.PUBLISHING, WindowState.READY):
        svc._set_state(state)
        assert svc._current_window_state == state


@pytest.mark.asyncio
async def test_train_and_publish_bumps_checkpoint_n(tmp_path):
    svc = _make_service(tmp_path)
    initial_checkpoint = svc._state.checkpoint_n

    # Open a window so there's an active batcher + seal_event to drive
    svc._open_window()

    # Mock the checkpoint store to avoid HF calls
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=None)
    from reliquary.validator.checkpoint import ManifestEntry
    fake_entry = ManifestEntry(
        checkpoint_n=initial_checkpoint + 1,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_x",
        signature="ed25519:x",
    )
    svc._checkpoint_store.publish = AsyncMock(return_value=fake_entry)

    # Mock storage.upload_window_dataset to avoid R2
    import reliquary.validator.service as svc_mod
    original_upload = svc_mod.storage.upload_window_dataset
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    try:
        await svc._train_and_publish()
    finally:
        svc_mod.storage.upload_window_dataset = original_upload

    assert svc._state.checkpoint_n == initial_checkpoint + 1
    assert svc._current_window_state == WindowState.READY
    assert svc._active_batcher is None
    svc._checkpoint_store.publish.assert_awaited_once()


def test_open_window_wires_checkpoint_hash_into_batcher(tmp_path):
    svc = _make_service(tmp_path)
    from reliquary.validator.checkpoint import ManifestEntry
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=ManifestEntry(
        checkpoint_n=5,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_005",
        signature="ed25519:sig",
    ))
    svc._open_window()
    assert svc._active_batcher.current_checkpoint_hash == "rev_sha_005"


def test_open_window_empty_hash_pre_first_publish(tmp_path):
    svc = _make_service(tmp_path)
    # No checkpoint published yet → current_manifest returns None
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=None)
    svc._open_window()
    assert svc._active_batcher.current_checkpoint_hash == ""


@pytest.mark.asyncio
async def test_publish_every_n_windows(tmp_path):
    """With _publish_every=3, publish is called only on windows 3 (first due to
    None manifest) then ... actually on windows 1 (first, manifest is None) and
    then next on window 3 (3 % 3 == 0). Verify: 5 _train_and_publish calls
    produce exactly 2 publish calls when publish_every=3 and manifest starts None."""
    import reliquary.validator.service as svc_mod
    from reliquary.validator.checkpoint import ManifestEntry

    svc = _make_service(tmp_path)
    svc._publish_every = 3

    # Start with no manifest so first call always publishes.
    mock_store = MagicMock()
    mock_store.current_manifest = MagicMock(return_value=None)

    published_entries = []

    async def _fake_publish(checkpoint_n, model):
        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            repo_id="aivolutionedge/reliquary-sn",
            revision=f"rev_{checkpoint_n:03d}",
            signature="ed25519:sig",
        )
        published_entries.append(entry)
        # After first publish, current_manifest returns the latest entry.
        mock_store.current_manifest.return_value = entry
        return entry

    mock_store.publish = AsyncMock(side_effect=_fake_publish)
    svc._checkpoint_store = mock_store

    original_upload = svc_mod.storage.upload_window_dataset
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    try:
        for _ in range(5):
            svc._open_window()
            svc._active_batcher.seal_event.set()
            await svc._train_and_publish()
    finally:
        svc_mod.storage.upload_window_dataset = original_upload

    # window_n increments: 1,2,3,4,5.
    # Publish fires when: window_n==1 (manifest is None), window_n==3 (3%3==0).
    # Windows 2,4,5 skip. checkpoint_n advances only on publish.
    assert mock_store.publish.await_count == 2
    assert published_entries[0].checkpoint_n == 1  # first publish: next_n = 0+1 = 1
    assert published_entries[1].checkpoint_n == 2  # second publish: next_n = 1+1 = 2
