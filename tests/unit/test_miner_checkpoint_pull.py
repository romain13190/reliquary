"""Miner detects new checkpoint_n via /window/state and downloads."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.miner.engine import maybe_pull_checkpoint


@pytest.mark.asyncio
async def test_pull_when_remote_n_higher():
    """Remote checkpoint_n > local → download triggered."""
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_url = "https://r2/5.safetensors"
    state.checkpoint_hash = "sha256:abc"

    download_fn = AsyncMock(return_value="/tmp/5.safetensors")
    load_fn = MagicMock(return_value="loaded_model_5")

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=4, local_hash="sha256:old", local_model="old_model",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_hash == "sha256:abc"
    assert new_model == "loaded_model_5"
    download_fn.assert_awaited_once_with("https://r2/5.safetensors")
    load_fn.assert_called_once_with("/tmp/5.safetensors")


@pytest.mark.asyncio
async def test_no_pull_when_local_up_to_date():
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_url = "https://r2/5.safetensors"
    state.checkpoint_hash = "sha256:abc"

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=5, local_hash="sha256:abc", local_model="cached",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_hash == "sha256:abc"
    assert new_model == "cached"
    download_fn.assert_not_called()


@pytest.mark.asyncio
async def test_no_pull_when_url_missing():
    """Pre-first-publish: state has checkpoint_n=0 and url=None — don't try to download."""
    state = MagicMock()
    state.checkpoint_n = 0
    state.checkpoint_url = None
    state.checkpoint_hash = None

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=0, local_hash="", local_model="initial_model",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 0
    assert new_model == "initial_model"
    download_fn.assert_not_called()


@pytest.mark.asyncio
async def test_pull_from_zero_local_to_nonzero_remote():
    """Fresh miner (local_n=0) joins active subnet (remote_n=7)."""
    state = MagicMock()
    state.checkpoint_n = 7
    state.checkpoint_url = "https://r2/7.safetensors"
    state.checkpoint_hash = "sha256:seven"

    download_fn = AsyncMock(return_value="/tmp/7.safetensors")
    load_fn = MagicMock(return_value="model_7")

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=0, local_hash="", local_model=None,
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 7
    assert new_hash == "sha256:seven"
    assert new_model == "model_7"
