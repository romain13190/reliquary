"""Tests for the GRPO dataset upload helper in storage.py."""

from __future__ import annotations

import gzip
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.infrastructure.storage import upload_window_dataset


@pytest.mark.asyncio
async def test_upload_window_dataset_writes_gzipped_json_at_expected_key() -> None:
    captured: dict[str, object] = {}

    fake_client = MagicMock()

    async def _put(Bucket: str, Key: str, Body: bytes) -> None:
        captured["bucket"] = Bucket
        captured["key"] = Key
        captured["body"] = Body

    fake_client.put_object = _put

    @asynccontextmanager
    async def _fake_get_client(**kwargs):
        yield fake_client

    data = {
        "window_start": 1000,
        "randomness": "deadbeef",
        "environment": "gsm8k",
        "slots": [
            {
                "slot_index": i,
                "prompt_id": f"p{i}",
                "prompt": f"Q{i}",
                "ground_truth": str(i),
                "settled": True,
                "completions": [
                    {"miner_hotkey": "m", "tokens": [1, 2, 3], "completion_text": "x", "reward": 1.0},
                ],
            }
            for i in range(8)
        ],
    }

    with patch("reliquary.infrastructure.storage.get_s3_client", _fake_get_client):
        ok = await upload_window_dataset(1000, data, bucket_name="testbucket")

    assert ok is True
    assert captured["bucket"] == "testbucket"
    assert captured["key"] == "reliquary/dataset/window-1000.json.gz"
    decompressed = gzip.decompress(captured["body"])
    restored = json.loads(decompressed)
    assert restored == data


@pytest.mark.asyncio
async def test_upload_window_dataset_uses_env_bucket_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("R2_BUCKET_ID", "from-env")
    captured: dict[str, object] = {}

    async def _put(Bucket: str, **_: object) -> None:
        captured["bucket"] = Bucket

    fake_client = MagicMock(put_object=_put)

    @asynccontextmanager
    async def _fake_get_client(**kwargs):
        yield fake_client

    with patch("reliquary.infrastructure.storage.get_s3_client", _fake_get_client):
        await upload_window_dataset(42, {"slots": []})

    assert captured["bucket"] == "from-env"


@pytest.mark.asyncio
async def test_upload_window_dataset_always_uses_flat_key() -> None:
    """Archive paths are flat — no hotkey prefix. validator_hotkey lives in the
    archive body for provenance, not in the object key."""
    captured: dict[str, object] = {}

    async def _put(Bucket: str, Key: str, Body: bytes) -> None:
        captured["key"] = Key

    fake_client = MagicMock(put_object=_put)

    @asynccontextmanager
    async def _fake_get_client(**kwargs):
        yield fake_client

    with patch("reliquary.infrastructure.storage.get_s3_client", _fake_get_client):
        await upload_window_dataset(
            7987940, {"slots": [], "validator_hotkey": "5Cxxx"}, bucket_name="b"
        )

    assert captured["key"] == "reliquary/dataset/window-7987940.json.gz"
