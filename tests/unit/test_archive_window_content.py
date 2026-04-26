"""_archive_window includes prompt + rollout content on R2."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.validator.batcher import ValidSubmission
from reliquary.protocol.submission import RolloutSubmission


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 100
    def get_problem(self, i):
        return {"prompt": f"question {i}", "ground_truth": f"answer {i}", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _rollout(r=1.0):
    return RolloutSubmission(
        tokens=[1, 2, 3, 4, 5],
        reward=r,
        commit={"tokens": [1, 2, 3, 4, 5], "proof_version": "v5"},
    )


def _valid_submission(prompt_idx, k=4, hotkey="hk"):
    import math
    rollouts = [_rollout(r=1.0 if i < k else 0.0) for i in range(8)]
    p = k / 8
    sigma = math.sqrt(p * (1 - p))
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        signed_round=100,
        merkle_root_bytes=b"\x00" * 32,
        sigma=sigma,
        rollouts=rollouts,
        completion_texts=[f"text_{i}" for i in range(8)],
    )


@pytest.mark.asyncio
async def test_archive_includes_prompt_and_rollout_content():
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )

    batcher = MagicMock()
    batcher.window_start = 42
    batcher.randomness = "0xdeadbeef"
    batcher.window_opened_at = 100.0

    batch = [_valid_submission(prompt_idx=7, k=4, hotkey="hk1"),
             _valid_submission(prompt_idx=13, k=5, hotkey="hk2")]
    batch[0].arrived_at = 102.5  # 2.5 s after window open
    batch[1].arrived_at = 107.0  # 7.0 s after window open

    captured = {}

    async def fake_upload(window_start, data, **kwargs):
        captured["data"] = data
        return True

    with patch(
        "reliquary.infrastructure.storage.upload_window_dataset",
        new=fake_upload,
    ):
        await svc._archive_window(batcher, batch)

    archive = captured["data"]
    assert archive["window_start"] == 42
    assert archive["environment"] == "fake"
    assert len(archive["batch"]) == 2

    import math
    entry0 = archive["batch"][0]
    assert entry0["prompt_idx"] == 7
    assert entry0["prompt"] == "question 7"
    assert entry0["ground_truth"] == "answer 7"
    expected_sigma = math.sqrt((4 / 8) * (1 - 4 / 8))  # Bernoulli(p=0.5) → 0.5
    assert abs(entry0["sigma"] - expected_sigma) < 1e-9
    assert len(entry0["rollouts"]) == 8
    assert entry0["rollouts"][0]["tokens"] == [1, 2, 3, 4, 5]
    assert entry0["rollouts"][0]["completion_text"] == "text_0"
    assert entry0["rollouts"][0]["reward"] == 1.0

    # cooldown rebuild backward-compat: still has window_start and batch[*].prompt_idx
    assert {"window_start", "batch"}.issubset(archive.keys())
    assert all("prompt_idx" in e for e in archive["batch"])

    # response_time: seconds between window-open and submission-accepted.
    assert entry0["response_time"] == pytest.approx(2.5)
    assert archive["batch"][1]["response_time"] == pytest.approx(7.0)
