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


def _rollout(r=1.0, eos=False):
    tokens = [1, 2, 3, 4, 5]
    if eos:
        tokens = tokens + [99]  # 99 = fake eos
    return RolloutSubmission(
        tokens=tokens,
        reward=r,
        commit={
            "tokens": tokens, "proof_version": "v5",
            "rollout": {
                "prompt_length": 2,
                "completion_length": len(tokens) - 2,
                "token_logprobs": [],
            },
        },
    )


def _valid_submission(prompt_idx, k=4, hotkey="hk", eos_first=False):
    import math
    rollouts = [
        _rollout(r=1.0 if i < k else 0.0, eos=(eos_first and i == 0))
        for i in range(8)
    ]
    p = k / 8
    sigma = math.sqrt(p * (1 - p))
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        signed_round=100,
        merkle_root_bytes=b"\xab" * 32,
        sigma=sigma,
        rollouts=rollouts,
        completion_texts=[f"text_{i}" for i in range(8)],
        sketch_diff_max=412,
        lp_dev_max=0.00037,
        dist_q10_min=0.74,
        claimed_checkpoint_hash="sha256:fake",
    )


@pytest.mark.asyncio
async def test_archive_includes_prompt_and_rollout_content():
    from reliquary.validator.service import ValidationService

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )

    batcher = MagicMock()
    batcher.window_start = 42
    batcher.randomness = "0xdeadbeef"
    batcher.window_opened_at = 100.0
    batcher.reject_counts = {"out_of_zone": 3, "logprob_mismatch": 1}

    batch = [
        _valid_submission(prompt_idx=7, k=4, hotkey="hk1", eos_first=True),
        _valid_submission(prompt_idx=13, k=5, hotkey="hk2"),
    ]
    batch[0].arrived_at = 102.5  # 2.5 s after window open
    batch[1].arrived_at = 107.0  # 7.0 s after window open

    runner = _valid_submission(prompt_idx=99, k=4, hotkey="hk_runner")
    runner.arrived_at = 110.0
    batcher.valid_submissions.return_value = list(batch) + [runner]

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
    assert entry0["rollouts"][0]["tokens"] == [1, 2, 3, 4, 5, 99]  # first is eos-terminated
    assert entry0["rollouts"][1]["tokens"] == [1, 2, 3, 4, 5]
    assert entry0["rollouts"][0]["completion_text"] == "text_0"
    assert entry0["rollouts"][0]["reward"] == 1.0

    # cooldown rebuild backward-compat: still has window_start and batch[*].prompt_idx
    assert {"window_start", "batch"}.issubset(archive.keys())
    assert all("prompt_idx" in e for e in archive["batch"])

    # response_time: seconds between window-open and submission-accepted.
    assert entry0["response_time"] == pytest.approx(2.5)
    assert archive["batch"][1]["response_time"] == pytest.approx(7.0)

    # filter telemetry passed through verbatim.
    assert entry0["sketch_diff_max"] == 412
    assert entry0["lp_dev_max"] == pytest.approx(0.00037)
    assert entry0["dist_q10_min"] == pytest.approx(0.74)

    # forensic fields.
    assert entry0["merkle_root"] == "ab" * 32
    assert entry0["claimed_checkpoint_hash"] == "sha256:fake"

    # eos detection: rollout 0 of entry 0 ends with eos_token_id=99 → True.
    assert entry0["rollouts"][0]["eos_terminated"] is True
    assert entry0["rollouts"][1]["eos_terminated"] is False
    assert entry0["rollouts"][0]["completion_length"] == 4  # 6 tokens − 2 prompt
    assert entry0["rollouts"][1]["completion_length"] == 3

    # runners_up: validated submissions that didn't make the batch — metadata only.
    assert "runners_up" in archive
    assert len(archive["runners_up"]) == 1
    ru = archive["runners_up"][0]
    assert ru["hotkey"] == "hk_runner"
    assert ru["prompt_idx"] == 99
    assert ru["response_time"] == pytest.approx(10.0)
    assert "rollouts" not in ru and "prompt" not in ru  # metadata only

    # reject_summary persisted from batcher.
    assert archive["reject_summary"] == {"out_of_zone": 3, "logprob_mismatch": 1}
