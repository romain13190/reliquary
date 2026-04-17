"""Tests for the GRPO submission Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reliquary.constants import COMPLETIONS_PER_SUBMISSION, PROMPTS_PER_WINDOW
from reliquary.protocol.submission import (
    CompletionSubmission,
    SlotState,
    SubmissionRequest,
    SubmissionResponse,
    WindowStateResponse,
)


def _valid_completion(token_offset: int = 0) -> dict:
    return {
        "tokens": [1 + token_offset, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "commit": {"proofs": [], "signature": "deadbeef", "version": "v5"},
    }


def _valid_request_payload() -> dict:
    return {
        "window_start": 1000,
        "slot_index": 3,
        "prompt_id": "abc123def4567890",
        "miner_hotkey": "5HXYZ" + "a" * 43,
        "completions": [_valid_completion(i) for i in range(COMPLETIONS_PER_SUBMISSION)],
    }


class TestCompletionSubmission:
    def test_valid_roundtrip(self) -> None:
        c = CompletionSubmission(**_valid_completion())
        dumped = c.model_dump()
        restored = CompletionSubmission(**dumped)
        assert restored == c

    def test_tokens_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            CompletionSubmission(tokens=[], commit={"x": 1})

    def test_tokens_must_be_ints(self) -> None:
        with pytest.raises(ValidationError):
            CompletionSubmission(tokens=["not", "ints"], commit={})

    def test_commit_must_be_dict(self) -> None:
        with pytest.raises(ValidationError):
            CompletionSubmission(tokens=[1, 2], commit="not a dict")  # type: ignore[arg-type]


class TestSubmissionRequest:
    def test_valid_request_roundtrip(self) -> None:
        r = SubmissionRequest(**_valid_request_payload())
        assert r.window_start == 1000
        assert r.slot_index == 3
        assert len(r.completions) == COMPLETIONS_PER_SUBMISSION
        # JSON roundtrip
        payload = r.model_dump_json()
        SubmissionRequest.model_validate_json(payload)

    def test_slot_index_lower_bound(self) -> None:
        bad = _valid_request_payload() | {"slot_index": -1}
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_slot_index_upper_bound(self) -> None:
        bad = _valid_request_payload() | {"slot_index": PROMPTS_PER_WINDOW}
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_slot_index_accepts_zero(self) -> None:
        ok = _valid_request_payload() | {"slot_index": 0}
        assert SubmissionRequest(**ok).slot_index == 0

    def test_slot_index_accepts_max(self) -> None:
        ok = _valid_request_payload() | {"slot_index": PROMPTS_PER_WINDOW - 1}
        assert SubmissionRequest(**ok).slot_index == PROMPTS_PER_WINDOW - 1

    def test_completion_count_empty_rejected(self) -> None:
        bad = _valid_request_payload()
        bad["completions"] = []
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_completion_count_one_accepted(self) -> None:
        ok = _valid_request_payload()
        ok["completions"] = ok["completions"][:1]
        assert len(SubmissionRequest(**ok).completions) == 1

    def test_completion_count_under_plafond_accepted(self) -> None:
        """Any size between 1 and COMPLETIONS_PER_SUBMISSION is valid."""
        from reliquary.constants import COMPLETIONS_PER_SUBMISSION
        ok = _valid_request_payload()
        extras = [_valid_completion(i) for i in range(10, 10 + COMPLETIONS_PER_SUBMISSION - len(ok["completions"]))]
        ok["completions"] = ok["completions"] + extras
        assert len(SubmissionRequest(**ok).completions) == COMPLETIONS_PER_SUBMISSION

    def test_completion_count_over_plafond_rejected(self) -> None:
        from reliquary.constants import COMPLETIONS_PER_SUBMISSION
        bad = _valid_request_payload()
        extras = [_valid_completion(i) for i in range(10, 10 + COMPLETIONS_PER_SUBMISSION)]
        bad["completions"] = bad["completions"] + extras  # guaranteed > plafond
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_window_start_must_be_non_negative(self) -> None:
        bad = _valid_request_payload() | {"window_start": -1}
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_prompt_id_must_be_non_empty(self) -> None:
        bad = _valid_request_payload() | {"prompt_id": ""}
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)

    def test_miner_hotkey_must_be_non_empty(self) -> None:
        bad = _valid_request_payload() | {"miner_hotkey": ""}
        with pytest.raises(ValidationError):
            SubmissionRequest(**bad)


class TestSubmissionResponse:
    def test_valid_accept(self) -> None:
        r = SubmissionResponse(accepted=True, reason="ok", settled=False, slot_count=4)
        assert r.accepted is True
        assert r.slot_count == 4

    def test_valid_reject(self) -> None:
        r = SubmissionResponse(
            accepted=False, reason="slot_full", settled=False, slot_count=32
        )
        assert r.accepted is False
        assert r.reason == "slot_full"

    def test_slot_count_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SubmissionResponse(accepted=True, reason="ok", settled=False, slot_count=-1)


class TestSlotStateAndWindowState:
    def test_slot_state_valid(self) -> None:
        s = SlotState(slot_index=2, prompt_id="abc", count=12, settled=False)
        assert s.count == 12
        assert s.rewards == {}  # default empty histogram

    def test_slot_state_with_rewards_histogram(self) -> None:
        s = SlotState(
            slot_index=0,
            prompt_id="p0",
            count=28,
            settled=False,
            rewards={"1.0": 28, "0.0": 0},
        )
        assert s.rewards["1.0"] == 28

    def test_window_state_response_roundtrip(self) -> None:
        slots = [
            SlotState(
                slot_index=i,
                prompt_id=f"p{i}",
                count=i * 4,
                settled=(i == 0),
                rewards={"1.0": i * 2, "0.0": i * 2},
            )
            for i in range(PROMPTS_PER_WINDOW)
        ]
        w = WindowStateResponse(window_start=2000, slot_states=slots)
        payload = w.model_dump_json()
        restored = WindowStateResponse.model_validate_json(payload)
        assert restored == w
        assert len(restored.slot_states) == PROMPTS_PER_WINDOW
        assert restored.slot_states[2].rewards == {"1.0": 4, "0.0": 4}
