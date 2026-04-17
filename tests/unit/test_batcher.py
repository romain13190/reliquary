"""Tests for reliquary.validator.batcher.WindowBatcher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reliquary.constants import (
    DIVERSITY_PREFIX_LEN,
    GROUP_SIZE,
    MINER_BATCH_SIZE,
    PROMPTS_PER_WINDOW,
    SLOT_DEADLINE_SECONDS,
)
from reliquary.validator.batcher import AcceptedCompletion, ProblemSlot, WindowBatcher
from reliquary.protocol.submission import SubmissionRequest, CompletionSubmission

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

WINDOW_START = 1000
RANDOMNESS = "deadbeefdeadbeef"
PROMPT_TEXT = "Q"
PROMPT_ID = "abc123"
GROUND_TRUTH = "42"

# Fake tokenizer: encode returns [ord(c) % 1000 for c in text],
# decode returns "".join(chr(t) for t in tokens if t < 128).

class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if t < 128)


class FakeEnv:
    name = "test"

    def __len__(self) -> int:
        return 100

    def get_problem(self, index: int) -> dict:
        return {"prompt": PROMPT_TEXT, "ground_truth": GROUND_TRUTH, "id": PROMPT_ID}

    def compute_reward(self, problem: dict, completion: str) -> float:
        return 1.0 if GROUND_TRUTH in completion else 0.0


def make_fake_model() -> MagicMock:
    return MagicMock()


def make_slot(slot_index: int = 0) -> ProblemSlot:
    problem = {"prompt": PROMPT_TEXT, "ground_truth": GROUND_TRUTH, "id": PROMPT_ID}
    return ProblemSlot(
        slot_index=slot_index,
        prompt_id=PROMPT_ID,
        problem=problem,
    )


class _FakeClock:
    """Monotonic clock mock — advance() by a delta, now() returns current."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        self._t += delta


def make_batcher(
    slots: list[ProblemSlot] | None = None,
    verify_commitment_proofs_fn=None,
    verify_signature_fn=None,
    verify_proof_version_fn=None,
    time_fn=None,
) -> WindowBatcher:
    if slots is None:
        slots = [make_slot(i) for i in range(PROMPTS_PER_WINDOW)]

    # Default mocks: everything passes.
    if verify_commitment_proofs_fn is None:
        verify_commitment_proofs_fn = MagicMock(return_value=(True, 32, 32))
    if verify_signature_fn is None:
        verify_signature_fn = MagicMock(return_value=True)
    if verify_proof_version_fn is None:
        verify_proof_version_fn = MagicMock(return_value=True)

    return WindowBatcher(
        window_start=WINDOW_START,
        slots=slots,
        randomness=RANDOMNESS,
        env=FakeEnv(),
        model=make_fake_model(),
        tokenizer=FakeTokenizer(),
        verify_commitment_proofs_fn=verify_commitment_proofs_fn,
        verify_signature_fn=verify_signature_fn,
        verify_proof_version_fn=verify_proof_version_fn,
        time_fn=time_fn,
    )


# ---------------------------------------------------------------------------
# Token helpers
#
# PROMPT_TEXT = "Q"  → encode = [ord('Q') % 1000] = [81]
# prompt_length = 1
#
# For reward=1.0 we need "42" in decoded text.
# chr(52) = '4', chr(50) = '2' → tokens [52, 50] decode to "42".
# So tail tokens [52, 50, <filler>] → decoded includes "42" → reward=1.0
# ---------------------------------------------------------------------------

PROMPT_TOKENS = [ord(c) % 1000 for c in PROMPT_TEXT]   # [81]
PROMPT_LEN = len(PROMPT_TOKENS)                          # 1

# Diverse prefixes (DIVERSITY_PREFIX_LEN = 8): just 8 distinct token values.
_DIVERSE_STARTS = [
    [100, 101, 102, 103, 104, 105, 106, 107],
    [200, 201, 202, 203, 204, 205, 206, 207],
    [300, 301, 302, 303, 304, 305, 306, 307],
    [400, 401, 402, 403, 404, 405, 406, 407],
]

# Tail tokens that decode to contain "42": chr(52)='4', chr(50)='2'
_REWARD_TAIL = [52, 50]  # decodes to "42" → reward 1.0 from FakeEnv
_WRONG_TAIL = [33]        # decodes to "!" → no "42" substring → reward 0.0


def make_tokens(diverse_idx: int, correct: bool = True) -> list[int]:
    """Build a full token list: [prompt] + [diverse_prefix_8] + [tail]."""
    tail = _REWARD_TAIL if correct else _WRONG_TAIL
    return PROMPT_TOKENS + _DIVERSE_STARTS[diverse_idx] + tail


def make_commit(tokens: list[int], prompt_length: int = PROMPT_LEN) -> dict:
    return {
        "tokens": tokens,
        "commitments": [{}] * len(tokens),
        "signature": "aabbcc",
        "proof_version": "v5",
        "rollout": {"prompt_length": prompt_length},
    }


def make_completions(
    diverse_starts: list[list[int]] | None = None,
    prompt_length: int = PROMPT_LEN,
    correct_flags: list[bool] | None = None,
) -> list[CompletionSubmission]:
    """Build 4 CompletionSubmission objects with distinct prefixes.

    ``correct_flags`` controls per-completion reward: True → reward=1.0,
    False → reward=0.0. Default is all-correct (legacy behaviour).
    """
    if diverse_starts is None:
        diverse_starts = _DIVERSE_STARTS
    if correct_flags is None:
        correct_flags = [True] * MINER_BATCH_SIZE
    result = []
    for i in range(MINER_BATCH_SIZE):
        tail = _REWARD_TAIL if correct_flags[i] else _WRONG_TAIL
        tokens = PROMPT_TOKENS + diverse_starts[i] + tail
        commit = make_commit(tokens, prompt_length)
        result.append(CompletionSubmission(tokens=tokens, commit=commit))
    return result


def make_request(
    hotkey: str = "miner_A",
    slot_index: int = 0,
    prompt_id: str = PROMPT_ID,
    window_start: int = WINDOW_START,
    completions: list[CompletionSubmission] | None = None,
) -> SubmissionRequest:
    if completions is None:
        completions = make_completions()
    return SubmissionRequest(
        window_start=window_start,
        slot_index=slot_index,
        prompt_id=prompt_id,
        miner_hotkey=hotkey,
        completions=completions,
    )


def _diverse_set(offset: int) -> list[list[int]]:
    """Build 4 distinct prefixes for one miner's batch, offset to avoid cross-miner collisions."""
    base = 10_000 + offset * 1000
    return [
        [base + i * 10 + j for j in range(8)]
        for i in range(MINER_BATCH_SIZE)
    ]


def fill_slot_class(
    batcher: WindowBatcher,
    slot_index: int,
    correct: bool,
    start_miner_idx: int,
    batch_count: int,
) -> None:
    """Submit ``batch_count`` batches (each MINER_BATCH_SIZE completions of
    the same class) from sequential hotkeys. Assumes residual capacity is
    sufficient."""
    for i in range(batch_count):
        m = start_miner_idx + i
        req = make_request(
            hotkey=f"miner_{m}",
            slot_index=slot_index,
            completions=make_completions(
                diverse_starts=_diverse_set(m),
                correct_flags=[correct] * MINER_BATCH_SIZE,
            ),
        )
        resp = batcher.accept_submission(req)
        assert resp.accepted is True, f"miner_{m} rejected: {resp.reason}"


# Number of miner batches needed to fill half of GROUP_SIZE with one class.
# With GROUP_SIZE=32 and MINER_BATCH_SIZE=4, that's 4 batches → 16 completions.
BATCHES_PER_HALF = (GROUP_SIZE // 2) // MINER_BATCH_SIZE


def fully_fill_slot(batcher: WindowBatcher, slot_index: int) -> None:
    """Fill a slot to GROUP_SIZE with a balanced mix (half corrects, half wrongs)
    to trigger auto-finalize on capacity. All hotkeys are distinct.
    """
    fill_slot_class(
        batcher, slot_index, correct=True,
        start_miner_idx=0, batch_count=BATCHES_PER_HALF,
    )
    fill_slot_class(
        batcher, slot_index, correct=False,
        start_miner_idx=BATCHES_PER_HALF, batch_count=BATCHES_PER_HALF,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidSubmission:
    def test_valid_submission_accepted(self) -> None:
        """Happy path: all 4 completions verify → accepted, hotkey added, count=4."""
        batcher = make_batcher()
        req = make_request()
        resp = batcher.accept_submission(req)

        assert resp.accepted is True
        assert resp.reason == "ok"
        assert resp.slot_count == MINER_BATCH_SIZE
        assert resp.settled is False

    def test_same_hotkey_can_submit_multiple_batches(self) -> None:
        """A single hotkey may submit multiple batches to the same slot,
        bounded only by slot capacity and prefix-distinct dedup."""
        batcher = make_batcher()

        first = batcher.accept_submission(
            make_request(
                hotkey="miner_A",
                completions=make_completions(diverse_starts=_diverse_set(0)),
            )
        )
        assert first.accepted is True

        # Same hotkey, fresh prefixes → should now be accepted (dedup removed).
        second = batcher.accept_submission(
            make_request(
                hotkey="miner_A",
                completions=make_completions(diverse_starts=_diverse_set(1)),
            )
        )
        assert second.accepted is True
        assert batcher.slots[0].count == 2 * MINER_BATCH_SIZE

    def test_zero_reward_still_accepted(self) -> None:
        """reward=0 does not reject; slot count grows as normal."""
        # FakeEnv returns 0.0 when "42" is NOT in the decoded text.
        # Build completions whose tail tokens don't include chr(52)+chr(50).
        no_reward_tail = [999, 998]  # both >=128 → decode to ""

        diverse = [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [200, 201, 202, 203, 204, 205, 206, 207],
            [300, 301, 302, 303, 304, 305, 306, 307],
            [400, 401, 402, 403, 404, 405, 406, 407],
        ]
        completions = []
        for prefix in diverse:
            tokens = PROMPT_TOKENS + prefix + no_reward_tail
            commit = make_commit(tokens)
            completions.append(CompletionSubmission(tokens=tokens, commit=commit))

        batcher = make_batcher()
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is True
        assert resp.slot_count == MINER_BATCH_SIZE
        # Rewards are all 0.0
        for ac in batcher.slots[0].accepted_completions:
            assert ac.reward == 0.0


class TestRejectionCases:
    def test_window_mismatch_rejected(self) -> None:
        batcher = make_batcher()
        req = make_request(window_start=WINDOW_START + 1)
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "window_mismatch"

    def test_invalid_slot_index_rejected(self) -> None:
        """slot_index >= PROMPTS_PER_WINDOW → invalid_slot."""
        batcher = make_batcher()
        # Bypass Pydantic by mutating after construction.
        req = make_request(slot_index=0)
        req.__dict__["slot_index"] = PROMPTS_PER_WINDOW  # out of range
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_slot"

    def test_settled_slot_rejects_new_submission(self) -> None:
        """Fill slot to GROUP_SIZE → auto-finalizes → extra submission → slot_full."""
        batcher = make_batcher()
        fully_fill_slot(batcher, slot_index=0)

        assert batcher.slots[0].finalized is True
        assert batcher.slots[0].settled is True

        # Another miner with totally fresh prefixes still rejected.
        extra_miner_offset = 2 * BATCHES_PER_HALF + 5
        resp = batcher.accept_submission(
            make_request(
                hotkey=f"miner_{extra_miner_offset}",
                completions=make_completions(
                    diverse_starts=_diverse_set(extra_miner_offset),
                    correct_flags=[True] * MINER_BATCH_SIZE,
                ),
            )
        )
        assert resp.accepted is False
        assert resp.reason == "slot_full"

    def test_batch_overflowing_capacity_rejected_as_slot_full(self) -> None:
        """A batch that would push slot.count past GROUP_SIZE is rejected
        atomically with reason slot_full. No per-class quota in this model —
        only the capacity guard applies."""
        batcher = make_batcher()
        # Fill GROUP_SIZE - 2 with mixed hotkeys so 1 more batch of 4 would overflow.
        # BATCHES_PER_HALF * 2 * MINER_BATCH_SIZE = GROUP_SIZE already.
        # Instead bring the slot to GROUP_SIZE - 2 by doing (BATCHES_PER_HALF*2 - 1)
        # full batches + a size-2 manual final submission would be needed, but the
        # scenario is cleaner by just filling to GROUP_SIZE - MINER_BATCH_SIZE and
        # adding a size-(MINER_BATCH_SIZE) batch that fits exactly.
        # Here we just verify the rejection for overflow.
        needed_full_batches = (GROUP_SIZE // MINER_BATCH_SIZE) - 1
        # Fill with needed_full_batches × 4 completions; all corrects.
        fill_slot_class(
            batcher, 0, correct=True, start_miner_idx=0,
            batch_count=needed_full_batches,
        )
        # slot.count = needed_full_batches * 4 = GROUP_SIZE - 4.
        # A fresh batch of 4 should fit exactly.
        fitting = batcher.accept_submission(
            make_request(
                hotkey="fits_exactly",
                completions=make_completions(
                    diverse_starts=_diverse_set(needed_full_batches + 1),
                    correct_flags=[False] * MINER_BATCH_SIZE,
                ),
            )
        )
        assert fitting.accepted is True
        assert batcher.slots[0].count == GROUP_SIZE
        assert batcher.slots[0].settled is True  # auto-finalize

    def test_prompt_mismatch_rejected(self) -> None:
        batcher = make_batcher()
        req = make_request(prompt_id="wrong_prompt_id")
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "prompt_mismatch"

    def test_diversity_violation_rejected(self) -> None:
        """2 completions in the batch share the same prefix → diversity_violation."""
        batcher = make_batcher()
        # Completions 0 and 1 share the same diverse prefix.
        dup_diverse = [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [100, 101, 102, 103, 104, 105, 106, 107],  # duplicate
            [300, 301, 302, 303, 304, 305, 306, 307],
            [400, 401, 402, 403, 404, 405, 406, 407],
        ]
        completions = make_completions(diverse_starts=dup_diverse)
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "diversity_violation"
        # Slot state unchanged — miner can retry.
        assert batcher.slots[0].count == 0

    def test_cross_miner_duplicate_prefix_rejected(self) -> None:
        """Second miner reusing the first miner's prefixes → duplicate_prefix.

        This is the copycat replay block: a second hotkey can't claim emission
        credit by submitting the same answer (or any answer with the same
        first-N generated tokens) as someone already in the slot.
        """
        batcher = make_batcher()
        # First miner: default prefixes, accepted.
        first = batcher.accept_submission(make_request(hotkey="miner_A"))
        assert first.accepted is True
        assert len(batcher.slots[0].accepted_prefixes) == MINER_BATCH_SIZE

        # Second miner with a different hotkey but the SAME prefixes.
        second = batcher.accept_submission(make_request(hotkey="miner_B"))
        assert second.accepted is False
        assert second.reason == "duplicate_prefix"
        # Slot count unchanged — miner_B can retry with fresh prefixes.
        assert batcher.slots[0].count == MINER_BATCH_SIZE

    def test_cross_miner_partial_overlap_rejects_whole_batch(self) -> None:
        """One overlapping prefix in a batch of 4 → whole batch rejected.

        Same all-or-nothing semantics as the proof verification: if any
        completion collides with the slot's history, the miner doesn't get
        credit for the other three either.
        """
        batcher = make_batcher()
        first = batcher.accept_submission(make_request(hotkey="miner_A"))
        assert first.accepted is True

        # Build a batch where prefix #2 collides with miner_A's prefix #2,
        # but #0, #1, #3 are fresh.
        partial = [
            [500, 501, 502, 503, 504, 505, 506, 507],   # fresh
            [600, 601, 602, 603, 604, 605, 606, 607],   # fresh
            _DIVERSE_STARTS[2],                          # collides
            [700, 701, 702, 703, 704, 705, 706, 707],   # fresh
        ]
        resp = batcher.accept_submission(
            make_request(hotkey="miner_B", completions=make_completions(diverse_starts=partial))
        )
        assert resp.accepted is False
        assert resp.reason == "duplicate_prefix"
        assert batcher.slots[0].count == MINER_BATCH_SIZE

    def test_accepted_prefixes_tracked_per_slot(self) -> None:
        """A successful accept extends slot.accepted_prefixes by exactly 4 entries."""
        batcher = make_batcher()
        assert batcher.slots[0].accepted_prefixes == set()

        batcher.accept_submission(make_request(hotkey="miner_A"))
        assert len(batcher.slots[0].accepted_prefixes) == MINER_BATCH_SIZE

        # Other slots untouched.
        for s in batcher.slots[1:]:
            assert s.accepted_prefixes == set()

    def test_token_mismatch_between_top_and_commit_rejected(self) -> None:
        """c.tokens != c.commit['tokens'] → token_mismatch."""
        batcher = make_batcher()
        completions = make_completions()
        # Tamper: top-level tokens differ from commit tokens on completion 0.
        bad_tokens = completions[0].tokens[:]
        bad_tokens.append(9999)
        completions[0] = CompletionSubmission(
            tokens=bad_tokens,
            commit=completions[0].commit,  # commit still has old tokens
        )
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "token_mismatch"

    def test_invalid_proof_version_rejected(self) -> None:
        """verify_proof_version_fn returns False → invalid_proof_version."""
        bad_version = MagicMock(return_value=False)
        batcher = make_batcher(verify_proof_version_fn=bad_version)
        req = make_request()
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_proof_version"

    def test_invalid_signature_rejected(self) -> None:
        """verify_signature_fn returns False → invalid_signature."""
        bad_sig = MagicMock(return_value=False)
        batcher = make_batcher(verify_signature_fn=bad_sig)
        req = make_request()
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_signature"

    def test_invalid_prompt_rejected(self) -> None:
        """Completion tokens don't start with the expected prompt tokens → invalid_prompt."""
        batcher = make_batcher()
        # Replace prompt tokens with something wrong.
        wrong_prompt = [9999]  # not [81]
        diverse = _DIVERSE_STARTS
        completions = []
        for i in range(MINER_BATCH_SIZE):
            tokens = wrong_prompt + diverse[i] + _REWARD_TAIL
            commit = make_commit(tokens, prompt_length=1)  # declares prompt_length=1
            completions.append(CompletionSubmission(tokens=tokens, commit=commit))
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_prompt"

    def test_invalid_proof_rejects_whole_batch(self) -> None:
        """Proof fails on 3rd completion → whole batch rejected, slot count unchanged."""
        call_count = [0]

        def selective_proof_fail(commit, model, randomness):
            call_count[0] += 1
            # Fail on the 3rd call.
            if call_count[0] == 3:
                return (False, 0, 32)
            return (True, 32, 32)

        batcher = make_batcher(verify_commitment_proofs_fn=selective_proof_fail)
        req = make_request()
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "invalid_proof"
        assert batcher.slots[0].count == 0  # no completions added


class TestWindowState:
    def test_is_window_complete_only_when_all_slots_settled(self) -> None:
        """Fully fill 7 slots (both quotas) → False; fill the 8th → True.

        Note: to limit test cost we fill slots via a per-slot miner-index
        offset. `fill_slot_class` with the same `start_miner_idx` in
        different slots is fine because the miner_hotkey namespace is
        shared but the (hotkey, slot) tuple is what's deduped.
        Use a fresh start_miner_idx per slot to keep hotkeys unique.
        """
        batcher = make_batcher()
        assert batcher.is_window_complete() is False

        for slot_idx in range(PROMPTS_PER_WINDOW - 1):
            base = slot_idx * 2 * BATCHES_PER_HALF  # leave room for both quotas
            fill_slot_class(batcher, slot_idx, correct=True,
                            start_miner_idx=base, batch_count=BATCHES_PER_HALF)
            fill_slot_class(batcher, slot_idx, correct=False,
                            start_miner_idx=base + BATCHES_PER_HALF, batch_count=BATCHES_PER_HALF)

        assert batcher.is_window_complete() is False

        # Fill the last slot.
        last = PROMPTS_PER_WINDOW - 1
        base = last * 2 * BATCHES_PER_HALF
        fill_slot_class(batcher, last, correct=True,
                        start_miner_idx=base, batch_count=BATCHES_PER_HALF)
        fill_slot_class(batcher, last, correct=False,
                        start_miner_idx=base + BATCHES_PER_HALF, batch_count=BATCHES_PER_HALF)

        assert batcher.is_window_complete() is True

    def test_get_window_state_reflects_counts(self) -> None:
        """After submitting 4 completions to slot 0, state shows count=4 for slot 0."""
        batcher = make_batcher()
        req = make_request(slot_index=0)
        batcher.accept_submission(req)

        state = batcher.get_window_state()
        assert state.window_start == WINDOW_START
        assert len(state.slot_states) == PROMPTS_PER_WINDOW

        slot0 = next(s for s in state.slot_states if s.slot_index == 0)
        assert slot0.count == MINER_BATCH_SIZE
        assert slot0.settled is False

        for s in state.slot_states:
            if s.slot_index != 0:
                assert s.count == 0
                assert s.settled is False

    def test_is_slot_settled(self) -> None:
        batcher = make_batcher()
        assert batcher.is_slot_settled(0) is False
        fully_fill_slot(batcher, slot_index=0)
        assert batcher.is_slot_settled(0) is True


class TestAdvantageScoring:
    """Advantage scoring over all accepted completions (no kept-subset trimming)."""

    def test_auto_finalize_on_group_size_reached_balanced(self) -> None:
        """Fill slot to GROUP_SIZE with a balanced mix → auto-finalize →
        |adv|=1.0 each."""
        batcher = make_batcher()
        fully_fill_slot(batcher, slot_index=0)
        assert batcher.slots[0].finalized is True

        scores = batcher.get_miner_scores()
        for miner_idx in range(2 * BATCHES_PER_HALF):
            assert scores[f"miner_{miner_idx}"] == pytest.approx(
                MINER_BATCH_SIZE * 1.0
            )

    def test_imbalanced_timeout_pays_everyone_advantage_weighted(self) -> None:
        """3 batches correct + 1 batch wrong accepted → timeout → advantage scoring.

        Slot: 12 ones + 4 zeros, n=16.
        mean = 0.75, pop_std = sqrt(12*0.25²/16 + 4*0.75²/16) = sqrt(0.1875) ≈ 0.4330
        |adv| correct = 0.25/0.4330 ≈ 0.5774
        |adv| wrong   = 0.75/0.4330 ≈ 1.7321 — rare class earns ~3× per completion.

        No kept-subset trimming: every accepted miner is paid according to
        their information value.
        """
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)

        for m in range(3):
            assert batcher.accept_submission(
                make_request(
                    hotkey=f"correct_{m}",
                    completions=make_completions(
                        diverse_starts=_diverse_set(m),
                        correct_flags=[True] * 4,
                    ),
                )
            ).accepted is True
        assert batcher.accept_submission(
            make_request(
                hotkey="wrong_X",
                completions=make_completions(
                    diverse_starts=_diverse_set(99),
                    correct_flags=[False] * 4,
                ),
            )
        ).accepted is True

        # Timeout finalize.
        clock.advance(SLOT_DEADLINE_SECONDS + 1)
        assert batcher.finalize_due_slots() > 0

        scores = batcher.get_miner_scores()
        for m in range(3):
            assert scores[f"correct_{m}"] == pytest.approx(4 * 0.5774, abs=1e-3)
        assert scores["wrong_X"] == pytest.approx(4 * 1.7321, abs=1e-3)
        # Rare class earns ~3× per completion.
        assert scores["wrong_X"] / scores["correct_0"] == pytest.approx(3.0, abs=0.01)

    def test_degenerate_slot_pays_zero(self) -> None:
        """Only one class present → std=0 → all scores 0 → budget fully burns."""
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)
        for m in range(2):
            assert batcher.accept_submission(
                make_request(
                    hotkey=f"miner_{m}",
                    completions=make_completions(
                        diverse_starts=_diverse_set(m),
                        correct_flags=[True] * 4,
                    ),
                )
            ).accepted is True
        clock.advance(SLOT_DEADLINE_SECONDS + 1)
        batcher.finalize_due_slots()
        scores = batcher.get_miner_scores()
        assert scores.get("miner_0", 0.0) == 0.0
        assert scores.get("miner_1", 0.0) == 0.0

    def test_singleton_slot_pays_zero(self) -> None:
        """A slot with a single accepted completion → std undefined → score 0."""
        from reliquary.validator.batcher import _compute_slot_scores, AcceptedCompletion, ProblemSlot

        slot = ProblemSlot(slot_index=0, prompt_id="p", problem={})
        slot.accepted_completions.append(
            AcceptedCompletion(
                miner_hotkey="solo", tokens=[1], commit={}, reward=1.0, completion_text="x",
            )
        )
        assert _compute_slot_scores(slot) == {}

    def test_finalize_due_slots_noop_before_deadline(self) -> None:
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)
        assert batcher.finalize_due_slots() == 0
        clock.advance(SLOT_DEADLINE_SECONDS - 1)
        assert batcher.finalize_due_slots() == 0
        assert all(not s.finalized for s in batcher.slots)

    def test_finalize_due_slots_marks_all_remaining_slots(self) -> None:
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)
        clock.advance(SLOT_DEADLINE_SECONDS + 1)
        assert batcher.finalize_due_slots() == PROMPTS_PER_WINDOW
        assert all(s.finalized for s in batcher.slots)
        assert batcher.finalize_due_slots() == 0  # idempotent

    def test_burn_score_empty_window_equals_full_notional(self) -> None:
        """No signal → full notional budget burns."""
        batcher = make_batcher()
        assert batcher.get_burn_score() == float(PROMPTS_PER_WINDOW * GROUP_SIZE)

    def test_burn_score_balanced_slot_zero_burn_for_that_slot(self) -> None:
        """One fully-balanced slot emits GROUP_SIZE × |adv|=1.0 signal → burn =
        notional - GROUP_SIZE."""
        batcher = make_batcher()
        fully_fill_slot(batcher, slot_index=0)
        expected_burn = float(PROMPTS_PER_WINDOW * GROUP_SIZE - GROUP_SIZE)
        assert batcher.get_burn_score() == pytest.approx(expected_burn)

    def test_burn_score_imbalanced_slot_captures_deficit(self) -> None:
        """Imbalanced slot emits less signal than notional → deficit burns."""
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)
        for m in range(3):
            batcher.accept_submission(
                make_request(
                    hotkey=f"correct_{m}",
                    completions=make_completions(
                        diverse_starts=_diverse_set(m),
                        correct_flags=[True] * 4,
                    ),
                )
            )
        batcher.accept_submission(
            make_request(
                hotkey="wrong_X",
                completions=make_completions(
                    diverse_starts=_diverse_set(99),
                    correct_flags=[False] * 4,
                ),
            )
        )
        clock.advance(SLOT_DEADLINE_SECONDS + 1)
        batcher.finalize_due_slots()
        # Slot signal: 12*0.5774 + 4*1.7321 ≈ 13.858
        scores = batcher.get_miner_scores()
        miner_total = sum(scores.values())
        notional = float(PROMPTS_PER_WINDOW * GROUP_SIZE)
        assert batcher.get_burn_score() == pytest.approx(notional - miner_total)

    def test_scores_sum_across_slots(self) -> None:
        """A miner contributing to multiple balanced slots accumulates scores."""
        clock = _FakeClock()
        batcher = make_batcher(time_fn=clock)

        batcher.accept_submission(make_request(
            hotkey="miner_A", slot_index=0,
            completions=make_completions(diverse_starts=_diverse_set(0), correct_flags=[True] * 4),
        ))
        batcher.accept_submission(make_request(
            hotkey="miner_B", slot_index=0,
            completions=make_completions(diverse_starts=_diverse_set(1), correct_flags=[False] * 4),
        ))
        batcher.accept_submission(make_request(
            hotkey="miner_A", slot_index=1,
            completions=make_completions(diverse_starts=_diverse_set(0), correct_flags=[False] * 4),
        ))
        batcher.accept_submission(make_request(
            hotkey="miner_C", slot_index=1,
            completions=make_completions(diverse_starts=_diverse_set(1), correct_flags=[True] * 4),
        ))

        clock.advance(SLOT_DEADLINE_SECONDS + 1)
        batcher.finalize_due_slots()

        # Each slot is 4+4=8 balanced → |adv|=1.0 each.
        scores = batcher.get_miner_scores()
        assert scores["miner_A"] == pytest.approx(8.0)
        assert scores["miner_B"] == pytest.approx(4.0)
        assert scores["miner_C"] == pytest.approx(4.0)


class TestRewardHistogramExposed:
    def test_window_state_includes_reward_histogram(self) -> None:
        batcher = make_batcher()
        # 1 miner × 4 corrects, 1 miner × 4 wrongs in slot 0.
        batcher.accept_submission(
            make_request(
                hotkey="miner_A",
                slot_index=0,
                completions=make_completions(
                    diverse_starts=_diverse_set(0), correct_flags=[True] * 4
                ),
            )
        )
        batcher.accept_submission(
            make_request(
                hotkey="miner_B",
                slot_index=0,
                completions=make_completions(
                    diverse_starts=_diverse_set(1), correct_flags=[False] * 4
                ),
            )
        )

        state = batcher.get_window_state()
        slot0 = next(s for s in state.slot_states if s.slot_index == 0)
        assert slot0.rewards == {"1.0": 4, "0.0": 4}

        # Empty slots have empty histogram (default).
        slot1 = next(s for s in state.slot_states if s.slot_index == 1)
        assert slot1.rewards == {}

    def test_histogram_omits_unseen_reward_values(self) -> None:
        """Slot with 4 corrects, 0 wrongs → histogram has only {"1.0": 4}."""
        batcher = make_batcher()
        batcher.accept_submission(make_request(hotkey="miner_A", slot_index=0))
        state = batcher.get_window_state()
        slot0 = next(s for s in state.slot_states if s.slot_index == 0)
        assert slot0.rewards == {"1.0": 4}


class TestScoresAndArchive:
    def test_get_archive_data_shape(self) -> None:
        """Archive data has correct top-level keys; empty slots are skipped."""
        batcher = make_batcher()
        batcher.accept_submission(make_request(slot_index=0))

        archive = batcher.get_archive_data()
        assert archive["window_start"] == WINDOW_START
        assert archive["randomness"] == RANDOMNESS
        assert archive["environment"] == "test"
        assert "slots" in archive

        # Only slot 0 has completions; others are empty → skipped.
        assert len(archive["slots"]) == 1
        slot_entry = archive["slots"][0]
        assert slot_entry["slot_index"] == 0
        assert slot_entry["prompt_id"] == PROMPT_ID
        assert "prompt" in slot_entry
        assert "ground_truth" in slot_entry
        assert "settled" in slot_entry
        assert len(slot_entry["completions"]) == MINER_BATCH_SIZE

        comp = slot_entry["completions"][0]
        assert "miner_hotkey" in comp
        assert "tokens" in comp
        assert "completion_text" in comp
        assert "reward" in comp
