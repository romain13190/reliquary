# Rejected Submissions in R2 Archive — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `rejected[]` list to the validator's R2 window archive so rejected miners can self-diagnose, while denying cheaters a calibration signal on `GRAIL_FAIL` and capping anti-spam blow-up.

**Architecture:**
- `GrpoWindowBatcher` already counts rejections in `reject_counts: dict[str,int]` and discards everything else. We add a sibling list `rejected_submissions: list[RejectedSubmission]` capturing `{hotkey, prompt_idx, reason, optional diagnostics}` at rejection time.
- The batcher's private `_reject(reason)` is widened to receive request context + the latest computed diagnostics from `_accept_locked`. For `GRAIL_FAIL`, `sketch_diff_max` is intentionally omitted (anti-tuning). Per-hotkey cap (`REJECTED_LIST_CAP_PER_HOTKEY = 5`) prevents a single attacker flooding the archive — over the cap, only `reject_counts` is incremented.
- `ValidationService._archive_window` reads `batcher.rejected_submissions` and writes a `rejected` field next to the existing `runners_up` and `reject_summary` fields.

**Tech Stack:** Python 3.11, `dataclasses`, `pytest`, `pytest-asyncio`. No new deps.

---

## File Structure

**Modified:**
- `reliquary/constants.py` — add `REJECTED_LIST_CAP_PER_HOTKEY` constant.
- `reliquary/validator/batcher.py` — add `RejectedSubmission` dataclass; thread context into `_reject(...)`; accumulate per-hotkey-capped list.
- `reliquary/validator/service.py` — include `rejected[]` in archive payload built by `_archive_window`.
- `tests/unit/test_grpo_window_batcher.py` — new tests for rejection metadata + cap + grail anti-tuning omission.
- `tests/unit/test_archive_window_content.py` — extend the archive test with the new `rejected` key.

**No new files.** The new dataclass lives in `batcher.py` next to `ValidSubmission` because it is a sibling concept.

---

## Task 1: Add `REJECTED_LIST_CAP_PER_HOTKEY` constant

**Files:**
- Modify: `reliquary/constants.py`

- [ ] **Step 1: Add the cap constant**

Append to the file (place near the other validator-side knobs, e.g. after `MAX_NEW_TOKENS_PROTOCOL_CAP`):

```python
# Soft cap on per-hotkey entries persisted to ``archive["rejected"]`` per
# window. Beyond this, ``reject_counts`` still increments but no metadata is
# appended — protects the R2 payload size against a flood of garbage
# submissions from a single attacker.
REJECTED_LIST_CAP_PER_HOTKEY = 5
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from reliquary.constants import REJECTED_LIST_CAP_PER_HOTKEY; print(REJECTED_LIST_CAP_PER_HOTKEY)"`
Expected: `5`

- [ ] **Step 3: Commit**

```bash
git add reliquary/constants.py
git commit -m "chore(validator): add REJECTED_LIST_CAP_PER_HOTKEY constant"
```

---

## Task 2: Add `RejectedSubmission` dataclass + state in batcher

**Files:**
- Modify: `reliquary/validator/batcher.py:53-75` (add dataclass next to `ValidSubmission`)
- Modify: `reliquary/validator/batcher.py:140-145` (add list to `__init__`)

- [ ] **Step 1: Write the failing test** (new file or append to `test_grpo_window_batcher.py`)

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_rejected_submissions_list_initialised_empty():
    from reliquary.validator.batcher import GrpoWindowBatcher, RejectedSubmission
    b = _make_batcher()  # existing helper in this file
    assert hasattr(b, "rejected_submissions")
    assert b.rejected_submissions == []
    # Confirm the dataclass exists and has the documented fields.
    fields = {f.name for f in RejectedSubmission.__dataclass_fields__.values()}
    assert {
        "hotkey", "prompt_idx", "reason",
        "sketch_diff_max", "lp_dev_max", "dist_q10_min",
    }.issubset(fields)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_rejected_submissions_list_initialised_empty -v`
Expected: FAIL — `ImportError: cannot import name 'RejectedSubmission'`

- [ ] **Step 3: Add the dataclass**

In `reliquary/validator/batcher.py`, immediately after the `ValidSubmission` class (after line 75), add:

```python
@dataclass
class RejectedSubmission:
    """A submission that did NOT pass verification.

    Persisted to the R2 archive (subject to per-hotkey cap) so rejected
    miners can self-diagnose. Diagnostics are best-effort: only fields
    computed before the rejection point are populated.

    Anti-tuning: ``sketch_diff_max`` is intentionally LEFT NONE for
    ``GRAIL_FAIL`` rejections. Surfacing the exact diff would let a cheater
    calibrate against ``PROOF_SKETCH_TOLERANCE_BASE``. Other reject reasons
    are not threshold-tunable, so their diagnostics are surfaced verbatim.
    """

    hotkey: str
    prompt_idx: int
    reason: str  # RejectReason.value
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
```

- [ ] **Step 4: Initialise the list in `__init__`**

In `reliquary/validator/batcher.py`, find the block right after `self.reject_counts: dict[str, int] = {}` (around line 145) and append:

```python
        # Per-hotkey-capped metadata for rejected submissions. Persisted in
        # the R2 archive next to ``reject_counts`` so a rejected miner can
        # see *which* of their submissions failed and why, instead of just
        # an aggregate count. Cap protects against single-attacker flooding.
        self.rejected_submissions: list[RejectedSubmission] = []
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_rejected_submissions_list_initialised_empty -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(validator): introduce RejectedSubmission dataclass + accumulator"
```

---

## Task 3: Thread context into `_reject` and populate the list

**Files:**
- Modify: `reliquary/validator/batcher.py:356-358` (widen `_reject` signature + body)
- Modify: `reliquary/validator/batcher.py:175-320` (update every `return self._reject(...)` callsite to pass `request` + diagnostics)

- [ ] **Step 1: Write a failing test for grail_fail anti-tuning omission**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_rejected_grail_fail_omits_sketch_diff_max(monkeypatch):
    """GRAIL_FAIL must NOT expose sketch_diff_max — anti-tuning."""
    from reliquary.validator.verifier import ProofResult
    from reliquary.protocol.submission import RejectReason

    b = _make_batcher()  # existing helper

    # Stub verify_commitment to return a failing proof with a known diff.
    def fake_verify(commit, model, randomness):
        return ProofResult(
            all_passed=False,
            passed=2,
            checked=4,
            sketch_diff_max=4242,  # MUST NOT leak into archive
            logits=_empty_logits(),
        )
    b._verify_commitment = fake_verify
    b._verify_signature = lambda commit, hk: True

    req = _build_request(hotkey="hk_grail", prompt_idx=3)  # existing helper
    resp = b.accept_submission(req)
    assert resp.reason == RejectReason.GRAIL_FAIL

    assert len(b.rejected_submissions) == 1
    rec = b.rejected_submissions[0]
    assert rec.hotkey == "hk_grail"
    assert rec.prompt_idx == 3
    assert rec.reason == "grail_fail"
    assert rec.sketch_diff_max is None  # ← anti-tuning invariant
```

If `_make_batcher`, `_build_request`, `_empty_logits` helpers don't exist in the file, write them inline at the top of the test (find the existing batcher fixture used by `test_reject_grail_fail` at line 172 and reuse it). Show the helper bodies you add explicitly:

```python
def _empty_logits():
    import torch
    return torch.empty(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_rejected_grail_fail_omits_sketch_diff_max -v`
Expected: FAIL — `assert len(b.rejected_submissions) == 1` (list still empty because `_reject` does not append).

- [ ] **Step 3: Widen `_reject` signature + populate list**

Replace the existing `_reject` body in `reliquary/validator/batcher.py` (lines 356-358):

```python
    def _reject(
        self,
        reason: RejectReason,
        *,
        hotkey: str | None = None,
        prompt_idx: int | None = None,
        sketch_diff_max: int | None = None,
        lp_dev_max: float | None = None,
        dist_q10_min: float | None = None,
    ) -> BatchSubmissionResponse:
        from reliquary.constants import REJECTED_LIST_CAP_PER_HOTKEY

        self.reject_counts[reason.value] = self.reject_counts.get(reason.value, 0) + 1

        if hotkey is not None and prompt_idx is not None:
            already = sum(
                1 for r in self.rejected_submissions if r.hotkey == hotkey
            )
            if already < REJECTED_LIST_CAP_PER_HOTKEY:
                # Anti-tuning: never surface the GRAIL sketch diff to miners.
                # All other reasons get the diagnostics computed up to the
                # rejection point.
                if reason is RejectReason.GRAIL_FAIL:
                    sketch_diff_max = None
                self.rejected_submissions.append(
                    RejectedSubmission(
                        hotkey=hotkey,
                        prompt_idx=prompt_idx,
                        reason=reason.value,
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
                    )
                )
        return BatchSubmissionResponse(accepted=False, reason=reason)
```

- [ ] **Step 4: Update every `_reject` callsite in `_accept_locked` to pass `request` + diagnostics**

In `reliquary/validator/batcher.py:175-320`, every `return self._reject(REASON)` becomes a kwargs call. Show each callsite explicitly:

| Line (approx.) | Old | New |
|---|---|---|
| 179 | `return self._reject(RejectReason.WINDOW_MISMATCH)` | `return self._reject(RejectReason.WINDOW_MISMATCH, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 183 | `return self._reject(RejectReason.WRONG_CHECKPOINT)` | `return self._reject(RejectReason.WRONG_CHECKPOINT, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 185 | `return self._reject(RejectReason.BAD_PROMPT_IDX)` | `return self._reject(RejectReason.BAD_PROMPT_IDX, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 189 | `return self._reject(RejectReason.PROMPT_IN_COOLDOWN)` | `return self._reject(RejectReason.PROMPT_IN_COOLDOWN, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 197 | `return self._reject(RejectReason.SUPERSEDED)` | `return self._reject(RejectReason.SUPERSEDED, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 205 | `return self._reject(RejectReason.REWARD_MISMATCH)` | `return self._reject(RejectReason.REWARD_MISMATCH, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 209 | `return self._reject(RejectReason.OUT_OF_ZONE)` | `return self._reject(RejectReason.OUT_OF_ZONE, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 235 | `return self._reject(RejectReason.BAD_SCHEMA)` | `return self._reject(RejectReason.BAD_SCHEMA, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 239 | `return self._reject(RejectReason.BAD_TOKENS)` | `return self._reject(RejectReason.BAD_TOKENS, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 248 | `return self._reject(RejectReason.PROMPT_MISMATCH)` | `return self._reject(RejectReason.PROMPT_MISMATCH, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 250 | `return self._reject(RejectReason.BAD_SIGNATURE)` | `return self._reject(RejectReason.BAD_SIGNATURE, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx)` |
| 263 | `return self._reject(RejectReason.GRAIL_FAIL)` | `return self._reject(RejectReason.GRAIL_FAIL, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx, sketch_diff_max=proof.sketch_diff_max)` |
| 272 | `return self._reject(RejectReason.BAD_TERMINATION)` | `return self._reject(RejectReason.BAD_TERMINATION, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx, sketch_diff_max=sketch_diff_max)` |
| 300 | `return self._reject(RejectReason.LOGPROB_MISMATCH)` | `return self._reject(RejectReason.LOGPROB_MISMATCH, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx, sketch_diff_max=sketch_diff_max, lp_dev_max=lp_dev_max)` |
| 319 | `return self._reject(RejectReason.DISTRIBUTION_SUSPICIOUS)` | `return self._reject(RejectReason.DISTRIBUTION_SUSPICIOUS, hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx, sketch_diff_max=sketch_diff_max, lp_dev_max=lp_dev_max, dist_q10_min=dist_q10_min)` |

- [ ] **Step 5: Run the failing test to verify it now passes**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_rejected_grail_fail_omits_sketch_diff_max -v`
Expected: PASS

- [ ] **Step 6: Run the full batcher suite to verify no regression**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v`
Expected: All previously passing tests still pass (signature change is keyword-only and backward-compatible for tests that did not pass kwargs).

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(validator): record per-hotkey rejection metadata, omit grail diff"
```

---

## Task 4: Per-hotkey cap test

**Files:**
- Modify: `tests/unit/test_grpo_window_batcher.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_rejected_submissions_capped_per_hotkey(monkeypatch):
    """6th rejection from same hotkey must NOT grow the list (cap = 5)."""
    from reliquary.protocol.submission import RejectReason
    from reliquary.constants import REJECTED_LIST_CAP_PER_HOTKEY

    assert REJECTED_LIST_CAP_PER_HOTKEY == 5  # plan invariant

    b = _make_batcher()
    # Trigger BAD_PROMPT_IDX repeatedly — cheapest reject path that needs no
    # heavy stubbing (just send prompt_idx >= len(env)).
    spam_hotkey = "hk_spam"
    for i in range(REJECTED_LIST_CAP_PER_HOTKEY + 3):
        req = _build_request(
            hotkey=spam_hotkey,
            prompt_idx=10_000 + i,  # past env size to force BAD_PROMPT_IDX
        )
        resp = b.accept_submission(req)
        assert resp.reason == RejectReason.BAD_PROMPT_IDX

    # List capped, but counter keeps climbing.
    assert len(b.rejected_submissions) == REJECTED_LIST_CAP_PER_HOTKEY
    assert b.reject_counts["bad_prompt_idx"] == REJECTED_LIST_CAP_PER_HOTKEY + 3

    # Different hotkey gets its own quota.
    other_req = _build_request(hotkey="hk_other", prompt_idx=99_999)
    b.accept_submission(other_req)
    assert len(b.rejected_submissions) == REJECTED_LIST_CAP_PER_HOTKEY + 1
    assert b.rejected_submissions[-1].hotkey == "hk_other"
```

- [ ] **Step 2: Run test to verify it passes (cap was implemented in Task 3)**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_rejected_submissions_capped_per_hotkey -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_grpo_window_batcher.py
git commit -m "test(validator): cover per-hotkey rejection cap behaviour"
```

---

## Task 5: Surface `rejected[]` in the R2 archive

**Files:**
- Modify: `reliquary/validator/service.py:467-475` (build the new field in the archive dict)

- [ ] **Step 1: Write the failing test in `test_archive_window_content.py`**

In `tests/unit/test_archive_window_content.py`, modify the existing `test_archive_includes_prompt_and_rollout_content` to assert the new field. Right after the `# reject_summary persisted from batcher.` block at line 160, append:

```python
    # rejected[] persisted from batcher.rejected_submissions — metadata only.
    assert "rejected" in archive
    assert archive["rejected"] == [
        {
            "hotkey": "hk_evict",
            "prompt_idx": 4,
            "reason": "out_of_zone",
            "sketch_diff_max": None,
            "lp_dev_max": None,
            "dist_q10_min": None,
        },
        {
            "hotkey": "hk_grail_cheater",
            "prompt_idx": 5,
            "reason": "grail_fail",
            "sketch_diff_max": None,  # anti-tuning: never surfaced
            "lp_dev_max": None,
            "dist_q10_min": None,
        },
    ]
```

And update the batcher mock setup near line 85 to provide the rejected list:

```python
    from reliquary.validator.batcher import RejectedSubmission
    batcher.reject_counts = {"out_of_zone": 3, "logprob_mismatch": 1}
    batcher.rejected_submissions = [
        RejectedSubmission(
            hotkey="hk_evict", prompt_idx=4, reason="out_of_zone",
        ),
        RejectedSubmission(
            hotkey="hk_grail_cheater", prompt_idx=5, reason="grail_fail",
            # sketch_diff_max intentionally None — set by _reject() in prod.
        ),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_archive_window_content.py -v`
Expected: FAIL — `KeyError: 'rejected'` or `assert "rejected" in archive`.

- [ ] **Step 3: Add the field in `_archive_window`**

In `reliquary/validator/service.py`, locate the archive dict construction (around line 467-475). Replace the `archive = {...}` block with:

```python
        rejected_payload = [
            {
                "hotkey": r.hotkey,
                "prompt_idx": r.prompt_idx,
                "reason": r.reason,
                "sketch_diff_max": r.sketch_diff_max,
                "lp_dev_max": r.lp_dev_max,
                "dist_q10_min": r.dist_q10_min,
            }
            for r in getattr(batcher, "rejected_submissions", [])
        ]

        archive = {
            "window_start": batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": batch_entries,
            "runners_up": runners_up,
            "reject_summary": dict(getattr(batcher, "reject_counts", {})),
            "rejected": rejected_payload,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_archive_window_content.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_archive_window_content.py
git commit -m "feat(validator): publish rejected[] metadata in window archive"
```

---

## Task 6: Verify all suites green + manual sanity check on log volume

**Files:** none modified.

- [ ] **Step 1: Run the full unit suite**

Run: `pytest tests/unit/ -x -q`
Expected: all green.

- [ ] **Step 2: Run any integration tests touching the archive or batcher**

Run: `pytest tests/integration/test_v21_window_loop.py tests/integration/test_miner_validator.py -x -q`
Expected: all green.

- [ ] **Step 3: Eyeball the log volume**

The `grail_fail diag …` log line at `batcher.py:257-262` is unchanged by this work (still WARNING, still local). Confirm no test output explosion:

Run: `pytest tests/unit/test_grpo_window_batcher.py -q 2>&1 | wc -l`
Expected: <500 lines.

- [ ] **Step 4: Final commit only if any tail-end fixes were needed**

If steps 1-3 surfaced regressions, fix and commit per failing case. Otherwise no commit.

---

## Self-Review

**Spec coverage:**
- [x] Rejected miners can self-diagnose → archive `rejected[]` includes hotkey + prompt_idx + reason (Task 5).
- [x] Anti-tuning on `GRAIL_FAIL` → `_reject` strips `sketch_diff_max` for that reason (Task 3 step 3); test enforces it (Task 3 step 1).
- [x] Spam cap → `REJECTED_LIST_CAP_PER_HOTKEY = 5`, applied in `_reject`; test in Task 4.
- [x] Backward compat → archive consumers can ignore the new field; existing `reject_summary` and `runners_up` are unchanged.

**Placeholder scan:** every step contains either an exact code block or an exact command + expected output. No "TBD", no "similar to", no naked references.

**Type consistency:**
- `RejectedSubmission.reason` is `str` (the `RejectReason.value`) throughout — matches what is asserted in Task 5.
- `_reject` kwargs are all `Optional[...]` and consistently named across signature, callsites, and tests.
- Archive payload dict keys (`sketch_diff_max`, `lp_dev_max`, `dist_q10_min`) match the dataclass field names.
