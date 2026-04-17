# Advantage-Based Miner Scoring — Design

Date: 2026-04-17
Branch: `feat/grpo-batching`
Status: Design approved (option A in brainstorm)

## Problem

Current scoring is `miner_score = sum(c.reward for c in accepted_completions)`.

This rewards **correctness only**, not **information contribution to the GRPO
training signal**. Pathological cases:

- Slot ends with `[1]×32` → variance 0 → zero GRPO signal → all 32 miners
  paid for producing data with no training value.
- Slot ends with `[1]×30 + [0]×2` → the 2 wrongs are extremely informative
  (huge advantage) but earn 0; the 30 corrects each earn 1.0 despite being
  largely redundant.
- Sybil attack: one entity with 10 hotkeys submits 10 correct completions to
  an easy slot — total reward = 10×, total compute = 10×, total information
  contribution to GRPO = ~0.

The metric we should be paying for is the **information value** of each
completion to the GRPO advantage computation that downstream training
will perform.

## Approved approach

### Scoring formula

For each accepted completion `c` in a settled slot with completions
`{c_1, ..., c_n}` and rewards `R = {r_1, ..., r_n}`:

```
mean = mean(R)
std  = stdev(R)
score(c) = |c.reward - mean| / std    if std > 0
         = 0                          if std == 0 (degenerate slot)
```

Per-miner per-window score = sum of `score(c)` across all their accepted
completions in all 8 slots.

Per-miner cumulative score (for weight submission, every `ROLLING_WINDOWS`
windows) = sum of per-window scores.

`compute_weights` keeps its current shape: `score → score^SUPERLINEAR_EXPONENT
→ normalize`. The superlinear exponent stays for concentration emphasis,
but its sybil-resistance role is now redundant — advantage scoring is
already sybil-resistant by construction (rare-class advantage diminishes
as more identical-class completions land).

### Why this works

1. **Self-balancing market.** Rare reward classes pay more. Miners who can
   see the slot's running distribution submit the rare class until balance,
   then stop. Slot converges to high-entropy.
2. **Degenerate-slot burn (implicit).** When all rewards are identical,
   `std == 0` → all scores 0 → that slot's emission share evaporates from
   the cumulative miner totals. Equivalent to burn without an explicit UID-0
   transfer: the unused share is simply not credited to any miner, and the
   normalization in `compute_weights` redistributes nothing extra (the
   emission is per-window-aggregate, not per-slot, so the budget gap goes
   to the chain's emission accounting). For an explicit burn, this can be
   added later in `_submit_weights` by detecting unused slot budget — out
   of scope for this iteration.
3. **Sybil resistance from the math.** Adding the Nth identical correct
   completion when the slot is `[1]×(N-1) + [0]×k` shifts the mean toward
   1.0 and shrinks std → reduces every existing correct's score. Marginal
   utility of duplicating diminishes rapidly.
4. **Easy/hard prompts handled symmetrically.** A "hard" slot that's mostly
   wrong heavily rewards the rare correct; an "easy" slot mostly correct
   heavily rewards the rare wrong. Either way the slot's GRPO signal is
   maximized.

### Live state exposure

`WindowStateResponse.slot_states[i]` gains a `rewards` field — the histogram
of accepted completion rewards. Format: `{reward_value: count}`, e.g.
`{1.0: 28, 0.0: 0}` for a slot with 28 corrects and no wrongs yet.

A miner GETs `/window/{n}/state`, sees the distribution per slot, and can
strategically choose which slot to target and what reward class to aim for.
We do NOT expose individual completion content — that stays private until
the dataset is published at window end.

### Settlement — minimal change

Slots still settle at `GROUP_SIZE = 32` accepted completions. Window still
ends at the time deadline. The change is **only in scoring** — what a
completion is *worth*.

Slots that don't reach 32 by deadline are still settled with whatever
they have (down to `n=2` minimum for std to be defined). Slots with `n < 2`
contribute no scores.

### What does NOT change

- The accept/reject path in `WindowBatcher.accept_submission` is unchanged
  (cross-miner prefix dedup stays as the only diversity gate).
- The GRAIL proof verification path is unchanged.
- The dataset archive shape (`get_archive_data`) is unchanged.
- The HTTP API for `/submit` is unchanged.
- Window cadence (`PROMPTS_PER_WINDOW`, `GROUP_SIZE`, etc.) unchanged.
- `compute_weights` signature is unchanged.

## Files touched

| File | Change |
|---|---|
| `grail/validator/batcher.py` | `get_miner_scores()` rewritten to compute advantage-based scores. New helper `_compute_slot_scores(slot) -> dict[hotkey, float]`. |
| `grail/protocol/submission.py` | `SlotState` gains `rewards: dict[str, int]` (str-keyed because JSON; e.g. `{"1.0": 28, "0.0": 0}`). |
| `grail/validator/server.py` | No change — passes `WindowStateResponse` through. |
| `tests/unit/test_batcher.py` | New tests: degenerate slot → 0 scores; balanced slot → equal scores; skewed slot → rare class earns more. |
| `tests/unit/test_submission.py` | Test the new `rewards` field on `SlotState`. |
| `tests/unit/test_validator_server.py` | Already snapshot-mocked — minor update to fixture if needed. |

## Test plan

### `test_batcher.py`

1. `test_get_miner_scores_balanced_slot_equal_payouts` — 16 corrects, 16 wrongs, all from different miners → each miner's score = 1.0 (|advantage| = 1).
2. `test_get_miner_scores_skewed_slot_rare_class_paid_more` — 30 corrects + 2 wrongs → rare-class miner score per completion ~3.96; common-class score ~0.13.
3. `test_get_miner_scores_degenerate_slot_zero` — all rewards identical → all scores 0.
4. `test_get_miner_scores_singleton_slot_zero` — only 1 completion in slot → std undefined → score 0.
5. `test_get_miner_scores_sums_across_slots` — same miner contributes to multiple slots → score is sum.
6. `test_window_state_includes_reward_histogram` — after submissions, the WindowStateResponse exposes the reward distribution per slot.

### Backward-compat

The existing `test_get_miner_scores_sums_rewards` test (which assumes
score = sum of raw rewards) becomes wrong. Rewrite it to use the
advantage-based expectation.

## Edge cases & decisions

- **Slot with all corrects but only 2 completions** (`[1, 1]`): std=0 → score 0. Acceptable — slot didn't generate signal.
- **Continuous rewards** (when an env returns 0.5, 0.7, etc., not just 0/1): the formula still works. Currently GSM8K is binary so this is a future concern.
- **Negative rewards** (some envs penalize bad completions): formula handles them; |advantage| is always non-negative.
- **Score precision**: float64 is plenty.
- **Score overflow at superlinear x^4**: for ~32 completions per slot × |adv| up to ~5 × 8 slots × 12 windows = ~1500. ^4 = ~5e12. No overflow risk.

## Out of scope (explicit non-goals)

- **Strategic miner submission logic** (look at slot state, pick rare class). The validator exposes the state; what the miner does with it is the miner operator's choice. The current `MiningEngine` will continue to round-robin all 8 slots without strategic selection.
- **Explicit UID-0 burn** for unused slot budget. Implicit zero-credit is sufficient for this iteration.
- **Continuous-reward calibration** (e.g. partial credit for almost-correct numeric answers). Env contract stays binary; advantage scoring works for any bounded reward.
- **Cross-window advantage normalization** (would require knowing all windows' distributions before scoring). Sum across windows is sufficient.

## Migration path

Single commit on `feat/grpo-batching`. No protocol-level breaking change
for miners — the `accept_submission` API is unchanged, only the score
computation differs. Miners can keep submitting as they do; the new
`rewards` field on `WindowStateResponse` is additive.
