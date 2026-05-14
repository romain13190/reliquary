# Rollout hash deduplication

## Problem

Miners pre-compute rollout submissions and replay them across windows.
The current validation pipeline (`GrpoWindowBatcher._accept_locked` in
`reliquary/validator/batcher.py:220`) accepts byte-identical rollouts
indefinitely once their prompt comes out of cooldown — a prompt that
exits the cooldown horizon (`BATCH_PROMPT_COOLDOWN_WINDOWS = 72`) can
be re-submitted with the exact same `(tokens, reward, commit)` triple
and pass every check (GRAIL, logprob, distribution, reward,
termination) because the validation is purely stateless w.r.t.
previously-seen content.

Result observed in production: a small pool of pre-computed
(prompt, rollouts) pairs cycles through the training batch every ~72
windows. The trained policy sees the same data repeatedly, training
signal degrades.

## Goal

Reject any submission containing a rollout whose token content matches
any rollout that has already entered a sealed training batch within
the cooldown horizon. Force miners who want to keep contributing to
either (a) generate fresh content from a recent checkpoint or (b)
maintain a pool large enough to exhaust the validator's hash memory —
both of which raise the marginal cost of replay.

Combined with `BATCH_PROMPT_COOLDOWN_WINDOWS` raised from 72 → 200,
the dedup horizon grows enough that natural model drift (catched by
the existing `evaluate_token_distribution` filter, see
`reliquary/validator/verifier.py:338`) starts to bite stale replays
that escape the byte-equal check via perturbation.

## Approach

Maintain an in-memory `RolloutHashSet` of `(rollout_hash, window)`
entries, bounded by the cooldown horizon. The hash is `SHA256` over
each rollout's `tokens` list (prompt + completion, packed as
big-endian `uint32`).

The set is:

- **Read** at submission time, after the cheap structural checks and
  before the expensive GRAIL forward pass — any hit returns the new
  `RejectReason.HASH_DUPLICATE`.
- **Written** at window seal time, by adding the hash of every rollout
  in every batched submission.
- **Persisted** by embedding each batched rollout's hash directly in
  the existing R2 archive payload (`_archive_window` in
  `reliquary/validator/service.py:519`) — no new R2 prefix.
- **Bootstrapped** at validator startup by scanning recent archives,
  exactly like `_rebuild_cooldown_from_history`.

The set's retention horizon equals `BATCH_PROMPT_COOLDOWN_WINDOWS`
(post-bump: 200). Entries older than the horizon are pruned at every
seal.

## Components

### New: `reliquary/validator/dedup.py`

```
class RolloutHashSet:
    def __init__(self, retention_windows: int) -> None: ...
    def add(self, h: bytes, window: int) -> None: ...
    def __contains__(self, h: bytes) -> bool: ...
    def prune(self, current_window: int) -> None: ...
    def rebuild_from_history(
        self, archives: list[dict], current_window: int
    ) -> None: ...
    def __len__(self) -> int: ...

def compute_rollout_hash(tokens: list[int]) -> bytes: ...
```

Internal state: `dict[bytes, int]` mapping hash → window-added.
`__contains__` is O(1). `prune` drops keys where
`current_window - window >= retention_windows`. `rebuild_from_history`
follows the contract of `CooldownMap.rebuild_from_history`: takes a
list of archive dicts (each carrying `window_start` and `batch`),
extracts every rollout hash from `batch[*].rollouts[*].hash`, and
indexes it. If an archive lacks the new `hash` field (pre-feature
data), the hash is recomputed by calling `compute_rollout_hash` on
`rollouts[*].tokens` from the archive (the only token list persisted
there). For honest submissions `commit["tokens"]` and `tokens` are
identical, so the rebuild is consistent with live accepts; a
malicious miner who shipped divergent values in the past would have
been verified against `commit["tokens"]` but archived under `tokens` —
acceptable, since their hash would simply fail to match a future
identical replay's `commit["tokens"]`, which is the conservative
direction.

`compute_rollout_hash` serialises tokens deterministically — each
`int` packed as a 4-byte big-endian unsigned integer, concatenated,
then hashed with SHA256. Returns the 32-byte digest.

### Modified: `reliquary/validator/batcher.py`

`GrpoWindowBatcher.__init__` accepts an optional `hash_set:
RolloutHashSet | None = None`. When `None`, dedup is disabled (for
test fixtures). When present, the batcher:

1. In `_accept_locked`, immediately after the `SUPERSEDED` check
   (around line 244), iterates `request.rollouts`; for each, computes
   `compute_rollout_hash(rollout.commit["tokens"])` and checks
   membership against (a) the persistent `hash_set` and (b) a local
   `set[bytes]` seeded empty per submission. First hit → `_reject(
   RejectReason.HASH_DUPLICATE, hotkey=hk, prompt_idx=pi)`. Otherwise
   the local set is updated and the hash kept aside in a per-submission
   list for reuse.
2. The list of computed hashes is stored on the `ValidSubmission`
   dataclass via a new `rollout_hashes: list[bytes] = field(
   default_factory=list)` attribute, so `seal_batch` and
   `_archive_window` reuse them without recomputing from
   `commit["tokens"]`.
3. In `seal_batch`, after the `select_batch` call and after the
   cooldown record loop, for every submission in the returned batch,
   for every `h` in `sub.rollout_hashes`, calls
   `hash_set.add(h, window_start)`, then calls
   `hash_set.prune(window_start)`.

The local set (step 1.b) ensures within-submission duplicates also
reject — two intra-submission collisions are themselves a strong
signal of pool replay.

### Modified: `reliquary/validator/service.py`

`ValidationService.__init__` constructs `self._hash_set =
RolloutHashSet(retention_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)`
alongside `self._cooldown_map`.

New method `_rebuild_hashes_from_history`, called from `run()` right
after `_rebuild_cooldown_from_history`, reads the same recent
archives and delegates to `self._hash_set.rebuild_from_history`.

`open_grpo_window` gains a `hash_set` parameter and forwards it to
`GrpoWindowBatcher`. The service-side `_open_window` passes
`self._hash_set`.

`_archive_window` extends the per-rollout dict produced by the local
`_rollout_payload(s, with_text=True)` helper (line 531) to include a
`hash` field — hex-encoded SHA256 read directly from
`s.rollout_hashes[i]` (the value computed at accept time). Only
batched rollouts get this field; runners-up keep the existing
metadata-only shape (no change).

### Modified: `reliquary/protocol/submission.py`

Add `HASH_DUPLICATE = "hash_duplicate"` to the `RejectReason` enum.
Position: alphabetical between `GRAIL_FAIL` and `LOGPROB_MISMATCH`,
matching the existing ordering convention.

### Modified: `reliquary/constants.py`

`BATCH_PROMPT_COOLDOWN_WINDOWS: 72 → 200`. The accompanying comment is
updated to reference the dedup horizon coupling.

## Data flow

```
Miner /submit
  ↓
batcher._accept_locked():
  WINDOW_MISMATCH check       (existing)
  WRONG_CHECKPOINT check      (existing)
  BAD_PROMPT_IDX check        (existing)
  PROMPT_IN_COOLDOWN check    (existing)
  SUPERSEDED check            (existing)
  HASH_DUPLICATE check        ← NEW: per-rollout SHA256(tokens)
  REWARD_MISMATCH (per rollout, existing)
  OUT_OF_ZONE (sigma, existing)
  per-rollout: schema, tokens, prompt match, signature, GRAIL,
               termination, logprob, distribution (all existing)
  ↓
  _valid.append(...)

Window seal:
  seal_batch():
    batch = select_batch(...)
    for sub in batch:
        cooldown.record_batched(sub.prompt_idx, window_start)  (existing)
        for rollout in sub.rollouts:
            hash_set.add(compute_rollout_hash(rollout.tokens),
                         window_start)                          ← NEW
  ↓
  _archive_window():
    each batch_entries[i].rollouts[j] gains a "hash" hex field   ← NEW
    archive uploaded to R2 (existing path)
  ↓
  hash_set.prune(window_start)                                  ← NEW

Validator restart:
  run() →
    _rebuild_cooldown_from_history()   (existing)
    _rebuild_hashes_from_history()     ← NEW, same archives, same horizon
```

## Edge cases

**Hash collision within a submission.** Two rollouts in the same
submission with identical tokens. Caught by maintaining a local
`set[bytes]` during the per-rollout loop in `_accept_locked` and
reject `HASH_DUPLICATE` if a hash is seen twice in either the local
set or the persistent `RolloutHashSet`.

**Random SHA256 collision.** Probability ~2^-256. Not handled
specially; the false-positive risk is below any meaningful threshold.

**Backwards-compat with old archives.** Archives written before this
change have no `hash` field on their rollouts. `rebuild_from_history`
checks for the field and, when absent, recomputes
`compute_rollout_hash(rollout["tokens"])` from the `tokens` already
present in the archive. No re-upload of old archives needed.

**Multi-validator coherence.** All validators reconstruct the same
hash set from the same R2 archives. The cooldown rebuild already
relies on this property; the hash rebuild inherits it.

**New validator joining mid-subnet.** Same as cooldown:
`_rebuild_hashes_from_history` populates the set from the last 200
windows in R2. A fresh validator catches up without coordination.

**Disabled dedup (tests).** Passing `hash_set=None` to
`GrpoWindowBatcher` short-circuits both the check and the
post-seal write. Existing batcher tests keep working unchanged.

**Pre-feature `BATCH_PROMPT_COOLDOWN_WINDOWS` raise.** Bumping
from 72 → 200 may temporarily increase the cooldown set size,
shrinking the eligible prompt pool. With a 12k-prompt env and
`B_BATCH = 8`, locked prompts at steady state are `8 × 200 = 1600`
≈ 13 % — well within capacity. No code change needed for this
specifically; covered by the constants bump.

## Non-goals

- **Robustness to single-token perturbation.** A miner who edits one
  token of a stored rollout produces a different SHA256. This is
  intentional out-of-scope; the existing distribution filter
  (`SAMPLING_LOW_Q10_MAX`, currently 0.025) handles perturbations
  that drift the chosen-token probabilities, and the wider cooldown
  horizon (200 windows) accumulates more training drift, sharpening
  that filter's effective signal. A future fingerprint mechanism
  (MinHash on token n-grams) is tracked separately.
- **Dedup against runners-up.** Only batched rollouts are stored.
  Pre-validated runners-up are not.
- **Cross-validator hash sharing in real time.** Each validator
  builds its hash set from R2 archives written by all validators,
  but no live hash gossip. Acceptable given the async R2 cadence.

## Testing

Unit tests for `RolloutHashSet`:

- `add` + `__contains__` round-trip
- `prune` drops only entries past the retention horizon
- `rebuild_from_history` correctly indexes archives with and without
  the new `hash` field (compat)
- `compute_rollout_hash` is deterministic for a given input list and
  differs for any single-token change (verifies the BE-32
  serialisation contract has no collisions on adjacent inputs)

Integration tests for `GrpoWindowBatcher`:

- Replay test: first submission accepted, identical second submission
  rejected with `HASH_DUPLICATE`
- Fresh content test: same prompt, different tokens, both accepted
- Intra-submission duplicate test: rollout `i == j` in the same
  submission → `HASH_DUPLICATE`
- Disabled test: `hash_set=None` accepts identical second submission
  (no behaviour change vs current)

Integration test for service bootstrap:

- Build a fake archive list with rollouts that lack `hash` fields
- Call `_rebuild_hashes_from_history`
- Assert the resulting `RolloutHashSet` rejects a subsequent matching
  submission

## Migration

No coordinated cutover. Validators that deploy this change start
populating archives with `hash` fields immediately. Validators on the
old code keep working; their archives just lack the new field. Old
archives stay readable. The cooldown bump (72 → 200) is the only
behaviour change visible to miners — same `PROMPT_IN_COOLDOWN`
response, longer effective duration.
