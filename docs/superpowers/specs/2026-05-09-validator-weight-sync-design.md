# Aligning validator weight submissions on subnet epoch

## Problem

Each validator on netuid 81 currently submits weights on its own clock:
`WeightOnlyValidator.run()` (`reliquary/validator/weight_only.py:46-61`)
tracks `_last_submit_block` and submits every `WEIGHT_SUBMISSION_INTERVAL = 360`
blocks since *its own* last submit. Two validators that boot at different
times drift apart, read different snapshots of the R2 archive set, and
end up submitting different weights on chain. Symptom: divergent weights
between validators, observed in production.

## Goal

All validators of the subnet submit weights inside the same short window
each epoch. Then they read near-identical R2 archive sets, replay the
same deterministic EMA, and converge on near-identical weight vectors.

## Approach

Anchor submissions to the bittensor subnet tempo, not to a per-validator
clock.

bittensor exposes `subtensor.blocks_until_next_epoch(netuid)`. All
validators of a given netuid see the same epoch boundary because the
formula is purely a function of `(netuid, current_block, tempo)`.

Submit when `blocks_until_next_epoch(netuid) <= EPOCH_SUBMIT_LEAD_BLOCKS`
(default 20 blocks ≈ 4 min on 12 s/block). At most one submit per epoch,
tracked by `_last_submit_epoch`.

A freshly-booted validator submits immediately on its first poll
regardless of where in the epoch it lands — surfaces weights on chain
right after a restart, then converges to the synced cadence from the
next epoch.

## Changes

**`reliquary/infrastructure/chain.py`** — add async wrapper:

```python
async def blocks_until_next_epoch(subtensor, netuid: int) -> int | None:
    return await asyncio.wait_for(
        asyncio.to_thread(subtensor.blocks_until_next_epoch, netuid),
        timeout=SUBSTRATE_CALL_TIMEOUT,
    )
```

Same timeout treatment as the other chain wrappers (`get_current_block`,
`set_weights`).

**`reliquary/validator/weight_only.py`** — replace the block-interval
gate in `run()`:

```python
# state
self._last_submit_epoch: int | None = None  # was: _last_submit_block

# loop body
blocks_until = await chain.blocks_until_next_epoch(subtensor, self.netuid)
current_block = await chain.get_current_block(subtensor)
current_epoch = (current_block + blocks_until + 1)  # absolute end-of-epoch block, used as epoch id

if self._last_submit_epoch == current_epoch:
    await asyncio.sleep(POLL_INTERVAL_SECONDS)
    continue

bootstrap = self._last_submit_epoch is None
if not bootstrap and blocks_until > EPOCH_SUBMIT_LEAD_BLOCKS:
    await asyncio.sleep(POLL_INTERVAL_SECONDS)
    continue

if await self.submit_once(subtensor):
    self._last_submit_epoch = current_epoch
```

Bootstrap path covers "validator just started" — submits at the next
poll regardless of position in the epoch.

**`reliquary/constants.py`**

- Remove `WEIGHT_SUBMISSION_INTERVAL = 360`
- Add `EPOCH_SUBMIT_LEAD_BLOCKS = 20`
- `ROLLING_WINDOWS` (currently `WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH`)
  becomes `tempo // WINDOW_LENGTH` computed at startup. To keep the EMA
  replay window stable, hardcode `ROLLING_WINDOWS_FALLBACK = 72` and use
  the dynamic value when the subtensor query succeeds.

## Tests

Existing weight-only tests in `tests/` use a stub subtensor — extend the
stub with `blocks_until_next_epoch` returning a configurable value.
Cases:
- bootstrap: first call submits regardless of `blocks_until`
- inside lead window: submits, sets `_last_submit_epoch`
- outside lead window after bootstrap: skips
- second call same epoch: skips
- next epoch: submits again
- substrate timeout on `blocks_until_next_epoch`: existing reconnect path
  catches it (covered by current tests)

## Out of scope

- Reading weights *from chain* to detect already-submitted-this-epoch.
  In-process state is enough; if the validator restarts mid-epoch, the
  bootstrap re-submit re-asserts the same EMA — duplicate submit is
  cheap, the chain accepts it.
- Changing the trainer side. `service.py` already delegates to a
  concurrent `WeightOnlyValidator`; modifying `WeightOnlyValidator`
  fixes both deployment modes.

## Risks

Within the 20-block window, the earliest and latest submitter see up to
~4 min of new R2 archives. Given typical window cadence (1+ min), that's
0–4 extra archives — EMA divergence on the order of `EMA_ALPHA * 1/B_BATCH`
per missed window. Acceptable for now; for perfect convergence, anchor
the EMA replay set to a deterministic snapshot (e.g. `archives where
window_start < epoch_start_block / blocks_per_window`). Out of scope.
