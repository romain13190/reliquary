# GRAIL Security Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 security vulnerabilities identified in the GRAIL V1 subnet, from critical to low severity.

**Architecture:** Each fix is self-contained and targets a specific file. Fixes are ordered by severity (critical first). Each task includes a test-first step, the implementation, and a verification step.

**Tech Stack:** Python 3.11+, pytest, torch, hashlib, bittensor

---

### Task 1: Drand beacon signature verification — fail-closed + fix DST

**Files:**
- Modify: `grail/infrastructure/drand.py:115-177` (verify_beacon_signature)
- Modify: `tests/unit/test_security.py:312-341` (TestBeaconSignatureVerification)

**Problem:** The fallback hash-based check at `drand.py:161-166` is a no-op: an attacker crafts `fake_randomness = SHA256(fake_sig)` and the check passes. Also, the DST at line 155 uses `BN254G1` instead of `BLS12381G1` for quicknet, so even with blst installed, verification always fails and falls through to the broken fallback.

- [ ] **Step 1: Update the security test to assert fail-closed behavior**

The existing test `test_accepts_matching_sha256_randomness` asserts the broken fallback *passes*. Flip it: without blst, crafted SHA256-matching pairs must be rejected. Add a test that verifies fail-closed when no BLS library is available.

In `tests/unit/test_security.py`, replace `TestBeaconSignatureVerification` with:

```python
class TestBeaconSignatureVerification:
    """Attack: MITM injects fake beacon without valid BLS signature.
    Fix: verify_beacon_signature rejects missing/invalid signatures.
    Without a BLS library (blst), verification must fail-closed (return False).
    """

    def test_rejects_none_signature(self):
        from grail.infrastructure.drand import verify_beacon_signature
        assert verify_beacon_signature("abc", 1, "dead" * 8, None) is False

    def test_rejects_empty_signature(self):
        from grail.infrastructure.drand import verify_beacon_signature
        assert verify_beacon_signature("abc", 1, "dead" * 8, "") is False

    def test_rejects_wrong_randomness(self):
        from grail.infrastructure.drand import verify_beacon_signature
        with patch("grail.infrastructure.drand._fetch_chain_pubkey", return_value=b"\x00" * 48):
            assert verify_beacon_signature("abc", 1, "bb" * 32, "aa" * 48) is False

    def test_no_bls_library_fails_closed(self):
        """Without blst, verification must return False — not fall through
        to a hash-based check that any attacker can satisfy."""
        from grail.infrastructure.drand import verify_beacon_signature
        sig_bytes = bytes.fromhex("cc" * 48)
        crafted_rand = hashlib.sha256(sig_bytes).hexdigest()
        with patch("grail.infrastructure.drand._fetch_chain_pubkey", return_value=b"\x01" * 48):
            # Even though SHA256(sig) == randomness, this MUST reject
            assert verify_beacon_signature("abc", 1, crafted_rand, "cc" * 48) is False

    def test_correct_dst_for_quicknet(self):
        """The DST must use BLS12381G1, not BN254G1."""
        import grail.infrastructure.drand as drand_mod
        source = inspect.getsource(drand_mod.verify_beacon_signature)
        assert "BLS12381G1" in source or "BLS_SIG_BLS12381G1" in source
        assert "BN254G1" not in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_security.py::TestBeaconSignatureVerification -v`
Expected: `test_no_bls_library_fails_closed` FAILS (currently returns True), `test_correct_dst_for_quicknet` FAILS.

- [ ] **Step 3: Fix verify_beacon_signature — remove fallback, fix DST**

In `grail/infrastructure/drand.py`, replace the `verify_beacon_signature` function body (lines 138-177) with:

```python
    try:
        import hashlib as _hl
        import struct as _st

        round_bytes = _st.pack(">Q", round_number)
        message = _hl.sha256(round_bytes).digest()

        sig_bytes = bytes.fromhex(signature_hex)

        # Try blst (fast C library) — the only supported verification path.
        try:
            from blst import P1_Affine, P2_Affine  # type: ignore[import-untyped]

            sig = P1_Affine(sig_bytes)
            pk = P2_Affine(pubkey)
            # DST for drand quicknet (BLS12-381 G1, RFC 9380)
            dst = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"
            result = sig.core_verify(pk, True, message, dst)
            return result == 0  # BLST_SUCCESS
        except ImportError:
            # SECURITY: No BLS library available — fail closed.
            # A hash-based fallback would be trivially forgeable.
            logger.error(
                "[Drand] No BLS library (blst) installed — cannot verify beacon. "
                "Install blst: pip install blst"
            )
            return False

    except Exception as e:
        logger.warning("[Drand] beacon signature verification failed: %s", e)
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_security.py::TestBeaconSignatureVerification -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add grail/infrastructure/drand.py tests/unit/test_security.py
git commit -m "security: drand beacon verification fail-closed + fix BLS DST

Remove hash-based fallback that any attacker could satisfy by crafting
SHA256(fake_sig)==randomness. Fix DST from BN254G1 to BLS12381G1 for
quicknet. Without blst installed, verification now returns False."
```

---

### Task 2: Deterministic beacon round for miner/validator synchronization

**Files:**
- Modify: `grail/infrastructure/chain.py:61-72` (compute_window_randomness)
- Modify: `grail/infrastructure/drand.py:446-464` (get_beacon)
- Modify: `grail/validator/service.py:92-101` (_process_window)
- Modify: `grail/miner/engine.py:57-71` (mine_window)
- Modify: `grail/constants.py` (add DRAND_ROUND_OFFSET)
- Create: `tests/unit/test_beacon_sync.py`

**Problem:** Both miner and validator call `get_beacon("latest")` at different times, getting different drand rounds. Different round = different randomness = legitimate miners' proofs fail. The round must be deterministically derived from the window's block number.

- [ ] **Step 1: Write test for deterministic round computation**

Create `tests/unit/test_beacon_sync.py`:

```python
"""Tests for deterministic beacon round selection.

Miner and validator must agree on the same drand round for a given window.
The round is derived from the window_start block number, not fetched as 'latest'.
"""

from grail.constants import BLOCK_TIME_SECONDS, WINDOW_LENGTH


class TestDeterministicBeaconRound:
    def test_compute_window_randomness_includes_round(self):
        """Window randomness must bind the drand round number to prevent
        a miner from choosing a favorable round."""
        from grail.infrastructure.chain import compute_window_randomness

        block_hash = "aa" * 32
        drand_rand = "bb" * 32

        r1 = compute_window_randomness(block_hash, drand_rand, drand_round=100)
        r2 = compute_window_randomness(block_hash, drand_rand, drand_round=101)
        r_no_round = compute_window_randomness(block_hash, drand_rand, drand_round=None)

        assert r1 != r2, "Different rounds must produce different randomness"
        assert r1 != r_no_round, "Providing a round must change the result"

    def test_compute_drand_round_for_window(self):
        """Round selection must be deterministic from window_start and chain params."""
        from grail.infrastructure.chain import compute_drand_round_for_window

        genesis_time = 1000
        period = 3

        # Window at block 100: timestamp = 100 * 12 = 1200
        # Expected round = 1 + (1200 - 1000) // 3 = 1 + 66 = 67
        r = compute_drand_round_for_window(100, genesis_time, period)
        assert r == 67

        # Same input = same output (deterministic)
        assert compute_drand_round_for_window(100, genesis_time, period) == 67

        # Different window = different round
        assert compute_drand_round_for_window(130, genesis_time, period) != 67

    def test_compute_drand_round_before_genesis_returns_1(self):
        from grail.infrastructure.chain import compute_drand_round_for_window

        # Window at block 10: timestamp = 120, before genesis at 1000
        r = compute_drand_round_for_window(10, 1000, 3)
        assert r == 1  # Clamp to round 1 (the first valid round)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_beacon_sync.py -v`
Expected: ImportError / AttributeError (functions don't exist yet)

- [ ] **Step 3: Add compute_drand_round_for_window to chain.py, update compute_window_randomness**

In `grail/infrastructure/chain.py`, replace `compute_window_randomness` and add the new function:

```python
def compute_drand_round_for_window(
    window_start_block: int, genesis_time: int, period: int
) -> int:
    """Deterministically compute which drand round to use for a window.

    Both miner and validator call this with the same inputs to agree
    on a single round — no "latest" fetch needed.

    Returns:
        The drand round number (>= 1).
    """
    window_timestamp = window_start_block * BLOCK_TIME_SECONDS
    if window_timestamp < genesis_time:
        return 1  # Clamp: can't go before round 1
    return 1 + (window_timestamp - genesis_time) // period


def compute_window_randomness(
    block_hash: str,
    drand_randomness: str | None = None,
    drand_round: int | None = None,
) -> str:
    """Combine block hash, drand randomness, and round into window randomness.

    Including the round number prevents a miner from choosing a round
    whose randomness is favorable.
    """
    clean_hash = block_hash.replace("0x", "")
    if drand_randomness:
        material = bytes.fromhex(clean_hash) + bytes.fromhex(drand_randomness)
        if drand_round is not None:
            material += drand_round.to_bytes(8, "big")
        combined = hashlib.sha256(material).hexdigest()
        return combined
    return clean_hash
```

Also add `from grail.constants import BLOCK_TIME_SECONDS` to the imports.

- [ ] **Step 4: Update validator service.py to use deterministic round**

In `grail/validator/service.py`, replace `_process_window` lines 92-101 with:

```python
    async def _process_window(self, subtensor, target_window: int):
        """Process a window: discover miners, verify each as files appear."""
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            from grail.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                target_window, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)
```

Update the import at top of file: add `from grail.constants import BLOCK_TIME_SECONDS` (already has chain import).

- [ ] **Step 5: Update miner engine.py to use deterministic round**

In `grail/miner/engine.py`, replace `mine_window` lines 63-71 with:

```python
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            from grail.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)
```

- [ ] **Step 6: Run all tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_beacon_sync.py tests/unit/test_security.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add grail/infrastructure/chain.py grail/validator/service.py grail/miner/engine.py tests/unit/test_beacon_sync.py
git commit -m "security: deterministic drand round for miner/validator sync

Derive the drand round from window_start_block * BLOCK_TIME_SECONDS
instead of fetching 'latest'. Include drand round number in
window_randomness to prevent round-shopping attacks."
```

---

### Task 3: Add rollout count and token length limits

**Files:**
- Modify: `grail/constants.py` (add MAX_ROLLOUTS_PER_FILE, MAX_TOKENS_PER_ROLLOUT)
- Modify: `grail/validator/service.py:203-214` (_verify_miner)
- Create: `tests/unit/test_rollout_limits.py`

**Problem:** A miner can submit a 350MB file with thousands of rollouts or rollouts with 8192-token sequences, causing excessive GPU time during verification. No limit on rollout count or per-rollout token length.

- [ ] **Step 1: Write tests for rollout limits**

Create `tests/unit/test_rollout_limits.py`:

```python
"""Tests for rollout count and token length limits."""

from grail.constants import MAX_ROLLOUTS_PER_FILE, MAX_TOKENS_PER_ROLLOUT


class TestRolloutLimitsConstants:
    def test_max_rollouts_per_file_defined(self):
        assert isinstance(MAX_ROLLOUTS_PER_FILE, int)
        assert MAX_ROLLOUTS_PER_FILE > 0

    def test_max_tokens_per_rollout_defined(self):
        assert isinstance(MAX_TOKENS_PER_ROLLOUT, int)
        assert MAX_TOKENS_PER_ROLLOUT > 0


class TestRolloutFiltering:
    def test_excess_rollouts_truncated(self):
        """If a miner submits more than MAX_ROLLOUTS_PER_FILE, excess are dropped."""
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": i, "commit": {"tokens": list(range(10))}}
            for i in range(MAX_ROLLOUTS_PER_FILE + 500)
        ]
        filtered = _filter_rollouts(rollouts)
        assert len(filtered) <= MAX_ROLLOUTS_PER_FILE

    def test_oversized_tokens_rejected(self):
        """Rollouts with tokens exceeding MAX_TOKENS_PER_ROLLOUT are dropped."""
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": 0, "commit": {"tokens": list(range(MAX_TOKENS_PER_ROLLOUT + 1))}},
            {"dataset_index": 1, "commit": {"tokens": list(range(100))}},
        ]
        filtered = _filter_rollouts(rollouts)
        assert len(filtered) == 1
        assert filtered[0]["dataset_index"] == 1

    def test_valid_rollouts_pass_through(self):
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": i, "commit": {"tokens": list(range(100))}}
            for i in range(10)
        ]
        assert len(_filter_rollouts(rollouts)) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_rollout_limits.py -v`
Expected: ImportError (constants and function don't exist)

- [ ] **Step 3: Add constants**

In `grail/constants.py`, after `MAX_ROLLOUT_FILE_SIZE_BYTES` (line 97), add:

```python
# Maximum number of rollouts per submission file.
MAX_ROLLOUTS_PER_FILE = 6000

# Maximum token sequence length in a single rollout.
MAX_TOKENS_PER_ROLLOUT = MAX_NEW_TOKENS_PROTOCOL_CAP + 4096  # prompt + completion
```

- [ ] **Step 4: Add _filter_rollouts function and wire it into _verify_miner**

In `grail/validator/service.py`, add after the imports:

```python
from grail.constants import (
    ...,
    MAX_ROLLOUTS_PER_FILE,
    MAX_TOKENS_PER_ROLLOUT,
)
```

Add the function before the class:

```python
def _filter_rollouts(rollouts: list[dict]) -> list[dict]:
    """Drop rollouts that exceed protocol limits on count or token length."""
    filtered = []
    for rollout in rollouts:
        tokens = rollout.get("commit", {}).get("tokens", [])
        if len(tokens) > MAX_TOKENS_PER_ROLLOUT:
            continue
        filtered.append(rollout)
        if len(filtered) >= MAX_ROLLOUTS_PER_FILE:
            break
    return filtered
```

In `_verify_miner`, at the very top of the method (before dedup), add:

```python
        rollouts = _filter_rollouts(rollouts)
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_rollout_limits.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add grail/constants.py grail/validator/service.py tests/unit/test_rollout_limits.py
git commit -m "security: add rollout count and token length limits

Reject rollouts with token sequences exceeding MAX_TOKENS_PER_ROLLOUT
and truncate submissions beyond MAX_ROLLOUTS_PER_FILE. Prevents GPU
exhaustion from oversized miner submissions."
```

---

### Task 4: Enforce failure lookback — ban recently-failed miners

**Files:**
- Modify: `grail/validator/service.py` (ValidationService)
- Create: `tests/unit/test_failure_lookback.py`

**Problem:** `FAILURE_LOOKBACK_WINDOWS = 14` is declared in constants but never used. A gated miner can retry immediately next window with zero penalty.

- [ ] **Step 1: Write test for failure lookback**

Create `tests/unit/test_failure_lookback.py`:

```python
"""Tests for miner failure lookback — recently gated miners are excluded."""

from grail.constants import FAILURE_LOOKBACK_WINDOWS


class TestFailureLookback:
    def test_recently_gated_miner_excluded(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        svc._record_gating("hk_bad", window=100)
        assert svc._is_miner_excluded("hk_bad", current_window=100)
        assert svc._is_miner_excluded("hk_bad", current_window=100 + FAILURE_LOOKBACK_WINDOWS * 30 - 30)

    def test_old_gating_expires(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        svc._record_gating("hk_old", window=100)
        # After FAILURE_LOOKBACK_WINDOWS windows, the exclusion expires
        far_future = 100 + FAILURE_LOOKBACK_WINDOWS * 30 + 30
        assert not svc._is_miner_excluded("hk_old", current_window=far_future)

    def test_clean_miner_not_excluded(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        assert not svc._is_miner_excluded("hk_clean", current_window=200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_failure_lookback.py -v`
Expected: AttributeError (_record_gating not defined)

- [ ] **Step 3: Add gating history tracking to ValidationService**

In `grail/validator/service.py`, add to `__init__` after `self._validator_hotkey`:

```python
        # {hotkey: last_gated_window} — tracks when miners were last gated
        self._gated_history: dict[str, int] = {}
```

Add these methods to `ValidationService`:

```python
    def _record_gating(self, hotkey: str, window: int) -> None:
        """Record that a miner was gated in this window."""
        self._gated_history[hotkey] = window

    def _is_miner_excluded(self, hotkey: str, current_window: int) -> bool:
        """Check if a miner is excluded due to recent gating."""
        last_gated = self._gated_history.get(hotkey)
        if last_gated is None:
            return False
        lookback_blocks = FAILURE_LOOKBACK_WINDOWS * WINDOW_LENGTH
        return (current_window - last_gated) < lookback_blocks
```

Add `FAILURE_LOOKBACK_WINDOWS` to the imports from `grail.constants`.

In `_verify_miner`, after `if gated:` (line 290), add the recording:

```python
        if gated:
            self._record_gating(hotkey, self._last_processed_window + WINDOW_LENGTH)
            return set(), total_unique
```

In `_process_window`, after the `selected = rng.sample(...)` line, add filtering:

```python
        # Exclude miners who were recently gated
        selected = [
            hk for hk in selected
            if not self._is_miner_excluded(hk, target_window)
        ]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_failure_lookback.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add grail/validator/service.py tests/unit/test_failure_lookback.py
git commit -m "security: enforce failure lookback — ban recently-gated miners

Track when miners get gated and exclude them for FAILURE_LOOKBACK_WINDOWS
subsequent windows. Previously the constant was defined but never used."
```

---

### Task 5: Bound used_indices with window-based expiry

**Files:**
- Modify: `grail/constants.py` (add USED_INDICES_MAX_AGE_WINDOWS)
- Modify: `grail/validator/service.py` (_verify_miner, _process_window)
- Create: `tests/unit/test_index_expiry.py`

**Problem:** `_used_indices` grows monotonically without bound, eventually exhausting memory as hundreds of thousands of dataset indices accumulate.

- [ ] **Step 1: Write test for index expiry**

Create `tests/unit/test_index_expiry.py`:

```python
"""Tests for used_indices expiry — old entries are purged."""

from grail.constants import WINDOW_LENGTH


class TestIndexExpiry:
    def test_old_indices_purged(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}

        # Simulate indices from old window
        svc._used_indices[10] = "hk_old"
        svc._index_windows[10] = 100

        # Simulate indices from recent window
        svc._used_indices[20] = "hk_new"
        svc._index_windows[20] = 1000

        # Purge: current window is far in the future
        svc._purge_old_indices(current_window=2000)

        assert 10 not in svc._used_indices  # old → purged
        assert 20 in svc._used_indices  # recent → kept

    def test_recent_indices_survive_purge(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}

        svc._used_indices[1] = "hk"
        svc._index_windows[1] = 950

        svc._purge_old_indices(current_window=1000)
        assert 1 in svc._used_indices
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_index_expiry.py -v`
Expected: AttributeError

- [ ] **Step 3: Add USED_INDICES_MAX_AGE_WINDOWS constant**

In `grail/constants.py`, after `WEIGHT_SUBMISSION_INTERVAL` (line 123), add:

```python
# How many windows of index history to retain before purging.
# Indices older than this are freed for reuse.
USED_INDICES_MAX_AGE_WINDOWS = 100
```

- [ ] **Step 4: Implement index expiry in ValidationService**

In `grail/validator/service.py`, add `USED_INDICES_MAX_AGE_WINDOWS` to the constants import.

Add to `__init__`:

```python
        # {dataset_index: window_start} — tracks when each index was credited
        self._index_windows: dict[int, int] = {}
```

Add method:

```python
    def _purge_old_indices(self, current_window: int) -> None:
        """Remove used indices older than USED_INDICES_MAX_AGE_WINDOWS."""
        cutoff = current_window - USED_INDICES_MAX_AGE_WINDOWS * WINDOW_LENGTH
        to_remove = [
            idx for idx, w in self._index_windows.items() if w < cutoff
        ]
        for idx in to_remove:
            self._used_indices.pop(idx, None)
            self._index_windows.pop(idx, None)
        if to_remove:
            logger.info("Purged %d expired indices (cutoff window=%d)", len(to_remove), cutoff)
```

In `_verify_miner`, where indices are credited (the `for rollout in fresh_rollouts` loop at line 295), also record the window:

Change:
```python
            self._used_indices[idx] = hotkey
```
To:
```python
            self._used_indices[idx] = hotkey
            self._index_windows[idx] = self._last_processed_window + WINDOW_LENGTH
```

In `_process_window`, before persisting state (before `await storage.save_used_indices`), add:

```python
        self._purge_old_indices(target_window)
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_index_expiry.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add grail/constants.py grail/validator/service.py tests/unit/test_index_expiry.py
git commit -m "security: bound used_indices with window-based expiry

Purge dataset indices older than USED_INDICES_MAX_AGE_WINDOWS windows
to prevent unbounded memory growth."
```

---

### Task 6: Proof version consistency in signature verification

**Files:**
- Modify: `grail/protocol/signatures.py:97` (verify_commit_signature)
- Modify: `tests/unit/test_security.py` (add test)

**Problem:** `verify_commit_signature` accepts `v4` and `v5`, but `verify_proof_version` only accepts `v5`. Inconsistent: a v4 commit passes signature check but fails later. Should only accept the current protocol version.

- [ ] **Step 1: Write test**

Add to `tests/unit/test_security.py`:

```python
class TestProofVersionConsistency:
    """verify_commit_signature must only accept the current GRAIL_PROOF_VERSION."""

    def test_rejects_v4(self):
        from grail.protocol.signatures import verify_commit_signature
        commit = {
            "tokens": [1, 2, 3],
            "commitments": [{"sketch": 0}],
            "proof_version": "v4",
            "signature": "aa" * 64,
            "beacon": {"randomness": "bb" * 32},
            "model": {"name": "test", "layer_index": -1},
        }
        assert verify_commit_signature(commit, "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY") is False

    def test_rejects_unknown_version(self):
        from grail.protocol.signatures import verify_commit_signature
        commit = {
            "tokens": [1, 2, 3],
            "commitments": [{"sketch": 0}],
            "proof_version": "v99",
            "signature": "aa" * 64,
            "beacon": {"randomness": "bb" * 32},
            "model": {"name": "test", "layer_index": -1},
        }
        assert verify_commit_signature(commit, "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY") is False
```

- [ ] **Step 2: Run test to verify v4 case fails (currently v4 is accepted)**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_security.py::TestProofVersionConsistency::test_rejects_v4 -v`
Expected: The test might pass or fail depending on whether bittensor actually verifies — but the logic allows v4 through. Run to check.

- [ ] **Step 3: Fix signatures.py to only accept current version**

In `grail/protocol/signatures.py`, change line 97:

```python
        if not proof_version or proof_version not in ("v4", "v5"):
```

To:

```python
        if not proof_version or proof_version != GRAIL_PROOF_VERSION:
```

Add `from grail.constants import GRAIL_PROOF_VERSION` to the imports (inside the try block or at module level).

- [ ] **Step 4: Run tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_security.py::TestProofVersionConsistency tests/unit/test_signatures.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add grail/protocol/signatures.py tests/unit/test_security.py
git commit -m "security: proof version consistency — only accept current protocol version

verify_commit_signature now uses GRAIL_PROOF_VERSION instead of
hardcoded ('v4', 'v5'). Eliminates inconsistency with verify_proof_version."
```

---

### Task 7: Credit only verified rollouts (proportional extrapolation)

**Files:**
- Modify: `grail/validator/service.py:292-305` (_verify_miner)
- Create: `tests/unit/test_proportional_credit.py`

**Problem:** When 10% of rollouts are sampled and pass, 100% are credited. A miner with 20% forged rollouts has a good chance of passing the sample and being credited for everything.

- [ ] **Step 1: Write test for proportional credit**

Create `tests/unit/test_proportional_credit.py`:

```python
"""Tests for proportional credit — only credit based on verified pass rate."""

import random
from unittest.mock import MagicMock, patch


class TestProportionalCredit:
    def test_partial_pass_credits_proportionally(self):
        """If 80% of sampled rollouts pass, ~80% of fresh indices are credited."""
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}
        svc._last_processed_window = 0
        svc._gated_history = {}
        svc.model = MagicMock()
        svc.tokenizer = MagicMock()
        svc.dataset = MagicMock()

        # Create 100 fresh rollouts
        rollouts = [
            {"dataset_index": i, "nonce": i, "commit": {"tokens": list(range(10))}}
            for i in range(100)
        ]

        rng = random.Random(42)

        # Mock verify_rollout: 80% pass
        call_count = [0]
        def mock_verify(rollout, hotkey, model, tokenizer, randomness, seen_nonces, dataset=None):
            call_count[0] += 1
            if call_count[0] % 5 == 0:  # 20% fail
                return False, "proof_failed"
            return True, "ok"

        with patch("grail.validator.service.verify_rollout", side_effect=mock_verify):
            import asyncio
            new_indices, total = asyncio.get_event_loop().run_until_complete(
                svc._verify_miner("hk_test", rollouts, "aabb", rng)
            )

        # Should NOT credit all 100 — should be proportional to pass rate
        assert len(new_indices) < 100
        assert len(new_indices) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_proportional_credit.py -v`
Expected: AssertionError (currently credits all 100)

- [ ] **Step 3: Implement proportional credit**

In `grail/validator/service.py`, replace the post-gating credit block (lines 293-305):

```python
        if gated:
            self._record_gating(hotkey, self._last_processed_window + WINDOW_LENGTH)
            return set(), total_unique

        # Credit proportional to verified pass rate
        if verified_total == 0:
            return set(), total_unique

        pass_rate = verified_valid / verified_total

        # Deterministically select which fresh rollouts get credit
        # based on the pass rate. Shuffle with the same rng for reproducibility.
        credit_count = int(len(fresh_rollouts) * pass_rate)
        credit_indices_order = list(range(len(fresh_rollouts)))
        rng.shuffle(credit_indices_order)
        credit_indices_order = credit_indices_order[:credit_count]

        new_indices = set()
        current_window = self._last_processed_window + WINDOW_LENGTH
        for i in credit_indices_order:
            idx = fresh_rollouts[i]["dataset_index"]
            new_indices.add(idx)
            self._used_indices[idx] = hotkey
            self._index_windows[idx] = current_window

        logger.info(
            "Miner %s: %d/%d verified passed (%.0f%%) — crediting %d/%d fresh indices (%d total used)",
            hotkey[:8], verified_valid, verified_total,
            100 * pass_rate, len(new_indices), len(fresh_rollouts),
            len(self._used_indices),
        )
        return new_indices, total_unique
```

Also remove the old `self._used_indices[idx] = hotkey` and `self._index_windows[idx] = ...` lines from the old credit block since they're replaced above.

- [ ] **Step 4: Run tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_proportional_credit.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add grail/validator/service.py tests/unit/test_proportional_credit.py
git commit -m "security: proportional credit based on verified pass rate

Instead of crediting 100% of fresh indices when sampled rollouts pass,
credit proportionally to the actual pass rate. A miner with 80% pass
rate gets ~80% of their indices credited."
```

---

### Task 8: Token sequence length check in verifier

**Files:**
- Modify: `grail/validator/verifier.py:66-95` (verify_commitment_proofs)
- Add test to `tests/unit/test_security.py`

**Problem:** `verify_commitment_proofs` has no upper bound check on token sequence length before running a forward pass. A miner could submit a rollout with an extremely long token sequence that is technically under the file size limit but causes OOM on the validator's GPU.

- [ ] **Step 1: Write test**

Add to `tests/unit/test_security.py`:

```python
class TestTokenSequenceLengthCheck:
    """Rollouts with excessively long token sequences must be rejected
    before the forward pass to prevent GPU OOM."""

    def test_rejects_oversized_sequence(self):
        from grail.validator.verifier import verify_commitment_proofs
        from grail.constants import MAX_TOKENS_PER_ROLLOUT

        tokens = list(range(MAX_TOKENS_PER_ROLLOUT + 1))
        commit = {"tokens": tokens, "commitments": [{"sketch": 0}] * len(tokens)}
        result, passed, checked = verify_commitment_proofs(
            commit, _make_mock_model(), "aabb"
        )
        assert result is False
        assert checked == 0

    def test_accepts_valid_length(self):
        from grail.validator.verifier import verify_commitment_proofs

        seq_len = 64
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}
        # Will fail on proof verification but should not short-circuit on length
        with patch("grail.shared.forward.forward_single_layer",
                   return_value=(torch.randn(1, seq_len, HIDDEN_DIM), None)), \
             patch("grail.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM):
            result, passed, checked = verify_commitment_proofs(
                commit, _make_mock_model(), "aabb"
            )
            assert checked > 0  # Got past the length check
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_security.py::TestTokenSequenceLengthCheck -v`
Expected: FAIL (no length check exists)

- [ ] **Step 3: Add length check to verify_commitment_proofs**

In `grail/validator/verifier.py`, add `MAX_TOKENS_PER_ROLLOUT` to the constants import:

```python
from grail.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    MAX_TOKENS_PER_ROLLOUT,
)
```

In `verify_commitment_proofs`, after the commitments count check (line 95), add:

```python
    # SECURITY: Reject sequences that would cause GPU OOM.
    if seq_len > MAX_TOKENS_PER_ROLLOUT:
        logger.warning(
            "Token sequence too long: %d tokens (max %d)",
            seq_len, MAX_TOKENS_PER_ROLLOUT,
        )
        return False, 0, 0
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add grail/validator/verifier.py tests/unit/test_security.py
git commit -m "security: reject oversized token sequences before GPU forward pass

Add MAX_TOKENS_PER_ROLLOUT check in verify_commitment_proofs to prevent
GPU OOM from adversarially long sequences."
```
