# Batch-Driven Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v2.0's time-based window loop with a batch-driven state machine. Window seals when 8 valid distinct-prompt non-cooldown submissions land → train_step (stub) → publish new checkpoint → next window.

**Architecture:** Single validator. Validator owns the state machine `OPEN→TRAINING→PUBLISHING→READY`. `GrpoWindowBatcher` exposes a `seal_event: asyncio.Event` set on B-th valid submission. New `CheckpointStore` handles save/upload/sign + manifest. `train_step()` is stubbed (just bumps counter, reuses weights) — real training plugs in later.

**Tech Stack:** Python 3.11, Pydantic v2, asyncio, FastAPI, aiobotocore (R2), torch (model load/save), substrate-interface (signing).

**Spec reference:** `docs/superpowers/specs/2026-04-21-batch-driven-windows-design.md` (commit `52516e9`).

---

## File structure

**Modified:**
- `reliquary/constants.py` — add `WINDOW_TIMEOUT_SECONDS`, `CHECKPOINT_STATE_PATH_DEFAULT`
- `reliquary/protocol/submission.py` — add `WindowState` enum, extend `GrpoBatchState`, add `checkpoint_hash` to `BatchSubmissionRequest`, add `RejectReason.WRONG_CHECKPOINT`
- `reliquary/validator/batcher.py` — add `seal_event`, `current_checkpoint_hash`, gate on hash mismatch
- `reliquary/validator/service.py` — replace time-based loop with state machine; `window_n` counter + persistence + rebuild
- `reliquary/validator/server.py` — `/window/state` returns extended state; `/submit` gates on state; new `/checkpoint`
- `reliquary/miner/engine.py` — poll-loop on state.state; download/load checkpoint when `checkpoint_n` increments
- `reliquary/miner/submitter.py` — include `checkpoint_hash` in payload; new `get_checkpoint_v2` helper

**Created:**
- `reliquary/validator/checkpoint.py` — `CheckpointStore` class (save / hash / sign / upload to R2 / manifest)
- `reliquary/validator/training.py` — `train_step(batch, model) -> new_model_path` stub
- `reliquary/validator/state_persistence.py` — load/save `window_n` + `checkpoint_n` to local JSON

**Tests created:**
- `tests/unit/test_window_state_enum.py`
- `tests/unit/test_checkpoint_store.py`
- `tests/unit/test_training_stub.py`
- `tests/unit/test_seal_event.py`
- `tests/unit/test_checkpoint_hash_gating.py`
- `tests/unit/test_state_machine.py`
- `tests/unit/test_state_persistence.py`
- `tests/unit/test_miner_checkpoint_pull.py`
- `tests/integration/test_v21_window_loop.py`

---

## Task order rationale

Pure data types and standalone modules first (constants, schemas, CheckpointStore, training stub, persistence) so each can be unit-tested in isolation. Then the orchestrator (batcher seal_event, service state machine). Then I/O (server endpoints, miner loop, submitter). Cleanup + integration last.

---

## Task 1: Constants + WindowState enum

**Files:**
- Modify: `reliquary/constants.py`
- Modify: `reliquary/protocol/submission.py`
- Create: `tests/unit/test_window_state_enum.py`

- [ ] **Step 1: Append to `reliquary/constants.py`**

```python
# ────────────────  v2.1 BATCH-DRIVEN WINDOWS  ────────────────

# Safety-net timeout: a window auto-seals after this many seconds even
# if fewer than B valid submissions have landed. The unused slots burn.
# Set generously — this is a backstop, not the cadence.
WINDOW_TIMEOUT_SECONDS = 600

# Local JSON path for validator state (window_n counter + checkpoint_n).
# Resolved relative to the CWD if not absolute.
CHECKPOINT_STATE_PATH_DEFAULT = "reliquary/state/checkpoint.json"

# Local directory for staged checkpoint files before R2 upload.
CHECKPOINT_STAGING_DIR_DEFAULT = "reliquary/state/checkpoints"
```

- [ ] **Step 2: Add `WindowState` enum to `reliquary/protocol/submission.py`**

After the existing `RejectReason` definition, add:

```python
class WindowState(str, Enum):
    """Current phase of a batch-driven window."""

    OPEN = "open"             # accepting /submit
    TRAINING = "training"     # GRPO step running, no submissions
    PUBLISHING = "publishing" # uploading weights, no submissions
    READY = "ready"           # checkpoint published; transient — back to OPEN once next window opens
```

Also add `WRONG_CHECKPOINT = "wrong_checkpoint"` to the `RejectReason` enum.

- [ ] **Step 3: Create test file**

`tests/unit/test_window_state_enum.py`:

```python
"""WindowState enum + RejectReason.WRONG_CHECKPOINT availability."""

from reliquary.protocol.submission import RejectReason, WindowState


def test_window_state_values():
    assert WindowState.OPEN.value == "open"
    assert WindowState.TRAINING.value == "training"
    assert WindowState.PUBLISHING.value == "publishing"
    assert WindowState.READY.value == "ready"


def test_window_state_set_membership():
    submitting_states = {WindowState.OPEN}
    not_submitting = {WindowState.TRAINING, WindowState.PUBLISHING, WindowState.READY}
    assert submitting_states.isdisjoint(not_submitting)
    assert len(submitting_states | not_submitting) == 4


def test_wrong_checkpoint_reject_reason_present():
    assert RejectReason.WRONG_CHECKPOINT.value == "wrong_checkpoint"


def test_window_state_serialises_to_string():
    state = WindowState.OPEN
    # str enum serialises naturally as its value
    assert str(state.value) == "open"
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_window_state_enum.py -v
```

Expected: 4 PASS.

```
git add reliquary/constants.py reliquary/protocol/submission.py tests/unit/test_window_state_enum.py
git commit -m "feat(protocol): WindowState enum + WRONG_CHECKPOINT reject reason + v2.1 constants"
```

---

## Task 2: Extend `GrpoBatchState` + `BatchSubmissionRequest`

**Files:**
- Modify: `reliquary/protocol/submission.py`
- Create: `tests/unit/test_v21_schemas.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_v21_schemas.py`:

```python
"""v2.1 schema extensions: state in GrpoBatchState, checkpoint_hash in BatchSubmissionRequest."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    WindowState,
)


def _rollouts(k=4):
    return [
        RolloutSubmission(
            tokens=[1, 2, 3], reward=1.0 if i < k else 0.0,
            commit={"tokens": [1, 2, 3], "proof_version": "v5"},
        )
        for i in range(M_ROLLOUTS)
    ]


def test_grpo_batch_state_has_window_state_fields():
    s = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=42,
        anchor_block=12345,
        current_round=999,
        cooldown_prompts=[],
        valid_submissions=0,
        checkpoint_n=7,
        checkpoint_url="https://r2.example/cp/7.safetensors",
        checkpoint_hash="ed25519:abcdef",
    )
    assert s.state == WindowState.OPEN
    assert s.window_n == 42
    assert s.checkpoint_n == 7


def test_grpo_batch_state_checkpoint_url_optional_pre_first_publish():
    s = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=0,
        anchor_block=0,
        current_round=0,
        cooldown_prompts=[],
        valid_submissions=0,
        checkpoint_n=0,
        checkpoint_url=None,
        checkpoint_hash=None,
    )
    assert s.checkpoint_url is None


def test_batch_submission_requires_checkpoint_hash():
    with pytest.raises(ValidationError, match="checkpoint_hash"):
        BatchSubmissionRequest(
            miner_hotkey="hk", prompt_idx=0, window_start=0,
            signed_round=0, merkle_root="00" * 32, rollouts=_rollouts(),
            # missing checkpoint_hash
        )


def test_batch_submission_with_checkpoint_hash_parses():
    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=0, window_start=0,
        signed_round=0, merkle_root="00" * 32, rollouts=_rollouts(),
        checkpoint_hash="ed25519:abc",
    )
    assert req.checkpoint_hash == "ed25519:abc"
```

- [ ] **Step 2: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_v21_schemas.py -v
```

- [ ] **Step 3: Modify `reliquary/protocol/submission.py`**

Replace the existing `GrpoBatchState` class with:

```python
class GrpoBatchState(BaseModel):
    """Live window state for miners polling ``/window/{n}/state`` (v2.1)."""

    model_config = ConfigDict(extra="forbid")

    state: WindowState
    window_n: int = Field(..., ge=0)
    anchor_block: int = Field(..., ge=0)
    current_round: int = Field(..., ge=0)
    cooldown_prompts: list[int] = Field(default_factory=list)
    valid_submissions: int = Field(..., ge=0)
    checkpoint_n: int = Field(..., ge=0)
    checkpoint_url: str | None = None
    checkpoint_hash: str | None = None
```

(Note: `window_start` was the v2.0 name; v2.1 renames it to `window_n`. Service code that constructed `GrpoBatchState(window_start=...)` needs updating in Task 6 — for now keep the schema test green.)

Add `checkpoint_hash: str = Field(..., min_length=1)` to `BatchSubmissionRequest`.

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_v21_schemas.py -v
```

Expected: 4 PASS.

Run also the existing `test_batch_submission_schema.py` — those tests will break because they construct without `checkpoint_hash` and they expect old `GrpoBatchState`. This is expected. Update them in Task 6 along with the service rewrite. Do not fix here.

```
git add reliquary/protocol/submission.py tests/unit/test_v21_schemas.py
git commit -m "feat(protocol): extend GrpoBatchState + require checkpoint_hash on submission"
```

---

## Task 3: `CheckpointStore` class

**Files:**
- Create: `reliquary/validator/checkpoint.py`
- Create: `tests/unit/test_checkpoint_store.py`

**Context:** Encapsulates the produce → hash → sign → upload → manifest lifecycle. Pure orchestration; the actual model-saving primitive is injected so tests don't need torch.

- [ ] **Step 1: Failing test**

`tests/unit/test_checkpoint_store.py`:

```python
"""CheckpointStore: produce → hash → sign → upload to R2 → in-memory manifest."""

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.validator.checkpoint import CheckpointStore, ManifestEntry


class FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        def sign(self, data): return b"signature_for_" + data[:16]
    hotkey = _Hk()


def _save_weights_stub(model, path):
    """Pretend to save model weights — write deterministic bytes."""
    path.write_bytes(b"weights_for_" + str(id(model)).encode())


def test_initial_manifest_is_none():
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path="/tmp/test_cp",
    )
    assert store.current_manifest() is None


@pytest.mark.asyncio
async def test_publish_writes_uploads_signs_and_serves(tmp_path):
    fake_upload = AsyncMock(return_value="https://r2.example/cp/1.safetensors")
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    model = MagicMock(name="mock_model")
    entry = await store.publish(checkpoint_n=1, model=model)
    assert isinstance(entry, ManifestEntry)
    assert entry.checkpoint_n == 1
    assert entry.file_url == "https://r2.example/cp/1.safetensors"
    assert entry.file_hash.startswith("sha256:")
    assert entry.signature.startswith("ed25519:")
    fake_upload.assert_awaited_once()
    # current_manifest now returns this entry
    assert store.current_manifest() is entry


@pytest.mark.asyncio
async def test_publish_increments_overrides_previous(tmp_path):
    fake_upload = AsyncMock(side_effect=[
        "https://r2.example/cp/1.safetensors",
        "https://r2.example/cp/2.safetensors",
    ])
    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=_save_weights_stub,
    )
    await store.publish(checkpoint_n=1, model=MagicMock())
    e2 = await store.publish(checkpoint_n=2, model=MagicMock())
    assert store.current_manifest() is e2
    assert store.current_manifest().checkpoint_n == 2


@pytest.mark.asyncio
async def test_file_hash_deterministic_for_same_bytes(tmp_path):
    """Same model bytes → same file_hash."""
    captured_hashes = []
    
    async def fake_upload(local_path, key):
        captured_hashes.append(local_path)
        return f"https://r2.example/{key}"

    def deterministic_save(model, path):
        path.write_bytes(b"identical_bytes")

    store = CheckpointStore(
        validator_hotkey="5FHk",
        wallet=FakeWallet(),
        staging_dir_path=str(tmp_path),
        upload_fn=fake_upload,
        save_weights_fn=deterministic_save,
    )
    e1 = await store.publish(1, model=object())
    e2 = await store.publish(2, model=object())
    assert e1.file_hash == e2.file_hash  # same bytes → same hash
    assert e1.checkpoint_n != e2.checkpoint_n
```

- [ ] **Step 2: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_checkpoint_store.py -v
```

- [ ] **Step 3: Create `reliquary/validator/checkpoint.py`**

```python
"""CheckpointStore: produce → hash → sign → upload → manifest entry.

Single-validator (v2.1) implementation. The validator owns the
checkpoint lifecycle for the whole netuid; multi-validator consensus
on checkpoint hash is a v2.2 concern.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """A published checkpoint entry."""

    checkpoint_n: int
    file_url: str
    file_hash: str       # "sha256:<hex>"
    signature: str       # "ed25519:<hex>" — wallet signs (n || file_hash)


class _WalletLike(Protocol):
    """Minimal wallet shape — tests inject a stub, prod injects bittensor wallet."""

    class hotkey:
        ss58_address: str
        @staticmethod
        def sign(data: bytes) -> bytes: ...


class CheckpointStore:
    """Owns the in-memory current manifest + the publish lifecycle.

    Production wiring:
      * ``save_weights_fn`` defaults to ``torch.save(model.state_dict, path)``
      * ``upload_fn`` defaults to ``storage.upload_checkpoint`` (added in
        Task 4 below; uses the existing R2 client)

    Tests inject both as mocks to avoid torch + R2 deps.
    """

    def __init__(
        self,
        validator_hotkey: str,
        wallet: _WalletLike,
        staging_dir_path: str,
        *,
        upload_fn: Callable[[str, str], Awaitable[str]] | None = None,
        save_weights_fn: Callable[[Any, Path], None] | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.wallet = wallet
        self.staging_dir = Path(staging_dir_path)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._upload = upload_fn or _default_upload
        self._save_weights = save_weights_fn or _default_save_weights
        self._current: ManifestEntry | None = None

    def current_manifest(self) -> ManifestEntry | None:
        return self._current

    async def publish(self, checkpoint_n: int, model: Any) -> ManifestEntry:
        """Save → hash → sign → upload → install in manifest."""
        path = self.staging_dir / f"{checkpoint_n}.safetensors"
        self._save_weights(model, path)
        file_hash = self._sha256_file(path)
        sig_payload = f"{checkpoint_n}|{file_hash}".encode()
        sig_bytes = self.wallet.hotkey.sign(sig_payload)
        signature = "ed25519:" + sig_bytes.hex()

        key = f"reliquary/checkpoints/{self.validator_hotkey}/{checkpoint_n}.safetensors"
        url = await self._upload(str(path), key)

        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            file_url=url,
            file_hash=file_hash,
            signature=signature,
        )
        self._current = entry
        logger.info(
            "Published checkpoint %d (hash=%s, url=%s)",
            checkpoint_n, file_hash[:16], url,
        )
        return entry

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return "sha256:" + h.hexdigest()


# ---- production defaults (loaded lazily so tests don't drag torch in) ----

async def _default_upload(local_path: str, key: str) -> str:
    """Default upload using the existing R2 client."""
    from reliquary.infrastructure import storage
    return await storage.upload_checkpoint_file(local_path, key)


def _default_save_weights(model: Any, path: Path) -> None:
    """Default save: torch.save the state_dict."""
    import torch
    torch.save(model.state_dict(), path)
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_checkpoint_store.py -v
```

Expected: 4 PASS.

```
git add reliquary/validator/checkpoint.py tests/unit/test_checkpoint_store.py
git commit -m "feat(checkpoint): CheckpointStore — save/hash/sign/upload/manifest"
```

---

## Task 4: R2 upload helper for checkpoints

**Files:**
- Modify: `reliquary/infrastructure/storage.py` — add `upload_checkpoint_file`
- Modify: `tests/unit/test_list_recent_datasets.py` — append upload test (or create new file)

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_list_recent_datasets.py`:

```python
@pytest.mark.asyncio
async def test_upload_checkpoint_file(tmp_path):
    """upload_checkpoint_file streams a local file to R2 and returns the URL."""
    from unittest.mock import AsyncMock, patch
    from reliquary.infrastructure.storage import upload_checkpoint_file

    src = tmp_path / "checkpoint.safetensors"
    src.write_bytes(b"fake_weights_payload")

    mock_client = AsyncMock()
    mock_client.put_object = AsyncMock(return_value={})
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch(
        "reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx
    ), patch.dict("os.environ", {"R2_BUCKET_ID": "reliquary"}):
        url = await upload_checkpoint_file(str(src), "reliquary/checkpoints/hk/1.safetensors")
    assert "reliquary/checkpoints/hk/1.safetensors" in url
    mock_client.put_object.assert_awaited_once()
```

- [ ] **Step 2: Run, expect FAIL (ImportError)**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_list_recent_datasets.py::test_upload_checkpoint_file -v
```

- [ ] **Step 3: Append to `reliquary/infrastructure/storage.py`**

```python
async def upload_checkpoint_file(local_path: str, key: str, **client_kwargs) -> str:
    """Upload a checkpoint file (typically *.safetensors) to R2.

    Returns the public-style URL the miner will use to download
    (constructed from R2 endpoint + bucket + key). Streams the file in
    chunks so large checkpoints (multi-GB) don't OOM the validator.
    """
    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    endpoint = os.getenv(
        "R2_PUBLIC_URL",
        os.getenv("R2_ENDPOINT_URL", ""),
    )
    async with get_s3_client(**client_kwargs) as client:
        with open(local_path, "rb") as f:
            await client.put_object(Bucket=bucket, Key=key, Body=f.read())
    url = f"{endpoint.rstrip('/')}/{bucket}/{key}" if endpoint else f"r2://{bucket}/{key}"
    logger.info("Uploaded checkpoint to %s", url)
    return url
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_list_recent_datasets.py -v
```

```
git add reliquary/infrastructure/storage.py tests/unit/test_list_recent_datasets.py
git commit -m "feat(storage): upload_checkpoint_file — stream local checkpoint to R2"
```

---

## Task 5: `train_step` stub

**Files:**
- Create: `reliquary/validator/training.py`
- Create: `tests/unit/test_training_stub.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_training_stub.py`:

```python
"""train_step stub — does not modify weights, returns model unchanged."""

from unittest.mock import MagicMock

from reliquary.validator.training import train_step


def test_train_step_returns_same_model():
    model = MagicMock(name="model")
    batch = [MagicMock(name="batch_member") for _ in range(8)]
    result = train_step(model=model, batch=batch)
    assert result is model  # stub: no actual update


def test_train_step_with_empty_batch():
    """Empty batch (no submissions in window) → model unchanged."""
    model = MagicMock()
    result = train_step(model=model, batch=[])
    assert result is model


def test_train_step_logs_batch_size(caplog):
    import logging
    caplog.set_level(logging.INFO)
    train_step(model=MagicMock(), batch=[MagicMock() for _ in range(5)])
    assert any("5" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_training_stub.py -v
```

- [ ] **Step 3: Create `reliquary/validator/training.py`**

```python
"""GRPO training step — STUB for v2.1.

This stub validates the orchestration path (seal → train → publish →
pull). The real GRPO loss + optimizer update plugs in here in a
follow-up PR with no protocol change.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def train_step(model: Any, batch: list) -> Any:
    """Run one GRPO step on *batch*. STUB: returns the model unchanged.

    The real implementation will:
      1. Compute per-rollout advantages within each group
      2. Forward pass + log-prob extraction
      3. Compute clipped PPO/GRPO loss + KL term
      4. Backward + optimizer step
      5. Return the updated model (or same reference, mutated in-place)

    For v2.1 we only need the orchestration to work end-to-end; the
    weights stay frozen so miners and validator continue to share the
    same effective model after each "training step". The checkpoint_n
    counter still bumps so the manifest signals progress.

    Args:
        model: the model object (torch.nn.Module in production).
        batch: list of ValidSubmission entries from the sealed window.

    Returns:
        The same model object (stub). Real impl returns the updated
        model.
    """
    logger.info(
        "train_step (stub) called with batch of %d submissions — "
        "weights not modified",
        len(batch),
    )
    return model
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_training_stub.py -v
```

Expected: 3 PASS.

```
git add reliquary/validator/training.py tests/unit/test_training_stub.py
git commit -m "feat(training): GRPO train_step stub — no-op pending real loss impl"
```

---

## Task 6: State persistence (window_n + checkpoint_n)

**Files:**
- Create: `reliquary/validator/state_persistence.py`
- Create: `tests/unit/test_state_persistence.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_state_persistence.py`:

```python
"""ValidatorState: simple JSON-backed counters for window_n + checkpoint_n."""

from pathlib import Path

import pytest

from reliquary.validator.state_persistence import ValidatorState


def test_default_counters_are_zero(tmp_path: Path):
    s = ValidatorState(path=str(tmp_path / "s.json"))
    assert s.window_n == 0
    assert s.checkpoint_n == 0


def test_save_and_load_roundtrip(tmp_path: Path):
    p = str(tmp_path / "s.json")
    s1 = ValidatorState(path=p)
    s1.window_n = 42
    s1.checkpoint_n = 7
    s1.save()

    s2 = ValidatorState(path=p)
    s2.load()
    assert s2.window_n == 42
    assert s2.checkpoint_n == 7


def test_load_missing_file_keeps_defaults(tmp_path: Path):
    s = ValidatorState(path=str(tmp_path / "missing.json"))
    s.load()
    assert s.window_n == 0
    assert s.checkpoint_n == 0


def test_atomic_save(tmp_path: Path):
    """Save uses tmp + rename so partial writes don't corrupt."""
    import os
    p = tmp_path / "s.json"
    s = ValidatorState(path=str(p))
    s.window_n = 1
    s.save()
    assert p.exists()
    # No leftover .tmp files
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".s.")]
    assert leftovers == []
```

- [ ] **Step 2: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_state_persistence.py -v
```

- [ ] **Step 3: Create `reliquary/validator/state_persistence.py`**

```python
"""ValidatorState: tiny JSON store for window_n + checkpoint_n counters.

Local-first: validator restart loads from disk. Cooldown rebuild already
covers R2 fallback, so this file is purely a hot-path optimisation —
losing it just means starting counters from 0 (the rebuild loop will
then advance them through observed history).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class ValidatorState:
    """Tiny counters store. Mutate fields directly, then call ``save()``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.window_n: int = 0
        self.checkpoint_n: int = 0

    def load(self) -> None:
        """Load from disk if present; otherwise leave defaults."""
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            data = json.load(f)
        self.window_n = int(data.get("window_n", 0))
        self.checkpoint_n = int(data.get("checkpoint_n", 0))

    def save(self) -> None:
        """Atomic write via tmp + rename."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".s.", dir=os.path.dirname(self.path) or "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {"window_n": self.window_n, "checkpoint_n": self.checkpoint_n},
                    f,
                )
            os.replace(tmp_path, self.path)
        except Exception:
            os.unlink(tmp_path)
            raise
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_state_persistence.py -v
```

```
git add reliquary/validator/state_persistence.py tests/unit/test_state_persistence.py
git commit -m "feat(state): ValidatorState — JSON-backed window_n + checkpoint_n counters"
```

---

## Task 7: Batcher — `seal_event` + `checkpoint_hash` gating

**Files:**
- Modify: `reliquary/validator/batcher.py`
- Modify: `tests/unit/test_grpo_window_batcher.py`

- [ ] **Step 1: Append failing tests to `tests/unit/test_grpo_window_batcher.py`**

```python
# --- v2.1 seal_event + checkpoint_hash gating ---

import asyncio
import pytest


def _request_v21(prompt_idx=42, signed_round=1000, window_start=500,
                 rewards=None, hotkey="hk", checkpoint_hash="ed25519:abc"):
    """v2.1 request includes checkpoint_hash."""
    if rewards is None:
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    rollouts = []
    for i, r in enumerate(rewards):
        text = "CORRECT" if r > 0.5 else "wrong"
        rollouts.append(
            RolloutSubmission(
                tokens=[1, 2, 3, 4, 5], reward=r,
                commit={
                    "proof_version": "v5", "tokens": [1, 2, 3, 4, 5],
                    "completion_text_for_test": text,
                },
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=hotkey, prompt_idx=prompt_idx,
        window_start=window_start, signed_round=signed_round,
        merkle_root="00" * 32, rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
    )


def test_reject_wrong_checkpoint():
    """Submission with checkpoint_hash != batcher's current is rejected."""
    b = _make_batcher()
    b.current_checkpoint_hash = "ed25519:current"
    req = _request_v21(checkpoint_hash="ed25519:stale")
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WRONG_CHECKPOINT


def test_accept_matching_checkpoint():
    b = _make_batcher()
    b.current_checkpoint_hash = "ed25519:current"
    req = _request_v21(checkpoint_hash="ed25519:current")
    resp = b.accept_submission(req)
    assert resp.accepted is True


@pytest.mark.asyncio
async def test_seal_event_set_when_b_valid_distinct_landed():
    """seal_event fires the moment the B-th valid distinct-prompt
    non-cooldown submission is accepted."""
    b = _make_batcher(current_round=2000)
    b.current_checkpoint_hash = "ed25519:hash"
    assert not b.seal_event.is_set()
    for i in range(B_BATCH):
        req = _request_v21(
            prompt_idx=i, signed_round=1000 + i, hotkey=f"hk{i}",
            checkpoint_hash="ed25519:hash",
        )
        b.accept_submission(req)
    # Wait briefly for the event to propagate.
    await asyncio.wait_for(b.seal_event.wait(), timeout=0.1)
    assert b.seal_event.is_set()


@pytest.mark.asyncio
async def test_seal_event_not_set_with_only_duplicate_prompts():
    """Two submissions on same prompt → only first counts → seal_event not set."""
    b = _make_batcher(current_round=2000)
    b.current_checkpoint_hash = "ed25519:hash"
    for i in range(2):
        req = _request_v21(
            prompt_idx=42, signed_round=1000 + i, hotkey=f"hk{i}",
            checkpoint_hash="ed25519:hash",
        )
        b.accept_submission(req)
    # Only 1 distinct prompt → not enough for seal
    assert not b.seal_event.is_set()
```

(Update existing v2.0 tests in this file to include `checkpoint_hash="ed25519:..."` in the constructed requests, OR set `b.current_checkpoint_hash = ""` and accept any. Easier: in `_make_batcher`, default `current_checkpoint_hash = ""`; add a special-case in the batcher: empty string disables the gate. This preserves v2.0 tests without rewriting all of them.)

- [ ] **Step 2: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_grpo_window_batcher.py -v
```

- [ ] **Step 3: Modify `reliquary/validator/batcher.py`**

In `GrpoWindowBatcher.__init__`, add:

```python
        self.seal_event: asyncio.Event = asyncio.Event()
        self.current_checkpoint_hash: str = ""  # "" disables gate (test convenience)
```

Add a new method `_count_distinct_eligible_in_pool` and trigger event in `_accept_locked`:

In `_accept_locked`, after successful append to `_valid`:

```python
        # v2.1: trigger seal_event the moment B distinct, non-cooldown
        # prompts have been accepted. Service awaits this to advance the
        # state machine.
        distinct_eligible = len({
            s.prompt_idx for s in self._valid
            if not self._cooldown.is_in_cooldown(s.prompt_idx, self.window_start)
        })
        if distinct_eligible >= B_BATCH and not self.seal_event.is_set():
            self.seal_event.set()
```

In the cheap-checks block (early in `_accept_locked`), add:

```python
        # v2.1: checkpoint hash gate — empty string disables (test convenience).
        if self.current_checkpoint_hash and request.checkpoint_hash != self.current_checkpoint_hash:
            return self._reject(RejectReason.WRONG_CHECKPOINT)
```

Place this BEFORE the cooldown check (cheap, no I/O).

Imports to add at top of batcher.py:

```python
import asyncio
```

(`asyncio.Event` is the only stdlib addition.)

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_grpo_window_batcher.py -v
```

Expected: existing tests pass + 4 new pass.

```
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(batcher): seal_event + checkpoint_hash gate (v2.1)"
```

---

## Task 8: Service — state machine main loop

**Files:**
- Modify: `reliquary/validator/service.py`
- Create: `tests/unit/test_state_machine.py`

**Context:** Largest task. Replaces the time-based loop with the OPEN→TRAINING→PUBLISHING→READY state machine. window_n counter, persistence, training stub call, checkpoint publish, rebuild from R2 at startup.

- [ ] **Step 1: Read current `reliquary/validator/service.py`**

```bash
cat /home/ubuntu/Catalyst/reliquary/validator/service.py
```

Note: `_compute_target_window`, `_run_window`, `_submit_weights`, `run`. The `__init__` already creates `self._cooldown_map` and `self._batched_hotkeys`.

- [ ] **Step 2: Failing test**

`tests/unit/test_state_machine.py`:

```python
"""ValidationService state machine: OPEN → TRAINING → PUBLISHING → READY."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.protocol.submission import WindowState


@dataclass
class _FakeEnv:
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


@pytest.mark.asyncio
async def test_run_one_window_advances_state(tmp_path):
    """A full lap through OPEN→TRAINING→PUBLISHING→READY increments window_n."""
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._state_path = str(tmp_path / "s.json")  # use temp path
    initial_window_n = svc._state.window_n
    initial_checkpoint_n = svc._state.checkpoint_n

    # Patch the publish + train to be no-ops that complete instantly.
    svc._checkpoint_store = MagicMock()
    fake_publish = AsyncMock(return_value=MagicMock(
        checkpoint_n=initial_checkpoint_n + 1,
        file_url="https://r2/x",
        file_hash="sha256:abc",
        signature="ed25519:xyz",
    ))
    svc._checkpoint_store.publish = fake_publish

    # Force seal immediately by setting the batcher's seal_event.
    async def _open_and_seal():
        svc._open_window()
        # Simulate batch arrival — trigger seal_event externally.
        svc._active_batcher.seal_event.set()

    await _open_and_seal()
    await svc._train_and_publish()

    assert svc._state.window_n == initial_window_n + 1
    assert svc._state.checkpoint_n == initial_checkpoint_n + 1
    fake_publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_window_state_transitions_visible_via_get_state(tmp_path):
    """GrpoBatchState.state changes through OPEN/TRAINING/PUBLISHING/READY."""
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._state_path = str(tmp_path / "s.json")

    svc._open_window()
    assert svc._current_window_state == WindowState.OPEN

    svc._set_state(WindowState.TRAINING)
    assert svc._current_window_state == WindowState.TRAINING
    svc._set_state(WindowState.PUBLISHING)
    assert svc._current_window_state == WindowState.PUBLISHING
    svc._set_state(WindowState.READY)
    assert svc._current_window_state == WindowState.READY
```

- [ ] **Step 3: Run, expect FAIL**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_state_machine.py -v
```

- [ ] **Step 4: Rewrite `reliquary/validator/service.py`**

This is a big rewrite. Replace the existing class with the state-machine version. Keep `_serve_axon_on_chain`, `_derive_randomness`, `_submit_weights`, `_compute_target_window`, `_rebuild_cooldown_from_history` — they're reused. Replace `_run_window` and `run` with the state-machine variants. Add `_open_window`, `_seal_window`, `_train_and_publish`, `_set_state`, `_publish_checkpoint`.

Add to the `__init__`:

```python
        from reliquary.constants import (
            CHECKPOINT_STAGING_DIR_DEFAULT,
            CHECKPOINT_STATE_PATH_DEFAULT,
        )
        from reliquary.validator.checkpoint import CheckpointStore
        from reliquary.validator.state_persistence import ValidatorState

        self._state_path = CHECKPOINT_STATE_PATH_DEFAULT
        self._state = ValidatorState(self._state_path)
        self._state.load()
        self._checkpoint_store = CheckpointStore(
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
            staging_dir_path=CHECKPOINT_STAGING_DIR_DEFAULT,
        )
        self._active_batcher = None
        self._current_window_state = WindowState.READY  # start ready
```

Add methods:

```python
    def _set_state(self, s):
        from reliquary.protocol.submission import WindowState
        self._current_window_state = s

    def _open_window(self):
        """Create a new GrpoWindowBatcher and mark state OPEN."""
        from reliquary.validator.service import open_grpo_window
        self._state.window_n += 1
        bootstrap = is_bootstrap_window(
            window_start=self._state.window_n, subnet_start=SUBNET_START_BLOCK,
        )
        self._active_batcher = open_grpo_window(
            window_start=self._state.window_n,
            current_round=self._state.window_n,  # placeholder; drand wiring is followup #4
            env=self.env, model=self.model,
            cooldown_map=self._cooldown_map, tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
        cp = self._checkpoint_store.current_manifest()
        self._active_batcher.current_checkpoint_hash = (
            cp.file_hash if cp else ""
        )
        self.server.set_active_batcher(self._active_batcher)
        self._set_state(WindowState.OPEN)

    async def _train_and_publish(self):
        """TRAINING + PUBLISHING + READY phases."""
        from reliquary.validator.training import train_step
        self._set_state(WindowState.TRAINING)
        batch = self._active_batcher.seal_batch()
        for sub in batch:
            self._batched_hotkeys.append(sub.hotkey)
        self.model = train_step(self.model, batch)
        self._set_state(WindowState.PUBLISHING)
        new_n = self._state.checkpoint_n + 1
        await self._checkpoint_store.publish(checkpoint_n=new_n, model=self.model)
        self._state.checkpoint_n = new_n
        self._state.save()
        # Archive window dataset to R2 (cooldown source-of-truth).
        await self._archive_window(self._active_batcher, batch)
        self.server.set_active_batcher(None)
        self._active_batcher = None
        self._set_state(WindowState.READY)

    async def _archive_window(self, batcher, batch):
        archive = {
            "window_start": batcher.window_start,
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": [
                {"hotkey": s.hotkey, "prompt_idx": s.prompt_idx,
                 "signed_round": s.signed_round, "k": s.k}
                for s in batch
            ],
        }
        try:
            await storage.upload_window_dataset(
                batcher.window_start, archive,
                validator_hotkey=self.wallet.hotkey.ss58_address,
            )
        except Exception:
            logger.exception("Failed to archive window %d", batcher.window_start)
```

Replace `run`:

```python
    async def run(self, subtensor) -> None:
        await self.server.start()
        await self._serve_axon_on_chain(subtensor)
        await self._rebuild_cooldown_from_history(subtensor)
        logger.info(
            "Validator started (v2.1): env=%s, netuid=%d, http=%s:%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
        )
        try:
            while True:
                try:
                    self._open_window()
                    seal_or_timeout = await asyncio.wait_for(
                        self._active_batcher.seal_event.wait(),
                        timeout=WINDOW_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Window %d timed out at %ds — sealing partial",
                        self._state.window_n, WINDOW_TIMEOUT_SECONDS,
                    )
                except asyncio.CancelledError:
                    raise
                try:
                    await self._train_and_publish()
                except Exception:
                    logger.exception("train_and_publish failed; skipping window")
                    self.server.set_active_batcher(None)
                    self._active_batcher = None
                    self._set_state(WindowState.READY)
                # Weight cadence: every ROLLING_WINDOWS sealed windows.
                if self._state.window_n % ROLLING_WINDOWS == 0:
                    submitted = await self._submit_weights(subtensor)
                    if submitted:
                        self._batched_hotkeys.clear()
        finally:
            await self.server.stop()
```

Imports to add at top:

```python
import asyncio
from reliquary.constants import WINDOW_TIMEOUT_SECONDS
from reliquary.protocol.submission import WindowState
```

- [ ] **Step 5: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_state_machine.py tests/unit/test_service_v2.py -v
```

Existing service tests may break — fix what's necessary, note what isn't (some v2.0-shaped service tests may be obsolete). Don't fix unrelated breakage.

```
git add reliquary/validator/service.py tests/unit/test_state_machine.py
git commit -m "feat(service): batch-driven state machine — OPEN/TRAINING/PUBLISHING/READY"
```

---

## Task 9: Server — state-aware endpoints

**Files:**
- Modify: `reliquary/validator/server.py`
- Modify: `tests/unit/test_validator_server.py`

- [ ] **Step 1: Failing test**

Append to `tests/unit/test_validator_server.py`:

```python
def test_submit_rejects_when_state_not_open():
    """When the validator is TRAINING/PUBLISHING/READY, /submit returns
    a non-OPEN reject."""
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "ed25519:abc"
    server.set_active_batcher(batcher)
    server._current_state = WindowState.TRAINING
    client = TestClient(server.app)
    req = _request(prompt_idx=0)  # well-formed
    resp = client.post("/submit", json=req.model_dump(mode="json"))
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False
    assert resp.json()["reason"] == "window_not_active"


def test_state_endpoint_returns_window_state_enum():
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server._current_state = WindowState.OPEN
    client = TestClient(server.app)
    resp = client.get("/window/500/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "open"
    assert body["window_n"] == 500


def test_checkpoint_endpoint_returns_manifest_or_404():
    server = ValidatorServer()
    # No checkpoint yet
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 404

    # After a manifest is set
    from reliquary.validator.checkpoint import ManifestEntry
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=42,
        file_url="https://r2.example/cp/42.safetensors",
        file_hash="sha256:abc",
        signature="ed25519:xyz",
    ))
    resp = client.get("/checkpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_n"] == 42
    assert body["file_hash"] == "sha256:abc"
```

(Also update the existing `_request` helper in this file to include `checkpoint_hash="ed25519:abc"`. Existing v2.0-shaped tests need a `checkpoint_hash` field now.)

- [ ] **Step 2: Modify `reliquary/validator/server.py`**

Add to `ValidatorServer.__init__`:

```python
        from reliquary.protocol.submission import WindowState
        self._current_state: WindowState = WindowState.READY
        self._current_checkpoint = None  # ManifestEntry or None
```

Add public setters:

```python
    def set_current_state(self, state) -> None:
        self._current_state = state

    def set_current_checkpoint(self, entry) -> None:
        self._current_checkpoint = entry
```

In the `/submit` handler, before queuing, check state:

```python
        if self._current_state != WindowState.OPEN:
            return BatchSubmissionResponse(
                accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
            )
```

Update the `/window/state` handler to construct the new shape:

```python
        @app.get("/window/{window_start}/state", response_model=GrpoBatchState)
        async def window_state(window_start: int) -> GrpoBatchState:
            batcher = self.active_batcher
            if batcher is None or batcher.window_start != window_start:
                raise HTTPException(status_code=404, detail="window_not_active")
            cp = self._current_checkpoint
            return GrpoBatchState(
                state=self._current_state,
                window_n=batcher.window_start,
                anchor_block=batcher.window_start,  # placeholder; real anchor via service later
                current_round=batcher.current_round,
                cooldown_prompts=sorted(
                    batcher._cooldown.current_cooldown_set(batcher.window_start)
                ),
                valid_submissions=len(batcher.valid_submissions()),
                checkpoint_n=cp.checkpoint_n if cp else 0,
                checkpoint_url=cp.file_url if cp else None,
                checkpoint_hash=cp.file_hash if cp else None,
            )
```

Add new endpoint:

```python
        @app.get("/checkpoint")
        async def checkpoint():
            cp = self._current_checkpoint
            if cp is None:
                raise HTTPException(status_code=404, detail="no_checkpoint")
            return {
                "checkpoint_n": cp.checkpoint_n,
                "file_url": cp.file_url,
                "file_hash": cp.file_hash,
                "signature": cp.signature,
            }
```

Service must call `server.set_current_state(...)` and `server.set_current_checkpoint(...)` at the right state machine transitions. Add those calls in service.py's `_set_state` and at the end of `_train_and_publish`.

- [ ] **Step 3: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_validator_server.py -v
```

```
git add reliquary/validator/server.py reliquary/validator/service.py tests/unit/test_validator_server.py
git commit -m "feat(server): state-aware /submit + extended /window/state + new /checkpoint"
```

---

## Task 10: Miner — checkpoint pull + state polling

**Files:**
- Modify: `reliquary/miner/engine.py`
- Modify: `reliquary/miner/submitter.py`
- Create: `tests/unit/test_miner_checkpoint_pull.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_miner_checkpoint_pull.py`:

```python
"""Miner detects new checkpoint_n via /window/state and downloads."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.miner.engine import maybe_pull_checkpoint


@pytest.mark.asyncio
async def test_pull_when_remote_n_higher():
    """Remote checkpoint_n > local → download triggered."""
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_url = "https://r2/5.safetensors"
    state.checkpoint_hash = "sha256:abc"

    download_fn = AsyncMock(return_value="/tmp/5.safetensors")
    load_fn = MagicMock(return_value="loaded_model_5")

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=4, local_hash="sha256:old", local_model="old_model",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_hash == "sha256:abc"
    assert new_model == "loaded_model_5"
    download_fn.assert_awaited_once_with("https://r2/5.safetensors")
    load_fn.assert_called_once_with("/tmp/5.safetensors")


@pytest.mark.asyncio
async def test_no_pull_when_local_up_to_date():
    state = MagicMock()
    state.checkpoint_n = 5
    state.checkpoint_url = "https://r2/5.safetensors"
    state.checkpoint_hash = "sha256:abc"

    download_fn = AsyncMock()
    load_fn = MagicMock()

    new_local_n, new_hash, new_model = await maybe_pull_checkpoint(
        state=state, local_n=5, local_hash="sha256:abc", local_model="cached",
        download_fn=download_fn, load_fn=load_fn,
    )
    assert new_local_n == 5
    assert new_model == "cached"
    download_fn.assert_not_called()
```

- [ ] **Step 2: Modify `reliquary/miner/engine.py`**

Add module-level `maybe_pull_checkpoint`:

```python
async def maybe_pull_checkpoint(
    state, local_n: int, local_hash: str, local_model,
    *, download_fn, load_fn,
):
    """If remote checkpoint_n > local, download + load and return new model.

    Returns (new_local_n, new_local_hash, new_model). If no update is
    needed, returns the inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_url is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_url)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_hash, new_model
```

Modify `MiningEngine.mine_window` to:

1. Poll `/window/state` continuously — only submit when `state.state == WindowState.OPEN`
2. Call `maybe_pull_checkpoint` after each state poll
3. Pass the `local_checkpoint_hash` into the `BatchSubmissionRequest`

```python
    async def mine_window(self, subtensor, window_start, use_drand=True):
        import httpx
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2, download_checkpoint,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        # Resolve URL (unchanged)...
        # Initial state
        local_n = 0
        local_hash = ""

        async with httpx.AsyncClient(timeout=30) as client:
            while True:  # caller terminates the loop externally (signal handler)
                try:
                    state = await get_window_state_v2(url, window_start, client=client)
                except SubmissionError:
                    await asyncio.sleep(2)
                    continue

                # Pull new checkpoint if needed (works even when state != OPEN)
                local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                    state=state, local_n=local_n, local_hash=local_hash,
                    local_model=self.hf_model,
                    download_fn=lambda url: download_checkpoint(url, client=client),
                    load_fn=lambda path: self._load_checkpoint(path),
                )

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue

                # Pick a prompt and submit (existing flow, with checkpoint_hash added)
                ...
                request = BatchSubmissionRequest(
                    ...,
                    checkpoint_hash=local_hash,
                )
                await submit_batch_v2(url, request, client=client)
```

(Add `_load_checkpoint` method on MiningEngine that uses torch.load + state_dict; a stub returning `self.hf_model` is fine for the test phase.)

- [ ] **Step 3: Add `download_checkpoint` to `reliquary/miner/submitter.py`**

```python
async def download_checkpoint(url: str, *, client=None) -> str:
    """Download a checkpoint URL to a temp file. Returns local path."""
    import tempfile
    own = client is None
    cli = client or httpx.AsyncClient(timeout=300)
    try:
        resp = await cli.get(url)
        resp.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        return path
    finally:
        if own:
            await cli.aclose()
```

- [ ] **Step 4: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/unit/test_miner_checkpoint_pull.py tests/unit/test_miner_engine_v2.py -v
```

```
git add reliquary/miner/engine.py reliquary/miner/submitter.py tests/unit/test_miner_checkpoint_pull.py
git commit -m "feat(miner): poll-loop on WindowState + checkpoint pull on /state n change"
```

---

## Task 11: Update v2.0 schema tests for v2.1 shape

**Files:**
- Modify: `tests/unit/test_batch_submission_schema.py` (broken since Task 2 added required `checkpoint_hash`)
- Modify: any other test that constructs `BatchSubmissionRequest` or `GrpoBatchState`

- [ ] **Step 1: Find broken tests**

```bash
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/ 2>&1 | grep -E "FAILED|checkpoint_hash" | head -20
```

- [ ] **Step 2: For each test that constructs `BatchSubmissionRequest`, add `checkpoint_hash="ed25519:abc"` (or whatever) field**

Same for `GrpoBatchState` — add the new fields with sensible defaults.

- [ ] **Step 3: Run full suite + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: all green.

```
git add tests/
git commit -m "test: update existing schema tests to include v2.1 fields (checkpoint_hash, state)"
```

---

## Task 12: Integration smoke test

**Files:**
- Create: `tests/integration/test_v21_window_loop.py`

- [ ] **Step 1: Create**

```python
"""End-to-end smoke: open window → 8 valid land → seal → publish → next window."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_one_window_lap_increments_counters(tmp_path):
    """One full state-machine lap bumps window_n + checkpoint_n + leaves
    state == READY for the next iteration."""
    # ... orchestrate ValidationService with fully-mocked subtensor + R2 + model
    # Mark with @pytest.mark.skip if tooling for the full lap is too heavy;
    # at minimum exercise:
    #   - svc._open_window() sets state = OPEN
    #   - inject 8 valid submissions into svc._active_batcher
    #   - await svc._active_batcher.seal_event.wait()
    #   - svc._train_and_publish() bumps counters
    #   - state ends in READY


@pytest.mark.asyncio
async def test_window_timeout_seals_partial(tmp_path):
    """Window with only 3 valid submissions, timeout fires → seal partial."""
    # ... similar orchestration with 3 submissions + asyncio.wait_for timeout
```

(These can start as `pytest.mark.skip` if mocking the full state machine is too heavy — at minimum they should compile and serve as a TODO marker for the next iteration.)

- [ ] **Step 2: Run + commit**

```
/home/ubuntu/Catalyst/.venv/bin/python -m pytest tests/integration/ -v
```

```
git add tests/integration/test_v21_window_loop.py
git commit -m "test(integration): v2.1 window state-machine smoke (skipped placeholders)"
```

---

## Task 13: Docs refresh

**Files:**
- Modify: `README.md`
- Modify: `docs/mining.md`
- Modify: `docs/validating.md`

- [ ] **Step 1: Update each doc**

Replace v2.0 window mechanics descriptions:

- v2.1 windows are batch-driven, not time-driven
- Validator's main loop is OPEN→TRAINING→PUBLISHING→READY
- Miners poll `/window/state` and download checkpoints from `/checkpoint`
- `WINDOW_TIMEOUT_SECONDS = 600` is a safety net, not the cadence
- Real GRPO training loss is stubbed for now (checkpoint counter still advances)

- [ ] **Step 2: Commit**

```
git add README.md docs/mining.md docs/validating.md
git commit -m "docs: update for v2.1 batch-driven window state machine"
```

---

## Self-review notes

**Spec coverage:**
- Q1 timeout → Task 1 (constant) + Task 8 (asyncio.wait_for in run loop) ✓
- Q2 cooldown unit → Task 8 (window_n drives cooldown clock) ✓
- Q3 window_n monotonic counter → Task 6 (ValidatorState) + Task 8 ✓
- Q4a checkpoint signing → Task 3 (CheckpointStore.publish) ✓
- Q4b single validator → assumed throughout ✓
- Q4c miner gating → Task 7 (batcher) + Task 10 (miner sends hash) ✓
- Q5 WindowState exposed → Task 1 (enum) + Task 9 (server) ✓
- Q6 drand round retained → existing field, unchanged ✓

**Type consistency check:**
- `ManifestEntry.file_hash` is `"sha256:<hex>"` string — used as `checkpoint_hash` in submissions and gating
- `GrpoBatchState.checkpoint_hash` and `BatchSubmissionRequest.checkpoint_hash` are both strings, compared by equality
- `WindowState` enum used in both server state-tracking and miner state polling

**No placeholders.** Every step has the actual code. Tests are concrete.

**Out-of-scope not promised:**
- Real GRPO training (Task 5 is explicit stub, with TODO in code comment)
- Drand round wiring (still placeholder per v2.0)
- Multi-validator (single-validator assumed throughout)
