# Free Prompt Selection & Index-Based Copycat Detection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Miners freely choose dataset indices instead of being assigned fixed prompts, validators verify prompts match the dataset, and copycat detection works by comparing submitted indices instead of completion digests.

**Architecture:** Remove the deterministic seed-based prompt assignment. Miners pick any index from the 553M-row `karpathy/climbmix-400b-shuffle` dataset, generate one rollout per index, and include the `dataset_index` in each rollout. Validators load the dataset, verify `tokens[:prompt_length]` matches the text at that index, then deduplicate: if two miners submit the same index, only the first uploader (by S3 timestamp) gets credit. Score = count of unique valid indices.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace `datasets` (Arrow memory-map), aiobotocore (S3/R2), bittensor

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `grail/constants.py` | Modify | Remove `NUM_PROMPTS_PER_WINDOW`, `ROLLOUTS_PER_PROBLEM`, `COPYCAT_WINDOW_THRESHOLD`, `COPYCAT_INTERVAL_THRESHOLD`. Update `DATASET_NAME`. |
| `grail/dataset/loader.py` | Rewrite | Remove seed-based functions. Keep `load_dataset_cached()`. Add `get_prompt_by_index()`. |
| `grail/miner/engine.py` | Rewrite | Miner chooses random indices, generates 1 rollout per index, includes `dataset_index` in output. |
| `grail/validator/copycat.py` | Rewrite | Index-based dedup using S3 timestamps instead of digest overlap. |
| `grail/validator/verifier.py` | Modify | Add `verify_prompt()` hard check. |
| `grail/validator/service.py` | Modify | Wire prompt verification, new copycat, S3 timestamps, scoring by unique indices. |
| `grail/infrastructure/storage.py` | Modify | Return S3 `LastModified` timestamp from download. |
| `grail/shared/digest.py` | Delete | No longer needed. |
| `tests/unit/test_dataset.py` | Rewrite | Test new `get_prompt_by_index`. |
| `tests/unit/test_copycat.py` | Rewrite | Test index-based dedup. |
| `tests/unit/test_prompt_verification.py` | Create | Test new `verify_prompt`. |
| `tests/unit/test_digest.py` | Delete | Module deleted. |

---

### Task 1: Clean up constants

**Files:**
- Modify: `grail/constants.py`

- [ ] **Step 1: Remove unused constants and update dataset name**

```python
# In grail/constants.py, DELETE these lines (106-118):

COPYCAT_WINDOW_THRESHOLD = 0.05   # 5% overlap → flagged at window scope
COPYCAT_INTERVAL_THRESHOLD = 0.03  # 3% overlap → flagged at interval scope

# ...

DATASET_NAME = "nvidia/ClimbMix-400B"
DATASET_SPLIT = "train"
NUM_PROMPTS_PER_WINDOW = 4

# REPLACE the DATASET section (lines 114-118) with:

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"

# DELETE lines 72 and 106-108:
ROLLOUTS_PER_PROBLEM = 16
COPYCAT_WINDOW_THRESHOLD = 0.05
COPYCAT_INTERVAL_THRESHOLD = 0.03
```

The full DATASET section should now be:
```python
# ────────────────  DATASET  ────────────────

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"
```

And the ROLLOUT GENERATION section should now be:
```python
# ────────────────  ROLLOUT GENERATION  ────────────────

# Network-wide protocol cap on completion length.
MAX_NEW_TOKENS_PROTOCOL_CAP = 8192
```

And the COPYCAT DETECTION section should be deleted entirely.

- [ ] **Step 2: Verify nothing imports the removed constants**

Run: `cd /home/ubuntu/DR2 && grep -rn "NUM_PROMPTS_PER_WINDOW\|ROLLOUTS_PER_PROBLEM\|COPYCAT_WINDOW_THRESHOLD\|COPYCAT_INTERVAL_THRESHOLD" grail/ tests/ --include="*.py"`

Expected: Only hits in files we plan to modify (`miner/engine.py`, `validator/copycat.py`, `dataset/loader.py`). Those will be updated in later tasks. No unexpected consumers.

- [ ] **Step 3: Commit**

```bash
git add grail/constants.py
git commit -m "refactor: remove fixed prompt count, rollout count, and copycat threshold constants"
```

---

### Task 2: Rewrite dataset loader

**Files:**
- Rewrite: `grail/dataset/loader.py`
- Rewrite: `tests/unit/test_dataset.py`

- [ ] **Step 1: Write failing tests for the new loader**

```python
# tests/unit/test_dataset.py
from unittest.mock import MagicMock

from grail.dataset.loader import get_prompt_by_index


class TestGetPromptByIndex:
    def _make_dataset(self, texts: list[str]) -> MagicMock:
        """Create a mock dataset that supports indexing."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(texts))
        ds.__getitem__ = MagicMock(side_effect=lambda i: {"text": texts[i]})
        return ds

    def test_returns_text_at_index(self):
        ds = self._make_dataset(["hello world", "foo bar", "baz qux"])
        assert get_prompt_by_index(ds, 0) == "hello world"
        assert get_prompt_by_index(ds, 2) == "baz qux"

    def test_returns_none_for_out_of_range(self):
        ds = self._make_dataset(["only one"])
        assert get_prompt_by_index(ds, 5) is None
        assert get_prompt_by_index(ds, -1) is None

    def test_returns_none_for_missing_text_field(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=1)
        ds.__getitem__ = MagicMock(return_value={"other_field": "value"})
        assert get_prompt_by_index(ds, 0) is None

    def test_returns_none_for_empty_text(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=1)
        ds.__getitem__ = MagicMock(return_value={"text": ""})
        assert get_prompt_by_index(ds, 0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_dataset.py -v`
Expected: FAIL — `get_prompt_by_index` does not exist yet.

- [ ] **Step 3: Rewrite the loader**

```python
# grail/dataset/loader.py
"""Dataset loader for karpathy/climbmix-400b-shuffle."""

import logging
from typing import Any

from grail.constants import DATASET_NAME, DATASET_SPLIT

logger = logging.getLogger(__name__)


def load_dataset_cached():
    """Load dataset with HuggingFace Arrow memory-mapping (cached after first load)."""
    from datasets import load_dataset

    logger.info("Loading dataset %s (split=%s)...", DATASET_NAME, DATASET_SPLIT)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d examples", len(ds))
    return ds


def get_prompt_by_index(dataset: Any, index: int) -> str | None:
    """Return the text at the given dataset index, or None if invalid.

    Args:
        dataset: HuggingFace dataset (or mock with __len__ and __getitem__).
        index: Row index into the dataset.

    Returns:
        The text string, or None if index is out of range or text is empty/missing.
    """
    if index < 0 or index >= len(dataset):
        return None
    try:
        row = dataset[index]
        text = row.get("text")
        if not text:
            return None
        return text
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_dataset.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add grail/dataset/loader.py tests/unit/test_dataset.py
git commit -m "refactor: rewrite dataset loader — free index selection replaces seed-based prompts"
```

---

### Task 3: Add prompt verification to the verifier

**Files:**
- Modify: `grail/validator/verifier.py`
- Create: `tests/unit/test_prompt_verification.py`

- [ ] **Step 1: Write failing tests for verify_prompt**

```python
# tests/unit/test_prompt_verification.py
from unittest.mock import MagicMock

from grail.validator.verifier import verify_prompt


class TestVerifyPrompt:
    def _make_dataset(self, texts: list[str]) -> MagicMock:
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(texts))
        ds.__getitem__ = MagicMock(side_effect=lambda i: {"text": texts[i]})
        return ds

    def _make_tokenizer(self, mapping: dict[str, list[int]]) -> MagicMock:
        """Tokenizer that returns preset token lists for known texts."""
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda text, add_special_tokens=False: mapping.get(text, []))
        return tok

    def test_valid_prompt(self):
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [10, 20, 30, 40, 50, 60],
                "rollout": {"prompt_length": 3},
            },
        }
        assert verify_prompt(rollout, ds, tok) is True

    def test_wrong_tokens(self):
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [99, 99, 99, 40, 50],
                "rollout": {"prompt_length": 3},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_wrong_prompt_length(self):
        """Prompt length doesn't match actual tokenization of the text."""
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [10, 20, 40, 50],  # only 2 prompt tokens but text needs 3
                "rollout": {"prompt_length": 2},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_invalid_index(self):
        ds = self._make_dataset(["only one"])
        tok = self._make_tokenizer({})
        rollout = {
            "dataset_index": 999,
            "commit": {
                "tokens": [1, 2, 3],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_missing_dataset_index(self):
        ds = self._make_dataset(["hello"])
        tok = self._make_tokenizer({})
        rollout = {
            "commit": {
                "tokens": [1, 2],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_negative_index(self):
        ds = self._make_dataset(["hello"])
        tok = self._make_tokenizer({})
        rollout = {
            "dataset_index": -1,
            "commit": {
                "tokens": [1, 2],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_prompt_verification.py -v`
Expected: FAIL — `verify_prompt` does not exist yet.

- [ ] **Step 3: Implement verify_prompt**

Add this function to `grail/validator/verifier.py` (after the existing imports, before `verify_signature`):

```python
from grail.dataset.loader import get_prompt_by_index


def verify_prompt(rollout: dict, dataset: Any, tokenizer: Any) -> bool:
    """Hard check: rollout tokens start with the correct prompt from the dataset.

    The miner declares a dataset_index. We look up that index, tokenize the
    text, and compare against the first prompt_length tokens of the rollout.
    """
    dataset_index = rollout.get("dataset_index")
    if dataset_index is None or dataset_index < 0:
        return False

    text = get_prompt_by_index(dataset, dataset_index)
    if text is None:
        return False

    commit = rollout.get("commit", {})
    tokens = commit.get("tokens", [])
    prompt_length = commit.get("rollout", {}).get("prompt_length", 0)

    expected_tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(expected_tokens) != prompt_length:
        return False
    if len(tokens) < prompt_length:
        return False
    if tokens[:prompt_length] != expected_tokens:
        return False

    return True
```

Also add `from typing import Any` to the imports if not already present (it is already present at line 6).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_prompt_verification.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Wire verify_prompt into verify_rollout**

Modify `verify_rollout` in `grail/validator/verifier.py` to accept `dataset` and call `verify_prompt` as the first check:

```python
def verify_rollout(
    rollout: dict,
    hotkey: str,
    model: Any,
    tokenizer: Any,
    window_randomness: str,
    seen_nonces: set[int],
    dataset: Any = None,
) -> tuple[bool, str]:
    """Run all hard checks on a rollout."""
    # Prompt check (requires dataset)
    if dataset is not None:
        if not verify_prompt(rollout, dataset, tokenizer):
            return False, "invalid_prompt"

    commit = rollout.get("commit", {})

    if not verify_signature(commit, hotkey):
        return False, "invalid_signature"

    if not verify_proof_version(commit):
        return False, "invalid_proof_version"

    nonce = rollout.get("nonce", -1)
    if not verify_nonce_unique(nonce, seen_nonces):
        return False, "duplicate_nonce"

    all_passed, passed, checked = verify_commitment_proofs(
        commit, model, window_randomness
    )
    if not all_passed:
        return False, f"proof_failed ({passed}/{checked})"

    return True, "ok"
```

- [ ] **Step 6: Commit**

```bash
git add grail/validator/verifier.py tests/unit/test_prompt_verification.py
git commit -m "feat: add prompt verification — validator checks rollout tokens match dataset index"
```

---

### Task 4: Rewrite copycat detection — index-based dedup with S3 timestamps

**Files:**
- Rewrite: `grail/validator/copycat.py`
- Rewrite: `tests/unit/test_copycat.py`

- [ ] **Step 1: Write failing tests for index-based dedup**

```python
# tests/unit/test_copycat.py


class TestDetectIndexCopycats:
    def test_no_overlap(self):
        """Different indices → no one flagged."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 3}, "upload_time": 100.0},
            "miner_b": {"indices": {4, 5, 6}, "upload_time": 101.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_overlap_later_uploader_loses(self):
        """Two miners share index 5. miner_b uploaded later → loses index 5."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 5}, "upload_time": 100.0},
            "miner_b": {"indices": {3, 4, 5}, "upload_time": 200.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {"miner_b": {5}}

    def test_overlap_earlier_uploader_keeps(self):
        """miner_b uploaded first → miner_a loses the shared index."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 5}, "upload_time": 300.0},
            "miner_b": {"indices": {5, 6}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {"miner_a": {5}}

    def test_three_miners_same_index(self):
        """Three miners share index 10. Only the earliest keeps it."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {10}, "upload_time": 300.0},
            "miner_b": {"indices": {10}, "upload_time": 100.0},
            "miner_c": {"indices": {10}, "upload_time": 200.0},
        }
        rejected = detect_index_copycats(submissions)
        assert 10 in rejected.get("miner_a", set())
        assert 10 in rejected.get("miner_c", set())
        assert "miner_b" not in rejected or 10 not in rejected["miner_b"]

    def test_equal_timestamps_both_keep(self):
        """If timestamps are equal, neither is rejected (benefit of doubt)."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1}, "upload_time": 100.0},
            "miner_b": {"indices": {1}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_none_timestamp_not_rejected(self):
        """If upload_time is None, don't reject (can't determine direction)."""
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1}, "upload_time": None},
            "miner_b": {"indices": {1}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_empty_submissions(self):
        from grail.validator.copycat import detect_index_copycats

        assert detect_index_copycats({}) == {}

    def test_single_miner(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 3}, "upload_time": 100.0},
        }
        assert detect_index_copycats(submissions) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_copycat.py -v`
Expected: FAIL — `detect_index_copycats` does not exist.

- [ ] **Step 3: Implement index-based copycat detection**

```python
# grail/validator/copycat.py
"""Copycat detection via dataset index deduplication.

When two miners submit rollouts for the same dataset index, only the
miner who uploaded first (by S3 LastModified timestamp) keeps credit.
The later uploader's overlapping indices are rejected.

With 553M dataset rows and ~hundreds of miners each picking random
indices, collisions are extremely rare and almost certainly copying.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def detect_index_copycats(
    submissions: dict[str, dict],
) -> dict[str, set[int]]:
    """Detect and reject duplicate dataset indices across miners.

    For each index submitted by multiple miners, the earliest uploader
    (lowest upload_time) keeps it. All later uploaders have that index
    rejected. If timestamps are equal or either is None, no one is
    rejected for that index (benefit of the doubt).

    Args:
        submissions: {hotkey: {"indices": set[int], "upload_time": float | None}}

    Returns:
        {hotkey: set of rejected indices} — only hotkeys with rejections appear.
    """
    if len(submissions) < 2:
        return {}

    # Build reverse map: index → list of (hotkey, upload_time)
    index_to_miners: defaultdict[int, list[tuple[str, float | None]]] = defaultdict(list)
    for hotkey, sub in submissions.items():
        upload_time = sub.get("upload_time")
        for idx in sub.get("indices", set()):
            index_to_miners[idx].append((hotkey, upload_time))

    rejected: defaultdict[str, set[int]] = defaultdict(set)

    for idx, miners in index_to_miners.items():
        if len(miners) < 2:
            continue

        # Check if all timestamps are available and not equal
        times = [t for _, t in miners if t is not None]
        if len(times) < len(miners):
            # At least one miner has no timestamp — can't determine direction
            logger.warning(
                "Index %d shared by %d miners but timestamp unavailable, skipping",
                idx, len(miners),
            )
            continue

        unique_times = set(times)
        if len(unique_times) == 1:
            # All timestamps equal — can't determine who was first
            continue

        # Find the earliest uploader
        earliest_hotkey = min(miners, key=lambda m: m[1] if m[1] is not None else float("inf"))[0]

        # Reject the index for everyone except the earliest
        for hotkey, _ in miners:
            if hotkey != earliest_hotkey:
                rejected[hotkey].add(idx)
                logger.warning(
                    "Copycat index %d: %s rejected (earlier uploader: %s)",
                    idx, hotkey, earliest_hotkey,
                )

    return dict(rejected)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_copycat.py -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add grail/validator/copycat.py tests/unit/test_copycat.py
git commit -m "feat: rewrite copycat detection — index-based dedup with S3 timestamp priority"
```

---

### Task 5: Add S3 upload timestamp to storage download

**Files:**
- Modify: `grail/infrastructure/storage.py`

- [ ] **Step 1: Modify download_window_rollouts to return upload timestamp**

Change the function signature and implementation in `grail/infrastructure/storage.py`:

```python
async def download_window_rollouts(
    hotkey: str, window_start: int, **client_kwargs
) -> tuple[list[dict] | None, float | None]:
    """Download window rollouts and return S3 LastModified timestamp.

    Returns:
        Tuple of (rollouts, upload_time_unix). upload_time is from S3
        LastModified header. Both are None if file not found.
    """
    key = f"grail/windows/{hotkey}-window-{window_start}.json.gz"
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            data = json.loads(body)

            upload_time = None
            last_modified = resp.get("LastModified")
            if last_modified is not None:
                upload_time = last_modified.timestamp()

            if isinstance(data, list):
                return data, upload_time
            return None, None
    except Exception as e:
        logger.debug("download_window_rollouts failed for %s: %s", key, e)
        return None, None
```

- [ ] **Step 2: Commit**

```bash
git add grail/infrastructure/storage.py
git commit -m "feat: return S3 LastModified timestamp from download_window_rollouts"
```

---

### Task 6: Rewrite miner engine — free index selection

**Files:**
- Rewrite: `grail/miner/engine.py`

- [ ] **Step 1: Rewrite MiningEngine.mine_window**

```python
# grail/miner/engine.py
"""Miner engine — vLLM generation + HuggingFace proof construction."""

import logging
import random
from typing import Any

import torch

from grail.constants import (
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
)
from grail.infrastructure import chain, storage
from grail.infrastructure.drand import get_beacon
from grail.protocol.crypto import create_proof
from grail.protocol.grail_verifier import GRAILVerifier
from grail.protocol.signatures import sign_commit_binding
from grail.shared.forward import forward_single_layer
from grail.shared.hf_compat import resolve_hidden_size

logger = logging.getLogger(__name__)


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        dataset,
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        rollouts_per_window: int = 64,
    ):
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.dataset = dataset
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.rollouts_per_window = rollouts_per_window
        self._dataset_size = len(dataset)
        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    async def mine_window(
        self,
        subtensor,
        window_start: int,
        use_drand: bool = True,
    ) -> list[dict]:
        """Generate rollouts for a window and upload."""
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            beacon = get_beacon(use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)

        # Pick random unique indices from the dataset
        indices = random.sample(
            range(self._dataset_size), self.rollouts_per_window
        )

        logger.info(
            "Mining window %d with %d indices", window_start, len(indices)
        )

        all_rollouts = []

        for nonce, dataset_index in enumerate(indices):
            try:
                row = self.dataset[dataset_index]
                prompt_text = row.get("text", "")
                if not prompt_text:
                    continue

                rollout = self._generate_and_prove(
                    prompt_text, randomness, window_start, block_hash,
                    nonce, dataset_index,
                )
                all_rollouts.append(rollout)
            except Exception as e:
                logger.error("Rollout generation failed for index %d: %s", dataset_index, e)

        if all_rollouts:
            hotkey = self.wallet.hotkey.ss58_address
            await storage.upload_window_rollouts(
                hotkey, window_start, all_rollouts
            )
            logger.info(
                "Uploaded %d rollouts for window %d",
                len(all_rollouts), window_start,
            )

        return all_rollouts

    def _generate_and_prove(
        self,
        prompt: str,
        randomness: str,
        window_start: int,
        block_hash: str,
        nonce: int,
        dataset_index: int,
    ) -> dict:
        """Generate text with vLLM, construct proof with HF."""
        # Step 1: Generate with vLLM (GPU 0)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.vllm_model.generate(
                input_ids.to(self.vllm_model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
        all_tokens = outputs[0].tolist()

        # Step 2: HF forward pass for proof (GPU 1)
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Step 3: Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # Step 4: Compute logprobs from HF (not vLLM — bit-identical with validator)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        token_logprobs = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Step 5: Create proof and sign
        model_name = getattr(self.hf_model, "name_or_path", "unknown")

        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        # Step 6: Package rollout
        commit = {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": "v5",
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }

        return {
            "window_start": window_start,
            "dataset_index": dataset_index,
            "nonce": nonce,
            "block_hash": block_hash,
            "hotkey": self.wallet.hotkey.ss58_address,
            "commit": commit,
        }
```

Key changes from the old version:
- No more `NUM_PROMPTS_PER_WINDOW` or `ROLLOUTS_PER_PROBLEM` imports
- `rollouts_per_window` is configurable (default 64) — miner chooses how many to do
- Each rollout is 1 unique index → 1 completion (not 16 rollouts per prompt)
- `dataset_index` is included in the rollout output
- `_generate_and_prove` takes `dataset_index` and includes it in the packaged rollout

- [ ] **Step 2: Commit**

```bash
git add grail/miner/engine.py
git commit -m "feat: rewrite miner — free index selection, 1 rollout per index, includes dataset_index"
```

---

### Task 7: Rewrite validator service — wire everything together

**Files:**
- Rewrite: `grail/validator/service.py`

- [ ] **Step 1: Rewrite _process_window**

```python
# grail/validator/service.py
"""Validator main loop — window processing, verification, weight submission."""

import asyncio
import hashlib
import logging
import random
import time
from collections import defaultdict

from grail.constants import (
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from grail.infrastructure import chain, storage
from grail.infrastructure.drand import get_beacon
from grail.validator.copycat import detect_index_copycats
from grail.validator.verifier import verify_rollout
from grail.validator.weights import compute_weights

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH  # 12


class ValidationService:
    """Main validator service."""

    def __init__(self, wallet, model, tokenizer, dataset, netuid: int, use_drand: bool = True):
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.netuid = netuid
        self.use_drand = use_drand

        self._last_processed_window: int = -1
        self._miner_metrics: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"valid": 0, "unique": 0, "checked": 0}
        )
        self._windows_in_interval: int = 0

    async def run(self, subtensor):
        """Main validation loop."""
        logger.info(
            "Starting validation service (netuid=%d, use_drand=%s)",
            self.netuid, self.use_drand,
        )

        while True:
            try:
                current_block = await chain.get_current_block(subtensor)
                target_window = self._compute_target_window(current_block)

                if target_window <= self._last_processed_window:
                    await asyncio.sleep(6)
                    continue

                logger.info(
                    "Processing window %d (block=%d)", target_window, current_block
                )
                await self._process_window(subtensor, target_window)
                self._last_processed_window = target_window
                self._windows_in_interval += 1

                if self._windows_in_interval >= ROLLING_WINDOWS:
                    await self._submit_weights(subtensor)
                    self._miner_metrics.clear()
                    self._windows_in_interval = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Validation loop error: %s", e, exc_info=True)
                await asyncio.sleep(12)

    async def _process_window(self, subtensor, target_window: int):
        """Process a single window: fetch, verify, deduplicate, score."""
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            beacon = get_beacon(use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        active_hotkeys = list(meta.hotkeys)

        # Sample miners deterministically
        sample_size = max(
            MINER_SAMPLE_MIN,
            min(
                int(len(active_hotkeys) * MINER_SAMPLE_RATE),
                MINER_SAMPLE_MAX,
                len(active_hotkeys),
            ),
        )
        seed = int(hashlib.sha256(block_hash.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        selected = rng.sample(
            active_hotkeys, min(sample_size, len(active_hotkeys))
        )

        logger.info(
            "Selected %d/%d miners for validation",
            len(selected), len(active_hotkeys),
        )

        # Phase 1: Download and verify each miner's rollouts
        # Collect per-miner valid indices and upload times for copycat detection
        miner_valid_rollouts: dict[str, list[dict]] = {}
        miner_valid_indices: dict[str, set[int]] = {}
        miner_upload_times: dict[str, float | None] = {}

        for hotkey in selected:
            rollouts, upload_time = await storage.download_window_rollouts(
                hotkey, target_window
            )
            if rollouts is None:
                logger.debug("No rollouts found for miner %s", hotkey)
                continue

            miner_upload_times[hotkey] = upload_time

            seen_nonces: set[int] = set()
            seen_indices: set[int] = set()
            valid_rollouts = []

            for rollout in rollouts:
                # Deduplicate indices within a single miner
                dataset_index = rollout.get("dataset_index")
                if dataset_index is not None and dataset_index in seen_indices:
                    continue

                is_valid, reason = verify_rollout(
                    rollout, hotkey, self.model, self.tokenizer,
                    randomness, seen_nonces, dataset=self.dataset,
                )
                if is_valid:
                    valid_rollouts.append(rollout)
                    if dataset_index is not None:
                        seen_indices.add(dataset_index)
                else:
                    logger.debug(
                        "Rollout failed for %s: %s", hotkey[:16], reason
                    )

            miner_valid_rollouts[hotkey] = valid_rollouts
            miner_valid_indices[hotkey] = seen_indices

            logger.info(
                "Miner %s: %d/%d valid, %d unique indices",
                hotkey[:8], len(valid_rollouts), len(rollouts), len(seen_indices),
            )

        # Phase 2: Cross-miner index dedup (copycat detection)
        copycat_submissions = {
            hotkey: {
                "indices": miner_valid_indices.get(hotkey, set()),
                "upload_time": miner_upload_times.get(hotkey),
            }
            for hotkey in miner_valid_indices
        }
        rejected_indices = detect_index_copycats(copycat_submissions)

        # Phase 3: Score — unique valid indices minus rejected ones
        for hotkey in miner_valid_indices:
            valid_indices = miner_valid_indices[hotkey]
            rejected = rejected_indices.get(hotkey, set())
            final_unique = len(valid_indices - rejected)
            final_valid = len(miner_valid_rollouts.get(hotkey, [])) - len(rejected)

            if rejected:
                logger.warning(
                    "Miner %s: %d indices rejected by copycat detection",
                    hotkey[:8], len(rejected),
                )

            self._miner_metrics[hotkey]["valid"] += max(0, final_valid)
            self._miner_metrics[hotkey]["unique"] += final_unique
            self._miner_metrics[hotkey]["checked"] += len(
                miner_valid_rollouts.get(hotkey, [])
            )

    async def _submit_weights(self, subtensor):
        """Compute and submit weights on-chain."""
        scores = dict(self._miner_metrics)
        weights = compute_weights(scores)

        non_zero = {hk: w for hk, w in weights.items() if w > 0}
        logger.info("Submitting weights for %d miners", len(non_zero))
        for hk, w in sorted(non_zero.items(), key=lambda x: -x[1])[:10]:
            logger.info("  %s: %.6f", hk[:8], w)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))

        uids = []
        weight_vals = []
        for hk, w in weights.items():
            if hk in hotkey_to_uid and w > 0:
                uids.append(int(hotkey_to_uid[hk]))
                weight_vals.append(w)

        if uids:
            await chain.set_weights(
                subtensor, self.wallet, self.netuid, uids, weight_vals
            )

    def _compute_target_window(self, current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH
```

Key changes:
- `__init__` takes `dataset` parameter
- `_process_window` now has 3 phases: verify, copycat dedup, score
- Uses `detect_index_copycats` instead of `detect_copycats`
- Uses `download_window_rollouts` new return signature (rollouts, upload_time)
- Passes `dataset=self.dataset` to `verify_rollout`
- Deduplicates indices within a miner (only first rollout per index counts)
- Removes `compute_completion_digest` import and usage

- [ ] **Step 2: Commit**

```bash
git add grail/validator/service.py
git commit -m "feat: rewrite validator service — prompt verification, index dedup, S3 timestamp copycat"
```

---

### Task 8: Update CLI to pass dataset to validator

**Files:**
- Modify: `grail/cli/main.py`

- [ ] **Step 1: Read current CLI code**

Read `grail/cli/main.py` to find the `validate` command.

- [ ] **Step 2: Add dataset loading to the validate command**

In the `validate` command, after loading the model and tokenizer, add:

```python
from grail.dataset.loader import load_dataset_cached
dataset = load_dataset_cached()
```

And pass `dataset` to `ValidationService`:

```python
service = ValidationService(wallet, model, tokenizer, dataset, netuid, use_drand=use_drand)
```

- [ ] **Step 3: Commit**

```bash
git add grail/cli/main.py
git commit -m "feat: load dataset in validator CLI and pass to ValidationService"
```

---

### Task 9: Delete dead code

**Files:**
- Delete: `grail/shared/digest.py`
- Delete: `tests/unit/test_digest.py`

- [ ] **Step 1: Verify digest.py is no longer imported**

Run: `cd /home/ubuntu/DR2 && grep -rn "from grail.shared.digest\|import digest" grail/ tests/ --include="*.py"`

Expected: No hits (the old import in `validator/service.py` was removed in Task 7).

- [ ] **Step 2: Delete the files**

```bash
rm grail/shared/digest.py tests/unit/test_digest.py
```

- [ ] **Step 3: Run full test suite**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/ -v`

Expected: All tests pass. No imports of deleted modules.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead code — completion digest no longer needed"
```

---

### Task 10: Final integration verification

- [ ] **Step 1: Run full test suite**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 2: Verify no broken imports**

Run: `cd /home/ubuntu/DR2 && python -c "from grail.miner.engine import MiningEngine; from grail.validator.service import ValidationService; from grail.validator.verifier import verify_rollout, verify_prompt; from grail.validator.copycat import detect_index_copycats; from grail.dataset.loader import load_dataset_cached, get_prompt_by_index; print('All imports OK')"`

Expected: `All imports OK`

- [ ] **Step 3: Verify no references to removed constants**

Run: `cd /home/ubuntu/DR2 && grep -rn "NUM_PROMPTS_PER_WINDOW\|ROLLOUTS_PER_PROBLEM\|COPYCAT_WINDOW_THRESHOLD\|COPYCAT_INTERVAL_THRESHOLD\|compute_completion_digest\|detect_copycats" grail/ tests/ --include="*.py"`

Expected: No hits.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify clean build — all imports, tests, and references validated"
```
