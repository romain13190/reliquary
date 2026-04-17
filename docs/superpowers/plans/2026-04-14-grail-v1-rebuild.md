# GRAIL V1 Rebuild — Verifiable Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Bittensor subnet 81 (GRAIL) from scratch with clean architecture — verifiable inference only, no training, no environments.

**Architecture:** Miners run vLLM (GPU 1) for fast generation + HuggingFace (GPU 2) for bit-identical proof construction via `forward_single_layer`. Validators run HuggingFace (1 GPU) to verify proofs using the same `forward_single_layer`. Prompts are drawn deterministically from ClimbMix-400B using drand + block hash derived seeds. Scoring rewards unique valid proofs with superlinear exponent (no burn, no environment correctness).

**Tech Stack:** Python 3.11+, uv, PyTorch, transformers, bittensor, vLLM, typer, httpx/requests, aiobotocore, pydantic, pytest

---

## File Structure

```
grail/
├── pyproject.toml                          # Project config, dependencies
├── grail/
│   ├── __init__.py                         # Version
│   ├── constants.py                        # All protocol constants
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── crypto.py                       # PRF, indices_from_root, dot_mod_q, create_proof
│   │   ├── signatures.py                   # commit binding, sign/verify
│   │   ├── tokens.py                       # hash_tokens, int_to_bytes, verify_tokens
│   │   └── grail_verifier.py               # GRAILVerifier: sketch commitments + verification
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── hf_compat.py                    # resolve_hidden_size, resolve_vocab_size, resolve_max_context_length
│   │   ├── forward.py                      # forward_single_layer (THE shared function)
│   │   └── digest.py                       # compute_completion_digest for copycat
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── drand.py                        # Drand v2+v1 client with fallback
│   │   ├── chain.py                        # Bittensor: metagraph, set_weights, block hash
│   │   └── storage.py                      # R2/S3 upload/download
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── loader.py                       # ClimbMix-400B + deterministic prompt selection
│   ├── miner/
│   │   ├── __init__.py
│   │   └── engine.py                       # vLLM generate + HF proof + upload
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── service.py                      # Main loop: windows, fetch, validate, weights
│   │   ├── verifier.py                     # Hard checks + soft checks
│   │   ├── copycat.py                      # Digest comparison inter-miners
│   │   └── weights.py                      # Scoring formula + set_weights
│   └── cli/
│       ├── __init__.py
│       └── main.py                         # Typer CLI: grail mine / grail validate
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_crypto.py
    │   ├── test_grail_verifier.py
    │   ├── test_signatures.py
    │   ├── test_tokens.py
    │   ├── test_drand.py
    │   ├── test_weights.py
    │   ├── test_copycat.py
    │   └── test_dataset.py
    └── integration/
        ├── test_miner_validator.py
        └── test_forward_single_layer.py
```

---

### Task 1: Project Scaffold + Constants

**Files:**
- Create: `pyproject.toml`
- Create: `grail/__init__.py`
- Create: `grail/constants.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "grail"
version = "0.1.0"
description = "GRAIL V1 — Verifiable Inference Subnet"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "torch>=2.1.0",
    "numpy>=1.20.0",
    "transformers>=4.40.0",
    "safetensors>=0.3.0",
    "bittensor>=7.0.0",
    "pydantic>=2.3,<3.0.0",
    "requests>=2.28.0",
    "httpx>=0.25.0",
    "huggingface-hub>=0.20.0",
    "datasets>=2.14.0",
    "aiobotocore>=2.0.0",
    "botocore>=1.24.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "tenacity>=8.0.0",
    "pyarrow>=14.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[project.scripts]
grail = "grail.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create grail/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create grail/constants.py**

Copy constants from the grail source (`/home/ubuntu/grail/grail/protocol/constants.py`). Keep only the ones relevant to V1 (remove environment-specific ones like CURRENT_ENV_ID, reward sigmoid params, distribution check params, delta checkpoint, trust list). Add dataset constants.

Key constants to include:
- GRAIL_PROOF_VERSION = "v5"
- PRIME_Q, CHALLENGE_K, RNG_LABEL, LAYER_INDEX
- PROOF_BATCH_SIZE, PROOF_TOPK, PROOF_NUM_BUCKETS, PROOF_COEFF_RANGE
- PROOF_SKETCH_TOLERANCE_BASE, PROOF_SKETCH_TOLERANCE_GROWTH
- ATTN_IMPLEMENTATION
- WINDOW_LENGTH, BLOCK_TIME_SECONDS, BLOCK_TIME_VARIANCE, NETWORK_UPLOAD_LATENCY, UPLOAD_GRACE_PERIOD, DRAND_FUTURE_BUFFER
- ROLLOUTS_PER_PROBLEM, MAX_NEW_TOKENS_PROTOCOL_CAP
- SUPERLINEAR_EXPONENT, UNIQUE_ROLLOUTS_CAP, UNIQUE_ROLLOUTS_CAP_ENABLED
- MINER_SAMPLING_ENABLED, MINER_SAMPLE_RATE (0.25), MINER_SAMPLE_MIN, MINER_SAMPLE_MAX
- FAILURE_LOOKBACK_WINDOWS
- MIN_ROLLOUT_FILE_SIZE_BYTES, MAX_ROLLOUT_FILE_SIZE_BYTES
- CHECKPOINT_PREFIX
- WEIGHT_SUBMISSION_INTERVAL = 360
- NUM_PROMPTS_PER_WINDOW = 4
- DATASET_NAME = "nvidia/ClimbMix-400B"
- DATASET_SPLIT = "train"
- STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.51
- COPYCAT_WINDOW_THRESHOLD = 0.05
- COPYCAT_INTERVAL_THRESHOLD = 0.03

- [ ] **Step 4: Create empty __init__.py files for all packages**

Create `__init__.py` in: `grail/protocol/`, `grail/shared/`, `grail/infrastructure/`, `grail/dataset/`, `grail/miner/`, `grail/validator/`, `grail/cli/`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml grail/
git commit -m "feat: project scaffold with constants"
```

---

### Task 2: Protocol — crypto.py

**Files:**
- Create: `grail/protocol/__init__.py`
- Create: `grail/protocol/crypto.py`
- Create: `tests/unit/test_crypto.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write tests for crypto primitives**

```python
# tests/unit/test_crypto.py
import pytest
from grail.protocol.crypto import prf, r_vec_from_randomness, indices_from_root, dot_mod_q, create_proof
from grail.constants import PRIME_Q, CHALLENGE_K


class TestPrf:
    def test_deterministic(self):
        a = prf(b"test", b"data", out_bytes=32)
        b = prf(b"test", b"data", out_bytes=32)
        assert a == b

    def test_different_labels_differ(self):
        a = prf(b"label1", b"data", out_bytes=32)
        b = prf(b"label2", b"data", out_bytes=32)
        assert a != b

    def test_correct_length(self):
        for n in [0, 1, 16, 32, 64, 128, 256]:
            assert len(prf(b"test", out_bytes=n)) == n

    def test_empty_output(self):
        assert prf(b"test", out_bytes=0) == b""

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            prf(b"test", out_bytes=-1)

    def test_rejects_non_bytes(self):
        with pytest.raises(TypeError):
            prf("not bytes", out_bytes=32)  # type: ignore


class TestRVecFromRandomness:
    def test_shape(self):
        import torch
        vec = r_vec_from_randomness("abcdef1234567890" * 4, 4096)
        assert vec.shape == (4096,)
        assert vec.dtype == torch.int32

    def test_deterministic(self):
        a = r_vec_from_randomness("aabb", 128)
        b = r_vec_from_randomness("aabb", 128)
        import torch
        assert torch.equal(a, b)

    def test_different_randomness_differs(self):
        a = r_vec_from_randomness("aabb", 128)
        b = r_vec_from_randomness("ccdd", 128)
        assert not (a == b).all()

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            r_vec_from_randomness("", 128)

    def test_rejects_invalid_d_model(self):
        with pytest.raises(ValueError):
            r_vec_from_randomness("aabb", 0)


class TestIndicesFromRoot:
    def test_correct_count(self):
        tokens = list(range(100))
        idxs = indices_from_root(tokens, "abcd1234", 100, 10)
        assert len(idxs) == 10

    def test_sorted(self):
        tokens = list(range(200))
        idxs = indices_from_root(tokens, "abcd1234", 200, 32)
        assert idxs == sorted(idxs)

    def test_deterministic(self):
        tokens = list(range(100))
        a = indices_from_root(tokens, "abcd1234", 100, 10)
        b = indices_from_root(tokens, "abcd1234", 100, 10)
        assert a == b

    def test_within_range(self):
        tokens = list(range(50))
        idxs = indices_from_root(tokens, "ffff", 50, 10)
        assert all(0 <= i < 50 for i in idxs)

    def test_rejects_k_gt_seq_len(self):
        with pytest.raises(ValueError):
            indices_from_root([1, 2, 3], "abcd", 3, 5)


class TestDotModQ:
    def test_result_in_range(self):
        import torch
        h = torch.randn(128)
        r = torch.randint(-100, 100, (128,), dtype=torch.int32)
        result = dot_mod_q(h, r)
        assert 0 <= result < PRIME_Q

    def test_deterministic(self):
        import torch
        torch.manual_seed(42)
        h = torch.randn(128)
        r = torch.randint(-100, 100, (128,), dtype=torch.int32)
        a = dot_mod_q(h, r)
        b = dot_mod_q(h, r)
        assert a == b


class TestCreateProof:
    def test_has_indices(self):
        tokens = list(range(100))
        proof = create_proof(tokens, "aabbccdd", 100, k=CHALLENGE_K)
        assert "indices" in proof
        assert len(proof["indices"]) == CHALLENGE_K

    def test_has_beacon(self):
        tokens = list(range(100))
        proof = create_proof(tokens, "aabbccdd", 100)
        assert "round_R1" in proof
        assert proof["round_R1"]["randomness"] == "aabbccdd"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_crypto.py -v`
Expected: ImportError / ModuleNotFoundError

- [ ] **Step 3: Implement crypto.py**

Copy from `/home/ubuntu/grail/grail/protocol/crypto.py` — the file is pure algorithmic logic, keep it exactly. Update import path from `.constants` to `grail.constants`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/test_crypto.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add grail/protocol/crypto.py tests/
git commit -m "feat: protocol crypto primitives (PRF, indices, dot_mod_q)"
```

---

### Task 3: Protocol — tokens.py

**Files:**
- Create: `grail/protocol/tokens.py`
- Create: `tests/unit/test_tokens.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_tokens.py
from grail.protocol.tokens import int_to_bytes, hash_tokens


class TestIntToBytes:
    def test_zero(self):
        assert int_to_bytes(0) == b"\x00\x00\x00\x00"

    def test_one(self):
        assert int_to_bytes(1) == b"\x00\x00\x00\x01"

    def test_big_endian(self):
        assert int_to_bytes(256) == b"\x00\x00\x01\x00"

    def test_large_value(self):
        result = int_to_bytes(0xFFFFFFFF)
        assert result == b"\xff\xff\xff\xff"

    def test_always_4_bytes(self):
        for val in [0, 1, 255, 65535, 2**32 - 1]:
            assert len(int_to_bytes(val)) == 4


class TestHashTokens:
    def test_deterministic(self):
        tokens = [1, 2, 3, 4, 5]
        assert hash_tokens(tokens) == hash_tokens(tokens)

    def test_32_bytes(self):
        assert len(hash_tokens([1, 2, 3])) == 32

    def test_different_tokens_differ(self):
        assert hash_tokens([1, 2, 3]) != hash_tokens([3, 2, 1])

    def test_order_matters(self):
        assert hash_tokens([1, 2]) != hash_tokens([2, 1])
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement tokens.py**

Copy from `/home/ubuntu/grail/grail/protocol/tokens.py`. Keep `int_to_bytes`, `hash_tokens`, `verify_tokens`, `_validate_token_ids`, `_validate_sequence_length`. Update imports to use `grail.shared.hf_compat`.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 4: Protocol — signatures.py

**Files:**
- Create: `grail/protocol/signatures.py`
- Create: `tests/unit/test_signatures.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_signatures.py
import hashlib
from grail.protocol.signatures import hash_commitments, build_commit_binding, derive_env_seed


class TestHashCommitments:
    def test_deterministic(self):
        comms = [{"sketch": 42}, {"sketch": 99}]
        assert hash_commitments(comms) == hash_commitments(comms)

    def test_32_bytes(self):
        assert len(hash_commitments([{"sketch": 1}])) == 32

    def test_order_independent_keys(self):
        """JSON sort_keys=True ensures key order doesn't matter."""
        a = hash_commitments([{"b": 2, "a": 1}])
        b = hash_commitments([{"a": 1, "b": 2}])
        assert a == b


class TestBuildCommitBinding:
    def test_deterministic(self):
        tokens = [1, 2, 3]
        comms = [{"sketch": 10}]
        a = build_commit_binding(tokens, "aabb", "model-v1", -1, comms)
        b = build_commit_binding(tokens, "aabb", "model-v1", -1, comms)
        assert a == b

    def test_32_bytes(self):
        result = build_commit_binding([1], "ff", "m", 0, [{"s": 1}])
        assert len(result) == 32

    def test_different_tokens_differ(self):
        comms = [{"sketch": 1}]
        a = build_commit_binding([1, 2], "aa", "m", -1, comms)
        b = build_commit_binding([3, 4], "aa", "m", -1, comms)
        assert a != b


class TestDeriveEnvSeed:
    def test_deterministic(self):
        a = derive_env_seed("5abc...", "0xdeadbeef", 0)
        b = derive_env_seed("5abc...", "0xdeadbeef", 0)
        assert a == b

    def test_different_index(self):
        a = derive_env_seed("addr", "hash", 0)
        b = derive_env_seed("addr", "hash", 1)
        assert a != b
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement signatures.py**

Copy from `/home/ubuntu/grail/grail/protocol/signatures.py`. Keep all functions. Update import for `hash_tokens`.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 5: Protocol — grail_verifier.py

**Files:**
- Create: `grail/protocol/grail_verifier.py`
- Create: `tests/unit/test_grail_verifier.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_grail_verifier.py
import torch
import pytest
from grail.protocol.grail_verifier import (
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
    adaptive_sketch_tolerance,
    GRAILVerifier,
)
from grail.constants import PRIME_Q, PROOF_SKETCH_TOLERANCE_BASE


class TestLogMagnitudeBucket:
    def test_zero(self):
        assert log_magnitude_bucket(0.0) == 0

    def test_near_zero(self):
        assert log_magnitude_bucket(1e-7) == 0

    def test_positive(self):
        b = log_magnitude_bucket(5.0)
        assert b > 0

    def test_negative(self):
        b = log_magnitude_bucket(-5.0)
        assert b < 0

    def test_symmetry(self):
        assert log_magnitude_bucket(5.0) == -log_magnitude_bucket(-5.0)

    def test_nan_returns_zero(self):
        assert log_magnitude_bucket(float("nan")) == 0

    def test_inf(self):
        assert log_magnitude_bucket(float("inf")) == 7  # num_buckets - 1
        assert log_magnitude_bucket(float("-inf")) == -7


class TestLogMagnitudeBucketVectorized:
    def test_matches_scalar(self):
        values = torch.tensor([0.0, 1e-7, 5.0, -5.0, 100.0, -0.001])
        vec_result = log_magnitude_bucket_vectorized(values)
        scalar_result = torch.tensor([log_magnitude_bucket(v.item()) for v in values])
        assert torch.equal(vec_result, scalar_result)

    def test_nan_handling(self):
        values = torch.tensor([float("nan"), 1.0])
        result = log_magnitude_bucket_vectorized(values)
        assert result[0].item() == 0

    def test_2d(self):
        values = torch.randn(10, 16)
        result = log_magnitude_bucket_vectorized(values)
        assert result.shape == (10, 16)


class TestAdaptiveSketchTolerance:
    def test_position_zero(self):
        assert adaptive_sketch_tolerance(0, 100) == PROOF_SKETCH_TOLERANCE_BASE

    def test_increases_with_position(self):
        t0 = adaptive_sketch_tolerance(0, 1000)
        t100 = adaptive_sketch_tolerance(100, 1000)
        t1000 = adaptive_sketch_tolerance(1000, 1000)
        assert t0 < t100 < t1000


class TestGRAILVerifier:
    @pytest.fixture
    def verifier(self):
        return GRAILVerifier(hidden_dim=128)

    def test_generate_r_vec_shape(self, verifier):
        r = verifier.generate_r_vec("aabbccdd")
        assert r.shape == (16,)  # PROOF_TOPK
        assert r.dtype == torch.int8

    def test_generate_r_vec_deterministic(self, verifier):
        a = verifier.generate_r_vec("aabbccdd")
        b = verifier.generate_r_vec("aabbccdd")
        assert torch.equal(a, b)

    def test_create_commitment(self, verifier):
        h = torch.randn(128)
        r = verifier.generate_r_vec("aabb")
        commit = verifier.create_commitment(h, r)
        assert "sketch" in commit
        assert 0 <= commit["sketch"] < PRIME_Q

    def test_create_commitments_batch_matches_single(self, verifier):
        torch.manual_seed(42)
        h_layer = torch.randn(8, 128)
        r = verifier.generate_r_vec("aabb")
        batch = verifier.create_commitments_batch(h_layer, r)
        for i in range(8):
            single = verifier.create_commitment(h_layer[i], r)
            assert batch[i]["sketch"] == single["sketch"], f"Mismatch at position {i}"

    def test_verify_own_commitment(self, verifier):
        torch.manual_seed(42)
        h = torch.randn(128)
        r = verifier.generate_r_vec("aabb")
        commit = verifier.create_commitment(h, r)
        valid, diag = verifier.verify_commitment(h, commit, r, 100, 0)
        assert valid
        assert diag["sketch_diff"] == 0
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement grail_verifier.py**

Copy from `/home/ubuntu/grail/grail/protocol/grail_verifier.py`. Update imports to `grail.constants` and `grail.protocol.crypto`.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 6: Shared — hf_compat.py, forward.py, digest.py

**Files:**
- Create: `grail/shared/__init__.py`
- Create: `grail/shared/hf_compat.py`
- Create: `grail/shared/forward.py`
- Create: `grail/shared/digest.py`

- [ ] **Step 1: Write tests for digest**

```python
# tests/unit/test_digest.py
from grail.shared.digest import compute_completion_digest


class TestComputeCompletionDigest:
    def test_basic(self):
        commit = {"tokens": [10, 20, 30, 40, 50]}
        meta = {"prompt_length": 2}
        d = compute_completion_digest(commit, meta)
        assert isinstance(d, str)
        assert len(d) == 64  # SHA-256 hex

    def test_deterministic(self):
        commit = {"tokens": [1, 2, 3, 4, 5]}
        meta = {"prompt_length": 2}
        assert compute_completion_digest(commit, meta) == compute_completion_digest(commit, meta)

    def test_different_completions_differ(self):
        meta = {"prompt_length": 1}
        a = compute_completion_digest({"tokens": [1, 2, 3]}, meta)
        b = compute_completion_digest({"tokens": [1, 4, 5]}, meta)
        assert a != b

    def test_same_completion_same_digest(self):
        """Same completion tokens, different prompts → same digest."""
        a = compute_completion_digest({"tokens": [10, 20, 30]}, {"prompt_length": 1})
        b = compute_completion_digest({"tokens": [99, 20, 30]}, {"prompt_length": 1})
        assert a != b  # Different prompt token changes slice starting point...
        # Actually test: same completion with different prompt prefix
        a2 = compute_completion_digest({"tokens": [10, 20, 30]}, {"prompt_length": 1})
        b2 = compute_completion_digest({"tokens": [99, 20, 30]}, {"prompt_length": 1})
        # These differ because tokens[1:] = [20,30] vs [20,30] — same!
        # Wait, first is [20,30], second is [20,30]. Same digest.
        # But tokens[0] differs. prompt_length=1 so completion = tokens[1:]
        a3 = compute_completion_digest({"tokens": [10, 20, 30]}, {"prompt_length": 1})
        b3 = compute_completion_digest({"tokens": [77, 20, 30]}, {"prompt_length": 1})
        assert a3 == b3  # Same completion [20, 30]

    def test_empty_tokens(self):
        assert compute_completion_digest({"tokens": []}, {"prompt_length": 0}) is None

    def test_missing_tokens(self):
        assert compute_completion_digest({}, {"prompt_length": 0}) is None
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement all three files**

- `hf_compat.py`: Copy from `/home/ubuntu/grail/grail/shared/hf_compat.py` (resolve_hidden_size, resolve_vocab_size, resolve_max_context_length).
- `forward.py`: Copy from `/home/ubuntu/grail/grail/model/forward.py` (forward_single_layer). No import changes needed.
- `digest.py`: Copy from `/home/ubuntu/grail/grail/shared/digest.py` (compute_completion_digest).

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 7: Infrastructure — drand.py

**Files:**
- Create: `grail/infrastructure/__init__.py`
- Create: `grail/infrastructure/drand.py`
- Create: `tests/unit/test_drand.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_drand.py
from grail.infrastructure.drand import get_mock_beacon, get_round_at_time, get_beacon


class TestMockBeacon:
    def test_has_required_fields(self):
        b = get_mock_beacon()
        assert "round" in b
        assert "randomness" in b
        assert isinstance(b["randomness"], str)

    def test_incrementing_rounds(self):
        a = get_mock_beacon()
        b = get_mock_beacon()
        assert b["round"] > a["round"]

    def test_randomness_is_hex(self):
        b = get_mock_beacon()
        bytes.fromhex(b["randomness"])  # Should not raise


class TestGetRoundAtTime:
    def test_genesis(self):
        # quicknet genesis = 1692803367, period = 3
        r = get_round_at_time(1692803367)
        assert r == 1

    def test_after_genesis(self):
        r = get_round_at_time(1692803367 + 30)
        assert r == 11  # 1 + 30/3


class TestGetBeacon:
    def test_mock_mode(self):
        b = get_beacon(use_drand=False)
        assert "randomness" in b
        assert b["source"] == "mock"
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement drand.py**

Copy from `/home/ubuntu/grail/grail/drand.py` (the top-level, more complete version). This includes:
- DRAND_CHAINS config (quicknet + default)
- v2/v1 path fallback
- Retry session with User-Agent
- get_drand_beacon, get_mock_beacon, get_beacon, get_round_at_time
- set_chain, get_current_chain
- Bootstrap _ensure_params at import

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 8: Infrastructure — storage.py

**Files:**
- Create: `grail/infrastructure/storage.py`

- [ ] **Step 1: Implement storage.py**

Simplified from the original `comms.py`. Keep only what V1 needs:

```python
"""R2/S3 object storage for window rollout files."""

import asyncio
import gzip
import json
import logging
import os
from typing import Any

from aiobotocore.session import get_session
from botocore.config import Config

logger = logging.getLogger(__name__)

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = get_session()
    return _SESSION


def get_s3_client(
    account_id: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    bucket_name: str | None = None,
):
    """Create S3 client context for R2."""
    account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
    access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint = os.getenv("R2_ENDPOINT_URL") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "us-east-1")

    config = Config(
        connect_timeout=3,
        read_timeout=30,
        retries={"max_attempts": 2},
    )
    return _get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )


async def upload_json(key: str, data: Any, **client_kwargs) -> bool:
    """Upload JSON data to S3."""
    payload = json.dumps(data, separators=(",", ":")).encode()
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=payload)
    return True


async def download_json(key: str, **client_kwargs) -> dict | None:
    """Download and parse JSON from S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            return json.loads(body)
    except Exception as e:
        logger.debug("download_json failed for %s: %s", key, e)
        return None


async def file_exists(key: str, **client_kwargs) -> bool:
    """Check if file exists in S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            await client.head_object(Bucket=bucket, Key=key)
            return True
    except Exception:
        return False


async def upload_window_rollouts(
    hotkey: str, window_start: int, rollouts: list[dict], **client_kwargs
) -> bool:
    """Upload window rollouts as gzipped JSON."""
    key = f"grail/windows/{hotkey}-window-{window_start}.json.gz"
    payload = json.dumps(rollouts, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info("Uploaded %d rollouts for window %d (%d bytes)", len(rollouts), window_start, len(compressed))
    return True


async def download_window_rollouts(
    hotkey: str, window_start: int, **client_kwargs
) -> list[dict] | None:
    """Download window rollouts."""
    key = f"grail/windows/{hotkey}-window-{window_start}.json.gz"
    data = await download_json(key, **client_kwargs)
    if isinstance(data, list):
        return data
    return None
```

- [ ] **Step 2: Commit**

```bash
git add grail/infrastructure/storage.py
git commit -m "feat: R2/S3 storage client for window rollouts"
```

---

### Task 9: Infrastructure — chain.py

**Files:**
- Create: `grail/infrastructure/chain.py`

- [ ] **Step 1: Implement chain.py**

Simplified Bittensor chain interactions:

```python
"""Bittensor chain interactions for GRAIL."""

import asyncio
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NETUID = int(os.getenv("NETUID", "81"))
NETWORK = os.getenv("BT_NETWORK", "finney")


async def get_subtensor():
    """Create async subtensor."""
    import bittensor as bt
    subtensor = bt.async_subtensor(network=NETWORK)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


async def get_metagraph(subtensor, netuid: int = NETUID):
    """Get subnet metagraph."""
    return await subtensor.metagraph(netuid)


async def get_block_hash(subtensor, block_number: int) -> str:
    """Get block hash for a given block number."""
    return await subtensor.get_block_hash(block_number)


async def get_current_block(subtensor) -> int:
    """Get current block number."""
    return await subtensor.get_current_block()


async def set_weights(
    subtensor,
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
) -> bool:
    """Submit weights on-chain."""
    import bittensor as bt
    try:
        result = await subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
        )
        logger.info("set_weights result: %s", result)
        return True
    except Exception as e:
        logger.error("set_weights failed: %s", e)
        return False


def compute_window_randomness(block_hash: str, drand_randomness: str | None = None) -> str:
    """Combine block hash and optional drand randomness into window randomness."""
    if drand_randomness:
        combined = hashlib.sha256(
            bytes.fromhex(block_hash.replace("0x", ""))
            + bytes.fromhex(drand_randomness)
        ).hexdigest()
        return combined
    return block_hash.replace("0x", "")
```

- [ ] **Step 2: Commit**

---

### Task 10: Dataset — loader.py

**Files:**
- Create: `grail/dataset/__init__.py`
- Create: `grail/dataset/loader.py`
- Create: `tests/unit/test_dataset.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_dataset.py
from grail.dataset.loader import get_prompt_indices_for_window


class TestGetPromptIndicesForWindow:
    def test_deterministic(self):
        a = get_prompt_indices_for_window(b"seed123", 1000, 4)
        b = get_prompt_indices_for_window(b"seed123", 1000, 4)
        assert a == b

    def test_correct_count(self):
        indices = get_prompt_indices_for_window(b"seed", 10000, 8)
        assert len(indices) == 8

    def test_within_range(self):
        dataset_size = 500
        indices = get_prompt_indices_for_window(b"test", dataset_size, 4)
        assert all(0 <= i < dataset_size for i in indices)

    def test_different_seeds_differ(self):
        a = get_prompt_indices_for_window(b"seed_a", 10000, 4)
        b = get_prompt_indices_for_window(b"seed_b", 10000, 4)
        assert a != b
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement loader.py**

```python
"""Deterministic prompt selection from ClimbMix-400B."""

import logging
from hashlib import sha256
from typing import Any

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/ClimbMix-400B"
DATASET_SPLIT = "train"


def get_prompt_indices_for_window(
    challenge_seed: bytes, dataset_size: int, num_prompts: int
) -> list[int]:
    """Select prompt indices deterministically from challenge seed.

    Both miner and validator derive the same indices from the same
    challenge_seed (drand + block_hash), so they work on the same
    prompts without direct communication.
    """
    indices = []
    for i in range(num_prompts):
        prompt_seed = sha256(challenge_seed + i.to_bytes(4, "big")).digest()
        index = int.from_bytes(prompt_seed[:8], "big") % dataset_size
        indices.append(index)
    return indices


def load_dataset_cached():
    """Load ClimbMix-400B dataset (cached after first load)."""
    from datasets import load_dataset

    logger.info("Loading dataset %s (split=%s)...", DATASET_NAME, DATASET_SPLIT)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d examples", len(ds))
    return ds


def get_prompts_for_window(
    challenge_seed: bytes, dataset: Any, num_prompts: int
) -> list[str]:
    """Select prompts from dataset for a window."""
    indices = get_prompt_indices_for_window(challenge_seed, len(dataset), num_prompts)
    prompts = []
    for idx in indices:
        row = dataset[idx]
        # Adapt to the actual field name in ClimbMix-400B
        text = row.get("text") or row.get("content") or row.get("prompt") or str(row)
        prompts.append(text)
    return prompts
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 11: Validator — weights.py (Scoring)

**Files:**
- Create: `grail/validator/__init__.py`
- Create: `grail/validator/weights.py`
- Create: `tests/unit/test_weights.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_weights.py
from grail.validator.weights import compute_weights


class TestComputeWeights:
    def test_single_miner(self):
        scores = {"miner_a": {"unique": 100, "valid": 100}}
        weights = compute_weights(scores)
        assert "miner_a" in weights
        assert weights["miner_a"] > 0

    def test_more_unique_gets_more_weight(self):
        scores = {
            "miner_a": {"unique": 100, "valid": 100},
            "miner_b": {"unique": 50, "valid": 50},
        }
        weights = compute_weights(scores)
        assert weights["miner_a"] > weights["miner_b"]

    def test_superlinear(self):
        """2x unique should give much more than 2x weight (exponent=4)."""
        scores_a = {"m": {"unique": 200, "valid": 200}}
        scores_b = {"m": {"unique": 100, "valid": 100}}
        wa = compute_weights(scores_a)["m"]
        wb = compute_weights(scores_b)["m"]
        assert wa / wb > 10  # 2^4 = 16

    def test_zero_unique_zero_weight(self):
        scores = {"m": {"unique": 0, "valid": 100}}
        weights = compute_weights(scores)
        assert weights["m"] == 0.0

    def test_normalizes_to_one(self):
        scores = {
            "a": {"unique": 100, "valid": 100},
            "b": {"unique": 200, "valid": 200},
            "c": {"unique": 50, "valid": 50},
        }
        weights = compute_weights(scores)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_cap_applied(self):
        """Scores above cap should be clamped."""
        scores = {
            "a": {"unique": 999999, "valid": 999999},
            "b": {"unique": 999999, "valid": 999999},
        }
        weights = compute_weights(scores)
        assert abs(weights["a"] - weights["b"]) < 1e-6
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement weights.py**

```python
"""Weight computation for GRAIL V1 — no burn, no environment correctness."""

import logging

from grail.constants import SUPERLINEAR_EXPONENT, UNIQUE_ROLLOUTS_CAP, UNIQUE_ROLLOUTS_CAP_ENABLED

logger = logging.getLogger(__name__)


def compute_weights(
    miner_scores: dict[str, dict[str, int]],
    superlinear_exponent: float = SUPERLINEAR_EXPONENT,
    unique_cap: int = UNIQUE_ROLLOUTS_CAP,
    cap_enabled: bool = UNIQUE_ROLLOUTS_CAP_ENABLED,
) -> dict[str, float]:
    """Compute normalized weights from miner scores.

    Scoring formula (V1 — no environment, no burn):
        raw_score = min(unique_rollouts, cap) ^ superlinear_exponent
        weight_i = raw_score_i / sum(raw_score_j for all j)

    Args:
        miner_scores: {hotkey: {"unique": int, "valid": int}}
        superlinear_exponent: Sybil resistance exponent (default 4.0)
        unique_cap: Max unique rollouts that count (default 5000)
        cap_enabled: Whether to enforce the cap

    Returns:
        {hotkey: normalized_weight} summing to 1.0
    """
    raw_scores: dict[str, float] = {}

    for hotkey, scores in miner_scores.items():
        unique = scores.get("unique", 0)
        valid = scores.get("valid", 0)

        if unique <= 0 or valid <= 0:
            raw_scores[hotkey] = 0.0
            continue

        capped = min(unique, unique_cap) if cap_enabled else unique
        raw_scores[hotkey] = capped ** superlinear_exponent

    total = sum(raw_scores.values())
    if total == 0:
        return {hk: 0.0 for hk in miner_scores}

    return {hk: score / total for hk, score in raw_scores.items()}
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 12: Validator — copycat.py

**Files:**
- Create: `grail/validator/copycat.py`
- Create: `tests/unit/test_copycat.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_copycat.py
from collections import Counter
from grail.validator.copycat import detect_copycats


class TestDetectCopycats:
    def test_no_overlap(self):
        digests = {
            "miner_a": Counter({"aaa": 5, "bbb": 3}),
            "miner_b": Counter({"ccc": 5, "ddd": 3}),
        }
        cheaters = detect_copycats(digests, threshold=0.05)
        assert len(cheaters) == 0

    def test_full_overlap(self):
        digests = {
            "miner_a": Counter({"aaa": 5, "bbb": 3}),
            "miner_b": Counter({"aaa": 5, "bbb": 3}),
        }
        cheaters = detect_copycats(digests, threshold=0.05)
        assert len(cheaters) > 0

    def test_below_threshold(self):
        """1 shared out of 100 = 1% < 5% threshold."""
        a = Counter({f"a_{i}": 1 for i in range(100)})
        b = Counter({f"b_{i}": 1 for i in range(99)})
        b["a_0"] = 1  # 1 shared digest
        cheaters = detect_copycats({"a": a, "b": b}, threshold=0.05)
        assert len(cheaters) == 0

    def test_above_threshold(self):
        """10 shared out of 20 = 50% > 5% threshold."""
        shared = {f"s_{i}": 1 for i in range(10)}
        a = Counter({**shared, **{f"a_{i}": 1 for i in range(10)}})
        b = Counter({**shared, **{f"b_{i}": 1 for i in range(10)}})
        cheaters = detect_copycats({"a": a, "b": b}, threshold=0.05)
        assert len(cheaters) > 0
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement copycat.py**

```python
"""Copycat detection via completion digest overlap."""

import logging
from collections import Counter
from itertools import combinations

from grail.constants import COPYCAT_WINDOW_THRESHOLD

logger = logging.getLogger(__name__)


def detect_copycats(
    miner_digests: dict[str, Counter[str]],
    threshold: float = COPYCAT_WINDOW_THRESHOLD,
) -> set[str]:
    """Detect miners with suspiciously similar completions.

    For each pair of miners, compute the ratio of shared digests
    to the smaller miner's total. If above threshold, flag both.

    Args:
        miner_digests: {hotkey: Counter of completion digests}
        threshold: Overlap ratio to flag (default 5%)

    Returns:
        Set of hotkeys flagged as copycats.
    """
    cheaters: set[str] = set()

    for (hk_a, digests_a), (hk_b, digests_b) in combinations(miner_digests.items(), 2):
        total_a = sum(digests_a.values())
        total_b = sum(digests_b.values())
        if total_a == 0 or total_b == 0:
            continue

        shared = sum((digests_a & digests_b).values())
        denominator = min(total_a, total_b)
        ratio = shared / denominator

        if ratio >= threshold:
            logger.warning(
                "Copycat detected: %s <-> %s | shared=%d denom=%d ratio=%.3f threshold=%.3f",
                hk_a, hk_b, shared, denominator, ratio, threshold,
            )
            cheaters.add(hk_a)
            cheaters.add(hk_b)

    return cheaters
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 13: Validator — verifier.py (Hard Checks)

**Files:**
- Create: `grail/validator/verifier.py`

- [ ] **Step 1: Implement verifier.py**

```python
"""GRAIL proof verification — hard checks + soft checks."""

import hashlib
import logging
from typing import Any

import torch

from grail.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    PROOF_BATCH_SIZE,
    STOCHASTIC_CHECK_FAILURE_THRESHOLD,
)
from grail.protocol.crypto import indices_from_root
from grail.protocol.grail_verifier import GRAILVerifier
from grail.protocol.signatures import verify_commit_signature
from grail.protocol.tokens import hash_tokens
from grail.shared.forward import forward_single_layer
from grail.shared.hf_compat import resolve_hidden_size

logger = logging.getLogger(__name__)


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    return verify_commit_signature(commit, hotkey)


def verify_proof_version(commit: dict) -> bool:
    """Hard check: proof version must match protocol."""
    return commit.get("proof_version") == GRAIL_PROOF_VERSION


def verify_model_identity(commit: dict, expected_model: str) -> bool:
    """Hard check: model name must match checkpoint."""
    model_info = commit.get("model", {})
    return model_info.get("name", "") == expected_model


def verify_nonce_unique(nonce: int, seen_nonces: set[int]) -> bool:
    """Hard check: nonce must not be reused within a window."""
    if nonce in seen_nonces:
        return False
    seen_nonces.add(nonce)
    return True


def verify_commitment_proofs(
    commit: dict,
    model: Any,
    tokenizer: Any,
    window_randomness: str,
) -> tuple[bool, int, int]:
    """Hard check: verify GRAIL sketch commitments against model forward pass.

    Returns:
        (all_passed, passed_count, checked_count)
    """
    tokens = commit["tokens"]
    commitments = commit["commitments"]
    beacon = commit.get("beacon", {})
    randomness = beacon.get("randomness", window_randomness)

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    # Get challenge indices
    seq_len = len(tokens)
    challenge_indices = indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))

    # Forward pass to get hidden states
    input_ids = torch.tensor([tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        hidden_states, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)

    hidden_states = hidden_states[0]  # Remove batch dim: [seq_len, hidden_dim]

    passed = 0
    checked = 0
    for idx in challenge_indices:
        if idx >= len(commitments):
            continue
        checked += 1
        miner_commit = commitments[idx]
        validator_hidden = hidden_states[idx]
        valid, _ = verifier.verify_commitment(validator_hidden, miner_commit, r_vec, seq_len, idx)
        if valid:
            passed += 1

    all_passed = passed == checked and checked > 0
    return all_passed, passed, checked


def verify_rollout(
    rollout: dict,
    hotkey: str,
    model: Any,
    tokenizer: Any,
    window_randomness: str,
    expected_model: str,
    seen_nonces: set[int],
) -> tuple[bool, str]:
    """Run all hard checks on a rollout.

    Returns:
        (is_valid, failure_reason or "ok")
    """
    commit = rollout.get("commit", {})

    # Hard check 1: signature
    if not verify_signature(commit, hotkey):
        return False, "invalid_signature"

    # Hard check 2: proof version
    if not verify_proof_version(commit):
        return False, "invalid_proof_version"

    # Hard check 3: model identity
    if not verify_model_identity(commit, expected_model):
        return False, "model_mismatch"

    # Hard check 4: nonce uniqueness
    nonce = rollout.get("nonce", -1)
    if not verify_nonce_unique(nonce, seen_nonces):
        return False, "duplicate_nonce"

    # Hard check 5: commitment proofs
    all_passed, passed, checked = verify_commitment_proofs(
        commit, model, tokenizer, window_randomness
    )
    if not all_passed:
        return False, f"proof_failed ({passed}/{checked})"

    return True, "ok"
```

- [ ] **Step 2: Commit**

---

### Task 14: Validator — service.py

**Files:**
- Create: `grail/validator/service.py`

- [ ] **Step 1: Implement service.py**

```python
"""Validator main loop — window processing, verification, weight submission."""

import asyncio
import hashlib
import logging
import time
from collections import Counter, defaultdict, deque

from grail.constants import (
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    NUM_PROMPTS_PER_WINDOW,
    STOCHASTIC_CHECK_FAILURE_THRESHOLD,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from grail.infrastructure import chain, storage
from grail.infrastructure.drand import get_beacon
from grail.shared.digest import compute_completion_digest
from grail.validator.copycat import detect_copycats
from grail.validator.verifier import verify_rollout
from grail.validator.weights import compute_weights

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH  # 12


class ValidationService:
    """Main validator service."""

    def __init__(self, wallet, model, tokenizer, netuid: int, use_drand: bool = True):
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.netuid = netuid
        self.use_drand = use_drand

        self._last_processed_window: int = -1
        self._miner_metrics: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"valid": 0, "unique": 0, "checked": 0}
        )
        self._windows_in_interval: int = 0

    async def run(self, subtensor):
        """Main validation loop."""
        logger.info("Starting validation service (netuid=%d, use_drand=%s)", self.netuid, self.use_drand)

        while True:
            try:
                current_block = await chain.get_current_block(subtensor)
                target_window = self._compute_target_window(current_block)

                if target_window <= self._last_processed_window:
                    await asyncio.sleep(6)
                    continue

                logger.info("Processing window %d (block=%d)", target_window, current_block)
                await self._process_window(subtensor, target_window)
                self._last_processed_window = target_window
                self._windows_in_interval += 1

                # Submit weights at interval
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
        """Process a single window: fetch, verify, score."""
        # Get window randomness
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            beacon = get_beacon(use_drand=True)
            randomness = chain.compute_window_randomness(block_hash, beacon["randomness"])
        else:
            randomness = chain.compute_window_randomness(block_hash)

        # Get active miners
        meta = await chain.get_metagraph(subtensor, self.netuid)
        active_hotkeys = list(meta.hotkeys)

        # Sample miners
        sample_size = max(MINER_SAMPLE_MIN, min(
            int(len(active_hotkeys) * MINER_SAMPLE_RATE),
            MINER_SAMPLE_MAX,
            len(active_hotkeys),
        ))
        # Deterministic sampling using window hash
        seed = int(hashlib.sha256(block_hash.encode()).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)
        selected = rng.sample(active_hotkeys, min(sample_size, len(active_hotkeys)))

        logger.info("Selected %d/%d miners for validation", len(selected), len(active_hotkeys))

        # Validate each miner
        model_name = getattr(self.model, "name_or_path", "unknown")
        window_digests: dict[str, Counter[str]] = {}

        for hotkey in selected:
            rollouts = await storage.download_window_rollouts(hotkey, target_window)
            if rollouts is None:
                logger.debug("No rollouts found for miner %s", hotkey)
                continue

            seen_nonces: set[int] = set()
            valid_count = 0
            unique_digests: set[str] = set()

            for rollout in rollouts:
                is_valid, reason = verify_rollout(
                    rollout, hotkey, self.model, self.tokenizer,
                    randomness, model_name, seen_nonces,
                )
                if is_valid:
                    valid_count += 1
                    # Track unique completions
                    commit = rollout.get("commit", {})
                    rollout_meta = commit.get("rollout", {})
                    digest = compute_completion_digest(commit, rollout_meta)
                    if digest:
                        unique_digests.add(digest)
                else:
                    logger.debug("Rollout failed for %s: %s", hotkey, reason)

            # Accumulate metrics
            self._miner_metrics[hotkey]["valid"] += valid_count
            self._miner_metrics[hotkey]["unique"] += len(unique_digests)
            self._miner_metrics[hotkey]["checked"] += len(rollouts)

            # Track digests for copycat detection
            if unique_digests:
                window_digests[hotkey] = Counter(unique_digests)

            logger.info(
                "Miner %s: %d/%d valid, %d unique",
                hotkey[:8], valid_count, len(rollouts), len(unique_digests),
            )

        # Copycat detection
        copycats = detect_copycats(window_digests)
        if copycats:
            logger.warning("Copycats detected: %s", copycats)
            for hk in copycats:
                self._miner_metrics[hk]["valid"] = 0
                self._miner_metrics[hk]["unique"] = 0

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
            await chain.set_weights(subtensor, self.wallet, self.netuid, uids, weight_vals)

    def _compute_target_window(self, current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH
```

- [ ] **Step 2: Commit**

---

### Task 15: Miner — engine.py

**Files:**
- Create: `grail/miner/__init__.py`
- Create: `grail/miner/engine.py`

- [ ] **Step 1: Implement engine.py**

```python
"""Miner engine — vLLM generation + HuggingFace proof construction."""

import asyncio
import logging
import os
import time
from typing import Any

import torch

from grail.constants import (
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    NUM_PROMPTS_PER_WINDOW,
    PROOF_BATCH_SIZE,
    ROLLOUTS_PER_PROBLEM,
    WINDOW_LENGTH,
)
from grail.dataset.loader import get_prompts_for_window
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
    ):
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.dataset = dataset
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    async def mine_window(
        self,
        subtensor,
        window_start: int,
        use_drand: bool = True,
    ) -> list[dict]:
        """Generate rollouts for a window and upload."""
        # Get window randomness
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            beacon = get_beacon(use_drand=True)
            randomness = chain.compute_window_randomness(block_hash, beacon["randomness"])
        else:
            randomness = chain.compute_window_randomness(block_hash)

        challenge_seed = bytes.fromhex(randomness)

        # Get prompts
        prompts = get_prompts_for_window(challenge_seed, self.dataset, NUM_PROMPTS_PER_WINDOW)
        logger.info("Mining window %d with %d prompts", window_start, len(prompts))

        all_rollouts = []
        nonce = 0

        for prompt_idx, prompt in enumerate(prompts):
            for rollout_idx in range(ROLLOUTS_PER_PROBLEM):
                try:
                    rollout = self._generate_and_prove(
                        prompt, randomness, window_start, block_hash, nonce,
                    )
                    all_rollouts.append(rollout)
                except Exception as e:
                    logger.error("Rollout generation failed: %s", e)
                nonce += 1

        # Upload
        if all_rollouts:
            hotkey = self.wallet.hotkey.ss58_address
            await storage.upload_window_rollouts(hotkey, window_start, all_rollouts)
            logger.info("Uploaded %d rollouts for window %d", len(all_rollouts), window_start)

        return all_rollouts

    def _generate_and_prove(
        self,
        prompt: str,
        randomness: str,
        window_start: int,
        block_hash: str,
        nonce: int,
    ) -> dict:
        """Generate text with vLLM, construct proof with HF."""
        # Step 1: Generate with vLLM (GPU 0)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = input_ids.shape[1]

        # vLLM generation (simplified — real impl uses vLLM's engine API)
        with torch.no_grad():
            outputs = self.vllm_model.generate(
                input_ids.to(self.vllm_model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
        all_tokens = outputs[0].tolist()

        # Step 2: HF forward pass for proof (GPU 1)
        proof_input = torch.tensor([all_tokens], device=f"cuda:{self.proof_gpu}")
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Step 3: Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # Step 4: Compute logprobs from HF (not vLLM)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        token_logprobs = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Step 5: Create proof and sign
        proof = create_proof(all_tokens, randomness, len(all_tokens))
        model_name = getattr(self.hf_model, "name_or_path", "unknown")

        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX, commitments, self.wallet
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
            "nonce": nonce,
            "block_hash": block_hash,
            "hotkey": self.wallet.hotkey.ss58_address,
            "commit": commit,
        }
```

- [ ] **Step 2: Commit**

---

### Task 16: CLI — main.py

**Files:**
- Create: `grail/cli/__init__.py`
- Create: `grail/cli/main.py`

- [ ] **Step 1: Implement CLI**

```python
"""GRAIL V1 CLI — mine and validate commands."""

import asyncio
import logging
import os

import typer

app = typer.Typer(name="grail", help="GRAIL V1 — Verifiable Inference Subnet")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run GRAIL miner."""
    setup_logging(log_level)
    logger = logging.getLogger("grail.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info("Starting GRAIL miner (network=%s, netuid=%d)", network, netuid)

    async def _run():
        import bittensor as bt
        from grail.infrastructure.chain import get_subtensor
        from grail.dataset.loader import load_dataset_cached
        from grail.miner.engine import MiningEngine
        from grail.constants import WINDOW_LENGTH

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        # Load models
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading models from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        vllm_model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to("cuda:0").eval()
        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to("cuda:1").eval()

        dataset = load_dataset_cached()

        engine = MiningEngine(vllm_model, hf_model, tokenizer, wallet, dataset)

        logger.info("Miner ready. Entering main loop.")
        last_window = -1
        while True:
            current_block = await subtensor.get_current_block()
            window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
            if window_start > last_window:
                await engine.mine_window(subtensor, window_start, use_drand=use_drand)
                last_window = window_start
            await asyncio.sleep(6)

    asyncio.run(_run())


@app.command()
def validate(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run GRAIL validator."""
    setup_logging(log_level)
    logger = logging.getLogger("grail.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info("Starting GRAIL validator (network=%s, netuid=%d)", network, netuid)

    async def _run():
        import bittensor as bt
        from grail.infrastructure.chain import get_subtensor
        from grail.validator.service import ValidationService
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        logger.info("Loading model from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to("cuda:0").eval()

        service = ValidationService(wallet, model, tokenizer, netuid, use_drand=use_drand)
        await service.run(subtensor)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Commit**

---

### Task 17: Unit Tests — Complete Suite

**Files:**
- Verify all unit tests pass together

- [ ] **Step 1: Run full unit test suite**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Commit any test fixes**

---

### Task 18: Integration Test — Miner→Validator Proof Roundtrip

**Files:**
- Create: `tests/integration/test_miner_validator.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_miner_validator.py
"""Integration test: miner creates proof → validator verifies it.

Requires: GPU with enough memory for a small model.
"""
import pytest
import torch

from grail.protocol.crypto import create_proof, indices_from_root
from grail.protocol.grail_verifier import GRAILVerifier
from grail.protocol.signatures import build_commit_binding, sign_commit_binding
from grail.constants import CHALLENGE_K, LAYER_INDEX, PRIME_Q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestMinerValidatorRoundtrip:
    def test_proof_roundtrip_synthetic(self):
        """Synthetic test: same hidden states → proof verifies."""
        hidden_dim = 256
        seq_len = 64
        torch.manual_seed(42)

        # Simulate shared hidden states (miner and validator see the same)
        hidden_states = torch.randn(seq_len, hidden_dim)
        tokens = list(range(seq_len))
        randomness = "aabbccddee112233"

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        # Miner: create commitments
        commitments = verifier.create_commitments_batch(hidden_states, r_vec)
        assert len(commitments) == seq_len

        # Validator: verify at challenge positions
        challenge_indices = indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))

        for idx in challenge_indices:
            valid, diag = verifier.verify_commitment(
                hidden_states[idx], commitments[idx], r_vec, seq_len, idx
            )
            assert valid, f"Failed at position {idx}: {diag}"

    def test_different_hidden_states_fail(self):
        """Different hidden states → proof should fail."""
        hidden_dim = 256
        seq_len = 64
        torch.manual_seed(42)

        miner_hidden = torch.randn(seq_len, hidden_dim)
        validator_hidden = torch.randn(seq_len, hidden_dim)  # Different!
        tokens = list(range(seq_len))
        randomness = "aabbccddee112233"

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        commitments = verifier.create_commitments_batch(miner_hidden, r_vec)
        challenge_indices = indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))

        failures = 0
        for idx in challenge_indices:
            valid, _ = verifier.verify_commitment(
                validator_hidden[idx], commitments[idx], r_vec, seq_len, idx
            )
            if not valid:
                failures += 1

        # Most should fail with random hidden states
        assert failures > len(challenge_indices) * 0.5
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/integration/test_miner_validator.py -v`

- [ ] **Step 3: Commit**

---

### Task 19: Final Cleanup + README

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create .env.example**

```env
# Bittensor
BT_NETWORK=finney
NETUID=81
BT_WALLET_NAME=default
BT_HOTKEY=default

# R2/S3 Storage
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_ID=grail

# Model
CHECKPOINT_PATH=

# Optional
HF_TOKEN=
```

- [ ] **Step 2: Verify project installs**

Run: `cd /home/ubuntu/DR2 && pip install -e ".[dev]"`

- [ ] **Step 3: Run full test suite**

Run: `cd /home/ubuntu/DR2 && python -m pytest tests/ -v --ignore=tests/integration`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: GRAIL V1 — verifiable inference subnet rebuild"
```
