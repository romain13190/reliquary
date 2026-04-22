# Dockerize Reliquary Validator + Watchtower Auto-Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single `ghcr.io/romain13190/reliquary-validator:latest` image that runs in either trainer mode (GPU, HF write, full training loop) or weight-only mode (CPU only, R2 read, on-chain weight submission) based on a `RELIQUARY_TRAIN` env var; add a `--resume-from` trainer flag so validator restarts don't lose training progress; wire GitHub Actions to auto-publish on every push to main; provide a Watchtower compose stack so community weight-only validators stay current automatically.

**Architecture:** One Dockerfile with `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` as the base so trainer mode has everything it needs. The entrypoint script reads `RELIQUARY_TRAIN` and builds the appropriate CLI call. Weight-only deployments run the same image without `--gpus all` — torch is only imported when the trainer branch runs, so the heavy CUDA libs cost disk space but no runtime. Resume-from-checkpoint accepts `sha:<hex>` (download from HF at that commit) or `path:<dir>` (load local directory) and reconstructs the manifest so `/state` announces the resumed checkpoint on boot — without it, every validator restart silently resets miners to the base model. Watchtower polls GHCR every N minutes and recreates the validator container when a new `:latest` tag is published.

**Tech Stack:** Docker, `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`, Python 3.12, torch 2.7.0+cu128, flash-attn 2.8.3 prebuilt wheel, GitHub Actions, GHCR, `containrrr/watchtower`.

---

## File Structure

**Create:**
- `reliquary/validator/resume.py` — pure logic for parsing `--resume-from` source strings and resolving them to a local model directory + recovered `ManifestEntry`. Kept out of `service.py` so it's independently testable.
- `tests/unit/test_resume.py` — tests for the source parser + manifest reconstruction.
- `Dockerfile` — unified trainer/weight-only image (single stage, CUDA base).
- `docker/entrypoint.sh` — reads env vars, builds `reliquary validate` argv, execs it.
- `docker/.env.example.trainer` — template env file for the subnet owner's trainer validator.
- `docker/.env.example.weight-only` — template env file for a community weight-only validator.
- `docker/docker-compose.trainer.yml` — standalone trainer deployment (no watchtower — sensitive, manual restart only).
- `docker/docker-compose.weight-only.yml` — weight-only deployment with `containrrr/watchtower` sidecar.
- `.github/workflows/docker-image.yml` — CI that builds + pushes to GHCR on main.
- `docs/deployment-docker.md` — operator guide: pull image, mount wallet safely, populate `.env`, run.

**Modify:**
- `reliquary/cli/main.py` — add `--resume-from` CLI option wired through to `ValidationService`.
- `reliquary/validator/service.py` — accept `resume_from` in `__init__`, call the resume logic during bootstrap, install the recovered manifest.

---

## Phase 1 — Resume-from-checkpoint (prereq)

Restart-safety for the trainer. Without this, every validator restart silently drops the in-memory trained model back to the base — miners see `checkpoint_n` reset to 0 (no manifest set) and lose all training progress until the validator produces a new publish. With this, an operator can pass `--resume-from sha:<commit>` (or env var `RELIQUARY_RESUME_FROM`) and the validator starts fresh from that checkpoint, announces it to miners immediately, and keeps training from there.

### Task 1: `ResumeSource` parser

Pure logic to parse the source string into a structured value so downstream code doesn't redo string handling. Accepts `sha:<64-hex>` and `path:<absolute-or-relative-directory>`. Rejects malformed input with a clear error.

**Files:**
- Create: `reliquary/validator/resume.py`
- Test: `tests/unit/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_resume.py
"""Parser + resolver for the --resume-from source string."""

import pytest


def test_parse_sha_source():
    from reliquary.validator.resume import parse_resume_source, ShaSource
    r = parse_resume_source("sha:fa53996ed1533fadfc86be0e6158ddd8465acf34")
    assert isinstance(r, ShaSource)
    assert r.sha == "fa53996ed1533fadfc86be0e6158ddd8465acf34"


def test_parse_path_source(tmp_path):
    from reliquary.validator.resume import parse_resume_source, PathSource
    r = parse_resume_source(f"path:{tmp_path}")
    assert isinstance(r, PathSource)
    assert r.path == str(tmp_path)


def test_parse_path_source_relative():
    from reliquary.validator.resume import parse_resume_source, PathSource
    r = parse_resume_source("path:./state/checkpoints/ckpt_5")
    assert isinstance(r, PathSource)
    assert r.path == "./state/checkpoints/ckpt_5"


def test_parse_rejects_unknown_scheme():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="unknown scheme"):
        parse_resume_source("weird:xyz")


def test_parse_rejects_bare_string():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="expected scheme"):
        parse_resume_source("fa53996ed153")


def test_parse_rejects_malformed_sha():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="not a 40-char hex"):
        parse_resume_source("sha:notahex")


def test_parse_rejects_empty_path():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="path is empty"):
        parse_resume_source("path:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_resume.py -v`
Expected: all 7 FAIL with `ModuleNotFoundError: No module named 'reliquary.validator.resume'`

- [ ] **Step 3: Implement the parser**

```python
# reliquary/validator/resume.py
"""Parse and resolve ``--resume-from`` source strings.

Accepts two schemes:

* ``sha:<40-hex>`` — a HuggingFace commit SHA on the trainer's own
  ``--hf-repo-id`` repo. The loader downloads that revision.
* ``path:<dir>`` — a local directory that already contains the HF-format
  snapshot (``model.safetensors`` + ``config.json`` + tokenizer files).

Anything else raises ``ValueError`` so operators see the mistake loudly
instead of silently falling back to the base model (which would produce
a GRAIL mismatch the miners would hit on the very next submission).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ShaSource:
    sha: str


@dataclass(frozen=True)
class PathSource:
    path: str


_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def parse_resume_source(raw: str) -> ShaSource | PathSource:
    if ":" not in raw:
        raise ValueError(
            f"resume source {raw!r}: expected scheme (sha:<hex> or path:<dir>)"
        )
    scheme, _, rest = raw.partition(":")
    if scheme == "sha":
        if not _HEX40.match(rest):
            raise ValueError(
                f"resume source sha:{rest}: not a 40-char hex commit SHA"
            )
        return ShaSource(sha=rest)
    if scheme == "path":
        if not rest:
            raise ValueError("resume source path: path is empty")
        return PathSource(path=rest)
    raise ValueError(
        f"resume source {raw!r}: unknown scheme {scheme!r} "
        "(expected 'sha' or 'path')"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_resume.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/resume.py tests/unit/test_resume.py
git commit -m "feat(validator/resume): parse --resume-from source strings"
```

---

### Task 2: `resolve_resume_source` — download (for sha) and detect the checkpoint number

Given a parsed source, returns `(local_path, checkpoint_n)`. For `sha`, downloads the snapshot into the HF cache and parses the commit title (`"checkpoint 7"` → 7). For `path`, uses the directory as-is and expects the operator to pass a directory named like `ckpt_<N>` OR the caller to derive `N` externally.

**Files:**
- Modify: `reliquary/validator/resume.py`
- Test: `tests/unit/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_resume.py
from unittest.mock import MagicMock


def test_resolve_sha_downloads_and_extracts_n():
    """SHA mode: snapshot_download is called with the right args, and the
    checkpoint_n is parsed from the commit title."""
    from reliquary.validator.resume import (
        ShaSource, resolve_resume_source,
    )

    calls = {}

    def fake_download(repo_id, revision, **kwargs):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        return "/tmp/fake-hf-cache/snapshots/" + revision

    def fake_commit_title(repo_id, revision):
        calls["title_query"] = (repo_id, revision)
        return "checkpoint 7"

    local_path, checkpoint_n = resolve_resume_source(
        source=ShaSource(sha="a" * 40),
        hf_repo_id="myorg/repo",
        download_fn=fake_download,
        commit_title_fn=fake_commit_title,
    )
    assert checkpoint_n == 7
    assert local_path == "/tmp/fake-hf-cache/snapshots/" + "a" * 40
    assert calls["repo_id"] == "myorg/repo"
    assert calls["revision"] == "a" * 40


def test_resolve_sha_rejects_unparseable_title():
    """If the commit title isn't 'checkpoint N', refuse (don't silently
    pick a wrong N)."""
    from reliquary.validator.resume import (
        ShaSource, resolve_resume_source,
    )

    with pytest.raises(ValueError, match="could not parse"):
        resolve_resume_source(
            source=ShaSource(sha="b" * 40),
            hf_repo_id="myorg/repo",
            download_fn=lambda **kw: "/x",
            commit_title_fn=lambda **kw: "some random commit",
        )


def test_resolve_path_uses_dir_as_is(tmp_path):
    """path mode: the provided directory is returned verbatim; checkpoint_n
    is extracted from ``ckpt_<N>`` in the trailing component."""
    from reliquary.validator.resume import (
        PathSource, resolve_resume_source,
    )

    target = tmp_path / "ckpt_12"
    target.mkdir()
    local_path, checkpoint_n = resolve_resume_source(
        source=PathSource(path=str(target)),
        hf_repo_id="unused",
    )
    assert local_path == str(target)
    assert checkpoint_n == 12


def test_resolve_path_rejects_unparseable_dir(tmp_path):
    """path mode without ``ckpt_N`` → ValueError. Operators must name the
    directory so the checkpoint number is unambiguous."""
    from reliquary.validator.resume import (
        PathSource, resolve_resume_source,
    )
    target = tmp_path / "random_name"
    target.mkdir()
    with pytest.raises(ValueError, match="could not derive checkpoint_n"):
        resolve_resume_source(
            source=PathSource(path=str(target)),
            hf_repo_id="unused",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_resume.py -v -k "resolve"`
Expected: 4 FAIL with `ImportError` on `resolve_resume_source`.

- [ ] **Step 3: Implement the resolver**

```python
# Append to reliquary/validator/resume.py
import re as _re
from pathlib import Path
from typing import Callable, Optional


_CKPT_TITLE = _re.compile(r"^checkpoint\s+(\d+)\s*$", _re.IGNORECASE)
_CKPT_DIRNAME = _re.compile(r"^ckpt_(\d+)$")


def resolve_resume_source(
    source: ShaSource | PathSource,
    hf_repo_id: str,
    *,
    download_fn: Optional[Callable[..., str]] = None,
    commit_title_fn: Optional[Callable[..., str]] = None,
) -> tuple[str, int]:
    """Resolve a parsed source to ``(local_path, checkpoint_n)``.

    ``download_fn`` / ``commit_title_fn`` are injected for testing; the
    real callers pass the HuggingFace Hub equivalents.
    """
    if isinstance(source, PathSource):
        dirname = Path(source.path).name
        m = _CKPT_DIRNAME.match(dirname)
        if not m:
            raise ValueError(
                f"resume path {source.path!r}: could not derive "
                "checkpoint_n from trailing component — expected 'ckpt_<N>'"
            )
        return source.path, int(m.group(1))

    # SHA path.
    if download_fn is None or commit_title_fn is None:
        raise RuntimeError(
            "resolve_resume_source(sha): download_fn and commit_title_fn "
            "are required for SHA mode"
        )
    title = commit_title_fn(repo_id=hf_repo_id, revision=source.sha)
    m = _CKPT_TITLE.match(title or "")
    if not m:
        raise ValueError(
            f"resume sha:{source.sha}: could not parse checkpoint_n from "
            f"commit title {title!r} (expected 'checkpoint N')"
        )
    local_path = download_fn(repo_id=hf_repo_id, revision=source.sha)
    return local_path, int(m.group(1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_resume.py -v`
Expected: 11 passed (7 from Task 1 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/resume.py tests/unit/test_resume.py
git commit -m "feat(validator/resume): resolve source to (local_path, checkpoint_n)"
```

---

### Task 3: Wire resume into `ValidationService`

Adds a `resume_from: str | None` parameter to `ValidationService.__init__`; during `_bootstrap_state_from_external` (or at the very start of `run()`), if the param is set, resolve it, load the model in place of the base, and install a `ManifestEntry` so `/state` announces the resumed checkpoint to miners.

**Files:**
- Modify: `reliquary/validator/service.py`
- Test: `tests/unit/test_state_machine.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_state_machine.py
@pytest.mark.asyncio
async def test_resume_from_path_installs_manifest():
    """resume_from="path:/tmp/ckpt_3" loads the directory AND installs a
    manifest so /state announces checkpoint_n=3 to miners immediately."""
    import tempfile, os
    from unittest.mock import AsyncMock, MagicMock, patch
    from reliquary.validator.service import ValidationService

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = os.path.join(td, "ckpt_3")
        os.makedirs(ckpt_dir)
        # A stub load_fn avoids torch / transformers.
        load_calls = []

        def fake_load(path):
            load_calls.append(path)
            return MagicMock(name="resumed_model")

        svc = ValidationService(
            wallet=_FakeWallet(),
            model=MagicMock(name="base_model"),
            tokenizer=MagicMock(),
            env=_FakeEnv(),
            netuid=99,
            resume_from=f"path:{ckpt_dir}",
            load_model_fn=fake_load,  # test hook
        )
        # Run the resume step directly — the real path goes through run().
        await svc._apply_resume_from()

        # Model replaced
        assert svc.model is not None
        assert load_calls == [ckpt_dir]
        # Manifest installed (so /state announces it)
        mf = svc._checkpoint_store.current_manifest()
        assert mf is not None
        assert mf.checkpoint_n == 3
        assert svc._checkpoint_n == 3


@pytest.mark.asyncio
async def test_resume_from_none_is_noop():
    """No resume_from → service boots with the base model, no manifest."""
    from reliquary.validator.service import ValidationService
    svc = ValidationService(
        wallet=_FakeWallet(),
        model=MagicMock(),
        tokenizer=MagicMock(),
        env=_FakeEnv(),
        netuid=99,
    )
    await svc._apply_resume_from()
    assert svc._checkpoint_store.current_manifest() is None


@pytest.mark.asyncio
async def test_resume_from_load_failure_aborts():
    """If the resume source fails to load, abort — never fall back silently
    to the base model (would cause GRAIL mismatch on first submission)."""
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService
    import os, tempfile

    def failing_load(path):
        raise RuntimeError("corrupt checkpoint")

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = os.path.join(td, "ckpt_3")
        os.makedirs(ckpt_dir)
        svc = ValidationService(
            wallet=_FakeWallet(),
            model=MagicMock(),
            tokenizer=MagicMock(),
            env=_FakeEnv(),
            netuid=99,
            resume_from=f"path:{ckpt_dir}",
            load_model_fn=failing_load,
        )
        with pytest.raises(RuntimeError, match="corrupt checkpoint"):
            await svc._apply_resume_from()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_state_machine.py -v -k "resume_from"`
Expected: 3 FAIL (resume_from / load_model_fn not accepted by constructor; `_apply_resume_from` doesn't exist).

- [ ] **Step 3: Add the constructor argument + helper method**

In `reliquary/validator/service.py`:

```python
# In ValidationService.__init__ signature, add:
#     resume_from: str | None = None,
#     load_model_fn: Callable[[str], Any] | None = None,
# and store them:
self._resume_from = resume_from
self._load_model_fn = load_model_fn or _default_load_model

# Add the helper method, to be called from run() before the main loop:
async def _apply_resume_from(self) -> None:
    """If --resume-from was set, load the model from that source and
    install a manifest. No-op if unset."""
    if not self._resume_from:
        return
    from reliquary.validator.resume import (
        parse_resume_source,
        resolve_resume_source,
    )
    from reliquary.validator.checkpoint import ManifestEntry
    from huggingface_hub import HfApi, snapshot_download

    def _commit_title(repo_id, revision):
        api = HfApi()
        commits = api.list_repo_commits(repo_id=repo_id)
        for c in commits:
            if c.commit_id == revision:
                return c.title
        return ""

    def _download(repo_id, revision):
        return snapshot_download(repo_id=repo_id, revision=revision)

    source = parse_resume_source(self._resume_from)
    local_path, checkpoint_n = resolve_resume_source(
        source,
        hf_repo_id=self._checkpoint_store.repo_id,
        download_fn=_download,
        commit_title_fn=_commit_title,
    )
    # Load weights — this replaces the base model loaded at __init__.
    self.model = self._load_model_fn(local_path)
    # Reconstruct manifest so miners see the resumed checkpoint via /state.
    sig_payload = f"{checkpoint_n}|{self._resume_from}".encode()
    sig_bytes = self._wallet.hotkey.sign(sig_payload)
    entry = ManifestEntry(
        checkpoint_n=checkpoint_n,
        repo_id=self._checkpoint_store.repo_id,
        revision=self._resume_from,
        signature="ed25519:" + sig_bytes.hex(),
    )
    self._checkpoint_store._current = entry
    self._checkpoint_n = checkpoint_n
    try:
        self.server.set_current_checkpoint(entry)
    except AttributeError:
        pass
    logger.info(
        "Resumed from %s: checkpoint_n=%d",
        self._resume_from, checkpoint_n,
    )
```

Also add a module-level default loader:

```python
def _default_load_model(local_path: str):
    import torch
    from transformers import AutoModelForCausalLM
    from reliquary.constants import ATTN_IMPLEMENTATION
    return AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()
```

And call `await self._apply_resume_from()` at the start of `run()` before `_bootstrap_state_from_external`.

- [ ] **Step 4: Run tests**

Run: `/home/ubuntu/Catalyst/.venv/bin/pytest tests/unit/test_state_machine.py tests/unit/test_resume.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_state_machine.py
git commit -m "feat(validator): --resume-from loads model + installs manifest at boot

Trainer restarts previously reverted miners to the base model until the
next publish landed — 30 min+ of idle training. --resume-from sha:<hex>
(or path:<dir>) resolves to a local snapshot, replaces self.model in
place, and installs a ManifestEntry so /state announces the resumed
checkpoint immediately. Load failures abort (never silent fallback —
base model + resumed manifest would cause every miner submission to fail
GRAIL check)."
```

---

### Task 4: CLI flag + env var

Expose `--resume-from` on the CLI, default to `RELIQUARY_RESUME_FROM` env when unset (so Docker operators can configure via `.env`).

**Files:**
- Modify: `reliquary/cli/main.py`

- [ ] **Step 1: Add the CLI option**

In `reliquary/cli/main.py`, inside the `validate` command:

```python
resume_from: str = typer.Option(
    os.getenv("RELIQUARY_RESUME_FROM", ""),
    help=(
        "Resume trainer from a checkpoint instead of the base model. "
        "Accepts 'sha:<40-hex>' (HF commit on --hf-repo-id) or "
        "'path:<dir>' (local ckpt_<N> directory). Trainer mode only."
    ),
),
```

And inside `_run()` when building the trainer service:

```python
service = ValidationService(
    # ... existing kwargs ...
    resume_from=resume_from or None,
)
```

- [ ] **Step 2: Manually verify help**

Run: `/home/ubuntu/Catalyst/.venv/bin/reliquary validate --help | grep -A2 resume-from`
Expected: the option is listed with the help text.

- [ ] **Step 3: Commit**

```bash
git add reliquary/cli/main.py
git commit -m "feat(cli): --resume-from flag + RELIQUARY_RESUME_FROM env fallback"
```

---

## Phase 2 — Dockerfile + entrypoint

### Task 5: `Dockerfile`

Single stage, CUDA-base, installs everything we manually confirmed works on Targon. Operators pull this image and run with GPU or without depending on `RELIQUARY_TRAIN`.

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore`**

```
.git
.venv
__pycache__
*.pyc
*.pyo
.pytest_cache
docs/superpowers
tests
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so system pip stays clean
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8 (matches our Targon setup)
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -O /tmp/flash_attn.whl \
 && pip install /tmp/flash_attn.whl \
 && rm /tmp/flash_attn.whl

# Source + install
WORKDIR /opt/reliquary
COPY . /opt/reliquary
RUN pip install -e .

# bittensor 10.2.0 ships with async-substrate-interface 2.0 which conflicts
# with its own scalecodec import path — roll back to the 1.x line that matches.
RUN pip uninstall -y cyscale \
 && pip install 'async-substrate-interface<2.0.0' \
 && pip install --force-reinstall --no-deps scalecodec==1.2.12

# boto3 for R2 (weight-only mode + trainer archive uploads)
RUN pip install boto3

# Runtime
ENV GRAIL_ATTN_IMPL=flash_attention_2
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/opt/entrypoint.sh"]
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(docker): CUDA 12.8 + torch 2.7 + flash-attn 2.8.3 image"
```

---

### Task 6: `entrypoint.sh`

Reads env vars, builds the `reliquary validate` argv, validates required vars, execs the CLI.

**Files:**
- Create: `docker/entrypoint.sh`

- [ ] **Step 1: Write the entrypoint script**

```bash
#!/bin/bash
# Entrypoint for the Reliquary validator image.
#
# Reads environment variables to build the `reliquary validate` argv.
# Required: BT_WALLET_NAME, BT_HOTKEY.
# Mode is controlled by RELIQUARY_TRAIN=1 (trainer) or !=1 (weight-only).
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required (the wallet dir name under ~/.bittensor/wallets)}"
: "${BT_HOTKEY:?BT_HOTKEY is required (the hotkey file name under wallets/<name>/hotkeys/)}"

args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
)

if [[ "${RELIQUARY_TRAIN:-0}" == "1" ]]; then
  : "${RELIQUARY_HF_REPO_ID:?RELIQUARY_HF_REPO_ID required in trainer mode (target HF repo for checkpoints)}"
  args+=(
    --train
    --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}"
    --hf-repo-id   "${RELIQUARY_HF_REPO_ID}"
    --http-host    "${RELIQUARY_HTTP_HOST:-0.0.0.0}"
    --http-port    "${RELIQUARY_HTTP_PORT:-8080}"
  )
  [[ -n "${RELIQUARY_EXTERNAL_IP:-}" ]]     && args+=(--external-ip   "${RELIQUARY_EXTERNAL_IP}")
  [[ -n "${RELIQUARY_EXTERNAL_PORT:-}" ]]   && args+=(--external-port "${RELIQUARY_EXTERNAL_PORT}")
  [[ -n "${RELIQUARY_RESUME_FROM:-}" ]]     && args+=(--resume-from   "${RELIQUARY_RESUME_FROM}")
else
  args+=(--no-train)
fi

echo "Launching: reliquary validate ${args[*]}"
exec reliquary validate "${args[@]}"
```

- [ ] **Step 2: Commit**

```bash
git add docker/entrypoint.sh
git commit -m "feat(docker): entrypoint.sh — env-driven validate command"
```

---

### Task 7: Example `.env` + `docker-compose` files

Two compose examples — one trainer (no watchtower, manual restart), one weight-only (with watchtower sidecar).

**Files:**
- Create: `docker/.env.example.trainer`
- Create: `docker/.env.example.weight-only`
- Create: `docker/docker-compose.trainer.yml`
- Create: `docker/docker-compose.weight-only.yml`

- [ ] **Step 1: `.env.example.trainer`**

```bash
# Bittensor
BT_NETWORK=finney
BT_NETUID=81
BT_WALLET_NAME=subnet
BT_HOTKEY=hotkey1

# Trainer mode
RELIQUARY_TRAIN=1
RELIQUARY_CHECKPOINT=Qwen/Qwen3-4B-Instruct-2507
RELIQUARY_HF_REPO_ID=your-org/reliquary-sn
RELIQUARY_HTTP_HOST=0.0.0.0
RELIQUARY_HTTP_PORT=8080
# Optional — advertise on-chain so miners can discover the validator via metagraph:
# RELIQUARY_EXTERNAL_IP=1.2.3.4
# RELIQUARY_EXTERNAL_PORT=8080
# Optional — resume from a prior checkpoint so restart doesn't reset miners:
# RELIQUARY_RESUME_FROM=sha:fa53996ed1533fadfc86be0e6158ddd8465acf34
# RELIQUARY_RESUME_FROM=path:/state/checkpoints/ckpt_7

# HuggingFace (write access to RELIQUARY_HF_REPO_ID)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# R2 (write access — trainer archives windows + reads for bootstrap)
R2_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_BUCKET_ID=reliquary
R2_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxx
R2_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_ENDPOINT_URL=https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.cloudflarestorage.com
```

- [ ] **Step 2: `.env.example.weight-only`**

```bash
# Bittensor
BT_NETWORK=finney
BT_NETUID=81
BT_WALLET_NAME=val-operator-01
BT_HOTKEY=hotkey

# Weight-only mode — no GPU, no HF, no HTTP ingress.
RELIQUARY_TRAIN=0

# R2 (read-only access is sufficient)
R2_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_BUCKET_ID=reliquary
R2_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxx
R2_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_ENDPOINT_URL=https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.cloudflarestorage.com
```

- [ ] **Step 3: `docker-compose.trainer.yml`**

```yaml
# Trainer validator — no watchtower (sensitive, manual restarts only).
# Mount ONLY the hotkey file + coldkeypub; never mount the coldkey itself.
services:
  reliquary-trainer:
    image: ghcr.io/romain13190/reliquary-validator:latest
    container_name: reliquary-trainer
    restart: unless-stopped
    env_file: .env
    # NVIDIA Container Toolkit required on the host.
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "${RELIQUARY_HTTP_PORT:-8080}:8080"
    volumes:
      # Hotkey (readonly, needed to sign submissions + weights)
      - type: bind
        source: ${BT_WALLETS_DIR:?set BT_WALLETS_DIR to the host path of your wallets/}
        target: /root/.bittensor/wallets
        read_only: true
      # Persistent state (staging checkpoints, HF cache)
      - reliquary-state:/root/reliquary/state
      - hf-cache:/root/.cache/huggingface

volumes:
  reliquary-state:
  hf-cache:
```

- [ ] **Step 4: `docker-compose.weight-only.yml`**

```yaml
# Weight-only validator + watchtower auto-update.
# Runs on any Linux host with Docker — no GPU required.
services:
  reliquary-weight-only:
    image: ghcr.io/romain13190/reliquary-validator:latest
    container_name: reliquary-weight-only
    restart: unless-stopped
    env_file: .env
    volumes:
      - type: bind
        source: ${BT_WALLETS_DIR:?set BT_WALLETS_DIR to the host path of your wallets/}
        target: /root/.bittensor/wallets
        read_only: true
    labels:
      com.centurylinklabs.watchtower.enable: "true"

  watchtower:
    image: containrrr/watchtower:latest
    container_name: watchtower
    restart: unless-stopped
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    # Poll registry every 5 min, only update containers with the label above,
    # and remove old images to save disk.
    command: >
      --interval 300
      --label-enable
      --cleanup
      --include-restarting
```

- [ ] **Step 5: Commit**

```bash
git add docker/
git commit -m "feat(docker): .env + compose examples (trainer, weight-only+watchtower)"
```

---

## Phase 3 — Local test

### Task 8: Build + smoke-test on the Targon trainer host

Build the image on the validator Targon box (has GPU), verify the image runs end-to-end without breaking the live testnet.

**Files:** none (ops step).

- [ ] **Step 1: Build on the trainer host**

```bash
ssh -i ~/.ssh/distill_h100 wrk-0nkyr4av9l9d@ssh.deployments.targon.com \
  "cd /root/reliquary && git pull origin main && \
   DOCKER_BUILDKIT=1 docker build -t reliquary-validator:local ."
```

Expected: image builds, final "Successfully built ..." line. ~10-15 min on first build (downloads torch + flash-attn). Subsequent builds cached in seconds.

- [ ] **Step 2: Smoke-test weight-only mode (no GPU)**

```bash
# On the Targon host (pretend a different wallet to avoid clashing with the live trainer):
docker run --rm \
  -e BT_WALLET_NAME=probe \
  -e BT_HOTKEY=probe-hk \
  -e BT_NETUID=81 \
  -e BT_NETWORK=finney \
  -e RELIQUARY_TRAIN=0 \
  -e R2_ACCOUNT_ID=... (from .env) \
  -e R2_ACCESS_KEY_ID=... \
  -e R2_SECRET_ACCESS_KEY=... \
  -e R2_BUCKET_ID=reliquary \
  -e R2_ENDPOINT_URL=https://... \
  -v /tmp/probe-wallet:/root/.bittensor/wallets:ro \
  reliquary-validator:local 2>&1 | head -30
```

Expected: launcher prints `Launching: reliquary validate --network finney ... --no-train`, then the CLI boots, tries to load the wallet (may fail because `probe` wallet doesn't exist on this box — that's fine, we only need to confirm the image starts). Kill with Ctrl-C.

- [ ] **Step 3: Confirm resume-from mode doesn't break the live trainer**

Do NOT run the containerized trainer against the same wallet as the running live trainer (port conflict + duplicate /submit endpoint). Just confirm the argv construction is correct:

```bash
docker run --rm --entrypoint /bin/bash \
  -e BT_WALLET_NAME=subnet \
  -e BT_HOTKEY=hotkey1 \
  -e RELIQUARY_TRAIN=1 \
  -e RELIQUARY_HF_REPO_ID=R0mAI/reliquary-math \
  -e RELIQUARY_RESUME_FROM=sha:de8490f58de7bfb25d7e53a9e12e0a67a51e9e4c \
  reliquary-validator:local -c \
  'echo "dry run" && cat /opt/entrypoint.sh | head -30 && echo "---" && RELIQUARY_HF_REPO_ID=x BT_WALLET_NAME=x BT_HOTKEY=x RELIQUARY_TRAIN=1 bash -n /opt/entrypoint.sh'
```

Expected: no syntax errors from `bash -n`, and visual inspection of the entrypoint confirms the argv order.

- [ ] **Step 4: No commit** (ops step — nothing to commit).

---

## Phase 4 — CI + GHCR

### Task 9: GitHub Action — build + push on main

Triggers on every push to main, builds the image using the GitHub Actions runner, pushes to `ghcr.io/romain13190/reliquary-validator:latest` + a SHA tag for reproducibility.

**Files:**
- Create: `.github/workflows/docker-image.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Build and publish validator image

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  packages: write

jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository_owner }}/reliquary-validator
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=sha-,format=short

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/docker-image.yml
git commit -m "ci(docker): build + push reliquary-validator to GHCR"
git push origin main
```

- [ ] **Step 3: Verify in GitHub**

Open https://github.com/romain13190/reliquary/actions. The workflow should be running. When green, the image appears at https://github.com/romain13190/reliquary/pkgs/container/reliquary-validator with the `latest` tag.

- [ ] **Step 4: Make the package public (one-time, via GitHub UI)**

Go to the package page → Package settings → Danger Zone → Change visibility → Public. Otherwise operators need a PAT to pull.

- [ ] **Step 5: No commit** (ops step).

---

## Phase 5 — Watchtower deployment + docs

### Task 10: `docs/deployment-docker.md` + integration verification

Operator-facing guide that explains how to pull, configure, and run. Also verify that Watchtower does what we claim — push a no-op "bump" commit, wait 5 min, confirm the weight-only container restarts on its own.

**Files:**
- Create: `docs/deployment-docker.md`

- [ ] **Step 1: Write the operator guide**

```markdown
# Running a Reliquary Validator via Docker

Two supported deployments:

1. **Trainer validator** — subnet owner only. GPU, HF write, full training loop. Manual restart.
2. **Weight-only validator** — anyone with a registered hotkey. CPU only, reads R2, submits EMA-based weights on-chain. Auto-updated by Watchtower.

## Prerequisites (both)

- Docker 24+ with Compose plugin.
- A Bittensor wallet registered on netuid 81 (coldkey stays offline — only the hotkey reaches this box).
- R2 credentials (bucket `reliquary`, the read-only keys are enough for weight-only).

## Trainer deployment (GPU host)

Additional prerequisites:

- NVIDIA driver + CUDA 12.8 on the host.
- NVIDIA Container Toolkit installed and configured.
- HF_TOKEN with write access to your checkpoint repo.
- 150 GB disk for staging checkpoints + HF cache.

```bash
git clone https://github.com/romain13190/reliquary.git
cd reliquary/docker
cp .env.example.trainer .env
# Edit .env — fill in BT_WALLET_NAME, BT_HOTKEY, RELIQUARY_HF_REPO_ID, HF_TOKEN, R2 creds.
export BT_WALLETS_DIR=/home/you/.bittensor/wallets
docker compose -f docker-compose.trainer.yml up -d
docker logs -f reliquary-trainer
```

To resume from an existing checkpoint after a restart, set `RELIQUARY_RESUME_FROM` in `.env` to a HF commit SHA (`sha:<40-hex>`) or a local checkpoint directory (`path:/root/reliquary/state/checkpoints/ckpt_<N>`).

**Security:** the wallet mount is readonly and contains only the hotkey + `coldkeypub.txt`. Never mount your coldkey into the container.

## Weight-only deployment (any Linux host)

```bash
git clone https://github.com/romain13190/reliquary.git
cd reliquary/docker
cp .env.example.weight-only .env
# Edit .env — fill in BT_WALLET_NAME, BT_HOTKEY, R2 creds.
export BT_WALLETS_DIR=/home/you/.bittensor/wallets
docker compose -f docker-compose.weight-only.yml up -d
```

Watchtower polls GHCR every 5 minutes and, when a new `:latest` tag appears, pulls it and restarts the validator container. No operator action required.

To verify Watchtower is alive:

```bash
docker logs watchtower | tail -20
```

You should see periodic `Checking containers for updated images` lines.

## Security notes on the wallet mount

```yaml
volumes:
  - ${BT_WALLETS_DIR}:/root/.bittensor/wallets:ro
```

- `:ro` (read-only) — the container cannot write to the wallet directory.
- The wallet directory should contain **only** `coldkeypub.txt` + `hotkeys/<your-hotkey>`. Not the coldkey private file.
- A good layout is to have a separate `operator-wallet/` folder with just the hotkey copied in and point `BT_WALLETS_DIR` at that, keeping the main `~/.bittensor/wallets` (with the coldkey) elsewhere.
```

- [ ] **Step 2: Verify end-to-end on a throwaway host**

On a CPU-only Linux VM:

1. Install Docker + Compose.
2. Copy a test hotkey into a dedicated directory.
3. `docker compose -f docker-compose.weight-only.yml up -d`.
4. Confirm `docker ps` shows both `reliquary-weight-only` and `watchtower` running.
5. Confirm `docker logs reliquary-weight-only | head` shows `reliquary validate --no-train` starting.
6. Push an empty commit to main (`git commit --allow-empty -m "noop" && git push`).
7. Wait ~10 min (5 min for CI build + 5 min for watchtower poll).
8. Check `docker logs watchtower` — look for "Found new image" + "Stopping / Starting".
9. Check `docker ps` — `Created` timestamp on `reliquary-weight-only` should have updated.

- [ ] **Step 3: Commit**

```bash
git add docs/deployment-docker.md
git commit -m "docs(deployment): Docker + Watchtower operator guide"
```

---

## Self-Review

**1. Spec coverage.**
- Resume-from-checkpoint trainer flag → Phase 1 (Tasks 1-4).
- Unified Dockerfile driven by `RELIQUARY_TRAIN` → Phase 2 (Task 5).
- `entrypoint.sh` that builds the CLI argv from env → Phase 2 (Task 6).
- Two compose examples (trainer manual, weight-only with watchtower) → Phase 2 (Task 7).
- Local smoke test → Phase 3 (Task 8).
- CI build + GHCR push → Phase 4 (Task 9).
- Watchtower stack + operator docs → Phase 5 (Task 10).
- Key safety (hotkey-only mount, no coldkey in image, readonly mount) is covered in Tasks 7 and 10.

**2. Placeholder scan.** Every step has full code/commands. No `TBD` or "add error handling". The one thing that's not code is Task 8 Step 2 ("fill in R2 creds from .env") which is inherent to ops — can't put secrets in the plan.

**3. Type consistency.** `ResumeSource`, `ShaSource`, `PathSource`, `parse_resume_source`, `resolve_resume_source`, `_apply_resume_from`, `load_model_fn`, `download_fn`, `commit_title_fn` are all named consistently across Tasks 1-3. `ManifestEntry` (checkpoint_n/repo_id/revision/signature) matches what `CheckpointStore` already uses.

**4. Known loose end.** Task 3 stores `resume_from` directly as the `ManifestEntry.revision`. That's correct for `sha:<hex>` mode (the string after `sha:` IS the HF commit SHA) but slightly inaccurate for `path:` mode (the directory path is not a HF revision — miners can't `from_pretrained` it). This is acceptable because path mode is expected to be used only for fast local restarts on the same host, where miners should be offline or will simply reject the submission until a real HF publish updates the manifest. If this ever becomes a real use case (miners need to keep mining during a local-path resume), the implementer should publish the snapshot to HF as part of resume, then use the resulting SHA.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-22-docker-validator-watchtower.md`.

**Execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review, fast iteration.
2. **Inline Execution** — execute tasks in this session with batch checkpoints.

Which approach?
