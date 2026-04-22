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
