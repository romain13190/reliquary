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
