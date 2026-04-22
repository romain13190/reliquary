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
