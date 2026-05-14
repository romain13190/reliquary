"""Unit tests for hash-dedup primitives."""

from reliquary.protocol.submission import RejectReason


def test_hash_duplicate_reject_reason_exists():
    assert RejectReason.HASH_DUPLICATE.value == "hash_duplicate"
