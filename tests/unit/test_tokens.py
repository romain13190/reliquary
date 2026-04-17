from reliquary.protocol.tokens import int_to_bytes, hash_tokens


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

    def test_empty(self):
        result = hash_tokens([])
        assert len(result) == 32
