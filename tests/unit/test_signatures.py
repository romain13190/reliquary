from reliquary.protocol.signatures import (
    hash_commitments,
    build_commit_binding,
    derive_env_seed,
)


class TestHashCommitments:
    def test_deterministic(self):
        comms = [{"sketch": 42}, {"sketch": 99}]
        assert hash_commitments(comms) == hash_commitments(comms)

    def test_32_bytes(self):
        assert len(hash_commitments([{"sketch": 1}])) == 32

    def test_order_independent_keys(self):
        a = hash_commitments([{"b": 2, "a": 1}])
        b = hash_commitments([{"a": 1, "b": 2}])
        assert a == b

    def test_different_values_differ(self):
        a = hash_commitments([{"sketch": 1}])
        b = hash_commitments([{"sketch": 2}])
        assert a != b


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

    def test_different_model_differ(self):
        comms = [{"sketch": 1}]
        a = build_commit_binding([1], "aa", "model-a", -1, comms)
        b = build_commit_binding([1], "aa", "model-b", -1, comms)
        assert a != b

    def test_handles_0x_prefix(self):
        comms = [{"sketch": 1}]
        a = build_commit_binding([1], "0xaabb", "m", -1, comms)
        b = build_commit_binding([1], "aabb", "m", -1, comms)
        assert a == b


class TestDeriveEnvSeed:
    def test_deterministic(self):
        a = derive_env_seed("5abc", "0xdeadbeef", 0)
        b = derive_env_seed("5abc", "0xdeadbeef", 0)
        assert a == b

    def test_different_index(self):
        a = derive_env_seed("addr", "hash", 0)
        b = derive_env_seed("addr", "hash", 1)
        assert a != b

    def test_different_wallet(self):
        a = derive_env_seed("addr_a", "hash", 0)
        b = derive_env_seed("addr_b", "hash", 0)
        assert a != b

    def test_returns_int(self):
        result = derive_env_seed("addr", "hash", 0)
        assert isinstance(result, int)
        assert result >= 0
