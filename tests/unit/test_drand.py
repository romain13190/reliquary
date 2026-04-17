from reliquary.infrastructure.drand import get_mock_beacon, get_round_at_time, get_beacon


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

    def test_randomness_is_64_chars(self):
        b = get_mock_beacon()
        assert len(b["randomness"]) == 64


class TestGetRoundAtTime:
    def test_genesis(self):
        # quicknet genesis = 1692803367, period = 3
        r = get_round_at_time(1692803367)
        assert r == 1

    def test_after_genesis(self):
        r = get_round_at_time(1692803367 + 30)
        assert r == 11  # 1 + 30/3

    def test_before_genesis(self):
        r = get_round_at_time(0)
        assert r == 0


class TestGetBeacon:
    def test_mock_mode(self):
        b = get_beacon(use_drand=False)
        assert "randomness" in b
        assert b["source"] == "mock"

    def test_mock_has_all_fields(self):
        b = get_beacon(use_drand=False)
        assert "round" in b
        assert "randomness" in b
        assert "source" in b
