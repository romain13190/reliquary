"""Miner prompt-picking strategy: pull random in-range, skip cooldown."""

import random
import pytest

from reliquary.miner.engine import pick_prompt_idx


class FakeEnv:
    def __len__(self):
        return 100


def test_pick_prompt_in_range():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_prompt_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(0, 95))  # only 5 choices free: 95..99
    for _ in range(20):
        idx = pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)
        assert idx not in cooldown


def test_pick_prompt_all_cooldown_raises():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(100))
    with pytest.raises(RuntimeError, match="no eligible prompt"):
        pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)


def test_engine_default_max_new_tokens_is_protocol_cap(monkeypatch):
    """The env-var override is removed; max_new_tokens is the protocol cap."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
    from reliquary.miner.engine import MiningEngine

    monkeypatch.setenv("RELIQUARY_MAX_NEW_TOKENS", "512")
    # Constructing MiningEngine should NOT pick up the env var.
    # We stub all heavy deps; the goal is just to read the default value.
    eng = MiningEngine.__new__(MiningEngine)  # avoid full __init__
    # Trigger the default-value branch: instantiating with no arg.
    import inspect
    sig = inspect.signature(MiningEngine.__init__)
    default = sig.parameters["max_new_tokens"].default
    assert default == MAX_NEW_TOKENS_PROTOCOL_CAP
