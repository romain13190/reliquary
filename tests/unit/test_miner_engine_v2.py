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
