"""Drand-round-anchored batch selection + emission distribution tests."""

from dataclasses import dataclass

from reliquary.validator.batch_selection import select_batch_and_distribute
from reliquary.validator.cooldown import CooldownMap


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    drand_round: int
    merkle_root: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, drand_round, merkle_root=None):
    return FakeSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        drand_round=drand_round,
        merkle_root=merkle_root or hotkey.encode().ljust(32, b"\x00"),
    )


def test_empty_pool_returns_empty():
    cd = CooldownMap(cooldown_windows=50)
    batch, rewards = select_batch_and_distribute(
        submissions=[], b=8, cooldown_map=cd, current_window=100,
    )
    assert batch == [] and rewards == {}


def test_round_1_prompts_win_over_round_2():
    """Earlier drand rounds fill the batch first — that's the whole point."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("late_a", prompt_idx=1, drand_round=2),
        _sub("late_b", prompt_idx=2, drand_round=2),
        _sub("early_a", prompt_idx=3, drand_round=1),
        _sub("early_b", prompt_idx=4, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    # All 4 fit (b=8, only 4 distinct (round, prompt)). Each is its own slot.
    assert len(batch) == 4
    assert all(abs(rewards[hk] - 1/8) < 1e-9 for hk in rewards)
    # Burned pool = 4 unfilled slots × 1/8 = 0.5. Distributed = 0.5.
    assert abs(sum(rewards.values()) - 0.5) < 1e-9


def test_fills_8_slots_chronologically():
    """Two rounds, 10 distinct prompts. First 8 by round order fill the batch."""
    cd = CooldownMap(cooldown_windows=50)
    subs = []
    # Round 1: prompts 0-4 (5 slots)
    for p in range(5):
        subs.append(_sub(f"r1_p{p}", prompt_idx=p, drand_round=1))
    # Round 2: prompts 5-9 (5 slots, only first 3 should fit)
    for p in range(5, 10):
        subs.append(_sub(f"r2_p{p}", prompt_idx=p, drand_round=2))
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 8
    # All round 1 prompts must win.
    batched_prompts = {s.prompt_idx for s in batch}
    assert {0, 1, 2, 3, 4}.issubset(batched_prompts)
    # Pool sums to 8 × 1/8 = 1.0 (full pool used).
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_within_slot_split_K_miners():
    """K miners on the same (round, prompt) split that slot's share."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("alice", prompt_idx=42, drand_round=1),
        _sub("bob", prompt_idx=42, drand_round=1),
        _sub("carol", prompt_idx=42, drand_round=1),
        _sub("dave", prompt_idx=7, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    # 2 distinct (round, prompt) slots, each worth 1/8.
    assert len(batch) == 2
    # Prompt 42 slot: 3 miners split 1/8 → 1/24 each.
    assert abs(rewards["alice"] - 1/24) < 1e-9
    assert abs(rewards["bob"] - 1/24) < 1e-9
    assert abs(rewards["carol"] - 1/24) < 1e-9
    # Prompt 7 slot: 1 miner takes 1/8.
    assert abs(rewards["dave"] - 1/8) < 1e-9


def test_same_prompt_different_rounds_first_wins():
    """When the same prompt appears in multiple rounds, only the earliest
    round's slot is claimed; later rounds' submissions earn nothing."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("early", prompt_idx=42, drand_round=1),
        _sub("late", prompt_idx=42, drand_round=2),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 1
    assert batch[0].hotkey == "early"
    assert "late" not in rewards
    assert abs(rewards["early"] - 1/8) < 1e-9


def test_cooldown_excludes_prompt():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=100)
    subs = [
        _sub("a", prompt_idx=42, drand_round=1),  # cooldown'd
        _sub("b", prompt_idx=7, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=110, pool=1.0,
    )
    assert [s.prompt_idx for s in batch] == [7]
    assert "a" not in rewards
    assert abs(rewards["b"] - 1/8) < 1e-9


def test_cooldown_not_mutated_by_select():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub("hk", prompt_idx=42, drand_round=1)]
    select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100,
    )
    assert cd.is_in_cooldown(42, 100) is False


def test_unfilled_slots_burn_share():
    """If only 5 slots fill, pool * 5/8 is distributed, 3/8 is burned."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i, drand_round=1) for i in range(5)]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 5
    total = sum(rewards.values())
    assert abs(total - 5/8) < 1e-9


def test_sybil_neutral_on_same_round_same_prompt():
    """N sybils on the same (round, prompt) earn the same total as 1 hotkey."""
    cd = CooldownMap(cooldown_windows=50)

    # Case A: lone attacker.
    lone = [
        _sub("attacker", prompt_idx=42, drand_round=1),
        _sub("honest", prompt_idx=7, drand_round=1),
    ]
    _, rewards_a = select_batch_and_distribute(
        submissions=lone, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    # Case B: 5 sybils on the same prompt-round.
    sybils = [
        _sub(f"sybil{i}", prompt_idx=42, drand_round=1) for i in range(5)
    ]
    sybils.append(_sub("honest", prompt_idx=7, drand_round=1))
    _, rewards_b = select_batch_and_distribute(
        submissions=sybils, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    attacker_a = rewards_a["attacker"]
    attacker_b_total = sum(v for k, v in rewards_b.items() if k.startswith("sybil"))
    assert abs(attacker_a - attacker_b_total) < 1e-9
    # The honest miner on a different prompt is unaffected.
    assert abs(rewards_a["honest"] - rewards_b["honest"]) < 1e-9


def test_canonical_representative_deterministic_across_input_order():
    """Validators with different network-arrival orders pick the same
    representative submission for each (round, prompt) slot."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=1, drand_round=1) for i in range(5)]

    batch_a, _ = select_batch_and_distribute(
        submissions=list(subs), b=8, cooldown_map=cd, current_window=100,
    )
    batch_b, _ = select_batch_and_distribute(
        submissions=list(reversed(subs)), b=8, cooldown_map=cd, current_window=100,
    )
    assert [s.hotkey for s in batch_a] == [s.hotkey for s in batch_b]


def test_pool_scaling():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i, drand_round=1) for i in range(4)]

    _, rewards_1 = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    _, rewards_100 = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=100.0,
    )
    for hk in rewards_1:
        assert abs(rewards_100[hk] - rewards_1[hk] * 100.0) < 1e-7
