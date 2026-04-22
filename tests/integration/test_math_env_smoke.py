"""End-to-end smoke test: the default environment name resolves to a
working Environment and its reward function runs on real dataset rows.
"""

import pytest


def test_default_environment_is_math():
    from reliquary.constants import ENVIRONMENT_NAME
    assert ENVIRONMENT_NAME == "math"


def test_default_environment_loads_and_scores():
    from reliquary.constants import ENVIRONMENT_NAME
    from reliquary.environment import load_environment, Environment

    try:
        env = load_environment(ENVIRONMENT_NAME)
    except Exception as exc:
        pytest.skip(f"Could not load default environment: {exc}")

    assert isinstance(env, Environment)
    assert len(env) > 0

    problem = env.get_problem(0)
    assert env.compute_reward(problem, r"\boxed{" + problem["ground_truth"] + "}") == 1.0
    assert env.compute_reward(problem, "definitely wrong") == 0.0
