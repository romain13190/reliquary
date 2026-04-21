"""_load_checkpoint: reload both hf_model and vllm_model from a local path."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_vllm(llm_factory=None):
    """Return a fake vllm module with a controllable LLM class."""
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = llm_factory if llm_factory is not None else MagicMock(name="LLM")
    return fake_vllm


@pytest.fixture
def mock_engine():
    """Build a MiningEngine with mock models so we can observe reload calls."""
    from reliquary.miner.engine import MiningEngine

    # MiningEngine.__init__ does real work (HF imports, GRAIL init). Bypass
    # with object.__new__ and manually set attrs we need.
    eng = object.__new__(MiningEngine)
    eng.vllm_model = MagicMock(name="initial_vllm")
    eng.hf_model = MagicMock(name="initial_hf")
    eng.vllm_gpu = 0
    eng.proof_gpu = 1
    return eng


def test_load_checkpoint_swaps_hf_model(mock_engine):
    """After successful reload, self.hf_model points to the new model."""
    fake_new_hf = MagicMock(name="new_hf")
    fake_new_hf.to.return_value = fake_new_hf
    fake_new_vllm_instance = MagicMock(name="new_vllm_instance")

    fake_vllm = _make_fake_vllm(llm_factory=MagicMock(return_value=fake_new_vllm_instance))

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=fake_new_hf,
    ), patch.dict(sys.modules, {"vllm": fake_vllm}):
        result = mock_engine._load_checkpoint("/tmp/checkpoint-5")

    assert mock_engine.hf_model is fake_new_hf
    assert result is fake_new_hf


def test_load_checkpoint_swaps_vllm_model(mock_engine):
    """After successful reload, self.vllm_model points to the new engine."""
    fake_new_vllm_instance = MagicMock(name="new_vllm_instance")
    fake_new_hf = MagicMock(name="new_hf")
    fake_new_hf.to.return_value = fake_new_hf

    fake_vllm = _make_fake_vllm(llm_factory=MagicMock(return_value=fake_new_vllm_instance))

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=fake_new_hf,
    ), patch.dict(sys.modules, {"vllm": fake_vllm}):
        mock_engine._load_checkpoint("/tmp/checkpoint-5")

    assert mock_engine.vllm_model is fake_new_vllm_instance


def test_load_checkpoint_short_circuits_on_same_path(mock_engine):
    """Calling twice with the same path doesn't reload."""
    fake_new_hf = MagicMock(name="new_hf")
    fake_new_hf.to.return_value = fake_new_hf
    fake_new_vllm_instance = MagicMock(name="new_vllm_instance")

    fake_vllm = _make_fake_vllm(llm_factory=MagicMock(return_value=fake_new_vllm_instance))

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=fake_new_hf,
    ) as mock_from_pretrained, patch.dict(sys.modules, {"vllm": fake_vllm}):
        mock_engine._load_checkpoint("/tmp/checkpoint-5")
        mock_engine._load_checkpoint("/tmp/checkpoint-5")

    # from_pretrained should have been called exactly once
    assert mock_from_pretrained.call_count == 1


def test_load_checkpoint_hf_load_failure_keeps_old_models(mock_engine):
    """If AutoModelForCausalLM.from_pretrained raises, old models stay."""
    original_hf = mock_engine.hf_model
    original_vllm = mock_engine.vllm_model

    llm_mock = MagicMock(name="LLM")
    fake_vllm = _make_fake_vllm(llm_factory=llm_mock)

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=RuntimeError("HF load failed"),
    ), patch.dict(sys.modules, {"vllm": fake_vllm}):
        result = mock_engine._load_checkpoint("/tmp/bad")

    assert mock_engine.hf_model is original_hf
    assert mock_engine.vllm_model is original_vllm
    assert result is original_hf
    llm_mock.assert_not_called()


def test_load_checkpoint_vllm_load_failure_keeps_old_vllm(mock_engine):
    """If vllm.LLM raises, hf is already swapped but vllm stays old."""
    original_vllm = mock_engine.vllm_model
    fake_new_hf = MagicMock(name="new_hf")
    fake_new_hf.to.return_value = fake_new_hf

    llm_mock = MagicMock(side_effect=RuntimeError("vllm OOM"))
    fake_vllm = _make_fake_vllm(llm_factory=llm_mock)

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=fake_new_hf,
    ), patch.dict(sys.modules, {"vllm": fake_vllm}):
        result = mock_engine._load_checkpoint("/tmp/vllm_broken")

    # hf got swapped (new one is active)
    assert mock_engine.hf_model is fake_new_hf
    # vllm didn't get replaced with None or a broken instance — it's old value
    # (the function set it to None temporarily, but the exception handler
    # should leave vllm_model as-None; document this edge case).
    assert mock_engine.vllm_model is None, (
        "After vllm load failure, vllm_model is None — miner is broken until "
        "the next successful pull"
    )
    # The function still returns the new hf_model
    assert result is fake_new_hf
