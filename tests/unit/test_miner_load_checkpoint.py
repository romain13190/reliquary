"""_load_checkpoint: reload both hf_model and vllm_model from a local path."""

from unittest.mock import MagicMock, patch

import pytest


def _make_hf_mock(name):
    m = MagicMock(name=name)
    m.to.return_value = m
    m.eval.return_value = m
    return m


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


def test_load_checkpoint_swaps_both_models(mock_engine):
    """After successful reload, both hf_model and vllm_model are swapped."""
    mock_hf = _make_hf_mock("new_hf")
    mock_gen = _make_hf_mock("new_gen")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[mock_hf, mock_gen],
    ):
        result = mock_engine._load_checkpoint("/tmp/checkpoint-5")

    assert mock_engine.hf_model is mock_hf
    assert mock_engine.vllm_model is mock_gen
    assert result is mock_hf


def test_load_checkpoint_short_circuits_on_same_path(mock_engine):
    """Calling twice with the same path doesn't reload."""
    mock_hf = _make_hf_mock("new_hf")
    mock_gen = _make_hf_mock("new_gen")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[mock_hf, mock_gen],
    ) as mock_from_pretrained:
        mock_engine._load_checkpoint("/tmp/checkpoint-5")
        mock_engine._load_checkpoint("/tmp/checkpoint-5")

    # from_pretrained should have been called exactly twice (once per model, first call only)
    assert mock_from_pretrained.call_count == 2


def test_load_checkpoint_hf_load_failure_keeps_old_models(mock_engine):
    """If AutoModelForCausalLM.from_pretrained raises on hf load, old models stay."""
    original_hf = mock_engine.hf_model
    original_vllm = mock_engine.vllm_model

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=RuntimeError("HF load failed"),
    ):
        result = mock_engine._load_checkpoint("/tmp/bad")

    assert mock_engine.hf_model is original_hf
    assert mock_engine.vllm_model is original_vllm
    assert result is original_hf


def test_load_checkpoint_vllm_load_failure_sets_none(mock_engine):
    """If the second from_pretrained (gen GPU) raises, hf is swapped but vllm is None."""
    mock_hf = _make_hf_mock("new_hf")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[mock_hf, RuntimeError("gen GPU OOM")],
    ):
        result = mock_engine._load_checkpoint("/tmp/vllm_broken")

    # hf got swapped (new one is active)
    assert mock_engine.hf_model is mock_hf
    # vllm_model becomes None after gen-GPU load failure
    assert mock_engine.vllm_model is None, (
        "After vllm load failure, vllm_model is None — miner is broken until "
        "the next successful pull"
    )
    # _loaded_checkpoint_path reset so next pull retries
    assert mock_engine._loaded_checkpoint_path is None
    # The function still returns the new hf_model
    assert result is mock_hf
