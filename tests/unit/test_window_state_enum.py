"""WindowState enum + RejectReason.WRONG_CHECKPOINT availability."""

from reliquary.protocol.submission import RejectReason, WindowState


def test_window_state_values():
    assert WindowState.OPEN.value == "open"
    assert WindowState.TRAINING.value == "training"
    assert WindowState.PUBLISHING.value == "publishing"
    assert WindowState.READY.value == "ready"


def test_window_state_set_membership():
    submitting_states = {WindowState.OPEN}
    not_submitting = {WindowState.TRAINING, WindowState.PUBLISHING, WindowState.READY}
    assert submitting_states.isdisjoint(not_submitting)
    assert len(submitting_states | not_submitting) == 4


def test_wrong_checkpoint_reject_reason_present():
    assert RejectReason.WRONG_CHECKPOINT.value == "wrong_checkpoint"


def test_window_state_serialises_to_string():
    state = WindowState.OPEN
    assert str(state.value) == "open"
