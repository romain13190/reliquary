"""Bootstrap phase: wider zone, shorter cooldown, smaller M for first
BOOTSTRAP_WINDOWS windows after SUBNET_START_BLOCK."""

from reliquary.constants import BOOTSTRAP_WINDOWS
from reliquary.validator.service import is_bootstrap_window


def test_bootstrap_active_at_start():
    assert is_bootstrap_window(window_start=100, subnet_start=100) is True


def test_bootstrap_active_within_horizon():
    assert is_bootstrap_window(
        window_start=100 + BOOTSTRAP_WINDOWS - 1, subnet_start=100
    ) is True


def test_bootstrap_expires_at_horizon():
    assert is_bootstrap_window(
        window_start=100 + BOOTSTRAP_WINDOWS, subnet_start=100
    ) is False


def test_bootstrap_inactive_before_start():
    assert is_bootstrap_window(window_start=50, subnet_start=100) is False
