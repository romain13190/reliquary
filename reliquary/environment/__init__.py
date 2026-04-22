"""Reliquary environment module.

Provides the Environment protocol and a factory function to instantiate
concrete environments by name.
"""

from reliquary.environment.base import Environment
from reliquary.environment.math import MATHEnvironment


def load_environment(name: str) -> Environment:
    """Return a concrete Environment instance for the given *name*.

    Raises:
        ValueError: if *name* is not a recognised environment.
    """
    if name == "math":
        return MATHEnvironment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = [
    "Environment",
    "load_environment",
]
