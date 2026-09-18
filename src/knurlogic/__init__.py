"""Knurlogic -- make local model runtimes manageable.

An artifact that a person cannot load is worth nothing. This package resolves
the settings and verifies the environment that stand between a downloaded
model and a working one.

Scope on purpose: it does not detect machines, fit models, or score them.
"""

__version__ = "0.1.0.dev0"

from .artifact import Artifact          # noqa: E402,F401
from .resolve import Resolution, resolve  # noqa: E402,F401
