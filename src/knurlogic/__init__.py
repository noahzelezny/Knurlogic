"""Knurlogic -- make local model runtimes manageable.

An artifact that a person cannot load is worth nothing. This package resolves
the settings and verifies the environment that stand between a downloaded
model and a working one.

Scope on purpose: it does not detect machines, fit models, or score them.
"""

__version__ = "0.1.0.dev0"

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import Resolution, resolve
