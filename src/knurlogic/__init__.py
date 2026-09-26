"""Knurlogic -- serve local models on Apple Silicon.

An artifact that a person cannot load is worth nothing. This package serves
models (an OpenAI- and Anthropic-compatible server, a page for people, an
MCP for agents) and resolves what stands between a downloaded model and a
working one: the environment, the settings, and whether it fits.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # pyproject.toml is the one place the version is written; a second copy
    # here said 0.1.0.dev0 on the 0.1.0 wheel
    __version__ = _version("knurlogic")
except PackageNotFoundError:           # a source tree that is not installed
    __version__ = "0+unknown"

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import Resolution, resolve
