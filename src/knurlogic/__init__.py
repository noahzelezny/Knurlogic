"""Knurlogic -- serve local models on Apple Silicon.

An artifact that a person cannot load is worth nothing. This package serves
models (an OpenAI- and Anthropic-compatible server, a page for people, an
MCP for agents) and resolves what stands between a downloaded model and a
working one: the environment, the settings, and whether it fits.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version


def _source_version(here=None):
    """The version in the pyproject.toml beside `src/` when this package
    runs from a source tree (src/knurlogic/..), else None. A source copy
    run by PYTHONPATH has no metadata of its own, and any it finds belongs
    to some other install (one test install reported 0.1.0.dev0)."""
    import re
    from pathlib import Path
    pkg = Path(here or __file__).resolve().parent
    if pkg.parent.name != "src":
        return None
    try:
        text = (pkg.parent.parent / "pyproject.toml").read_text()
    except OSError:
        return None
    section = None
    for line in text.splitlines():
        head = re.match(r"\s*\[([^\]]+)\]\s*$", line)
        if head:
            section = head.group(1).strip()
            continue
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m and section == "project":
            return m.group(1)
    return None


# pyproject.toml is the one place the version is written; a second copy
# here said 0.1.0.dev0 on the 0.1.0 wheel
__version__ = _source_version()
if __version__ is None:
    try:
        __version__ = _version("knurlogic")
    except PackageNotFoundError:       # neither a source tree nor installed
        __version__ = "0+unknown"

# public API, after __version__
from knurlogic.machine.artifact import Artifact  # noqa: E402,F401
from knurlogic.tuning.resolve import Resolution, resolve  # noqa: E402,F401
