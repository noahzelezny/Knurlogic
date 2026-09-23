"""The stack is pinned (design D2): the installed mlx and mlx-lm are the
versions pyproject.toml pins, and mlx-lm's server.py is the file the seam's
wraps were verified against.

WHY A DIGEST AS WELL AS VERSIONS. The seam wraps `ResponseGenerator`
methods by name and the vision key rides the prompt trie; v1 of the vision
design cited mlx-lm lines from a build that was not the one serving
(critique B4). A local patch or a fork installed under the same version
string changes server.py without changing the version -- the digest sees it.

One home: the pins live in pyproject.toml ([project] dependencies and
[tool.knurlogic.pins]); this test reads them and holds the install to them.
A failure says what to re-verify before moving the pin.
"""
import hashlib
import importlib.metadata as md
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REVERIFY = (
    "Before moving the pin, re-verify against the new build: "
    "(1) every seam wrap still resolves by name -- ResponseGenerator."
    "_tokenize (returns prompt, segments, segment_types, initial_state), "
    "_serve_single, _generate's fetch_nearest_cache / segment trim / "
    "insert_segments, BatchGenerator; "
    "(2) mlx_lm.models.cache.PromptTrie / LRUPromptCache still accept "
    "non-int hashable tokens (tests/test_vision_key.py runs the real one); "
    "(3) process_message_content still rejects non-text parts the way "
    "the _tokenize wrap expects; "
    "(4) the full suite. Then update pyproject.toml's dependency pins and "
    "[tool.knurlogic.pins] together.")


def _pyproject():
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _dep_pin(deps, name):
    for d in deps:
        m = re.match(rf"^{re.escape(name)}\s*==\s*([^;\s]+)", d)
        if m:
            return m.group(1)
    return None


def test_pyproject_pins_exact_versions_and_agrees_with_itself():
    pp = _pyproject()
    deps = pp["project"]["dependencies"]
    pins = pp["tool"]["knurlogic"]["pins"]
    for name in ("mlx", "mlx-lm"):
        v = _dep_pin(deps, name)
        assert v, f"{name} is not pinned with == in [project] dependencies"
        assert v == pins[name], (
            f"{name}: dependencies say {v}, [tool.knurlogic.pins] says "
            f"{pins[name]} -- move them together")


@pytest.mark.parametrize("dist", ["mlx", "mlx-lm"])
def test_installed_version_is_the_pin(dist):
    try:
        have = md.version(dist)
    except md.PackageNotFoundError:
        pytest.skip(f"{dist} not installed in this interpreter")
    want = _pyproject()["tool"]["knurlogic"]["pins"][dist]
    assert have == want, (
        f"{dist} {have} is installed but knurlogic is pinned to {want}. "
        + REVERIFY)


def test_mlx_lm_server_is_the_verified_file():
    try:
        import mlx_lm
    except ImportError:
        pytest.skip("mlx-lm not installed in this interpreter")
    server = Path(mlx_lm.__file__).parent / "server.py"
    have = hashlib.sha256(server.read_bytes()).hexdigest()
    want = _pyproject()["tool"]["knurlogic"]["pins"]["mlx_lm_server_sha256"]
    assert have == want, (
        f"{server} has sha256 {have}, not the verified {want} (mlx-lm "
        f"{md.version('mlx-lm')}). The seam wraps this file's methods. "
        + REVERIFY)
