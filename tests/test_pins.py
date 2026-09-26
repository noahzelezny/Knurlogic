"""The stack is pinned (design D2): the installed mlx and mlx-lm are the
versions pyproject.toml pins, and the mlx-lm files knurlogic builds on are
the files it was verified against.

WHY A DIGEST AS WELL AS VERSIONS. MTPBatchGenerator subclasses mlx-lm's
BatchGenerator and reads its internals (the queue tuple, the response
shapes), and the image pins read LRUPromptCache's order. A local patch or
a fork installed under the same version string changes those files
without changing the version -- the digest sees it.

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
    "(1) BatchGenerator: `_unprocessed_sequences` entries are still (uid, "
    "segments, max_tokens, cache, all_tokens, sampler, procs, sm), "
    "insert_segments/extract_cache/remove/close keep their contracts, and "
    "PromptProcessingBatch/GenerationBatch.Response keep their fields; "
    "(2) LRUPromptCache: fetch_nearest_cache/insert_cache/trim_to and "
    "`_lru._lrus` (engine/vision/cachehook.py), and non-int hashable tokens "
    "(tests/test_vision_key.py runs the real one); (3) SequenceStateMachine "
    "and TokenizerWrapper's think/tool fields (engine/runtime/request.py); "
    "(4) the full suite and tests/api on a served model. Then update "
    "pyproject.toml's dependency pins and [tool.knurlogic.pins] together.")


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


@pytest.mark.parametrize("rel,key", [
    ("generate.py", "mlx_lm_generate_sha256"),
    ("models/cache.py", "mlx_lm_cache_sha256")])
def test_the_mlx_lm_files_we_build_on_are_the_verified_ones(rel, key):
    try:
        import mlx_lm
    except ImportError:
        pytest.skip("mlx-lm not installed in this interpreter")
    f = Path(mlx_lm.__file__).parent / rel
    have = hashlib.sha256(f.read_bytes()).hexdigest()
    want = _pyproject()["tool"]["knurlogic"]["pins"][key]
    assert have == want, (
        f"{f} has sha256 {have}, not the verified {want} (mlx-lm "
        f"{md.version('mlx-lm')}). " + REVERIFY)
