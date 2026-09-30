"""engine/families/: one folder per model family, listed explicitly.

The maps the generic engine reads are BUILT from the family manifests;
there are no hand-kept tables. These tests hold the manifests to their
contract and make a listed family tested by being listed.
"""
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from knurlogic.engine import families  # noqa: E402


def test_every_family_folder_is_listed_and_every_listed_one_exists():
    here = Path(families.__file__).parent
    folders = {p.name for p in here.iterdir()
               if (p / "__init__.py").is_file()}
    assert folders == set(families.FAMILIES), (
        f"unlisted: {folders - set(families.FAMILIES)}, "
        f"missing: {set(families.FAMILIES) - folders}")


def test_manifests_are_data_and_import_no_mlx():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; from knurlogic.engine import families; "
         "families.build_maps(); "
         "print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "('mlx', 'mlx_lm', 'mlx_vlm', 'PIL')))"],
        capture_output=True, text=True, env={"PYTHONPATH": str(SRC)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_every_architecture_depends_only_on_its_own_family():
    for m in families.manifests():
        mods = set(m["architectures"])
        for mod, a in m["architectures"].items():
            assert set(a.get("depends_on", [])) <= mods, (m["name"], mod)
        for mod in (m.get("vision") or {}).get("architectures", []):
            assert mod in mods, (m["name"], mod)


def test_the_spellings_that_have_bitten_before_resolve():
    """The tables are built from the manifests now; these are the rows a
    wrong manifest would break silently -- the `_text` spelling (it hid
    vision on all 20 released rungs once), the subclass chains, the GLM
    host, the prefill widths, the two GLM head names."""
    from knurlogic.engine import arch
    from knurlogic.engine.mtp import registry as mreg
    from knurlogic.engine.vision import registry as vreg
    from knurlogic.tuning import settings
    a = arch.ARCH_FOR_MODEL_TYPE
    assert a["qwen3_5_text"] == "qwen3_5" and a["gemma4_text"] == "gemma4_text"
    assert a["gemma4"] == "gemma4" and a["glm5_next_text"] == "glm5_next"
    assert arch.ARCH_DEPENDS_ON == {"qwen3_5_moe": ["qwen3_5"],
                                    "gemma4": ["gemma4_text"]}
    assert arch.ARCH_HOST == {}        # every architecture is mlx_lm's name
    assert {k: v[0] for k, v in settings.PREFILL_CHUNK_MEASURED.items()} == {
        "glm5_next": 2048, "qwen3_5": 4096, "qwen3_5_moe": 2048,
        "qwen4_exp": 2048}
    w, why = settings.prefill_chunk_for("qwen3_5_text")
    assert w == 4096 and why.startswith("measured for qwen3_5: ")
    assert set(vreg.FAMILIES) == {"qwen3_5", "qwen3_5_moe", "qwen4_exp",
                                  "gemma4", "gemma4_text", "glm5_next"}
    assert vreg.family_of("gemma4_text") == "gemma4_text"
    assert vreg.FAMILIES["gemma4_text"] == vreg.FAMILIES["gemma4"]
    assert {"glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5",
            "qwen3_5_moe"} <= set(mreg.FAMILIES)


# --- packaging tripwires -----------------------------------------------------------

def test_pins_are_not_empty_on_a_checkout():
    """Every pin test SKIPS when the pins file is missing -- so pins that
    stop shipping (a moved file, a package-data glob that no longer
    matches) would pass silently. On a checkout they must load."""
    from knurlogic.engine import arch
    assert arch.PINNED_SHA256, "pins.json did not load: pin tests would skip"


def _package_data_globs():
    try:
        import tomllib
    except ModuleNotFoundError:            # 3.10
        import tomli as tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return cfg["tool"]["setuptools"]["package-data"]["knurlogic"]


def test_every_data_file_in_the_package_ships():
    """A non-.py file under src/knurlogic that no package-data glob matches
    is left out of the wheel, silently. pins.json, PROVENANCE.md, rungs.json
    and the page are all such files."""
    globs = _package_data_globs()
    pkg = SRC / "knurlogic"
    missed = []
    for p in pkg.rglob("*"):
        if not p.is_file() or p.suffix in (".py", ".pyc") \
                or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(pkg).as_posix()
        if not any(fnmatch.fnmatch(rel, g) or
                   fnmatch.fnmatch(rel, g.replace("**/", "")) for g in globs):
            missed.append(rel)
    assert not missed, f"data files no package-data glob ships: {missed}"


# --- conformance: being listed is enough to be tested -----------------------------

def test_every_vision_family_has_an_end_to_end_rig():
    """tests/test_vision_e2e.py drives each vision architecture through
    mlx-lm's real server with a tiny model, and runs the encode-twice gate.
    A family whose manifest declares vision but has no rig there fails here,
    instead of quietly shipping untested."""
    sys.path.insert(0, str(ROOT / "tests" / "support"))
    import test_vision_e2e as e2e
    vision = families.build_maps()["vision"]
    assert set(e2e.FAMILIES) <= set(vision)
    # every tower builder is driven by at least one rig
    assert {vision[a] for a in e2e.FAMILIES} == set(vision.values())


def test_every_head_a_manifest_names_imports():
    import importlib
    for name, h in families.build_maps()["heads"].items():
        mod, _, attr = h["head"].partition(":")
        assert hasattr(importlib.import_module(mod), attr), (name, h["head"])
        assert h["cache_semantics"] in ("reassign", "copy"), name


def test_every_architecture_registers_without_mlx_vlm():
    """What `pip install knurlogic` has: no mlx-vlm. Every vendored module,
    in its dependency order, must register and import anyway."""
    code = (
        "import sys\n"
        "class H:\n"
        "    def find_spec(self, n, p=None, t=None):\n"
        "        if n == 'mlx_vlm': raise ImportError('blocked')\n"
        "sys.meta_path.insert(0, H())\n"
        "from knurlogic.engine import register\n"
        "done = register.register(*register.available())\n"
        "import importlib\n"
        "for m in register.available():\n"
        "    importlib.import_module(f'mlx_lm.models.{m}')\n"
        "print(sorted(done))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={"PYTHONPATH": str(SRC)})
    assert out.returncode == 0, out.stderr[-2000:]
    from knurlogic.engine import register
    assert out.stdout.strip() == str(sorted(register.available()))


def test_every_pins_file_parses():
    """arch._load_pins skips a file it cannot read, so ONE corrupt pins.json
    would silently turn its family UNPINNED; the non-empty tripwire above
    only catches all of them failing."""
    import json
    for d in families.architecture_dirs():
        p = d / "pins.json"
        assert p.is_file(), p
        for mod, row in json.loads(p.read_text()).items():
            assert "sha256" in row, (p, mod)


def test_no_serve_module_imports_a_sibling_the_package_shadows():
    """`from . import X` reads the PACKAGE's attribute X. When __init__
    re-exports a function with a submodule's name, that import yields the
    function: `serve` crashed on its first real start that way."""
    import ast
    import types
    from pathlib import Path

    import knurlogic.engine.serve as pkg
    for f in Path(pkg.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 \
                    and node.module is None:
                for a in node.names:
                    assert isinstance(getattr(pkg, a.name, None),
                                      types.ModuleType), (f.name, a.name)


def test_a_head_name_claimed_by_two_families_is_refused(monkeypatch):
    """Like a thinking dialect: two families naming one head would leave
    whichever loaded last deciding the other's cache semantics."""
    import pytest
    two = [{"name": f, "architectures": {f"m_{f}": {
        "model_types": [f"t_{f}"], "head": {"names": ["mtp"], "k": f}}}}
        for f in ("a", "b")]
    monkeypatch.setattr(families, "manifests", lambda: two)
    with pytest.raises(ValueError, match="claimed twice"):
        families.build_maps()


def test_required_modules_follow_dependencies_of_dependencies(monkeypatch):
    from knurlogic.engine import arch
    monkeypatch.setitem(arch.ARCH_FOR_MODEL_TYPE, "t_x", "x")
    monkeypatch.setattr(arch, "ARCH_DEPENDS_ON", {"x": ["y"], "y": ["z"]})
    assert arch.required_modules("t_x") == ["x", "y", "z"]


def test_one_malformed_pin_costs_only_itself(tmp_path, monkeypatch):
    import json
    from knurlogic.engine import arch
    (tmp_path / "pins.json").write_text(json.dumps(
        {"good": {"sha256": "ab"}, "bad": "TODO"}))
    monkeypatch.setattr(arch._families, "architecture_dirs",
                        lambda: [tmp_path])
    assert arch._load_pins() == {"good": "ab"}
