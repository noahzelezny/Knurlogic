"""engine/families/: one folder per model family, listed explicitly.

The maps the generic engine reads are BUILT from the family manifests.
Until every reader has switched over, the built maps must equal the
hand-kept tables they replace -- the migration's safety net.
"""
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    assert arch.ARCH_HOST == {"glm5_next": "mlx_vlm"}
    assert settings.PREFILL_CHUNK_BY_FAMILY == {
        "glm5_next": 2048, "qwen3_5": 4096, "qwen3_5_moe": 4096}
    w, why = settings.prefill_chunk_for("qwen3_5_text")
    assert w == 4096 and why.startswith("measured for qwen3_5: ")
    assert set(vreg.FAMILIES) == {"qwen3_5", "qwen3_5_moe", "qwen4_exp",
                                  "gemma4", "glm5_next"}
    assert {"glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5",
            "qwen3_5_moe"} <= set(mreg.FAMILIES)


# --- packaging tripwires -----------------------------------------------------------

def test_pins_are_not_empty_on_a_checkout():
    """Every pin test SKIPS when the pins file is missing -- so pins that
    stop shipping (a moved file, a package-data glob that no longer
    matches) would pass silently. On a checkout they must load."""
    from knurlogic.engine import arch
    assert arch.PINNED_SHA256, "PINS.json did not load: pin tests would skip"


def _package_data_globs():
    try:
        import tomllib
    except ModuleNotFoundError:            # 3.10
        import tomli as tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return cfg["tool"]["setuptools"]["package-data"]["knurlogic"]


def test_every_data_file_in_the_package_ships():
    """A non-.py file under src/knurlogic that no package-data glob matches
    is left out of the wheel, silently. PINS.json, PROVENANCE.md, rungs.json
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
