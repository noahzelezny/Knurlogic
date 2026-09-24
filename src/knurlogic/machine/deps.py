"""What knurlogic stands on, per interpreter, and whether each piece is stock.

`knurlogic deps` and the MCP `deps` tool both answer from here. The question
is never "is mlx installed" -- it is WHICH mlx, in WHICH environment, because
this stack runs in more than one: knurlogic serves from its own interpreter,
and exo runs in its own (python 3.13, its own rust bindings). A fix that lives
in one of them is absent from the other, and nothing else says so.

Every verdict is read off an ARTIFACT of the fix, never off a version string.
A version names a build; it does not say what is in it. So:

  jaccl self-heal   the string JACCL_COLLECTIVE_TIMEOUT_MS inside the
                    installed libjaccl.dylib -- the fix's own env read,
                    compiled in (noahzelezny/mlx, branch jaccl-selfheal)
  mlx-lm fork       mlx_lm/models/qwen4_exp.py in the installed package
                    (noahzelezny/mlx-lm, exo-qwen4-exp)
  exo fork          exo/worker/engines/mlx/mtp/ in the installed package
                    (noahzelezny/exo, mtp-stage1)

Each interpreter is asked in a SUBPROCESS running only the stdlib. That is
not caution for its own sake: this module runs in knurlogic's interpreter,
and a check that imports mlx here answers for this interpreter every time --
the exact mistake that once made a version probe over several environments
report the system Python on every iteration.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: What each piece is, where the fork lives, and what it carries that
#: upstream does not. One home for this; the README and CONTEXT.md point here.
PIECES = {
    "mlx": {
        "role": "array framework and Metal kernels; jaccl is its RDMA ring",
        "fork": "github.com/noahzelezny/mlx @ jaccl-selfheal "
                "(local: ~/mlx-jaccl-fork; wheels: ~/mlx-wheels)",
        "carries": "ring self-heal: a no-progress deadline on RDMA "
                   "collectives, read live from JACCL_COLLECTIVE_TIMEOUT_MS, "
                   "so a wedged ring throws instead of hanging every rank",
        "needed_for": "multi-node over Thunderbolt RDMA (jaccl). One box and "
                      "TCP rings do not touch it.",
        "portable": False,
        "why_not_ported": "compiled C++ inside mlx itself; it ships as a "
                          "wheel, not as Python a package can carry",
    },
    "mlx-lm": {
        "role": "model architectures, the loader, and the OpenAI server "
                "`knurlogic serve` runs",
        "fork": "github.com/noahzelezny/mlx-lm @ exo-qwen4-exp",
        "carries": "the qwen4_exp architecture and PipelineMixin on "
                   "qwen3_5 / qwen3_5_moe",
        "needed_for": "nothing, for knurlogic: the architectures it needs "
                      "are vendored and pinned by digest "
                      "(engine/families/*/architecture/PROVENANCE.md). exo still installs the "
                      "fork, because exo imports them from mlx-lm directly.",
        "portable": True,
        "why_not_ported": "",
    },
    "mlx-vlm": {
        "role": "multimodal architectures; glm5_next is looked up here",
        "fork": "",
        "carries": "",
        "needed_for": "nothing, for knurlogic. Vision for every released "
                      "family, GLM-5.3 included, is vendored under "
                      "engine/families/*/vision/ with provenance; glm5_siblings() is "
                      "empty and a test keeps it so. It was needed until "
                      "2026-09-23, when glm5_next imported nine of its "
                      "modules.",
        "portable": True,
        "why_not_ported": "",
    },
    "exo": {
        "role": "placement, sharding and the ring for multi-node; knurlogic "
                "wraps it and never requires it",
        "fork": "github.com/noahzelezny/exo @ mtp-stage1",
        "carries": "MTP drafting (sequential and batched), the per-family "
                   "prefill table, EXO_MLX_CACHE_LIMIT_GB and "
                   "EXO_MLX_MEM_LIMIT_GB, KV-pool token budget, jaccl "
                   "phase-aware deadline, subnet pinning, VQ codebook "
                   "sharding",
        "needed_for": "multi-node. One box needs no exo at all.",
        "portable": True,
        "why_not_ported": "MTP, batch MTP and the prefill widths are ported "
                          "(knurlogic.engine.mtp; each family's manifest in "
                          "knurlogic.engine.families). Placement, sharding "
                          "and networking still run through exo today; "
                          "knurlogic is replacing them (docs/PLAN.md, "
                          "'Next: replace exo').",
    },
}

#: Run by each interpreter. Stdlib only, and it must stay that way: it is the
#: probe, and a probe that imports what it measures measures its own import.
_PROBE = r"""
import importlib.metadata as md, importlib.util as u, json, os, sys
SIBLINGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
out = {"python": sys.version.split()[0], "executable": sys.executable}
def dist(n):
    try:
        d = md.distribution(n)
    except md.PackageNotFoundError:
        return None
    du = d.read_text("direct_url.json")
    return {"version": d.version, "source": json.loads(du) if du else None}
def pkgdir(mod):
    s = u.find_spec(mod)
    return list(s.submodule_search_locations)[0] if s and s.submodule_search_locations else None
def contains(path, needle):
    try:
        with open(path, "rb") as f:
            return needle in f.read()
    except OSError:
        return None
for n, mod in (("mlx", "mlx"), ("mlx-lm", "mlx_lm"), ("mlx-vlm", "mlx_vlm"), ("exo", "exo")):
    d = dist(n)
    if d:
        d["dir"] = pkgdir(mod)
    out[n] = d
m = out.get("mlx")
if m and m["dir"]:
    lib = os.path.join(m["dir"], "lib", "libjaccl.dylib")
    m["jaccl_selfheal"] = contains(lib, b"JACCL_COLLECTIVE_TIMEOUT_MS") if os.path.exists(lib) else None
l = out.get("mlx-lm")
if l and l["dir"]:
    l["fork"] = os.path.exists(os.path.join(l["dir"], "models", "qwen4_exp.py"))
v = out.get("mlx-vlm")
if v and v["dir"]:
    base = os.path.join(v["dir"], "models")
    v["glm5_missing"] = [m for m in SIBLINGS
        if not (os.path.exists(os.path.join(base, *m.split(".")) + ".py")
                or os.path.isdir(os.path.join(base, *m.split("."))))]
e = out.get("exo")
if e and e["dir"]:
    e["fork"] = os.path.isdir(os.path.join(e["dir"], "worker", "engines", "mlx", "mtp"))
print(json.dumps(out))
"""


def glm5_siblings() -> list:
    """The mlx_vlm modules the vendored glm5_next imports, read off its
    source. Derived every time, because a list written down beside the
    vendoring went stale: it named 8, the code imports 9 different ones."""
    import re
    root = (Path(__file__).resolve().parents[1] / "engine" / "families"
            / "glm5" / "architecture" / "glm5_next")
    found = set()
    for f in root.glob("*.py"):
        for m in re.finditer(r"^\s*from \.\.([\w.]+) import", f.read_text(),
                             re.M):
            found.add(m.group(1))
    return sorted(found)


def probe(python: str) -> dict:
    """What this interpreter has installed, asked of the interpreter."""
    try:
        r = subprocess.run([python, "-c", _PROBE, json.dumps(glm5_siblings())],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"executable": python, "error": f"{type(e).__name__}: {e}"}


def exo_python() -> str | None:
    """exo's interpreter: the one running it, else the one `exo` names."""
    try:
        ps = subprocess.run(["ps", "-xo", "command="], capture_output=True,
                            text=True, timeout=5).stdout
        for ln in ps.splitlines():
            parts = ln.split()
            if (len(parts) >= 2 and "python" in os.path.basename(parts[0])
                    and os.path.basename(parts[1]) == "exo"):
                return parts[0]
    except Exception:
        pass
    exe = shutil.which("exo")
    if exe:
        try:
            first = Path(exe).read_text(errors="replace").splitlines()[0]
            if first.startswith("#!"):
                return first[2:].strip().split()[0]
        except Exception:
            pass
    return None


def _verdicts(env: dict) -> list:
    """(piece, state, detail) for one interpreter, read off artifacts."""
    out = []
    m = env.get("mlx")
    if m:
        heal = m.get("jaccl_selfheal")
        out.append(("mlx", m["version"],
                    "jaccl self-heal: YES (fork)" if heal else
                    "jaccl self-heal: no -- stock ring; a wedged RDMA "
                    "collective hangs every rank" if heal is False else
                    "no libjaccl in this build"))
    else:
        out.append(("mlx", "-", "not installed"))
    lm = env.get("mlx-lm")
    out.append(("mlx-lm", lm["version"] if lm else "-",
                ("fork (carries qwen4_exp)" if lm.get("fork") else "stock")
                if lm else "not installed"))
    vlm = env.get("mlx-vlm")
    if vlm:
        miss = vlm.get("glm5_missing") or []
        out.append(("mlx-vlm", vlm["version"],
                    "optional: knurlogic vendors its own vision code"
                    if not miss else
                    f"knurlogic's vendored glm5_next (taken from mlx-vlm "
                    f"0.7.1) cannot load here: missing {', '.join(miss)}. "
                    f"exo loads its own copy, so this is about serving "
                    f"GLM-5.3 through knurlogic from this interpreter"))
    else:
        out.append(("mlx-vlm", "-", "not installed -- not needed; vision is vendored"))
    ex = env.get("exo")
    if ex:
        src = (ex.get("source") or {}).get("url", "")
        out.append(("exo", ex["version"],
                    ("fork (carries MTP)" if ex.get("fork") else "stock")
                    + (f", editable from {src[7:]}" if src.startswith("file://")
                       else "")))
    return out


def survey() -> dict:
    """Both interpreters, probed, with verdicts."""
    envs = {"knurlogic": probe(sys.executable)}
    exo_py = exo_python()
    if exo_py and os.path.realpath(exo_py) != os.path.realpath(sys.executable):
        envs["exo"] = probe(exo_py)
    doc = {"interpreters": {}, "pieces": PIECES}
    for name, env in envs.items():
        doc["interpreters"][name] = {
            "executable": env.get("executable"),
            "python": env.get("python"),
            "error": env.get("error"),
            "pieces": [{"piece": p, "version": v, "verdict": d}
                       for p, v, d in _verdicts(env)] if "error" not in env
            else [],
        }
    if "exo" not in envs:
        doc["note"] = ("no exo interpreter found (not running, not on PATH): "
                       "one box needs none")
    return doc


def render(doc: dict) -> str:
    lines = []
    for name, env in doc["interpreters"].items():
        lines.append(f"{name}  {env['executable']}  (python {env['python']})")
        if env.get("error"):
            lines.append(f"  could not probe: {env['error']}")
        for p in env["pieces"]:
            lines.append(f"  {p['piece']:<8} {p['version']:<32} {p['verdict']}")
        lines.append("")
    if doc.get("note"):
        lines.append(doc["note"])
    return "\n".join(lines).rstrip()


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="knurlogic deps",
                                description="what this stack stands on, per "
                                            "interpreter, and what is stock")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    doc = survey()
    print(json.dumps(doc, indent=1) if a.json else render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
