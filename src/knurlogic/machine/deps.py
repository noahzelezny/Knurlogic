"""What knurlogic stands on, and whether each piece is stock.

`knurlogic deps` and the MCP `deps` tool answer from here. Every verdict is
read off an artifact of the fix, never off a version string:

  jaccl self-heal   JACCL_COLLECTIVE_TIMEOUT_MS inside the installed
                    libjaccl.dylib (mlx fork, branch jaccl-selfheal)
  mlx-lm fork       mlx_lm/models/qwen4_exp.py in the installed package

The interpreter is asked in a subprocess running only the stdlib: a probe
that imports what it measures measures its own import, and machine/ may
not import mlx.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: What each piece is, where the fork lives, and what it carries that
#: upstream does not. One home for this; the README points here.
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
        "role": "the library under knurlogic's server: model classes, the "
                "loader, tokenizer, caches, BatchGenerator",
        "fork": "github.com/noahzelezny/mlx-lm @ exo-qwen4-exp",
        "carries": "the qwen4_exp architecture and PipelineMixin on "
                   "qwen3_5 / qwen3_5_moe",
        "needed_for": "nothing, for knurlogic: the architectures it needs "
                      "are vendored and pinned by digest "
                      "(engine/families/*/architecture/PROVENANCE.md).",
        "portable": True,
        "why_not_ported": "",
    },
}

#: Run by each interpreter. Stdlib only, and it must stay that way: it is the
#: probe, and a probe that imports what it measures measures its own import.
_PROBE = r"""
import importlib.metadata as md, importlib.util as u, json, os, sys
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
    if s and s.submodule_search_locations:
        return list(s.submodule_search_locations)[0]
    return None
def contains(path, needle):
    try:
        with open(path, "rb") as f:
            return needle in f.read()
    except OSError:
        return None
for n, mod in (("mlx", "mlx"), ("mlx-lm", "mlx_lm")):
    d = dist(n)
    if d:
        d["dir"] = pkgdir(mod)
    out[n] = d
m = out.get("mlx")
if m and m["dir"]:
    lib = os.path.join(m["dir"], "lib", "libjaccl.dylib")
    m["jaccl_selfheal"] = (contains(lib, b"JACCL_COLLECTIVE_TIMEOUT_MS")
                           if os.path.exists(lib) else None)
l = out.get("mlx-lm")
if l and l["dir"]:
    l["fork"] = os.path.exists(os.path.join(l["dir"], "models", "qwen4_exp.py"))
print(json.dumps(out))
"""


def glm5_siblings() -> list:
    """Modules OUTSIDE its own package that the vendored glm5_next imports
    (relative `from ..X`), read off its source -- empty, and a test keeps it
    so: it registers under mlx_lm's name, where such an import would reach
    into mlx-lm. Derived every time, because a list written down beside the
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
        r = subprocess.run([python, "-c", _PROBE],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return {"executable": python, "error": f"{type(e).__name__}: {e}"}


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
    return out


def survey() -> dict:
    """knurlogic's interpreter, probed, with verdicts."""
    envs = {"knurlogic": probe(sys.executable)}
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
                                description="what this stack stands on, "
                                            "and what is stock")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    doc = survey()
    print(json.dumps(doc, indent=1) if a.json else render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
