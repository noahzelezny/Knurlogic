"""engine/families/ -- one folder per model family: everything knurlogic
knows about that family, in one place.

  FAMILIES   the explicit list below. A family folder that is not listed is
             a test failure, not a silent skip (tests/test_families.py).
  <family>/__init__.py   MANIFEST, plain data: dicts, strings, numbers.
             Anything that runs is named by a "module:attr" string and
             imported only when used, so asking what a family supports
             imports no mlx.

Adding a family is one folder and one line here. Manifest shape and the
full checklist: docs/design/families.md.

Stdlib only.
"""
from __future__ import annotations

import importlib
from pathlib import Path

HERE = Path(__file__).parent

#: Every family, by package name under engine/families/.
FAMILIES = ("qwen", "gemma4", "glm5", "deepseek")


def manifests() -> list:
    return [importlib.import_module(f"{__name__}.{f}").MANIFEST
            for f in FAMILIES]


def build_maps() -> dict:
    """The tables the generic engine reads, built from the manifests."""
    arch_for_type, host, depends, prefill = {}, {}, {}, {}
    vision, heads, thinking, kvq = {}, {}, {}, {}
    for m in manifests():
        for mod, a in m["architectures"].items():
            for t in a["model_types"]:
                if t in arch_for_type:
                    raise ValueError(f"model_type {t!r} claimed twice")
                arch_for_type[t] = mod
            if a.get("host", "mlx_lm") != "mlx_lm":
                host[mod] = a["host"]
            if a.get("depends_on"):
                depends[mod] = list(a["depends_on"])
            if a.get("prefill_chunk"):
                prefill[mod] = a["prefill_chunk"]
            if a.get("kv_quant"):
                kvq[mod] = a["kv_quant"]
            h = a.get("head")
            if h:
                spec = {k: v for k, v in h.items() if k != "names"}
                for n in h["names"]:
                    if n in heads:
                        raise ValueError(f"drafting head {n!r} claimed twice")
                    heads[n] = spec
        for d, spec in (m.get("thinking") or {}).items():
            if d in thinking:
                raise ValueError(f"thinking dialect {d!r} claimed twice")
            thinking[d] = spec
        v = m.get("vision")
        if v:
            for mod in v["architectures"]:
                vision[mod] = v["build"]
    return {"arch_for_model_type": arch_for_type, "arch_host": host,
            "arch_depends_on": depends, "prefill_chunk": prefill,
            "vision": vision, "heads": heads, "thinking": thinking,
            "kv_quant": kvq}


def architecture_dir(family: str) -> Path:
    """Where a family's vendored architecture modules live."""
    return HERE / family / "architecture"


def family_of_module(module: str):
    """The family whose manifest lists this architecture module, or None."""
    for f in FAMILIES:
        m = importlib.import_module(f"{__name__}.{f}").MANIFEST
        if module in m["architectures"]:
            return f
    return None


def architecture_dirs() -> list:
    return [architecture_dir(f) for f in FAMILIES]
