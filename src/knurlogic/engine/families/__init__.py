"""engine/families/ -- one folder per model family: everything knurlogic
knows about that family, in one place.

  FAMILIES   the explicit list below. A family folder that is not listed is
             a test failure, not a silent skip (tests/test_families.py).
  <family>/__init__.py   MANIFEST, plain data (see below)

WHY AN EXPLICIT LIST, NOT A FOLDER SCAN. Implicit discovery already cost
this project a bug that passed by alphabetical luck (register.py,
_with_dependencies), and a list is what a person or an agent can grep:
`grep gemma4 engine/families/__init__.py`. Adding a family is one folder
and one line here.

THE MANIFEST is data, never code: dicts, strings, numbers. Anything that
runs is named by a dotted "module:attr" string and imported only when
used, so asking what a family supports imports no mlx. Its shape:

  name            the family ("qwen")
  architectures   {module: {...}}, one entry per architecture module --
                  the unit the engine registers and pins:
      host            the package it registers under ("mlx_lm"/"mlx_vlm")
      depends_on      modules whose arithmetic it inherits
      model_types     every config.json spelling that means this module
      prefill_chunk   (width, evidence) measured for it, or absent
      head            the MTP head, or absent: {names, head, capture,
                      draft_cache, cache_semantics, sidecar_name}
  vision          {"build": "module:attr", "architectures": [...]}, or None

Where a family quirk lives: code quirks in the family's own code;
declarative ones read by generic code in the manifest, with evidence;
facts readable from the artifact itself (tool-call dialect) nowhere here.

Stdlib only.
"""
from __future__ import annotations

import importlib

#: Every family, by package name under engine/families/.
FAMILIES = ("qwen", "gemma4", "glm5")


def manifests() -> list:
    return [importlib.import_module(f"{__name__}.{f}").MANIFEST
            for f in FAMILIES]


def build_maps() -> dict:
    """The tables the generic engine reads, built from the manifests."""
    arch_for_type, host, depends, prefill = {}, {}, {}, {}
    vision, heads = {}, {}
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
            h = a.get("head")
            if h:
                spec = {k: v for k, v in h.items() if k != "names"}
                for n in h["names"]:
                    heads[n] = spec
        v = m.get("vision")
        if v:
            for mod in v["architectures"]:
                vision[mod] = v["build"]
    return {"arch_for_model_type": arch_for_type, "arch_host": host,
            "arch_depends_on": depends, "prefill_chunk": prefill,
            "vision": vision, "heads": heads}
