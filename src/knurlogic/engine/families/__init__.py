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
  thinking        {dialect: {detect: {all: [...], none: [...]}, default,
                  native: [[ladder level, native name, template kwargs]]}}
                  -- keyed by CHAT-TEMPLATE DIALECT, not architecture: one
                  module can ship templates with different controls. The
                  ladder is OpenAI's: none minimal low medium high xhigh.

ADDING A FAMILY, the whole checklist:
  1. engine/families/<family>/__init__.py with its MANIFEST, and one line
     in FAMILIES below. Every config.json spelling of each architecture
     goes in `model_types` (the `_text` suffix has hidden vision on every
     released rung once).
  2. architecture/: `knurlogic vendor <module> --family <family> --python
     <env> --host <pkg>` copies the module and writes PROVENANCE.md; add
     its license row to THIRD-PARTY.md. Vendor the version the artifacts
     were BUILT on (GLM needed 0.6.17; 0.7.1 could not load it).
     `knurlogic smoke --pin` on a real artifact writes pins.json.
  3. vision/ if it sees images: a `build(model_path, text_model, config)`
     returning a Family (engine/vision/__init__.py has the protocol).
     Watch the processor's normalization and the chat template's image
     token spelling -- both were wrong once and only a real model showed it.
  4. heads/ if it ships an MTP head: the head class, and a `head` entry in
     the manifest (capture point, cache semantics MEASURED with
     mtp.caches.check_snapshot_semantics).
  5. tests/test_families.py runs over every listed family; the real-model
     gates are tools/vision_gate.py and tools/vq_gate.py.

Where a family quirk lives: code quirks in the family's own code;
declarative ones read by generic code in the manifest, with evidence;
facts readable from the artifact itself (tool-call dialect) nowhere here.

Stdlib only.
"""
from __future__ import annotations

import importlib
from pathlib import Path

HERE = Path(__file__).parent

#: Every family, by package name under engine/families/.
FAMILIES = ("qwen", "gemma4", "glm5")


def manifests() -> list:
    return [importlib.import_module(f"{__name__}.{f}").MANIFEST
            for f in FAMILIES]


def build_maps() -> dict:
    """The tables the generic engine reads, built from the manifests."""
    arch_for_type, host, depends, prefill = {}, {}, {}, {}
    vision, heads, thinking = {}, {}, {}
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
            "vision": vision, "heads": heads, "thinking": thinking}


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
