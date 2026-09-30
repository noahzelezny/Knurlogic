# Model families

## src/knurlogic/engine/families/__init__.py

WHY AN EXPLICIT LIST, NOT A FOLDER SCAN. Implicit discovery once let a bug
pass by alphabetical luck (register.py, _with_dependencies), and a list is
what a person or an agent can grep: `grep gemma4 engine/families/__init__.py`.
Adding a family is one folder and one line in FAMILIES.

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
      kv_quant        {"bits": [8, 6, 4], "why": ...} when its attention
                      K/V may be stored quantized (engine/kvquant.py), or
                      {"refused": reason}; absent reads as refused
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
     in FAMILIES. Every config.json spelling of each architecture goes in
     `model_types` (the `_text` suffix has hidden vision on a released
     rung).
  2. architecture/: `knurlogic vendor <module> --family <family> --python
     <env> --host <pkg>` copies the module and writes PROVENANCE.md; add
     its license row to THIRD-PARTY.md. Vendor the version the artifacts
     were BUILT on (GLM needs 0.6.17; 0.7.1 cannot load it).
     `knurlogic smoke --pin` on a real artifact writes pins.json.
  3. vision/ if it sees images: a `build(model_path, text_model, config)`
     returning a Family (engine/vision/__init__.py has the protocol).
     Check the processor's normalization and the chat template's image
     token spelling -- only a real model shows either being wrong.
  4. heads/ if it ships an MTP head: the head class, and a `head` entry in
     the manifest (capture point, cache semantics MEASURED with
     mtp.caches.check_snapshot_semantics).
  5. tests/test_families.py runs over every listed family; the real-model
     gate is tools/vision_gate.py.

Where a family quirk lives: code quirks in the family's own code;
declarative ones read by generic code in the manifest, with evidence;
facts readable from the artifact itself (tool-call dialect) nowhere here.
