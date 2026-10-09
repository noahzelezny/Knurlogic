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
      pipeline        {"core": trunk core class name, "restage":
                      "module:attr" or absent}: a pipeline split's trunk,
                      and what re-takes the indices it froze from the whole
                      layer list over a stage's layers (pipeline_stage.py)
      tensor          present when engine/runtime/tensor.py splits it:
                      {"divisible": [config keys the ranks must divide]}
      tensor_split    {"act_quant_block": n, "inputs": [{what, keys,
                      defaults}]}: linear inputs rounded in blocks a tensor
                      split must not cut (tuning/resolve.tensor_refusals)
  vision          {"build": "module:attr", "architectures": [...],
                   "signature": {"config": nested key | "config_layers":
                   flat count key, "tower_in_weights": prefix,
                   "tower_prefixes": [...]}}, or None
  chat_templates  {name: {file | base + prefix, model_types, stubs, marker,
                  when_config, parser, part_separator}}: templates
                  knurlogic serves in place of an artifact's own, their
                  variants and what selects them (engine/templates)
  sampling        {"non_thinking": {model_type: sampler set}}: the maker's
                  published sampling where generation_config.json cannot
                  carry it (machine/artifact.sampling_defaults)
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
  5. tests/engine/test_families.py runs over every listed family; the real-model
     gate is tools/vision_gate.py.

Where a family quirk lives: code quirks in the family's own code;
declarative ones read by generic code in the manifest, with evidence;
facts readable from the artifact itself (tool-call dialect) nowhere here.

## Checking a family against its maker's spec

Every maker-specific fact knurlogic holds about a family is in
engine/families/<family>/, so a reviewer checks one folder against the
maker's model card, config.json, reference code and encoder. Generic code
(engine/runtime, engine/templates, engine/vision/registry, tuning/,
machine/) holds no family names; a `if fam == ...` or a family's spelling
there is a finding. The checklist, per family:

  [ ] architectures: every config.json `model_type` spelling; the vendored
      module's PROVENANCE.md names the maker's revision and each edit
  [ ] thinking: each dialect's levels and kwargs as the maker's template
      or encoder defines them, its default as served
  [ ] chat_templates (if knurlogic supplies one): the port's PROVENANCE.md
      against the maker's encoder; `stubs` / `marker` / `model_types` /
      `when_config` select it from the artifact's config.json, never its
      folder name; `part_separator` per role as the encoder joins a
      message's list of parts (DeepSeek: a tool result's "\n\n")
  [ ] sampling: the maker's published sets, with the source cited
  [ ] vision: `signature` names the config key the maker's config uses and
      every tensor prefix of the tower in the maker's checkpoint and its
      conversions; the processor matches the maker's (vision/PROVENANCE.md)
  [ ] head: capture point and cache semantics measured; the sidecar's
      tensor layout
  [ ] pipeline / tensor_split: what the trunk freezes from the whole layer
      list; any block-wise activation rounding a split must respect
  [ ] prefill_chunk / kv_quant: measured, with the numbers
  [ ] speed: a served request's partition (usage.knurlogic.timing.spans_s,
      engine/runtime/spans.py) at a 2048-token prompt, n >= 3, recorded
      with the machine: TTFT, prefill and decode tok/s, the engine-only
      prefill rate (prompt / prefill_forward), and spans_unaccounted_s ~ 0
      (`vqlab bench serve-timeline` drives it)
