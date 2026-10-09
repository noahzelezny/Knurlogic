# Model families and adding a model

A model is supported by its family's folder under `engine/families/`.
The manifest's full shape and the per-family checklist are in
[families](../design/families.md); the definition of done for a new model
is [new-model](../design/new-model.md). Read both before starting. This
page says where things are.

## Where the code is

| path | what |
|---|---|
| `engine/families/__init__.py` | `FAMILIES` (the explicit list: `qwen`, `gemma4`, `glm5`, `deepseek`), `manifests()`, `build_maps()`: the tables generic code reads, built from the manifests |
| `engine/families/<family>/__init__.py` | `MANIFEST`: plain data. Anything that runs is a `"module:attr"` string, imported only when used |
| `engine/families/<family>/architecture/` | the vendored model code, with `PROVENANCE.md` |
| `engine/families/<family>/heads/` | MTP heads (see [drafting](drafting.md)) |
| `engine/families/<family>/vision/` | the vision tower and its `Family` (see [vision](vision.md)) |
| `engine/families/<family>/pipeline_stage.py` | `restage` functions for a pipeline split, where the trunk froze per-layer indices (qwen, glm5) |
| `engine/families/qwen/kvcache.py` | a family's own cache classes |
| `engine/register.py` | `register(*names)`: makes `mlx_lm.models.<name>` import knurlogic's vendored module without writing site-packages. Public: vqlab calls it (pinned by `tests/engine/test_register_public.py`) |
| `engine/arch.py` | which architecture module a `model_type` needs, and pins (`supported`, `required_modules`, `check`) |
| `engine/vendor.py` | `knurlogic vendor`: copy an architecture file and record where it came from |
| `engine/smoke.py` | `knurlogic smoke`: generate a token and prove where the code came from; `--pin` writes pins |
| `engine/templates/__init__.py` | the generic half of supplied chat templates; the templates themselves are in the manifests (`chat_templates`) |
| `engine/serve/thinking.py` | one thinking control translated to each template's dialect (from the manifest's `thinking`) |
| `machine/artifact.py` | what an artifact declares: `Artifact`, `context_length`, `sampling_defaults`, `identity` |

## Rules that keep it correct

- **Generic code names no family.** `engine/runtime`, `engine/templates`,
  `engine/vision/registry`, `tuning/` and `machine/` read the tables from
  `build_maps()`. A family's spelling or an `if family == ...` there is a
  bug.
- **The manifest is data.** It must import without mlx; `build_maps`
  raises if two families claim the same `model_type`, head name, thinking
  dialect, chat template or pipeline core.
- **Vendored files are not edited in place.** Re-vendor with
  `knurlogic vendor` and update `PROVENANCE.md` in the same change.
- **Every listed family is tested.** A family folder not in `FAMILIES`
  fails `tests/engine/test_families.py`.
- **Measured values carry their evidence.** A `prefill_chunk` is
  `(width, evidence)`; `cache_semantics` starts at `"copy"` until
  `engine/mtp/caches.check_snapshot_semantics` passes on a loaded model.

## Adding a model

To a family that exists: add its `model_types` spellings and anything the
new rung changes, then work through [new-model](../design/new-model.md)
(inventory, weights, chat, thinking, images, drafting, clusters, cache
saving, maker-spec numerics, live proof).

A new family: one folder and one line in `FAMILIES`; the steps are in
[families](../design/families.md), "Adding a family". Per feature, the
hooks it needs:

| feature | what to add |
|---|---|
| KV quantization | `kv_quant` in the architecture entry; its `caches` names the quantized factory of any family cache class (read by `engine/kvquant.family_caches`) |
| drafting | `head` entry and the class in `heads/` ([drafting](drafting.md)) |
| vision | `vision` entry and `vision/` with `build(...)` ([vision](vision.md)) |
| tensor split | `tensor` / `tensor_split` entries and rules in `engine/split/tensor_rules.py` ([splits](splits.md)) |
| pipeline split | `pipeline` entry, and a `restage` if the trunk froze layer indices |
| cache saving | the class round-trips in `tests/engine/test_prompt_disk.py` ([prompt-cache](prompt-cache.md)) |
| prefill width | a measured `prefill_chunk` |

## Notes

A model's support is spread by design across the family folder (the
manifest) and the generic tables that read it, but a few family facts sit
outside the folder: the tensor split rules are one table for every family
in `engine/split/tensor_rules.py`, and `tuning/` reads
the manifests through per-family helpers (`measured.prefill_chunk_for`,
`measured.kv_quant_for`, `context_window.long_context_family`).

## Tests

`tests/engine/test_families.py` (every family), `test_register_public.py`,
the per-family reference and parity tests (`test_qwen_reference.py`,
`test_gemma4_text_parity.py`, `test_glm5_next_reference.py`,
`test_deepseek_v4*.py`), with their golden builders in `tests/support/goldens/`.
Real-model gates are by hand: [tools](../design/tools.md).
