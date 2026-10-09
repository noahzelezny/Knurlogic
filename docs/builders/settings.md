# Settings: knobs, presets and resolve

Every setting a model runs with is decided in one package, `tuning/`, from
the model's config and the memory it has, with the measurement behind
each value. The why: [settings](../design/settings.md); memory:
[memory](../design/memory.md) and [memory-pacing](../design/memory-pacing.md).

## Where the code is

| file | what |
|---|---|
| `tuning/measured.py` | every measured constant with its provenance: decode and prefill chunks (`DECODE_CHUNK_*`, `PREFILL_CHUNK_*`, `prefill_chunk_for`), the fit reserve (`FIT_*`), cache limits (`CACHE_LIMIT_GB_DEFAULT`, `CACHE_LIMIT_GB_MAX`), low headroom (`low_headroom_bytes`), KV element size and the KV precisions a family takes (`kv_bytes_per_element`, `kv_quant_for`), the vision allowance (`VISION_*`, `BF16_BYTES`) |
| `tuning/numerics.py` | the VQ numerics flags: `NUMERICS_FLAGS`, `RUNTIME_PROFILES`, `NUMERICS_SOURCES`, `numerics_for` |
| `tuning/presets.py` | the presets (the tune axis): `TUNE_PROFILES`, `PRESETS` (`default`, `lean`), `preset_of`, `preset_or`, `preset_arg`, `preset_launch`, `preset_values`; the rows a custom set is saved as: `PRESET_ROWS`, `MTP_MODE`, `preset_row_values` |
| `tuning/knobs.py` | the knob registry: `KNOB_DOC`, `KNOB_HELP`, `KNOB_TITLES`, `KNOB_ALIASES`, `MODEL_KNOBS`, `ENGINE_KNOB_NAMES`, `knob_tier`, `KNOB_RANGE`, `KNOB_BOUNDS`; the readers of a set value (`on_off`, `vision_of`, `mtp_of`, `kv_bits_of`, `thinking_default_of`, `cross_chip_of`); aliases: `engine_settings`, `canonical_sets`, `legacy_mirror`, `default_alias` |
| `tuning/groups.py` | the knurlogic-wide groups read per request: `COMPACT_KNOBS`, `compact_settings`, `check_compact_knob`; `PROMPT_CACHE_KNOBS`, `check_prompt_cache_knob` |
| `tuning/context_window.py` | `model_window`, long context (YaRN): `long_context_of`, `long_context_refusal`, `context_ceiling`, `settle_context`, `long_context_config`, `with_long_context`, `long_context_room` |
| `tuning/checks.py` | every refusal: `check_knob` (one value), `clean_sets` and `PATH_KEYS` (a request's knobs, `launch_knobs`), `refuse_sets` (a set on one artifact), `settings_refusal`, `launch_refusal`, `launch_fit` (a whole launch, before a process starts) |
| `tuning/live.py` | live or restart: `LIVE_KNOBS` (what a running server can change), `knob_reach` (live, restart, or no effect on this artifact) |
| `tuning/fit.py` | the memory arithmetic: `kv_bytes_per_token`, `step_margin`, `fit_reserve`, `rank_margin`, `context_room`, `room_for`, `vision_budget`, `vision_freed_bytes`, `mtp_head_bytes`, `single_fit_check`; the widths the room allows: `decode_chunk_for`, `prefill_chunk_by_room` |
| `tuning/resolve.py` | `resolve(artifact, budget, ...)` returns a `Resolution` (env, notes, warnings, vision, ranges, preset), or a `ClusterResolution` when given nodes (`resolve_cluster`). Emitting: `emit`, `emit_cache_limit`. A preset applied: `preset_env`, `apply_preset_overrides`. Launch: `model_launch`, `kv_refusal` |
| `tuning/tensor_split.py` | the tensor split: `TENSOR_TYPES`, `tensor_refusals`, `tensor_placement`, `trunk_headers`, `tensor_split_refusals`, `tensor_unverified` |
| `tuning/pipeline_split.py` | the pipeline split: `PIPELINE_TYPES`, `pipeline_refusals`, `pipeline_layer_bytes`, `pipeline_shares`, `leader_bytes`, `chip_bandwidth_gbs` |
| `tuning/rank_order.py` | which machine leads and the ring's order: `rank_order`, `leader_key`, `LINK_SPEED` |
| `tuning/preferences.py` | the saved knurlogic-wide settings (`~/.config/knurlogic/settings.json`): `get`, `set`, `invalid`, `launch_sets`, `prompt_cache_env`, `compaction_env` |
| `tuning/strategy.py` | the machine's default preset (`strategy.json`) |

The import order inside the package, lowest first: `measured`,
`numerics`, `rank_order` -> `presets`, `fit`, `tensor_split` ->
`context_window`, `pipeline_split` -> `knobs` -> `groups`, `live` ->
`checks`, `resolve` -> `preferences`, `strategy`. A module reaches a
sibling's names through the module (`measured.PREFILL_CHUNK_DEFAULT`), so
a test that patches one patches it for every caller. `checks` imports
`resolve`, `fit` and `preferences` inside its launch functions, since
`preferences` checks its own values with `check_knob`.

Where the settings meet the rest (thin calls, no settings logic):

- `interfaces/serve.py`: `_parse_sets` (`--set KEY=VALUE`), the
  `tuning/checks` refusals before a load, then the resolved env for the
  server; on a live change it applies `LIVE_KNOBS` through the engine.
- `interfaces/page/documents.py`: the Settings panel's documents
  (`settings_document`, `knob_limit`, `machine_settings`,
  `compaction_document`, `strategy_doc`): page JSON built from the
  registry, `knob_reach` and `resolve`.
- `engine/serve/load.py`: `apply_live`, which applies a live knob to the
  running process (it needs mlx, so it stays in the engine).
- `machine/memory/allowance.py`: the most memory knurlogic may use. It stays in
  `machine/` because it lowers the load budget (`wired.load_budget`);
  `tuning/` takes that budget as an input.
- `interfaces/mcp/inspection.py`: the `settings` and `fit` tools.

## Rules that keep it correct

- **The resolver owns the final value.** It returns one dict; nothing
  writes env files and hopes one wins. Order: the preset, then the
  model's fit in memory, then explicit settings. An explicit value wins.
- **Every number has its evidence.** A constant in `measured.py` (or a
  family manifest) carries the measurement that set it; a value without
  one is a bug.
- **Headroom is an input.** `resolve` takes a byte count (or nodes); it
  does not measure the machine. The budget comes from `machine/`
  ([machine](machine.md)).
- **Only documented knobs pass.** A forwarded load may set only
  `launch_knobs()` (the `KNOB_DOC` names, their aliases and the numerics
  flags); `clean_sets` refuses the rest.
- **One check per knob, used everywhere.** The page, `serve`, the MCP and
  a cluster job call `tuning/checks` (`check_knob`, `refuse_sets`,
  `launch_refusal`), so a value is refused the same way on every surface.
- **A saved preference beats a server's environment**; an explicit
  `--set` of the cross-chip knob still beats it at that launch.

## Adding a knob

The tables a knob can appear in:

1. Its name in `KNOB_DOC`, `KNOB_HELP` and `KNOB_TITLES`; values in
   `KNOB_RANGE` or `KNOB_BOUNDS`; aliases in `KNOB_ALIASES` if it has a
   logical name (all in `knobs.py`). A value it may not take is refused
   in `checks.check_knob`.
2. If a preset sets it, a row in `PRESET_ROWS` (`presets.py`).
3. Its resolution in `resolve.py`, with a note saying why.
4. If it can change on a running server, add it to `live.LIVE_KNOBS` and
   apply it in `engine/serve/load.apply_live`; if it must reach every rank
   of a split, to `plan.SETS`.
5. A test in `tests/tuning/` (the settings audit checks claims against
   code).

## Tests

`tests/tuning/` (`test_tuning.py`, `test_presets.py`,
`test_settings_audit.py`, `test_fit_reserve.py`, `test_long_context.py`,
`test_vision_budget.py`, `test_allowance.py`, `test_doctor_tune.py`),
`tests/engine/test_resolve.py`, `tests/machine/test_machine_settings.py`
(the allowance and the strategy), `tests/interfaces/test_load_refusals.py`
(`launch_refusal`).
