# Settings: knobs, presets and resolve

Every setting a model runs with is decided in one place, `tuning/`, from
the model's config and the memory it has, with the measurement behind
each value. The why: [settings](../design/settings.md); memory:
[memory](../design/memory.md) and [memory-pacing](../design/memory-pacing.md).

## Where the code is

| file | what |
|---|---|
| `tuning/settings.py` | every measured constant with its provenance (prefill and decode chunks, cache limits, VQ flags); the knobs: `KNOB_DOC`, `KNOB_HELP`, `KNOB_TITLES`, `KNOB_ALIASES`, `KNOB_RANGE`, `KNOB_BOUNDS`, `MODEL_KNOBS`; presets: `PRESETS` (`default`, `lean`), `PRESET_ROWS`, `preset_values`, `preset_launch`; checks: `check_knob`, `check_compact_knob`, `check_prompt_cache_knob`, `clean_sets`, `launch_knobs`; groups: `COMPACT_KNOBS`, `PROMPT_CACHE_KNOBS`; long context: `settle_context`, `long_context_config` |
| `tuning/resolve.py` | `resolve(artifact, budget, ...)` returns a `Resolution` (env, notes, warnings, vision, ranges, preset), or a `ClusterResolution` when given nodes (`resolve_cluster`). Fit: `single_fit_check`, `context_room`, `kv_bytes_per_token`, `fit_reserve`, `step_margin`, `rank_margin`, `vision_budget`, `prefill_chunk_by_room`. Launch: `model_launch`, `preset_env`, `apply_preset_overrides`, `numerics_for`. Splits: `tensor_refusals`, `pipeline_layer_bytes` |
| `machine/preferences.py` | the saved knurlogic-wide settings (`~/.config/knurlogic/settings.json`): `get`, `set`, `launch_sets`, `prompt_cache_env`, `compaction_env` |
| `machine/strategy.py` | the machine's default preset |

Where the settings meet the rest:

- `interfaces/serve.py`: `launch_refusal`, `launch_fit`,
  `settings_refusal`, `_parse_sets` (`--set KEY=VALUE`), then the
  resolved env for the server.
- `interfaces/page/documents.py`: the Settings panel's documents:
  `settings_document`, `knob_limit`, `refuse_sets`, `knob_reach` (does a
  change apply live or need a restart), `machine_settings`,
  `compaction_document`.
- `engine/serve/load.py`: `LIVE_KNOBS` and `apply_live`, the knobs a
  running server can change. `settings.engine_settings` reads the
  engine's values out of a resolved environment, whichever alias they
  were emitted under.
- `interfaces/mcp.py`: the `settings` and `fit` tools.

## Rules that keep it correct

- **The resolver owns the final value.** It returns one dict; nothing
  writes env files and hopes one wins. Order: the preset, then the
  model's fit in memory, then explicit settings. An explicit value wins.
- **Every number has its evidence.** A constant in `settings.py` carries
  the measurement that set it; a value without one is a bug.
- **Headroom is an input.** `resolve` takes a byte count (or nodes); it
  does not measure the machine. The budget comes from `machine/`
  ([machine](machine.md)).
- **Only documented knobs pass.** A forwarded load may set only
  `launch_knobs()` (the `KNOB_DOC` names, their aliases and the numerics
  flags); `clean_sets` refuses the rest.
- **One check per knob, used everywhere.** The page, `serve` and the MCP
  call `check_knob` (and the compact and prompt-cache checks), so a value
  is refused the same way on every surface.
- **A saved preference beats a server's environment**; an explicit
  `--set` of the cross-chip knob still beats it at that launch.

## Adding a knob

The tables a knob can appear in:

1. Its name in `KNOB_DOC`, `KNOB_HELP` and `KNOB_TITLES`; values in
   `KNOB_RANGE` or `KNOB_BOUNDS`; aliases in `KNOB_ALIASES` if it has a
   logical name.
2. If a preset sets it, a row in `PRESET_ROWS`.
3. Its resolution in `resolve.py`, with a note saying why.
4. If it can change on a running server, add it to `LIVE_KNOBS`; if it
   must reach every rank of a split, to `plan.SETS`.
5. A test in `tests/tuning/` (the settings audit checks claims against
   code).

## Notes

Settings are spread across `tuning/settings.py` and `tuning/resolve.py`
(the decisions), `machine/preferences.py`, `machine/strategy.py` and
`machine/allowance.py` (what was saved), `interfaces/page/documents.py`
(the panel's documents and live-vs-restart logic), `interfaces/serve.py`
(launch checks) and `engine/serve/load.py` (`LIVE_KNOBS`).

## Tests

`tests/tuning/` (`test_tuning.py`, `test_presets.py`,
`test_settings_audit.py`, `test_fit_reserve.py`, `test_long_context.py`,
`test_vision_budget.py`, `test_allowance.py`, `test_doctor_tune.py`),
`tests/engine/test_resolve.py`, `tests/machine/test_machine_settings.py`.
