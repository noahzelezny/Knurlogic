# Architecture

Knurlogic is one Python package, `src/knurlogic/`, in six parts.

| package | what it does |
|---|---|
| `engine/` | Runs models. The only place that imports mlx: model families and architectures, the batch scheduler, prompt cache, KV cache, MTP drafting, vision, the VQ runtime, the serve loop inside one process. |
| `interfaces/` | How people and agents reach it. The CLI, the HTTP API (`http/`), the web page (`page/`), the MCP server, model loading and `doctor`. The only place that opens an HTTP server. |
| `machine/` | Facts about this Mac and what runs on it: memory (`machine/memory/`: the wired limit, the allowance, where memory went, pressure), installed models and artifacts, running servers, the load lock. No mlx, and it does not import `tuning/`. |
| `tuning/` | Every setting, in one package: the measured constants, the knob registry, the presets, the checks that refuse a value or a launch, which knobs apply live, the memory fit, the split arithmetic, the saved knurlogic-wide settings and strategy, and `resolve()`, which returns the final value of every knob with the measurement behind it. See [builders/settings](builders/settings.md). |
| `cluster/` | Several Macs as one: finding peers, link checks, launching one job across machines, watching its ranks, recovering after a failure. |
| `context_management/` | Compaction of long conversations so a harness does not have to manage its agents' context. |

## Who may import whom

```
interfaces  ->  cluster, context_management, engine, machine, tuning
cluster     ->  engine, machine, tuning
engine, machine, tuning   (a mutual core: they share data types and knobs;
                           machine does not import tuning)
context_management  ->  machine, tuning
```

Four rules are enforced by `tests/integration/test_layers.py`, which
parses every source file, function-level imports included:

1. Only `engine/` imports `mlx`, `mlx_lm` or `mlx_vlm`.
2. Only `interfaces/` creates an HTTP server (`http.server`, `socketserver`).
3. `engine/`, `machine/`, `tuning/` and `context_management/` never import
   `knurlogic.interfaces`.
4. `engine/`, `machine/` and `tuning/` never import `knurlogic.cluster`. A
   rank's progress calls from engine go through `engine/runtime/marker.py`;
   `interfaces/serve.py` sets the marker there.

Everything else in the diagram is convention. `cluster/` does call back
into `interfaces.serve` with a lazy import, to start a rank.

## How a request flows

```
page / MCP / CLI  ->  tuning.resolve  ->  interfaces.serve  ->  engine
   (you choose         (model + memory     (starts the model     (loads it,
    a model)            -> settings)        server process)       answers)
```

1. You pick a model in the page, through the MCP `load` tool, or with
   `knurlogic serve`.
2. `tuning.resolve` reads the model's `config.json` and the memory
   available on this machine and returns the settings: preset, context,
   KV bits, prefill size, cache limit. Anything you set explicitly wins.
3. `interfaces/serve.py` starts the model server with those settings as
   its environment. `machine/` records the server and holds the load lock
   so two loads do not race.
4. The server process (`interfaces/http/`) speaks the OpenAI-style API.
   Each chat request is handed to `engine/`: prompt and template, prompt
   cache, scheduler, generation with MTP drafting where the family has
   it, vision inputs through the image store. Compaction runs here too,
   per request, from `context_management/`.

## How a cluster job flows

1. You pick two or more machines on the page, or pass `machines` to the MCP
   `load` tool. The page you pressed Launch on is the coordinator.
2. Every machine's page checks the request against its own disk, memory,
   software and links (`cluster/launch.py`, `cluster/checks.py`). Nothing
   starts unless every machine can.
3. Each page starts its own ranks. Pipeline or tensor parallelism runs in
   `engine/`; ranks talk over the link found by `cluster/links.py`.
4. Each rank writes a heartbeat under `~/.cache/knurlogic/jobs/`; its page
   watches them (`cluster/jobs.py`) and tears the whole job down if a rank
   dies or stalls. `cluster/recovery.py` can relaunch it.

Peers are found by `cluster/discovery.py` and `cluster/peers.py`.

## Where settings live

- **Saved per machine**: `~/.config/knurlogic/settings.json`
  (`XDG_CONFIG_HOME` honoured), managed by `tuning/preferences.py`. It
  holds a custom preset's values, compaction, the disk prompt cache and
  cross-chip rounding; the default preset (the strategy) is beside it in
  `strategy.json` (`tuning/strategy.py`). Every model server on the
  machine reads them.
- **Per model**: chosen in the page's Settings panel and applied when that
  model loads. `--set KEY=VALUE` on the command line does the same for one
  launch and beats the saved value.
- **Presets**: named bundles of knob values (the default one and
  `lean` among them), defined in `tuning/presets.py`.
  `resolve()` starts from the preset, then the model's fit in memory, then
  your overrides.

## Design notes

Start with `engine.md` and `server.md` for the shape of a server.

- Engine and models: [engine](design/engine.md),
  [families](design/families.md), [drafting](design/drafting.md),
  [kv-cache](design/kv-cache.md)
- Prompt cache: [on disk](design/prompt-cache-disk.md),
  [shard-agnostic](design/prompt-cache-shards.md) (all its code is in
  `engine/prompt_cache/`)
- Vision: [vision](design/vision.md),
  [vision-contracts](design/vision-contracts.md)
- Server and interfaces: [server](design/server.md), [mcp](design/mcp.md)
- Settings and memory: [settings](design/settings.md),
  [memory](design/memory.md)
- Conversations: [compaction](design/compaction.md)
- Cluster: [cluster](design/cluster.md), [discovery](design/discovery.md)
- Real-model gates: [tools](design/tools.md)
- Usage and the control plane: [fleet](design/fleet.md),
  [telemetry](design/telemetry.md) (the contract clients share)

## Tests

`tests/` mirrors the packages (`engine/`, `interfaces/`, `machine/`,
`tuning/`, `cluster/`, `context_management/`); cross-cutting tests,
including the layer rules, are in `tests/integration/`. Helpers that are
not tests (fake ranks, tiny-model fixtures, goldens) are in
`tests/support/`.
