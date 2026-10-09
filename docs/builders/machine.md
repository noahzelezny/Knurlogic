# The machine: memory, models and running servers

`machine/` says what is true about this Mac, read rather than assumed:
its memory, what is installed, what is loaded and running. No mlx. The
why: [memory](../design/memory.md), [memory-ledger](../design/memory-ledger.md),
[fleet](../design/fleet.md).

## Where the code is

`src/knurlogic/machine/`:

| file | what |
|---|---|
| `wired.py` | the GPU wired limit (`read`, `advise`, `command_for`; never sets it), `detected_working_set_bytes`, `machine()`, and `load_budget()`: the one number every fit, settings and load answer is computed against |
| `allowance.py` | the most memory knurlogic may use on this Mac (`get`, `set`, `cap`), saved in `~/.config/knurlogic/allowance.json`. Only ever lowers the budget |
| `loaded.py` | what is resident now in every runtime (knurlogic, exo, ollama, mlx-lm): `survey`, `available_memory` (from `vm_stat`), `memory_map` (every big process, attributed, the remainder reported) |
| `metrics.py` | how hard the machine is working: CPU, GPU, swap, pressure, thermal (`sample`, `metrics`) |
| `servers.py` | the record of running knurlogic servers (`registry`, `save_registry`, `listening_serves`, `free_port`, `serve_log`) |
| `loadlock.py` | the model-load lock: `model_load(...)` holds `flock` on `~/.cache/knurlogic/load.lock`; `holder`, `Busy` |
| `artifact.py` | what a model directory is: `Artifact`, `identity` (content hash), `resolve_identity`, `context_length`, `sampling_defaults`, `on_network` |
| `discover.py` | every model on disk in every tool's store (`find`, `find_named`) |
| `folders.py` | the extra model folders this Mac remembers (`add`, `remove`, `roots`) |
| `identity.py` | which machine this is: a stable `id`, never the name |
| `status.py` | the snapshot `/status.json` serves (`snapshot`, `aggregate`) |
| `deps.py` | which mlx, mlx-lm, mlx-vlm builds are installed, read off the fix itself |
| `preferences.py`, `strategy.py` | saved settings (see [settings](settings.md)) |
| `ledger.py` | the request ledger (see [telemetry](telemetry.md)) |
| `disk_cache.py` | a small JSON cache so a start does not redo unchanged work |

Who uses it:

- `interfaces/loading.py` (`prepare`) checks fit against `load_budget`.
- `engine/runtime/host.py` and `engine/serve/load.py` take the load lock
  around a real load; the MCP's `ready` reads `loadlock.holder()`.
- The scheduler's memory guard counts against the server's working set,
  which the allowance lowers ([engine](engine.md)).
- The page and MCP read `loaded.survey`, `servers.registry` and
  `discover.find`.

## Rules that keep it correct

- **Every answer says how it was measured.** Each module's docstring
  records the case that was wrong while it was assumed; keep that when
  changing it.
- **One budget.** Fit, settings and load all use `wired.load_budget()`:
  the smaller of the GPU working set and what macOS would hand over now,
  capped by the allowance.
- **knurlogic never changes system settings.** `wired.py` prints the
  `sysctl` command and its cost; a person runs it.
- **One real load at a time.** The lock is released by the kernel however
  the holder dies, so it is never stale. `wait_s=0` fails at once with
  `Busy`. Tiny-fixture tests do not take it.
- **A runtime that is not running is absent, not an error** (`loaded`).
- **A node is its id.** Names collide.
- **Stdlib only** where the page reads it: allowance, loadlock,
  preferences, servers.

## Notes

Memory is spread across packages: `machine/wired.py`, `allowance.py` and
`loaded.py` (the machine), `tuning/resolve.py` (the fit), the scheduler's
guard in `engine/runtime/scheduler.py`, `engine/serve/load.py`
(`memory`, `gpu_in_use`, `set_cache_limit`), and
`cluster/launch.py` (`available_now`, `budget_of`, `gpu_working_set`)
for a peer's share.

## Tests

`tests/machine/` (`test_machine.py`, `test_loaded.py`, `test_loadlock.py`,
`test_ledger.py`, `test_discover.py`, `test_model_folders.py`,
`test_loading.py`, `test_machine_settings.py`,
`test_no_leaked_processes.py`), `tests/tuning/test_allowance.py`,
`tests/interfaces/test_memory_refresh.py`.
