# Building on the engine

The engine runs one model in one process: the model server that
`knurlogic serve` starts. Why it is one scheduler thread, the executor seam
and the memory guard: [server](../design/server.md) and
[engine](../design/engine.md). The KV cache's precision:
[kv-cache](../design/kv-cache.md).

## Where the code is

`src/knurlogic/engine/runtime/` is the server's core:

| file | what |
|---|---|
| `scheduler.py` | `Scheduler`: the ONE thread that owns the MLX stream. `submit(Job)` queues; `_run` / `_tick` take jobs, run commands, admit, step |
| `memory_guard.py` | `MemoryGuard`, the mixin Scheduler inherits: the memory guard (`_guard_memory`, `_make_room`, `_fit_next`, `memory_short`, the pressure warning), the transient lines it measures (`_measure`, `_line`), and `OutOfMemory` |
| `executor.py` | the seam: `Executor` protocol, `LocalExecutor`, and the events it returns (`Admission`, `Progress`, `Checkpoint`, `Token`, `Finished`, `RowFailure`) |
| `host.py` | `ModelHost`: the one served model and its state (empty, loading, ready, unloading, failed); loads on the scheduler thread |
| `prompt.py` | messages to tokens, cut into segments (`flatten`, `tokenize`, `ChatRequest`, `PromptArgs`) |
| `request.py` | `Request`: token events to reasoning, answer, tool calls, stop strings and usage; no mlx |
| `control.py` | `ControlMachine`: which part of an answer a token is in, and which sequence ends the row |
| `timing.py` | `usage.knurlogic.timing`: `rates` and where a request's time went, `Spans` (see [telemetry](telemetry.md)) |
| `tensor.py`, `pipeline.py`, `plan.py`, `tensor_rules.py`, `viability.py` | the splits (see [splits](splits.md)) |

Around it, in `engine/`:

| file | what |
|---|---|
| `serve/` | what the served model is: `load.py` (load, memory, `apply_live`, `tool_support`), `state.py` (process state: `SERVED`, `DRAFT`, `VISION`), `segments.py`, `thinking.py`, `drafting.py`, `vision.py`. Importing it imports no mlx |
| `kvquant.py` | `QuantKVCache`, `BatchQuantKVCache`, `install`: K/V stored at 8, 6 or 4 bits |
| `kvattn.py` | the 8-bit decode attention kernel (`decode_sdpa`, `patch_model`) |
| `crosschip.py` | identical results across chips (`KNURLOGIC_CROSS_CHIP`) |
| `templates/` | chat templates knurlogic supplies in place of an artifact's own |
| `prompt_cache/` | see [prompt-cache](prompt-cache.md) |
| `mtp/` | the batch engine and drafting (see [drafting](drafting.md)) |
| `vision/` | see [vision](vision.md) |

The process around it is `interfaces/http/` (see
[http-and-page](http-and-page.md)): `interfaces/http/__init__.py`
(`serve`, `scheduler_options`, `watch_ring`) builds the scheduler and the
server.

## A request's path

    HTTP thread --submit(Job)--> queue --> tokenize --> prompt cache
      --> executor.insert --> executor.step --> events --> Request
      --> the Job's outbox --> HTTP thread

`interfaces/http/openai.py` (`build_job`) turns a body into a `Job`; the
scheduler does the rest on its thread.

## Rules that keep it correct

- **One thread touches the model.** Everything that runs the model,
  loads it or changes the prompt cache runs on the scheduler thread.
  Other threads `submit` a job or queue a `Command` (`_command`,
  `_do_commands`), which runs between steps.
- **A failure is per request.** A failed tokenize or admit, a `RowFailure`
  event or a raising step fails only its rows; the executor is rebuilt
  and the thread lives. An executor reports a row's failure as an event,
  not an exception.
- **Memory is guarded before every step.** Outgrowing the GPU working set
  aborts the process, so `_guard_memory` (`memory_guard.py`) runs
  first: past the limit,
  freed buffers are released, then the prompt cache gives up entries
  (`trim_to`), then the newest rows are requeued (if not prefilled) or
  stopped with `OutOfMemory` (a 503). Admission estimates from what this
  model's caches measured (`_learn`, `_need`, `_room_for`, `_make_room`):
  a row is admitted with checkpoints, without them, waits, or with nothing
  running is refused.
- **Events stay small.** A token event carries the token and its logprob
  (top-k when asked), never a vocabulary row: the same executor runs on
  every rank of a split.
- **`engine/serve/` is the one door to the engine.** Code outside the
  engine asks `engine.serve` names, not mlx; version skew in mlx-lm is
  handled there (`load.load_unlocked`).
- **A live knob reaches every rank.** `tuning/live.LIVE_KNOBS` can change
  on a running server (`apply_live`); on a split, `Scheduler.share_live` journals them
  as `set` ops (`plan.SETS`).

## Extending

- A new request field: parse it in `interfaces/http/openai.py`
  (`build_job`), carry it on `Job` / `Admission`, and, if a split model
  must see it, in `plan.py`'s `admit` fields.
- A new scheduler command: a `Command` run in `_do_commands`; on a split,
  journal it (see [prompt-cache](prompt-cache.md) for the op checklist).
- A new cache class: it must work under `kvquant` (or be listed as not
  quantized in its family manifest) and save and restore bit for bit
  ([prompt-cache](prompt-cache.md), "A new cache class").

## Notes

Memory, by layer: the guard is `engine/runtime/memory_guard.py`; what mlx
reports and the cache limit, `engine/serve/load.py` (`memory`,
`set_cache_limit`); the fit and margins, `tuning/fit.py` (`step_margin`,
`rank_margin`, `fit_reserve`, `single_fit_check`); the machine's budget,
`machine/memory/` (`wired.load_budget`, the allowance,
`footprint.available_memory`). See [machine](machine.md) and
[settings](settings.md).

## Tests

`tests/engine/test_scheduler.py`, `test_executor.py`, `test_request.py`,
`test_prompt.py`, `test_segments.py`, `test_thinking.py`, `test_timing.py`,
`test_kvquant.py`, `test_kvattn.py`, `test_cross_chip.py`,
`test_warmup.py`, `test_thread_arrays.py`;
`tests/interfaces/test_http_server.py` for the server around it.
