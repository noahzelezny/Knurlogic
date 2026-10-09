# Telemetry: request ids, the ledger and timing

What a client learns about a request, and what the machine remembers of
it. The contract clients share (`telemetry: 1`):
[telemetry](../design/telemetry.md); the request record:
[fleet](../design/fleet.md).

## Where the code is

A request's path to its ledger row, in order:

1. `interfaces/http/server.py`: `Handler.do_POST` looks the path up in
   `INFERENCE` and opens a `T.Request` for each inference route (chat,
   completions, messages, responses, Ollama); `App.submit` copies its id
   and labels onto the job; `end_headers` answers `X-Request-Id`.
2. `engine/runtime/scheduler.py`: the job's `Spans` is charged at each
   step boundary; `_done` fills `usage.knurlogic.timing` with
   `timing.rates` and the spans, and `usage.knurlogic.cache` from
   `engine/prompt_cache/report.py`. The scheduler never writes the ledger.
3. `interfaces/http/openai.py` (`Reply`): the final usage goes to
   `Request.done`; streamed progress is `T.progress`.
4. `Handler.do_POST`'s `finally`: `Request.close()` writes one row through
   `machine/ledger.py`.

| file | what |
|---|---|
| `interfaces/http/telemetry.py` | request ids (`HEADER`, `valid_id`, `id_of`: a client's `X-Request-Id`, echoed exactly when valid, 1..128 printable ASCII); `Request`: opened at the handler, closed on its last byte into one ledger row; `progress(...)`: the SSE `knurlogic.progress` event; `TelemetryHandlers._usage`: `GET /v1/usage`, this machine's ledger (a mixin of `Handler`) |
| `machine/ledger.py` | `Ledger` (SQLite at `KNURLOGIC_HOME/ledger.db`, WAL, one table): `insert`, `rows`, `summary`, `prune`; `record(row)`; `ulid`; `labels` (the `X-Client*` headers). In `machine/`: what this Mac remembers; `engine/` never imports it |
| `engine/runtime/timing.py` | `rates`: queue, TTFT, prefill and decode rates, the contract's `*_ms`/`*_tps`; `Spans`: a partition of one request's wall time into named buckets (`step_bucket`); off with `KNURLOGIC_TIMING_SPANS=off` |
| `engine/prompt_cache/report.py` | `usage.knurlogic.cache` for one request (with the prompt cache, whose engine writes it) |

The page's router (`interfaces/page/server.py`) passes `X-Request-Id`
through with `T.id_of` / `T.valid_id`.

## Rules that keep it correct

- **Counts, never text.** A ledger row holds token counts, timings, the
  client's opaque labels and the outcome. Never prompt or answer text.
- **One row per request, written once** when it closes.
- **Only `machine/ledger.py` opens the ledger file;** `engine/` never
  imports it.
- **The ledger is a ring.** Rows older than `KNURLOGIC_LEDGER_DAYS` (30)
  go, and the oldest go while it is over `KNURLOGIC_LEDGER_MIB` (256).
- **One id everywhere.** The request id is the `X-Request-Id` answered,
  `usage.knurlogic.request_id` and the row's id.
- **Spans change nothing on the GPU.** Host clocks only; no `mx.eval` is
  added. Buckets sum to the wall time by construction.

## Extending

A new usage field: compute it on the scheduler thread, carry it on the
job, report it in `usage.knurlogic`; if the ledger keeps it, a column in
`machine/ledger.py`. A field clients rely on is part of the contract in
[telemetry](../design/telemetry.md): change the doc with it.

## Tests

`tests/interfaces/test_telemetry.py`, `test_request_id.py`,
`test_requests_report.py`, `tests/machine/test_ledger.py`,
`tests/engine/test_timing.py`.
