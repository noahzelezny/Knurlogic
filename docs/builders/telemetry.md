# Telemetry: request ids, the ledger and timing

What a client learns about a request, and what the machine remembers of
it. The contract clients share (`telemetry: 1`):
[telemetry](../design/telemetry.md); the request record:
[fleet](../design/fleet.md).

## Where the code is

| file | what |
|---|---|
| `interfaces/http/telemetry.py` | `Request`: opened at the handler, closed on its last byte into one ledger row; `progress(...)`: the SSE `knurlogic.progress` event |
| `interfaces/http/request_id.py` | `valid`, `of`: a client's `X-Request-Id`, echoed exactly when valid (1..128 printable ASCII) |
| `machine/ledger.py` | `Ledger` (SQLite at `KNURLOGIC_HOME/ledger.db`, WAL, one table): `insert`, `rows`, `summary`, `prune`; `record(row)`; `ulid`; `labels` (the `X-Client*` headers) |
| `engine/runtime/spans.py` | `Spans`: a partition of one request's wall time into named buckets (`step_bucket`); off with `KNURLOGIC_TIMING_SPANS=off` |
| `engine/prompt_cache/report.py` | `usage.knurlogic.cache` for one request |

Hooks:

- `interfaces/http/server.py`: `Handler.do_POST` opens a
  `T.Request` for each inference route (`CHAT_PATHS`, messages,
  responses, Ollama); `GET /v1/usage` (`_usage`) reads this machine's
  ledger.
- `engine/runtime/scheduler.py`: `_timing` and the spans fill
  `usage.knurlogic.timing` on the job; the scheduler never writes the
  ledger.

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

## Notes

Telemetry spans `interfaces/http/` (`telemetry.py`, `request_id.py`, the
handler), `machine/ledger.py` (storage), `engine/runtime/spans.py` and
the scheduler (timing), and `engine/prompt_cache/report.py` (cache usage).

## Tests

`tests/interfaces/test_telemetry.py`, `test_request_id.py`,
`test_requests_report.py`, `tests/machine/test_ledger.py`,
`tests/engine/test_spans.py`.
