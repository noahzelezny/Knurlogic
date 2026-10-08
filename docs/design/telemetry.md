# Telemetry contract

The one page shared, verbatim, between Knurlogic (the server) and any
client that wants attribution (an agent harness is the first). No shared code.

## Request headers (client → server)

| header | meaning | limit |
|---|---|---|
| `X-Client` | `<name>/<version>`, e.g. `client/0.0.1` | 128 bytes |
| `X-Client-Session` | an id the client uses for one conversation or job | 128 bytes |
| `X-Client-Run` | an id for a sub-unit (a worker, a sub-task), or `-` | 128 bytes |
| `X-Client-Role` | a short label the client chooses (`pm`, `worker`, `sidecar`, …) | 32 bytes |

All optional. The server stores them as opaque strings and groups by them.
A request with none is attributed to the key only, or `anonymous`.

## Response (server → client)

- `X-Request-Id`: the ledger row's id. If the client sent `X-Request-Id`
  (1..128 printable ASCII), it is echoed exactly and is the row's id;
  otherwise the server mints a ULID. An invalid value is ignored, not
  refused. A client that reuses an id replaces the earlier row. The page's
  router passes the header through.
- `usage.knurlogic.request_id`: the same, in the final usage object of
  every API (OpenAI, Messages, Responses, Ollama).
- `usage.knurlogic.timing`: `{queue_ms, prefill_ms, decode_ms,
  prefill_tps, decode_tps}` when the server has them.
- SSE `event: knurlogic.progress` during streaming:
  `{"request_id", "phase": "queue"|"prefill", "done", "total", "tps",
  "queue": {"ahead"}}`. Clients that do not know it ignore it.

## Joining

A client records `X-Request-Id` in its own trace beside its session/run
ids. Any of the server's ledger rows can then be matched to the client's
work record by `request_id`, and the server's per-session/per-run rollups
match the client's ids without either side importing the other.

## Versioning

This page carries `telemetry: 1`. A server advertises the version it
speaks in `GET /v1/models` (`knurlogic.telemetry`). Additive changes keep
the version; a renamed or removed field bumps it.

telemetry: 1
