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

| `X-Cache-Retain` | `pin`: this session's prompt-cache entries are never auto-deleted from disk (sticky for the session) | — |
| `X-Cache-Keep` | `latest`: this request's prompt-cache entries replace its session's earlier ones (memory and disk); its system-prompt checkpoint is shared, not the session's | — |

All optional. The server stores them as opaque strings and groups by them.
A request with none is attributed to the key only, or `anonymous`.
The session also owns the prompt-cache entries the request makes: only
entries with a session are saved to disk (prompt-cache-disk.md,
"Sessions"). The page's router passes all five headers up.

## Prompt-cache endpoints (loopback only; the page forwards them)

- `POST /v1/prompt-cache/save` `{"session"?}`: with a session, that
  session's newest (longest) entry if not on disk yet -- call it right
  after the context compacts; without, every session's entries.
  Counts: `saved, kept, skipped, not_worth, entries, bytes, seconds`.
- `POST /v1/prompt-cache/drop` `{"session"}`: out of memory and off disk
  (every model). `{memory, disk}`.
- `POST /v1/prompt-cache/drop` `{"sessionless": true[, "older_than_s"]}`:
  the loaded model's entries no session owns (calls without
  `X-Client-Session`, the shared system-prompt copies); with an age, only
  its files unused that long (memory untouched). On a split model every
  rank deletes the same files, by name.
- `POST /v1/prompt-cache/pin` `{"session", "pinned"}`.
- `POST /v1/prompt-cache/park` `{"session"}`: saved to disk (what is not
  there yet), then freed from memory; the next request reads it back.
  Returns `{saved, bytes, freed, in_memory}`. One Mac's models only.
- `GET /v1/prompt-cache`: `data: [{session, role, run, tokens, bytes,
  in_memory, on_disk, saved_at, pinned, hash, model, key_id}]`.

On the page, name the model with a `model` query or body field when it
serves more than one.

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
