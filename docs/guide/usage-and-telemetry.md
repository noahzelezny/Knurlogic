# Usage and telemetry

Every request a model server answers is counted in a **ledger** on that
Mac: token counts, timings and your client's labels. Never the prompt or
the answer text.

## In each response

`usage.knurlogic` in the final usage object of every API (OpenAI,
Messages, Responses, Ollama):

| field | |
|---|---|
| `request_id` | this request's id, the same as the `X-Request-Id` response header |
| `timing` | `queue_ms`, `prefill_ms`, `decode_ms`, `prefill_tps`, `decode_tps`, and more when measured (time to first token, cached vs computed prompt tokens, the prompt chunk) |
| `cache` | what the prompt cache supplied: `used` tokens, `disk` (tokens read back from disk), `diverged` when a session's prompt parted from its cache ([prompt-cache.md](prompt-cache.md)) |
| `context` | `{tokens, window}`: this prompt against the model's window (chat) |
| `thinking` | the thinking level requested and applied ([chat-and-api.md](chat-and-api.md)) |
| `sampling` | the sampling used, and which values came from the model's defaults |
| `compaction` | the summary pass, when compaction ran ([compaction.md](compaction.md)) |

Standard fields as usual: `prompt_tokens`, `completion_tokens`,
`prompt_tokens_details.cached_tokens`,
`completion_tokens_details.reasoning_tokens`.

`GET /v1/models` carries `"knurlogic": {"telemetry": 1}`, the version of
this contract.

## Labelling your requests

Send these headers to have your work grouped; all optional, stored as
opaque strings (at most 128 bytes; the role 32):

| header | |
|---|---|
| `X-Client` | `<name>/<version>` |
| `X-Client-Session` | one conversation or job |
| `X-Client-Run` | a sub-unit: a worker, a sub-task, or `-` |
| `X-Client-Role` | a short label: `pm`, `worker`, `sidecar`, ... |
| `X-Request-Id` | your own id for the request (1-128 printable ASCII). Echoed back exactly and used as the ledger row's id; reusing an id replaces the earlier row. Invalid values are ignored. |

Record the returned `X-Request-Id` beside your own ids and any ledger row
can be joined to your trace.

## Reading it back

```bash
curl 'http://127.0.0.1:8080/v1/usage?group=session&since=1759900000'
```

On the model server's port, from the Mac itself (loopback) only; anything
else gets 403.

| parameter | |
|---|---|
| `since`, `until` | epoch seconds (default: everything) |
| `group` | `model` (default), `key`, `client`, `session`, `run`, `role`, `api` |
| `key` | only this key's rows |

Each row of `summary`: the group value, `requests`, `prompt_tokens`,
`cached_tokens`, `output_tokens`, `errors`, `cancelled`, `queue_ms_avg`,
`prefill_tps_avg`, `decode_tps_avg`.

Every request is attributed to key `anonymous` today.

## The ledger file

`~/.knurlogic/ledger.db` (SQLite; `KNURLOGIC_HOME` moves it). One row per
request, written when it ends, with its `finish`: `stop`, `length`,
`tool_calls`, `error` or `cancelled`. It is a ring:

| variable | default | |
|---|---|---|
| `KNURLOGIC_LEDGER_DAYS` | 30 | rows older than this are deleted |
| `KNURLOGIC_LEDGER_MIB` | 256 | the oldest rows go while it is bigger |

## Live progress

Streaming responses carry `event: knurlogic.progress` while a request
waits (`phase: queue`, with how many are `ahead`) and while its prompt is
read (`phase: prefill`, `done` / `total` tokens, `tps`). See
[chat-and-api.md](chat-and-api.md#streaming) for which APIs send it.

## Requests running and waiting

`requests` in `/v1/residency`, the page and MCP `state`: `in_flight`,
`pending`, `capacity`, `oldest_pending_s`, and `holding` (why they wait:
`loading`, `memory`, `batch_full`, `queued`).

Design: [../design/telemetry.md](../design/telemetry.md) (the contract),
[../design/fleet.md](../design/fleet.md) (what is planned beyond it).
