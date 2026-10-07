# Fleet: request records, attribution, and the control plane

## What someone pays for

One person on one Mac gets everything for free and should. A team with
six Macs under a desk has questions the free product does not answer: who
used what, is the fleet healthy, which model is worth the memory it holds,
can one person's agent starve the others, what happened last night. Those
are the questions a paid tier answers, and all of them are server-side by
nature: every client goes through the server, so the server is the only
place the answers exist.

```
solo   (free)   one Mac, no keys, page + MCP as today, usage visible locally
team   (paid)   keys -> owners, per-key usage and quotas, fleet dashboard
                across peers, usage export, alerts
enterprise      SSO (OIDC) for the page and keys, retention policy, audit
                export, support
```

The free tier gets the **request ledger** too. It is what makes the paid
tier possible and it costs nothing to keep; a solo user sees their own
numbers and the product is better for it.

## Design rules

- **Count, do not keep.** The ledger stores token counts, timings and
  labels; never prompt or completion text. A row is safe to export by
  construction. Text capture is a separate, opt-in switch with its own
  retention, off by default, and the page says so where it is turned on.
- **Labels are opaque.** A client may send `X-Client-*` headers; the server
  stores the strings and groups by them. It does not know what a session
  or a role is, so every harness gets attribution and none is special.
- **Keys are the only identity.** A request is attributed to the key it
  presented, or to `anonymous`. Users, teams and SSO map onto keys; the
  ledger never changes shape when they arrive.
- **Quotas refuse at admission, in `interfaces`.** The engine never sees a
  key. A refused request is a 429 with the arithmetic attached (used,
  limit, resets), in the shape the client reads as an error.
- **The ledger is a ring.** Bounded by days and bytes, oldest rows go
  first, export before they do. Nothing grows without a limit.
- **Peers aggregate, they do not forward.** Each machine keeps its own
  ledger; the page asks peers for summaries over `/peer/v1/msg`, the same
  path it already uses for residency. No central database.

## Layering (architecture.md: who may import whom)

```
interfaces/http/server.py     opens a Request at the handler, closes it on
                              the last byte: labels, key, status, bytes
interfaces/http/{openai,messages,responses,ollama}.py
                              hand usage + timings to the open Request
engine/runtime/scheduler.py   already measures admit/prefill/decode; it
                              reports them on the job, as it reports
                              requests(); it never imports the ledger
machine/ledger.py             the ring: schema, insert, query, retention
machine/keys.py               keys file, lookup, quota counters
interfaces/page/              /usage.json, /keys.json, the Usage tab
interfaces/mcp.py             `usage` tool (read-only)
```

Rule 3 holds: `engine/` and `machine/` never import `interfaces/`. The
scheduler attaches timings to the job object; the HTTP layer reads them
off when the response closes and writes the row. `machine/ledger.py` is
the only module that opens the ledger file.

## The request record

One row per completed (or refused) request. Written once, at close.

| column | from |
|---|---|
| `id` | ULID, also returned as `X-Request-Id` and in `usage.knurlogic.request_id` |
| `ts_start`, `ts_end` | handler |
| `machine`, `job`, `rank` | `machine.identity`, cluster job if any |
| `model` | the served artifact id |
| `api` | `chat` / `messages` / `responses` / `ollama` / `completions` |
| `key_id` | the presenting key, or `anonymous` |
| `client`, `session`, `run`, `role` | `X-Client`, `X-Client-Session`, `X-Client-Run`, `X-Client-Role`, verbatim, each ≤ 128 bytes |
| `prompt_tokens`, `cached_tokens`, `output_tokens` | what `_usage` already computes |
| `queue_ms` | submitted → admitted (scheduler) |
| `prefill_ms`, `prefill_tps` | admitted → first token |
| `decode_ms`, `decode_tps` | first token → finish |
| `finish` | `stop` / `length` / `tool_calls` / `error` / `cancelled` / `refused` |
| `status` | HTTP status |
| `memory_at_admit` | working set bytes (`machine.metrics`) |
| `thinking` | level applied, if any |

Stored in SQLite at `KNURLOGIC_HOME/ledger.db` (WAL), one table, indexes on
`(ts_start)`, `(key_id, ts_start)`, `(model, ts_start)`. Retention: 30
days or 256 MiB, whichever first, settable.

## Keys

`KNURLOGIC_HOME/keys.json`, managed from the page and the CLI
(`knurlogic keys new|list|revoke`). A key is `kl_<32 url-safe chars>`;
the file holds its SHA-256, an owner label, a team label, created/revoked
times and quotas. Presented as `Authorization: Bearer` or `x-api-key`,
both already allowed by CORS.

Mode, per server: `--auth off | optional | required`. Default `optional`
everywhere: today's behaviour, dummy keys still work and are attributed to
`anonymous`. Nothing is required until the ledger and keys have been used
for real and proved themselves. The page shows a one-line notice when a
server is bound off-loopback with no key required, because such a server
is one anyone on the LAN can run out of memory; `required` stays a choice.

Quotas per key, each optional: `tokens_per_day`, `requests_per_minute`,
`concurrency`. Counted in `machine/keys.py` from the ledger plus an
in-memory window; enforced before `scheduler.submit`. Over quota →
`429 {"error": {"type": "rate_limit_error", "message": "...used/limit,
resets at..."}}`, a `refused` ledger row, and the page shows it.

## Reading it back

- `GET /v1/usage?since=&until=&group=key|model|client|session|role&key=`
  on the model server: summaries, JSON. Requires a key with `admin` or the
  loopback operator.
- `GET /metrics`: Prometheus text; counters and histograms for the same
  columns, labelled by model and key.
- Page `/usage.json` aggregates its own ledger and each peer's summary,
  and the **Usage** tab shows: tokens and requests per model, per key, per
  client over 1h/24h/7d; TTFT and decode tok/s p50/p95; queue depth and
  wait; memory headroom; refused requests with reasons; the last 50
  requests as a table. One screen, no drill-down to text (there is none).
- MCP `usage` tool: the same summary for an agent that wants to know what
  it is spending.
- `knurlogic usage [--since 24h] [--group model]` prints it.
- Export: `knurlogic usage export --since --until > rows.jsonl`.

## Live progress (the harness's request)

Prefill progress is already on the wire for OpenAI streaming: the
scheduler's hook puts `("progress", (done, total))` on the job outbox per
chunk (`engine/runtime/scheduler.py:1206`) and `http/openai.py:336` writes
`: keepalive <done>/<total>`. The Messages path flattens it to a bare
`ping` (`http/messages.py:326`). Promote it to a named event on every
streaming API:

```
event: knurlogic.progress
data: {"request_id": "...", "phase": "prefill", "done": 1840, "total": 3200,
       "tps": 410.2, "queue": {"ahead": 0}}
```

Non-streaming requests get nothing new. Clients that do not know the event
ignore it (SSE rule). the harness's status line reads it; so can the page's own
chat. Also `queue` while waiting for admission, with `ahead` from
`scheduler.requests()`, every second.

## Fleet

The page already knows its peers. Usage joins residency: `/usage.json`
asks each peer for `{summary, last_n}` with the same query, merges, and
marks which machine each row came from. A cluster job's requests carry
the job id, so a split model's usage is one line, not one per rank.
Alerts (team tier): a peer unreachable, memory headroom under a threshold,
a key over 80% of quota — shown on the page, posted to a webhook URL if
one is set. No email, no app; a webhook is enough for Slack.

## Order of work

1. **Ledger + labels** — `machine/ledger.py`, the Request object in
   `http/server.py`, scheduler timings on the job, `X-Request-Id`,
   `usage.knurlogic.request_id`. ~300 lines + tests. Every client gains
   attribution the day this lands.
2. **Progress event** — `on_chunk` → SSE `knurlogic.progress`. ~80 lines.
3. **`/v1/usage`, `/metrics`, `knurlogic usage`** — read side. ~250 lines.
4. **Page Usage tab** — `/usage.json` + one screen. Peers folded in.
5. **Keys + quotas** — `machine/keys.py`, `--auth`, 429 path, page keys
   panel. This is the paid line.
6. **Export, retention settings, alerts/webhook.**
7. **OIDC for the page** (enterprise), mapping groups → key owners.

## Tests

- `tests/machine/test_ledger.py`: ring retention by days and bytes; a row
  per finish kind; labels stored verbatim and truncated at 128.
- `tests/api/test_attribution.py`: headers → row; no headers → anonymous;
  `X-Request-Id` echoed; `usage.knurlogic.request_id` present on every API.
- `tests/api/test_quota.py`: over quota is 429 with the arithmetic, a
  `refused` row, and never reaches the scheduler.
- `tests/integration/test_layers.py` gains: only `machine/ledger.py`
  opens `ledger.db`; `engine/` never imports it.
- `tests/api/test_progress.py`: a 3-chunk prefill yields 3 progress events
  before the first token; non-streaming yields none.
- A privacy test: no column of the ledger ever contains request text, by
  inserting a sentinel string in a prompt and grepping the db file.
