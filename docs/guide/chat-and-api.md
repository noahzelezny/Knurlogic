# Chat and the HTTP API

Every loaded model is an HTTP server (port 8080 by default) that speaks the
OpenAI, Anthropic Messages, OpenAI Responses and Ollama shapes. The page
(port 8899) also forwards chat requests to whichever running model they
name.

## Two addresses

| address | use it when |
|---|---|
| the model's own port, e.g. `http://127.0.0.1:8080` | one model; everything below is served here |
| the page, `http://127.0.0.1:8899` | several models, here or on other Macs: it forwards `POST /v1/chat/completions`, `/v1/messages` and `/v1/messages/count_tokens` to the server whose model id the request's `model` names. Its `GET /v1/models` lists every model it can reach, with the machine each runs on. |

`GET /v1/models` on a model server returns the one model it serves. Use its
`id`, or on the page, the id you want routed to.

## Connect a client

The page's **Connect** panel shows these for the models running now, and
`knurlogic connect --port 8080` prints them for one server.

**Claude Code** (or any Anthropic Messages client):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=x \
ANTHROPIC_DEFAULT_OPUS_MODEL=local \
ANTHROPIC_DEFAULT_SONNET_MODEL=local \
ANTHROPIC_DEFAULT_HAIKU_MODEL=local \
API_TIMEOUT_MS=3000000 \
claude
```

The API key is not checked but must be set. The long timeout is there
because a local model is slower than the hosted one. Pointed at the page
(`:8899`), each tier can name a different running model.

**OpenAI-compatible clients** (Zed, Cline, Continue, Open WebUI): base URL
`http://127.0.0.1:8080/v1`, any API key, model `local`.

**Ollama clients**: `OLLAMA_HOST=http://127.0.0.1:8080`.

**MCP**, for an agent that manages models: see [mcp.md](mcp.md).

## Endpoints on a model server

| endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat (also `/chat/completions`) |
| `POST /v1/completions` | OpenAI text completion |
| `POST /v1/messages` | Anthropic Messages, including tool use and streaming |
| `POST /v1/messages/count_tokens` | Anthropic token count |
| `POST /v1/responses` | OpenAI Responses: function tools, streaming, reasoning summaries |
| `POST /api/chat`, `/api/generate`, `/api/show`; `GET /api/tags`, `/api/version` | Ollama (streaming NDJSON by default) |
| `GET /v1/models` | the model: `id`, `status` (`loading`, `ready`, `failed`, or why memory is short), `capabilities` (text, vision, thinking), `size_bytes`, `context_length`, `sampling_defaults`, `thinking` levels |
| `GET /health` | `{"status": "ok", ...}` |
| `GET /v1/residency` | the model, its memory, state, requests waiting and running, recovery |
| `POST /v1/ensure` | load or switch to a model ([loading-models.md](loading-models.md)) |
| `GET /v1/usage` | request rollups, loopback only ([usage-and-telemetry.md](usage-and-telemetry.md)) |
| `/v1/prompt-cache...` | the prompt cache, loopback only ([prompt-cache.md](prompt-cache.md)) |

The Responses API keeps no conversation: `previous_response_id` and
`store: true` are refused, so resend the input.

A request from a browser is answered only for the server's own page or an
origin allowed with `--allow-origin`.

## Sampling

A parameter you leave out takes the model's own recommendation (from its
`generation_config.json`), listed in `/v1/models` as `sampling_defaults`; a
model that recommends nothing is greedy. What was used is reported in
`usage.knurlogic.sampling`.

Accepted: `temperature`, `top_p`, `top_k`, `min_p`, `xtc_probability`,
`xtc_threshold`, `seed`, `repetition_penalty`, `presence_penalty`,
`frequency_penalty` (and their `*_context_size`), `logit_bias`, `stop`,
`logprobs`, `top_logprobs` (0-20), `max_tokens` or `max_completion_tokens`.
Without `max_tokens` the answer may use the rest of the context window.
`n` above 1 is refused: send the request n times (they are batched).

## Thinking

One control for every model: OpenAI's `reasoning_effort` (or
`reasoning: {"effort": ...}`), on the ladder
`none < minimal < low < medium < high < xhigh`. Knurlogic translates it to
the model's own chat-template controls. A level the template does not have
goes to the nearest one **at or above** it; `none` on a template with no off
switch goes to its lowest level. A word not on the ladder is refused (400).
There is no token budget.

- What was applied is in `usage.knurlogic.thinking`: `requested`,
  `applied`, `native`, and a `note` when it differs.
- `GET /v1/models` lists the model's levels under `thinking`; the MCP
  `models` tool says what each ladder level maps to.
- Reasoning is streamed by default. `reasoning: {"exclude": true}` leaves
  it out. Its count is `usage.completion_tokens_details.reasoning_tokens`.
- Your own `chat_template_kwargs` win, key by key.
- A request that names no level gets the model's **Thinking default**
  (Settings -> Models), or the template's own default.

On `/v1/messages`: `thinking: {"type": "disabled"}` asks for none;
`{"type": "enabled"}` keeps the model's default and returns thinking
blocks. `budget_tokens` is ignored. A `reasoning_effort` field is passed
on.

## Tools

Send OpenAI `tools` or Anthropic `tools`; calls come back as `tool_calls`
or `tool_use` blocks, streaming or not. Whether a model emits good calls is
the model's; `knurlogic doctor <model>` says which tool-call dialect its
template uses.

`tool_choice`: `"auto"`, `"none"`, `"required"` or
`{"type": "function", "function": {"name": ...}}`. On `/v1/messages`:
`auto`, `any` (= required), `none`, or `{"type": "tool", "name": ...}`.
Anything else is a 400.

## Images

Send `image_url` parts with `data:` URLs (or Anthropic base64 image
blocks) to a vision model. URLs are not fetched. See [vision.md](vision.md).

## Long conversations

- `usage.knurlogic.context = {tokens, window}` in every chat response says
  how full the window is.
- A prompt past the context length is refused (400), and `max_tokens` is
  trimmed to fit.
- The server can compact the conversation for you: [compaction.md](compaction.md).
- The prompt cache makes the next turn read only what is new:
  [prompt-cache.md](prompt-cache.md).

## Headers

Request headers, all optional:

| header | |
|---|---|
| `X-Request-Id` | your id for this request (1-128 printable ASCII); echoed back and used as its ledger row id. Otherwise the server makes one. |
| `X-Client` | `<name>/<version>` of your client |
| `X-Client-Session` | your conversation or job id; also owns its prompt-cache entries |
| `X-Client-Run` | a sub-unit (a worker, a sub-task) |
| `X-Client-Role` | a short label (`pm`, `worker`, ...) |
| `X-Cache-Keep`, `X-Cache-Retain` | prompt-cache policy ([prompt-cache.md](prompt-cache.md)) |

Response headers:

| header | |
|---|---|
| `X-Request-Id` | this request's id |
| `X-Knurlogic-Concurrency` | `rows=N, more=?1\|?0`: how many requests decode together now, and whether one more is measured to help (`more` left out when not measured) |

## Streaming

Server-sent events in each API's own shape. While a long prompt is read,
knurlogic sends progress:

```
event: knurlogic.progress
data: {"request_id": "...", "phase": "prefill", "done": 1840, "total": 3200, "tps": 410.2, "queue": {"ahead": 0}}
```

`phase` is `queue` while waiting to start (every second) and `prefill`
while reading the prompt. `/v1/messages` streams always carry it; on the
OpenAI shapes it is sent only to a client that names itself with
`X-Client` (the OpenAI SDK does not skip unknown events). Ollama streams
have no events.

A client that disconnects cancels its request, streaming or not.

## Errors

OpenAI-shaped: `{"error": {"message", "type", "param", "code"}}`.

| status | meaning |
|---|---|
| 400 | a bad request: invalid JSON, `n > 1`, an unknown `reasoning_effort`, a bad `tool_choice`, images sent to a model without vision, a prompt past the context length |
| 413 | the body is over `--max-request-mib`, or an image is too large |
| 503 + `Retry-After: 5` | no model loaded |
| 503 + `Retry-After: 10` | not enough memory now (`insufficient_memory`) |
| 503 + `Retry-After: 30` | a cluster job failed (`cluster_failed`) |

Nothing is refused for queue length: requests wait.
