# The HTTP server and the page

Two HTTP servers, both stdlib, both in `interfaces/` (the only package
that may open one):

- the **model server**, one per loaded model (`knurlogic serve`): the API
  clients call;
- the **page** (`knurlogic ui`, and `knurlogic` with no command): the
  view of every model and runtime on this Mac and its peers. It never
  imports mlx and holds no model; loading starts a model server as a
  child process.

The why: [server](../design/server.md). The clients' shared contract:
[telemetry](../design/telemetry.md). Downloads from Hugging Face:
[huggingface](../design/huggingface.md).

## Where the code is

The model server, `src/knurlogic/interfaces/http/`:

| file | what |
|---|---|
| `__init__.py` | `serve` (build the scheduler and server), `scheduler_options`, `watch_ring`, `switch`, `unload`, `bind_all` |
| `server.py` | `App` (the scheduler, what is served, hooks), `Handler` (`do_GET`, `do_POST`), `make_server`; `browser_refusal` and `host_is_local` (Origin and Host checks) |
| `openai.py` | `/v1/chat/completions`, `/v1/completions`, `/v1/models`: `build_job`, `Reply`, `ApiError`, `models_document` |
| `messages.py` | the Anthropic Messages API as a translation over the OpenAI one (`to_openai`, `from_openai`, `stream`) |
| `responses.py` | the OpenAI Responses API, the same way |
| `ollama.py` | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/version` |
| `residency.py` | `/v1/residency` and `/v1/ensure` for orchestration clients |
| `prompt_cache.py` | `/v1/prompt-cache` ([prompt-cache](prompt-cache.md)) |
| `compaction.py` | `CompactingChat`, `App`'s chat path through context management ([compaction](compaction.md)) |
| `telemetry.py` | request ids, the ledger `Request`, progress events, `GET /v1/usage`: see [telemetry](telemetry.md) |

`interfaces/serve.py` is `knurlogic serve`: it checks and resolves the
settings (`tuning/checks`: `launch_refusal`, `settings_refusal`,
`refuse_sets`), then
`run` starts the scheduler and server, or a follower rank.
`interfaces/load_checks.py` (`prepare`, `NotLoadable`) is the check every load
passes, at startup and on every switch. `interfaces/spawn.py` starts,
lists and stops the `knurlogic serve` children (`spawn`, `children`,
`loading`, `stop`, `SERVE_PORT`): the page and the MCP share it.

The page, `src/knurlogic/interfaces/page/`:

| file | what |
|---|---|
| `server.py` | `knurlogic ui`: `main`, `serve_ui`, `make_handler` (every route and the guards in front of them), `_wire` (what `cluster/launch` and `recovery` need of the page) |
| `nodes.py` | the machines this page sees: `PEERS`, Bonjour (`_start_discovery`), `/status.json` (`_status_fn`) and the light liveness document (`_status_light`, `hot`) |
| `loads.py` | load and unload from the page: `_load_fn` (POST `/loaded.json`), `tracked_load`, `forward_launch` (one peer), `cluster_launch`, `peer_launch` (the peer side); GET `/loaded.json` (`_loaded_fn`, `with_jobs`, `load_progress`) |
| `router.py` | the router to model servers: `route`, `ROUTE_PATHS`, `routable`, `route_models_document`, `proxy_chat`, `chat_targets`, `_stream` (with `cluster_failure`), `CLIENT_HEADERS` |
| `peers.py` | the peer gate (`peer_refusal`), what peers serve (`peer_residency`, `upstream`, `MSG_PATH`, `PEER_RELAY`), a machine's settings from any page (`peer_machine`, `machine_apply`) |
| `relay.py` | `peer_relay`: `/peer/v1/...`, a peer page reaching a model this machine started |
| `peek.py` | `/peek` and `/apply` (a model's or a peer page's settings), and their peer side `peer_settings` |
| `prompt_cache.py` | `prompt_cache_forward`: the page's `/v1/prompt-cache`, sent on to the model server that serves the model |
| `messages.py` | `peer_table`: the protocol kinds a page answers on `MSG_PATH` (`survey_here`, `read_here`) |
| `documents.py` | the routes shared by the page and `serve`: `routes(...)` (`/status.json`, `/settings.json`, `/models.json`, `/loaded.json`, `/connect.json`), `load_action`, `machine_settings`, `settings_document`, `knob_limit`, `compaction_document` (page JSON over `tuning/`) |
| `hub.py` | Hugging Face search, download, cancel, delete |
| `updates.py` | is a model or knurlogic out of date (asked once per page start) |
| `assets/` | the page itself: `index.html`, `app.js`, `api.js`, `views/`, `page.css`, shipped as package data |

## The page's routing

- **Router.** `POST /v1/messages`, `/v1/chat/completions` and
  `/v1/messages/count_tokens` on the page are forwarded (`route`) to the
  server whose model id is the body's `model`, here or on a peer. One
  base URL for a harness that names a model per tier.
- **Peer relay.** `/peer/v1/...` on a page reaches the model servers that
  page started, by model name (`peer_relay`). `upstream(base, path)` picks
  the server itself or the peer page's relay.
- **Control plane.** `/peer/v1/msg` (`MSG_PATH`) is the one page-to-page
  route ([cluster](cluster.md)). Peer routes pass `peer_refusal` first.
- **Settings on a peer.** `/peek` reads and `/apply` changes a peer's
  model settings through that peer's page.

## Rules that keep it correct

- **Every model operation goes to the scheduler.** A handler builds a
  `Job` and reads its outbox; it never touches the model. A failed write
  (the client hung up) cancels the job.
- **Translations stay translations.** Messages, Responses and Ollama are
  converted to and from the OpenAI chat shape and go through the same
  transport; a feature is built once on the OpenAI path.
- **Local by default.** A request with an Origin is answered only for
  this server's own origin or one allowed with `--allow-origin`; the Host
  header must name this machine (`host_is_local`).
- **The page holds no model and imports no mlx.** It starts `serve` as a
  child, which owns the model.
- **A refusal is an answer.** `ApiError` carries a status, a message and
  `retry_after` where waiting helps (a load in progress, memory short).

## Extending

- A new endpoint on the model server: a branch in `Handler.do_GET` /
  `do_POST` (or a mixin like `PromptCacheHandlers`), the logic in its
  own module.
- A new API shape: a translation module with `handler_over(transport,
  model)`, like `messages.py`.
- A new page document: a route in `documents.routes` (shared with
  `serve`) or in `page/server.py` (`serve_ui`) if only the page serves it, and the
  view in `assets/views/`.
- Something a peer must reach: through `peer_relay` or a protocol
  message, never a new direct route between pages.

## Tests

`tests/interfaces/` (`test_http_server.py`, `test_messages.py`,
`test_responses_ollama.py`, `test_tool_choice.py`, `test_ui_router.py`,
`test_ui_peer_relay.py`, `test_ui_peer_residency.py`, `test_ui_peek.py`,
`test_ui_load_progress.py`, `test_load_refusals.py`, `test_hub_delete.py`,
`test_updates.py`, `test_web_vision.py`, ...) and
`tests/api/test_conformance.py`.
