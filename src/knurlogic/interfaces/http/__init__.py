"""knurlogic's own server: HTTP here, the model in engine/runtime.

    knurlogic serve <artifact>

lifecycle.serve() builds the host and the scheduler, queues the load, and
answers HTTP at once: /status.json and the page work while the model
loads, and inference requests wait for it (they are queued, not refused).

  lifecycle.py     serve, switch, unload, the ring watch, binding
  server.py        the HTTP server: routes, one thread per connection
  guard.py         loopback, browser (Origin/Host) and hang-up checks
  respond.py       the JSON reply writer of the page's handlers
  dialect.py       what the translated dialects below share
  openai.py        /v1/chat/completions, /v1/completions, /v1/models
  messages.py      the Anthropic Messages API, over the OpenAI one
  responses.py     the OpenAI Responses API, over the OpenAI one
  ollama.py        the Ollama API, over the OpenAI one
  residency.py     what the server holds: capabilities, ensure, unload
  prompt_cache.py  /v1/prompt-cache: save, drop, pin, park, list
  compaction.py    the chat path through context_management/
  telemetry.py     request ids, the ledger row, progress, /v1/usage
"""
