# Changelog

## 0.1.2

DeepSeek-V4-Flash in full, MTP on a tensor split, and a cluster that stops
cleanly.

DeepSeek-V4-Flash
* Tool calls work: a release that ships a copy of knurlogic's DeepSeek
  template now gets the DSML tool-call parser (on 0.1.1 the calls came back
  as text, so agent harnesses could not use it). A reply that is only a
  tool call carries no blank text part. A DeepSeek-V4 shipped with no chat
  template at all gets knurlogic's instead of refusing chat.
* MTP drafting with its head beside the weights, exact (mxfp4) or VQ
  experts: about 1.4x faster decode on one Mac at ~0.95 acceptance.
* Tensor split across machines (2 or 4), with or without MTP.
* Thinking levels are DeepSeek's own: off, high, max (the official Think
  Max prompt).
* It reasons in the user's language: the chat page sends the date and
  browser language as DeepSeek's own reminder message.

Clusters
* Stopping a cluster job no longer leaves the GPUs reading 100% until a
  reboot -- while it generated, holding the job's memory too: rank 0 stops
  the ring between steps, and a load cancelled mid-read stops between
  batches, so every machine exits cleanly (unload, cancel and warm-up,
  tensor and pipeline).
* MTP drafting on a tensor split (the 397B VQ-2.4 runs ~37-40 tok/s on two
  Macs over RDMA). Flash-Next (qwen4_exp) splits tensor too.
* Tensor is offered from the model's own weight shapes, one rule table for
  the picker and the loader, and greys out (with the reason) when a
  machine cannot hold its share.
* The MTP head counts toward rank 0's share only when MTP is on.
* A cluster load's % measures each machine against its own share; the
  card shows warming up and stalled (naming the machine).
* `load(draft=false)` works on a cluster.
* A rank whose prompt processing fails ends the job with the reason
  instead of leaving the other machines waiting.

Requests
* A seeded request reproduces with MTP on.
* Thinking off uses the maker's thinking-off sampling (Qwen3.5).

Page
* The page fills the window however wide (zoomed out too): chats on the
  left edge, the model panel on the right, the chat bar at the bottom, and
  the machines across the middle.
* Split and link choices grey out while a launch is in flight.
* Instance cards show MTP when drafting, the load % in teal, one card per
  exo instance, and the GiB of every machine in a cluster job.
* The picker remembers each model's split answers across restarts and
  re-reads the other machines' model lists when it opens.

For tools
* `knurlogic.engine.register.register()` / `unregister()` are public API.

## 0.1.1

Bug fixes from the first day of real use, and simpler launch decisions.

Launching
* A launch either fits or is refused with the numbers (weights, MTP head,
  vision, margin, budget). When turning MTP off would make it fit, the page
  offers "Turn MTP off and launch"; an agent is told to retry with
  `draft=false`.
* Vision can be turned off at launch, like MTP, and the page shows what
  each costs (e.g. "MTP 6.1 GiB", "Vision 1.5 GiB"). Off frees that memory
  for headroom and a wider prompt chunk; an image sent to it is refused
  with a clear message.
* A model is found by its own folder in each store instead of a scan of
  every store: launching on a Mac whose models are on a network share went
  from about a minute to seconds.
* The model picker lists models in under a second after the page starts
  (was up to 40 s): each model's identity is remembered on disk and only
  recomputed when its files change.
* Unloading waits for the server to exit, and a launch re-reads every
  machine's memory first, so a load right after an unload is no longer
  refused for memory still being freed.
* A model reads "ready" only once its weights are in memory, and
  `/v1/models` carries `status: "loading"` until then, so a client timing
  its first request no longer times the load.

Speed
* Prompts are processed in the widest chunk the model's memory reserve
  allows (was often 512): prefill up to ~2x faster on large models, on one
  Mac and across a cluster (a cluster uses the smallest rank's chunk).
* Responses report the prompt chunk and how many prompt tokens came from
  the cache versus were computed (`usage.knurlogic.timing`).

Clusters
* A pipeline follower that sat idle no longer reads as over its memory
  limit and refuses every prompt.
* With one Thunderbolt cable, a cluster launch with no link chosen uses
  TCP over it; with several, it names them and asks.
* A cluster launch shows one card, even when the Macs hold the model under
  different folder names, and every Mac loads the folder that was picked.
* Each rank prints which runtime it runs (stock mlx-lm and its version, or
  the model's own `model.py`).

Memory display
* "Used" counts what macOS really cannot hand out (inactive anonymous
  pages are not free); swap is shown when present.
* The legend's catch-all row is "other".
* The memory map refreshes every 3 s around loads and requests.

Page
* A model answering a request shows a turning gear on its card.
* The memory diagram fits narrow windows and browser zoom without
  crushing; the context counter resets on a new chat.

## 0.1.0

First public release (alpha).

* `knurlogic ui` shows a gear in the macOS menu bar (models loaded, Open,
  Quit); `--no-menubar` opts out, and it is skipped without a GUI session.
* `knurlogic serve`: an OpenAI-compatible server (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`) with an Anthropic Messages endpoint
  (`/v1/messages`) for Claude Code and similar harnesses, the OpenAI
  Responses API (`/v1/responses`), and the Ollama API (`/api/chat`,
  `/api/generate`, `/api/tags`, `/api/show`, `/api/version`; set
  `OLLAMA_HOST=http://127.0.0.1:8080`).
* Settings resolved from each model's `config.json` and the memory
  available, with launch bundles (`--tune`) and per-setting overrides.
* `doctor`, `models`, `loaded`, `deps`, `smoke`: whether a model fits and
  will run, what is on disk, what is in memory in every runtime, and which
  build of the stack is installed.
* The page (`knurlogic ui`): models, memory, chat with images, settings,
  launch and stop.
* An MCP server (`knurlogic mcp`) for agents: `models`, `fit`, `settings`,
  `drafting`, `ready`, `load`, `state`, `unload`, `deps`.
* Model families: Qwen 3.5/3.6/3.8, Gemma 4, GLM-5, DeepSeek-V4, with
  vendored, pinned architectures; VQ-quantized models, each running the
  `model.py` it ships (knurlogic carries no VQ runtime of its own; a VQ
  model without one is refused and should be re-downloaded).
* An "update" tag in the model picker when the Hugging Face copy of a
  downloaded model has a newer revision; clicking it opens the download
  dialog. Checked once per page start; `knurlogic ui --offline` or
  `HF_HUB_OFFLINE=1` skips it.
* Multi-token-prediction drafting for models that ship a head; images for
  Qwen, Gemma 4 and GLM-5; 8-bit KV cache; YaRN long context for Qwen;
  context compaction.
* Clusters of 2 to 16 Macs: one model split by tensor or pipeline over
  Thunderbolt (TCP, or RDMA; RDMA beyond two Macs is experimental), with
  Bonjour discovery and `--host cluster`. Macs on Ethernet or Wi-Fi join
  only when named with `--peer` on each side. A versioned control protocol
  between machines: a Mac on an older knurlogic is named and refused, a
  restarted or lost machine stops the job, and launches are refused before
  loading when any Mac lacks the free memory for its share.
* Launch presets are `default` and `lean` (`--tune`, the page, the MCP);
  `fast` and `stable` name `default`, `safe` names `lean`. The Knurlogic
  settings tab has a row for each thing a preset sets, so a custom set is
  the preset plus the rows that differ; per-chip rounding is its own
  setting.
* A load that would be refused is refused before anything starts, from the
  page, the MCP and a cluster job, in the page's words. `serve` exits 78
  with a `REFUSING:` line and is never relaunched; a missing module is a
  refusal too.
* A context past a model's native window turns on YaRN where the family
  has it, and is lowered to what the model reaches otherwise.
* One cache limit, `KNURLOGIC_CACHE_LIMIT_GB`, mirrored to the VQ runtime's
  names. The page offers KV bf16 and 8-bit; 6 and 4 are `--set`.
* An omitted `max_tokens` on `/v1/chat/completions` and `/v1/completions`
  is the rest of the context window (it was 512); the page's chat sends no
  sampling settings.
* Hugging Face in the picker: search MLX models, see whether one runs here,
  download it, and manage downloads in the Downloads overlay (stop, resume,
  delete).
* A model's identity includes the `*.py` files it ships: identical copies
  are one model, and a local copy that differs from a shared one yields to
  the shared one, with an alert.
* The settings page is redesigned; models that are loading, preparing or
  failed have their own cards.

### Renamed settings

* `VQLAB_CACHE_LIMIT_GB` is now `VQ_CACHE_LIMIT_GB`, matching the VQ
  runtime. The old name is still accepted from saved settings and `--set`,
  and is still set for models whose bundled runtime reads only it; it will
  be removed in a later release.
* `VQLAB_PREFILL_CHUNK` is now `KNURLOGIC_PREFILL_CHUNK`. The old name is
  still accepted from saved settings and `--set`; it will be removed in a
  later release.
