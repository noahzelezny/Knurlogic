# Changelog

## Unreleased

* A per-model "Thinking default" setting: the thinking level a request
  that names none is served at ("model" keeps the template's own). For a
  client that sends no reasoning_effort, or whose control is broken --
  GLM-5.3's own default is max, and long conversations thought for hours
  without answering. A request that names a level still wins; live, so a
  change applies to the next request.
* GLM-5.3: a long prompt's prefill attends in the MLA latent over only the
  tokens the sparse indexer picked, instead of expanding every cached token
  into per-head K/V and masking it away: prefill memory no longer grows
  with the context (a 334k-token prompt had shrunk the chunk to 256 and was
  still killed at 98% on a 96 GB M3 Ultra). Same logits as before to
  rounding. Measured on one layer at GLM-5.3-Flash's shapes (M3 Ultra,
  chunk 2048): 131k tokens 0.47 s and 8.1 GiB against 2.72 s and 42 GiB;
  32k 0.30 s / 7.4 GiB against 0.67 s / 11.4 GiB.
  tools/bench_glm5_sparse_prefill.py measures your Mac. The memory guard
  reads a GLM prefill chunk as spanning what it reads (min(context, 2048)
  + context / 72), not the whole context: one warm-up sample no longer
  refuses a 339k-token prompt "even prefilled 128 tokens at a time".

## 0.1.4

Long generations hold up; `knurlogic` starts the page.
* A long prefill says how far it is: rank 0 logs `prefill <uid>: <done>/<total> tokens, chunk <c>, <tok/s>` every 8 chunks or 10 s, and every chunk's progress goes to the client as a keepalive, so a client that hung up mid-prefill is noticed. On a single Mac the prefill then stops at the next chunk; on a ring (tensor or pipeline) it runs to the end of that step, where the row is dropped -- every rank must run the same forwards.
* When macOS has compressed or swapped more than 1 GiB of the server's own memory (pressure from other processes: 2026-10-05 a 98 s prefill ran 17+ min), rank 0 logs one warning, its instance card and /v1/residency say so, and new requests get the insufficient_memory 503 until it is paged back in; running requests go on.
* A long prompt no longer leaves a reserve behind it: the memory margin is the transient predicted for the step about to run (that prompt's context and prefill chunk, read off the measured line), not the largest ever seen -- after one 59k-token GLM-5.3-Flash prefill every request was refused with 30-60 GiB unused; and a prompt whose step would not fit at the launch chunk is prefilled at a smaller one (halving to 128, the same chunk on every rank of a ring) instead of refused.
* A prompt refused for memory logs, and returns in its 503 (`error.memory`), every term of the limit per rank (working set, others and their readings, margin and the transients behind it, active, cache, peers' over-limit, room, need); a loaded server that could not admit a 1k-token prompt says why in /v1/models `status` and on its instance card instead of "ready".
* A peer page that is slow to answer no longer stops a cluster job whose ranks run: a job stops only when a rank dies, the ring fails, or the peer machine stops accepting connections at all; the liveness status, /loaded.json and /models.json refresh off the request (a stalling SMB model share or a busy rank 0 held them past the peers' timeouts), and a peer that gives up mid-answer prints no traceback.
* The page reads model folders only when you act -- opening the model picker (this Mac's and the picked peers'), launching, a download finishing, or a CLI/MCP call -- never on a timer: /models.json answers from its last read unless asked with `rescan=1`, a load rescans once only for a model the last read lacked, and the load indicator and server phases take a model's size from its launch record instead of reading its shards on every poll.
- Load panel: switching MTP off hides Dynamic MTP (and an artifact without a drafting head or vision tower hides those rows): the rows' flex display kept every one of them on screen.
- Chat: scrolling up while a reply streams keeps the view where you are; it follows the reply again once you scroll back to the bottom.
Starting knurlogic is one word: `knurlogic` on each Mac.

* `knurlogic` with no command starts the page and says where to open it;
  `--open` opens it in the browser (never over SSH). `knurlogic help` lists
  the commands.
* GLM no longer fails after ~40 minutes of one generation with
  `[metal::malloc] Resource limit (499000) exceeded`: two cache fields no
  forward reads (the MLA latent's zero-width V and its offset) grew a lazy
  graph every step, each link holding a GPU buffer, until Metal's cap on
  live buffers. They are evaluated every step now, drafting or not, on
  every rank.
* DeepSeek-V4's compressed pools grow in place, 256 rows at a time, instead
  of being copied whole every few tokens: long generations no longer slow
  down as the context grows or fill mlx's buffer cache until the machine
  stutters and the job is torn down.
* The page answers on the Thunderbolt link(s) and 127.0.0.1 by default
  (what `--host cluster` did), and never on Wi-Fi or Ethernet. A Mac with no
  Thunderbolt link starts the same way and answers on 127.0.0.1.
  `--host 127.0.0.1` keeps it to this Mac.
* Model folders are remembered per Mac: `knurlogic models add <folder>`,
  `knurlogic models remove <folder>`, `knurlogic models folders`, and the
  MCP tool `model_folders`. A folder on a drive that is not mounted is
  skipped until it is back.
* A Mac started with `EXO_MODELS_DIRS`, `EXO_MODELS_READ_ONLY_DIRS` or
  `KNURLOGIC_MODELS` remembers those folders on its first start (while none
  are saved), so the variable is not needed after that.
* The chat takes PDF attachments: the text of every page, and with a vision
  model the first 8 pages as images too (pdf.js 5.7.284 ships with the page).

## 0.1.3

DeepSeek-V4-Flash-Vision-Exp with images, its thinking levels and DSpark
drafting; every model held to its maker's reference; GLM-5.3 images.

* A model that ends its thinking with `</thinking>` instead of `</think>`
  (seen on Qwen3.6) now has the block closed the same way: the text after it
  is the reply and its tool calls are parsed, streamed or not; before,
  everything came back as reasoning with no content and no tool calls.
* DeepSeek-V4-Flash-Vision-Exp: images (one or several per message, kept
  in the prompt cache), its four thinking levels (off, low, high, max), tool
  calls, and DSpark drafting (5 tokens per step), on one Mac or a split,
  pipeline or tensor; images work on both splits (a request with an image
  runs undrafted).
  Needs an artifact that carries its tower; a conversion without its
  vision weights loads as the text model (the page grays its Vision switch:
  "this conversion has no vision weights", and images are refused), and
  one with only part of them is refused.
* A drafting request that ends partway through a step stores only the
  tokens it returned, so the next turn reuses its cache exactly.
* Runs on mlx 0.32.3 and mlx-lm 0.32.0 (was 0.31.2 / 0.31.3), so it
  computes what current mlx computes. Logits move by float rounding
  (mlx 0.32's kernels); greedy tokens on the test models are unchanged.
* Python 3.11 or newer (mlx-lm 0.32 needs it).
* The page shows "Update available" beside Settings when PyPI has a newer
  knurlogic (the version in its tooltip); clicking it copies `pip install -U knurlogic`. Asked once per
  page start; `--offline` skips it.

Models compute what their makers' references compute

Every vendored architecture was checked against its maker's reference
code on small test models; each now matches it to about 1e-5. Outputs of
every model below change, for the better.
* DeepSeek-V4 (Flash and Vision-Exp): the shared expert's SwiGLU clamp at
  10; the FP8 / FP4 rounding DeepSeek's inference applies to the attention
  cache, the compressor, the indexer and every quantized linear's input;
  the MoE's routing weights and sums in float32. And in bf16, the dtype
  it serves in: the router's scores, the logits (also the drafting
  heads') and the compressor's pooling in float32, every norm rounded
  once, the per-head query norm in bf16 as DeepSeek's.
* Qwen3.8-Flash-Next: its n-gram embedding hashes with seed 1234, the
  reference's (it used 0, so every token read the wrong rows; about 3%
  lower perplexity now), and it now takes the multipliers the checkpoint
  stores rather than rebuilding them.
* Qwen3.5 / 3.6 / 3.8: the linear-attention q/k normalization epsilon
  (it was 128 times too large).
* GLM-5.3: the SwiGLU clamp at 10 in every MLP, router logits in float32,
  two norm epsilons, and the indexer's scores in float32.
* Gemma 4: an image's bidirectional attention applies on the sliding
  layers only, and only for the models that use it (26B-A4B, 31B).

GLM-5.3
* Images work: they were silently ignored (the image rows never reached
  the model, one Mac or split), their patches are laid out as the model
  expects (one circle read as two ovals before), and an image is sized as
  GLM's own processor sizes it (aspect kept, 16 to 8000 tokens), framed
  by <|begin_of_image|> / <|end_of_image|> as its template frames it.
* The MTP draft head computes its layer as the model does (the SwiGLU
  clamp and float32 router): drafts closer to the model's own tokens.
* Its attention cache stores the compressed latent once (it was kept
  twice): 16918 bytes a token instead of 28182 in bf16, so a longer
  context fits.

Qwen3.8-Flash-Next
* With thinking off it samples with the set Qwen publishes for that mode
  (temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5), as
  Qwen3.5 / 3.6 already did.

DeepSeek-V4
* DSpark drafting is faster: no second forward to roll back a step, and it
  verifies only as many drafts as are likely to stand (from the head's own
  confidence). About a third faster than plain decoding (about 25 against
  18.6 tok/s, sampled, on one two-Mac split).
* A short prompt (under the 128-token window) beside other requests no
  longer fails its admission.
* A tensor split that would cut one of the model's 128-wide rounding
  blocks is refused, with the reason.

Clusters
* A tool call (any request ending partway through a drafting step) on a
  tensor split no longer desynchronizes the machines.
* A split's own per-request seed no longer forces drafting on, or the
  widest verify, whatever the measured cost.
* A job that failed after relaunching shows as one card, not one per
  attempt; its x (and the MCP's unload, by the id its load answered)
  clears it on every Mac.
* When the link between the machines fails mid-request (an RDMA send or
  receive error, the ranks out of step), the first Mac answers what is in
  flight and restarts the job instead of staying up answering errors.
* The Macs of a split line up over the network before the steps where one
  can lag behind (joining, after the load, waking from idle), so a slower
  Mac no longer loses the faster one's first RDMA message.

Drafting
* KNURLOGIC_MTP_VERIFY set to anything but a whole number is refused when
  the server starts, instead of being ignored.

MCP
* fit, settings and drafting take a model name as load does; an unknown
  name is a refusal, not an error.

Known issue
* On an RDMA (jaccl) split, a rank that dies in the middle of a step can
  leave the other Mac's GPU waiting in that step until it restarts: mlx
  0.32.3's jaccl has no timeout. A normal unload is unaffected.

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
