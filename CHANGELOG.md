# Changelog

## 0.1.0

First public release (alpha).

* `knurlogic serve`: an OpenAI-compatible server (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`) with an Anthropic Messages endpoint
  (`/v1/messages`) for Claude Code and similar harnesses.
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
* Two Macs: one model split by tensor or pipeline over Thunderbolt, with
  Bonjour discovery and `--host cluster`.
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
