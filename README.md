# Knurlogic

Knurlogic serves local language models on Apple Silicon. It works out the
settings a model needs and whether it fits in memory before loading it,
then runs it behind an OpenAI- and Anthropic-compatible server. People use
it through a web page; agents use it through an MCP server. One model can
also be split across two or more Macs.

## Requirements

* A Mac with Apple Silicon and enough unified memory for the model.
* Python 3.11 or newer.
* mlx 0.32.3 and mlx-lm 0.32.0, pinned exactly (pip installs them).

## Install

    pip install knurlogic
    knurlogic

`knurlogic` starts the page; open http://127.0.0.1:8899/ in your browser
(`knurlogic --open` opens it for you): every model on the disk with a
Launch button, a model picker for downloading more, a chat, and where the
memory went.

From source:

    git clone https://github.com/noahzelezny/Knurlogic
    cd Knurlogic && pip install -e .

## Quickstart

    pip install knurlogic
    knurlogic

`knurlogic` opens the page at http://127.0.0.1:8899/: every model on the
disk with a Launch button, what is loaded and where the memory went. Run
the same on each Mac; Macs joined by Thunderbolt find each other.

Models on an external drive: `knurlogic models add "/Volumes/My SSD/Models"`
once. The folder is remembered; while the drive is not mounted it is
skipped. `knurlogic models folders` lists the saved folders and
`knurlogic models remove <folder>` forgets one.

Get a model in MLX format: search and download it from the page's model
picker (Hugging Face), or with the `hf` CLI that comes with
`huggingface_hub`:

    pip install -U huggingface_hub
    hf download mlx-community/gemma-4-e4b-it-8bit \
      --local-dir ~/Knurlogic/Models/gemma-4-e4b-it-8bit

`knurlogic models` lists the models already on this Mac (Knurlogic's own
folder, the saved folders, the Hugging Face cache, LM Studio, Ollama and exo
folders).

Check it and serve it:

    knurlogic doctor ~/Knurlogic/Models/gemma-4-e4b-it-8bit
    knurlogic serve  ~/Knurlogic/Models/gemma-4-e4b-it-8bit

Then ask it something:

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'

The page is at http://127.0.0.1:8080/: what is loaded, where the memory
went, a chat and the settings. `knurlogic` (or `knurlogic ui`) opens the
page at http://127.0.0.1:8899/ without loading anything, with a Launch
button for every model on the disk, and a gear in the macOS menu bar (`--no-menubar`
skips it) that shows what is loaded and opens or quits the page.

## Connect a harness

`knurlogic connect --port 8080` prints these for a running server.

**Claude Code** (or any Anthropic Messages client):

    ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
    ANTHROPIC_API_KEY=x \
    ANTHROPIC_DEFAULT_OPUS_MODEL=local \
    ANTHROPIC_DEFAULT_SONNET_MODEL=local \
    ANTHROPIC_DEFAULT_HAIKU_MODEL=local \
    API_TIMEOUT_MS=3000000 \
    claude

**OpenAI-compatible clients** (Zed, Cline, Continue, Open WebUI): base URL
`http://127.0.0.1:8080/v1`, any API key, model `local`.

**Responses-API clients** (the OpenAI SDK's `client.responses`, Codex-style
tools): the same base URL; `POST /v1/responses` with function tools,
streaming and reasoning summaries. `previous_response_id` and `store: true`
are refused: the server keeps no conversation, so resend the input.

**Ollama clients** (Open WebUI's Ollama mode, Continue, the `ollama`
libraries): `OLLAMA_HOST=http://127.0.0.1:8080`. `/api/chat`, `/api/generate`
(streaming NDJSON by default), `/api/tags`, `/api/show` and `/api/version`
are served, with images in chat messages on a vision model.

**MCP**, for an agent that should manage models rather than talk to one:

    claude mcp add knurlogic -- knurlogic mcp

Its tools are `models`, `model_folders`, `fit`, `settings`, `drafting`,
`ready`, `load`, `state`, `unload` and `deps`; `knurlogic mcp --list` describes each.
`load` refuses a model that does not fit.

## Two or more Macs

To split one model across two or more Macs (up to 16) joined by Thunderbolt,
install the same
knurlogic version and the model on both, and run on each:

    knurlogic

The page answers on the Thunderbolt link(s) and on 127.0.0.1 only, never
on Wi-Fi or Ethernet (a Mac with no Thunderbolt link answers on 127.0.0.1).
`knurlogic --host 127.0.0.1` keeps it to this Mac.

The Macs find each other over Bonjour (`--peer HOST` names one directly, on each machine
(or link them with Thunderbolt);
`knurlogic doctor --cluster` says what is in the way). Then launch the
model from the page with the machines selected, or with the MCP `load`
tool's `machines` argument. The TCP ring works for any number of Macs;
RDMA (jaccl) needs every pair cabled with Thunderbolt 5, and beyond two
Macs it is experimental and untested.

## Settings

Every setting is resolved from the model's `config.json` and the memory
available, and shown in the page's Settings panel and at `/settings.json`.
Two presets cover most needs: `default` (fastest) and `lean` (more
context in less memory: 512-token prompt chunks, MTP off, 8-bit KV).
`serve --tune default|lean` picks one, `--kv-bits 8` stores the KV cache
in about half the memory, and `--set KEY=VALUE` overrides any single
setting. A context length past the model's native window turns on YaRN
for the Qwen families (up to 1,048,576 tokens).

## Supported models

| Family | `model_type` | Vision | Drafting (MTP) | 8-bit KV |
|---|---|---|---|---|
| Qwen 3.5 / 3.6 | `qwen3_5`, `qwen3_5_moe` | yes | if the model has a head | yes |
| Qwen 3.8 | `qwen4_exp` | yes | if the model has a head | yes |
| Gemma 4 | `gemma4`, `gemma4_text` | yes | no | yes |
| GLM-5 | `glm5_next` | yes | if the model has a head | yes |
| DeepSeek-V4 | `deepseek_v4` | no | no | no |

VQ-quantized models (published with their own `model.py`) are supported
for these families; each runs the `model.py` it ships. The page tags a
Hugging Face model "update" when the Hub has a newer revision (checked once
at start; `knurlogic ui --offline` or `HF_HUB_OFFLINE=1` skips it).

## Code

How the package is laid out and who may import whom: [docs/architecture.md](docs/architecture.md).

## Status

Alpha. Expect rough edges and breaking changes before 1.0; see
CHANGELOG.md.

## License

Apache-2.0. Vendored third-party code is listed in NOTICE.
