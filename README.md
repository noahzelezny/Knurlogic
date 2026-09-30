# Knurlogic

Knurlogic serves local language models on Apple Silicon. It works out the
settings a model needs and whether it fits in memory before loading it,
then runs it behind an OpenAI- and Anthropic-compatible server. People use
it through a web page; agents use it through an MCP server. One model can
also be split across two Macs.

## Requirements

* A Mac with Apple Silicon and enough unified memory for the model.
* Python 3.10 or newer.
* mlx 0.31.2 and mlx-lm 0.31.3, pinned exactly (pip installs them).

## Install

    pip install knurlogic

From source:

    git clone https://github.com/noahzelezny/Knurlogic
    cd Knurlogic && pip install -e .

## Quickstart

Get a model in MLX format: search and download it from the page's model
picker (Hugging Face), or with the `hf` CLI that comes with
`huggingface_hub`:

    pip install -U huggingface_hub
    hf download mlx-community/gemma-4-e4b-it-8bit \
      --local-dir ~/Knurlogic/Models/gemma-4-e4b-it-8bit

`knurlogic models` lists the models already on this Mac (Knurlogic's own
folder, the Hugging Face cache, LM Studio, Ollama and exo folders).

Check it and serve it:

    knurlogic doctor ~/Knurlogic/Models/gemma-4-e4b-it-8bit
    knurlogic serve  ~/Knurlogic/Models/gemma-4-e4b-it-8bit

Then ask it something:

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'

The page is at http://127.0.0.1:8080/: what is loaded, where the memory
went, a chat and the settings. `knurlogic ui` opens the page at
http://127.0.0.1:8899/ without loading anything, with a Launch button for
every model on the disk.

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

**MCP**, for an agent that should manage models rather than talk to one:

    claude mcp add knurlogic -- knurlogic mcp

Its tools are `models`, `fit`, `settings`, `drafting`, `ready`, `load`,
`state`, `unload` and `deps`; `knurlogic mcp --list` describes each.
`load` refuses a model that does not fit.

## Two Macs

To split one model across two Macs joined by Thunderbolt, install the same
knurlogic version and the model on both, and run on each:

    knurlogic ui --host cluster

The Macs find each other over Bonjour (`--peer HOST` names one directly;
`knurlogic doctor --cluster` says what is in the way). Then launch the
model from the page with both machines selected, or with the MCP `load`
tool's `machines` argument.

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
for these families.

## Status

Alpha. Expect rough edges and breaking changes before 1.0; see
CHANGELOG.md.

## License

Apache-2.0. Vendored third-party code is listed in NOTICE.
