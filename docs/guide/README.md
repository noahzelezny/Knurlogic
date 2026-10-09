# Knurlogic user guide

Knurlogic runs local language models on Apple Silicon Macs (MLX). It works
out whether a model fits and what settings it needs before loading it,
then serves it to people (a web page) and to programs (an OpenAI- and
Anthropic-style HTTP API, an MCP server, a CLI). One model can also be
split across two or more Macs.

Start with the first page; the rest can be read in any order.

| page | what it covers |
|---|---|
| [install-and-start.md](install-and-start.md) | `pip install`, the page, the menu-bar gear, every CLI command, `doctor` |
| [models.md](models.md) | where models come from: the Hugging Face picker, downloads, model folders, `knurlogic models` |
| [loading-models.md](loading-models.md) | picking a model, the room it leaves, MTP and vision switches, Launch, unload, `knurlogic serve` |
| [chat-and-api.md](chat-and-api.md) | the endpoints, connecting Claude Code and OpenAI clients, thinking, tools, sampling, headers, errors |
| [vision.md](vision.md) | sending images, limits, turning vision off |
| [drafting.md](drafting.md) | MTP drafting: which models have a head, on, off and dynamic |
| [compaction.md](compaction.md) | letting the server compact long conversations |
| [prompt-cache.md](prompt-cache.md) | the prompt cache: sessions, saving to disk, park, pin, drop |
| [settings-and-memory.md](settings-and-memory.md) | the Settings panel, presets, every launch setting, KV bits, context length, long context, the wired limit and the allowance |
| [clusters.md](clusters.md) | two or more Macs: finding each other, pipeline vs tensor, TCP vs RDMA, failure and recovery |
| [mcp.md](mcp.md) | the MCP server for agents that manage models |
| [usage-and-telemetry.md](usage-and-telemetry.md) | `GET /v1/usage`, the request ledger, `usage.knurlogic`, progress events |
| [troubleshooting.md](troubleshooting.md) | what an error means and what to do |

Why things work the way they do is in the design notes:
[../architecture.md](../architecture.md) and [../design/](../design/).
