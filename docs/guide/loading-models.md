# Loading and unloading models

A model is loaded into its own server process. Knurlogic picks the
settings from the model's `config.json` and the memory free, and refuses
a model that does not fit rather than trying it.

## From the page

1. **Pick the machine.** Click it in the Memory panel. One machine is a
   normal load; two or more is a cluster job ([clusters.md](clusters.md)).
2. **Choose a model.** Load model -> *choose a model* opens the picker:
   a search, the memory it has to fit in, families down the side
   (favorites and recents first), one row per model with its quantizations
   folded in. Only published models of a known architecture are offered.
3. **Read the room.** Under the pick, a line says what is left to talk in:
   the free working set less the weights and a step margin, in GiB and
   roughly how many tokens of context, shared by every conversation. It is
   amber when that is little ("little room for long conversations").
   Hover it for the arithmetic. A model that does not fit says "more space
   required" and Launch stays disabled with the reason.
4. **Set the switches** (shown when the model has them):

   | switch | what it does |
   |---|---|
   | MTP | draft with the model's MTP head: usually faster, costs the head's memory ([drafting.md](drafting.md)) |
   | Dynamic MTP | draft only where it measures faster; off drafts every step |
   | Vision | load the vision tower; off frees its memory and image requests get a 400 ([vision.md](vision.md)) |

   Switching MTP or vision off gives its memory back to the room line.
5. **Launch.** The model loads on port 8080 (the page's `--serve-port`),
   or the next free one.

Everything else (preset, KV cache bits, context length, thinking default)
is a per-model launch setting in Settings -> Models, used every time that
model is launched. See [settings-and-memory.md](settings-and-memory.md).

While a model loads, its `/v1/models` entry has `"status": "loading"`;
requests sent meanwhile wait. It becomes `ready` (or `failed`).

## Unloading

Press **Unload** on the model's card in Instances. A model on another Mac
is unloaded through that Mac's page; a cluster job stops on every machine.
Models of other runtimes (ollama, exo) are shown but not unloaded from
here.

Knurlogic never unloads one model to make room for another: unload first,
then load.

## Several models at once

Each loaded model is its own server on its own port. The page routes a
chat request to the right one by its `model` field
([chat-and-api.md](chat-and-api.md)).

## From the command line: knurlogic serve

```bash
knurlogic serve <model> [options]
```

| option | default | |
|---|---|---|
| `--port` | 8080 | |
| `--host` | 127.0.0.1 | an address (comma-separated for several), or `cluster` (loopback and Thunderbolt only) |
| `--tune` / `--preset` | default | `default` or `lean` ([settings-and-memory.md](settings-and-memory.md)) |
| `--kv-bits` | bf16 | `bf16`, `8`, `6` or `4` (6 and 4 only where the family takes them) |
| `--context-length N` | the model's window | the longest prompt + answer a request may use; a cap, nothing reserved |
| `--no-draft` | | do not use an MTP head (`KNURLOGIC_MTP=off`) |
| `--mtp-dynamic on\|off` | on | |
| `--set KEY=VALUE` | | force any setting (repeatable); beats the resolver and the preset |
| `--decode-concurrency N` | 32 | most requests decoding at once |
| `--prompt-cache-size N` | sized by memory | prompt-cache entries kept |
| `--prompt-cache-gib G` | half the room beside the model | cap the prompt cache's memory |
| `--image-store-gib G` | 0.25 | memory for encoded images |
| `--max-request-mib N` | 512 | largest request body (413 above it) |
| `--working-set-gib G` | asked of the framework | the GPU working set to plan against |
| `--allow-origin URL`, `--allow-host NAME` | | as for the page |
| `--profile v1.5\|v2` | none | force a VQ numerics profile; changes a model's outputs if it is not its own |

Most settings are read when the process starts, so changing them means a
new launch.

## From an agent

The MCP `load` tool does what Launch does, with the same refusals; `fit`
asks first; `unload` stops. See [mcp.md](mcp.md).

A model server also answers `POST /v1/ensure {"model", "wait"}`: it returns
at once if the model is ready, otherwise loads it (one load at a time) and
optionally waits. On a server already serving another model it is a
switch, refused with 409 while requests are in flight unless `"force":
true`.
