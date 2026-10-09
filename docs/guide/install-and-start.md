# Install and start

## What you need

- A Mac with Apple Silicon and enough unified memory for the model.
- Python 3.11 or newer.
- mlx and mlx-lm at the exact versions knurlogic pins (pip installs them).

## Install

```bash
pip install knurlogic
knurlogic
```

From source:

```bash
git clone https://github.com/noahzelezny/Knurlogic
cd Knurlogic && pip install -e .
```

## The page

`knurlogic` (the same as `knurlogic ui`) starts the page at
`http://127.0.0.1:8899/`. It loads nothing by itself. `knurlogic --open`
also opens it in your browser (not over SSH).

On the page:

| part | what it shows |
|---|---|
| Memory | each Mac's memory: what is loaded, what else holds memory, swap. Click a machine to pick it for Load model. |
| Instances | every model in memory, in every runtime (knurlogic, ollama, exo, other OpenAI ports), with an Unload button on knurlogic's own |
| Load model | the model picker, its switches and Launch ([loading-models.md](loading-models.md)) |
| Chat | a chat with any running model; the chats are kept in this browser only |
| Settings | presets, compaction, each machine's memory, each model's launch settings ([settings-and-memory.md](settings-and-memory.md)) |
| Connect | what to paste into Claude Code, an OpenAI client or an MCP client |
| Downloads | Hugging Face downloads in progress, stopped or failed ([models.md](models.md)) |

`knurlogic ui` options:

| option | default | |
|---|---|---|
| `--host` | `cluster` | `cluster`: answer on Thunderbolt link(s) and loopback only, never Wi-Fi or Ethernet. `127.0.0.1`: this Mac only. Or an address. |
| `--port` | 8899 | the page's port |
| `--serve-port` | 8080 | the port a model loaded from the page is served on |
| `--peer HOST[:PORT]` | | another Mac's page (repeatable); see [clusters.md](clusters.md) |
| `--allow-origin URL` | | a web page origin allowed to call the page's API from a browser |
| `--allow-host NAME` | | a DNS name this Mac is reached by, beyond localhost, IPs, `.local` and its hostname |
| `--offline` | | skip the check for newer Hugging Face revisions (`HF_HUB_OFFLINE=1` does the same) |
| `--no-menubar` | | no menu-bar icon |
| `--open` | | open the page in the browser once it is up |

## The menu-bar gear

While the page runs, a gear in the macOS menu bar shows how many models
are loaded and can open the page or quit it. One icon per Mac.
`--no-menubar` turns it off.

## Commands

`knurlogic --help` lists them; `knurlogic <command> --help` gives each
one's options.

| command | what it does |
|---|---|
| `knurlogic` / `knurlogic ui` | start the page |
| `knurlogic serve <model>` | serve one model on an OpenAI-compatible port (8080 by default), without the page |
| `knurlogic doctor <model>` | say whether a model will run here, and why not |
| `knurlogic doctor --cluster` | say what stops this Mac and the others finding each other |
| `knurlogic models` | the models already on this Mac, in every tool's folder |
| `knurlogic models add\|remove <folder>`, `models folders` | remember, forget or list extra model folders |
| `knurlogic mtp [<model>]` | which models have a drafting (MTP) head |
| `knurlogic connect` | print how to point a client at a running server |
| `knurlogic mcp` | the MCP server, on stdio |
| `knurlogic loaded` | what is in memory now, in every runtime on this Mac |
| `knurlogic deps` | mlx, mlx-lm, mlx-vlm: which are stock and which are forks |
| `knurlogic smoke`, `knurlogic vendor` | developer tools |
| `knurlogic --version` | the version |

## Check a model before loading: doctor

```bash
knurlogic doctor ~/Knurlogic/Models/gemma-4-e4b-it-8bit
```

It prints, for that model on this Mac:

- its type and size, and the working set (the memory the GPU may use;
  capped by your allowance if you set one);
- `room`: what is left to talk in once the weights are loaded, and
  `SMALL` when that is little;
- whether its architecture is installed and pinned (`ok`, `??` unpinned,
  `!!` drifted, `XX` missing);
- the settings it would launch with, notes and warnings;
- whether it has an MTP head, and which tool-call dialect its chat
  template uses (or that the template asks for tools and no parser
  matched: calls would come back as prose);
- the wired limit, and the `sudo sysctl` command to run if raising it
  would let the model fit ([settings-and-memory.md](settings-and-memory.md)).

It ends with `no blockers found` or `WILL NOT RUN as configured` (exit
code 1). `doctor` with no model lists what is on the disk.

| option | |
|---|---|
| `--tune default\|lean` | the preset to check against |
| `--working-set-gib N` | pretend the GPU may use N GiB |
| `--exports` | print only `export K=V` lines |
| `--cluster [--port 8899]` | the cluster check instead |

## Serve one model without the page

```bash
knurlogic serve ~/Knurlogic/Models/gemma-4-e4b-it-8bit
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'
```

The server has its own page at `http://127.0.0.1:8080/`. Its options are in
[loading-models.md](loading-models.md).

## Updates

The page shows a note in its top bar when PyPI has a newer knurlogic;
clicking it copies the upgrade command.

Knurlogic is alpha: expect breaking changes before 1.0 (see CHANGELOG.md).
