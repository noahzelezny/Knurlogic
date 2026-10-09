# The MCP server

For an agent that should manage models (see what is on the disk, check
fit, load, unload) rather than only talk to one. It runs on stdio; the
client starts it.

```bash
claude mcp add knurlogic -- knurlogic mcp     # Claude Code
codex mcp add knurlogic -- knurlogic mcp      # Codex CLI
knurlogic mcp --list                          # print the tools
```

Other clients:

```json
{"mcpServers": {"knurlogic": {"command": "knurlogic", "args": ["mcp"]}}}
```

The page's Connect panel shows the same. To talk to a model, use the HTTP
API instead ([chat-and-api.md](chat-and-api.md)).

## The rules it follows

- Every tool answers deterministically and says what it looked at.
- It refuses rather than gambles: a model that does not fit, or a load
  while another load is still moving memory, is a refusal with the reason.
- It never evicts a model to make room. Unload first.
- It never deletes or writes a model's files.

## Tools

| tool | arguments | what it does |
|---|---|---|
| `ready` | | is it safe to load now? Not while another load is moving memory; every blocker is named. Call it before `load`. |
| `models` | `fits_only` | every model on this Mac: whether it fits, whether it has a drafting head, what each `reasoning_effort` maps to |
| `model_folders` | `add`, `remove` | the extra model folders this Mac remembers, and whether each is mounted |
| `fit` | `artifact`, `draft`, `vision` | will it fit now: `fits`, `fits, low headroom` (loads with a narrower prompt chunk) or `will not fit`, with the headroom. `draft=false` / `vision=false` ask about MTP or vision off. |
| `settings` | `artifact`, `tune` | every resolved setting with the measurement behind it and whether it changes live or only at launch |
| `drafting` | `artifact` | whether it has an MTP head and what will happen to it |
| `load` | `artifact`, `port`, `tune`, `sets`, `force`, `draft`, `vision`, `machines`, `split`, `link`, `cable` | start a model (below) |
| `state` | | what is loaded, here and on every Mac this Mac's page sees |
| `unload` | `port`, `model`, `job`, `instance`, `machine` | stop a model knurlogic started |
| `deps` | | mlx, mlx-lm, mlx-vlm: stock or fork, by what is installed |

### load

- `artifact`: the model's name (for another Mac, its 16-hex identity).
- `port`: omitted, the first free port from 8080 up.
- `tune`: `default` or `lean`. `sets`: launch-only overrides, `{KEY: VALUE}`.
- `force`: load while another load is still moving memory (one Mac only).
  It never makes a model fit.
- `draft` (default true), `vision` (default true): `false` frees the MTP
  head's or the vision tower's memory.
- `machines`: empty is this Mac. One other Mac loads there, checked by
  that Mac. Two or more is a cluster job with `split` (`tensor` |
  `pipeline`) and `link` (`tcp` | `rdma`); placement, leader and cable are
  chosen for you. Anything past this Mac goes through this Mac's page, so
  `knurlogic ui` must be running (the MCP finds it at `KNURLOGIC_PAGE`,
  default `127.0.0.1:8899`).
- Role words in `machines` say where without knowing what the Macs are
  called (peers are found on their own): `here` (this Mac), `peers` (every
  peer answering this Mac's page; refused when none is), `all` (here +
  peers). They mix with names and ids; duplicates collapse; the order is
  this Mac first, then the peers in the page's order. A role or name that
  is a Mac not answering is refused with the reason.
- `fit`, alone (mixed with anything it is refused): this Mac when the
  model fits here (the `fit` tool's check), else the smallest set of
  answering Macs it fits on (the placement the page's Launch makes). It
  splits `pipeline` unless `split` names `tensor`, as does any role that
  makes a cluster. Nothing fits: refused with the reason, each Mac's
  `working_set_gib` / `available_gib` and this Mac's `fit` numbers.

A cluster load returns `job`, `port` (rank 0's, on the leader), `leader`,
`machines` and `placement {order, leader, layers, cable, cable_note}`, and
loads in the background: poll `state`. A refusal is
`{"loaded": false, "refused": "..."}` with nothing started: a share does
not fit, a machine is not answering, the model is not on a machine, RDMA
without a Thunderbolt 5 cable, another load in progress. When a model fits
only with MTP off, the refusal says so.

### state

`models`: one entry per resident model, here and on answering peers, each
with `name`, `machine`, `port`, `machines`, `split`, `link`, `job`,
`instance`, `leader`, `phase`, `requests` (`in_flight`, `pending`,
`capacity`, `oldest_pending_s`, `holding`) and `recovery`. A cluster job is
one entry. `machines` lists the peers asked and any that did not answer.
The rest is this Mac: every runtime, where the memory went, and
`started_here` (each server `load` started, with its phase: serving,
loading, stalled or exited, and its log tail).

### unload

`port` alone stops the server on that port of this Mac. `model`, `job` or
`instance` find it on any Mac the page sees (`machine` and `port` narrow
it). A cluster job stops on every Mac.

Without the page running, `load` and `unload` work on this Mac by port and
`state` lists this Mac only.

Design: [../design/mcp.md](../design/mcp.md),
[../design/server.md](../design/server.md) (The MCP across machines).
