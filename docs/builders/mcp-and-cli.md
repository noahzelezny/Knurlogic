# The MCP server and the CLI

An agent and a person reach the same things: the MCP tools mirror what
the page does. The why, and the rule that an agent is never left waiting
on silence: [mcp](../design/mcp.md).

## Where the code is

| file | what |
|---|---|
| `interfaces/mcp/` | `knurlogic mcp`: stdio JSON-RPC, stdlib only. Its `__init__` is the public API (the tool functions and `main`) |
| `interfaces/mcp/server.py` | the stdio loop: `_call`, `_serve_stdio`, `main`, `INSTRUCTIONS` (sent at `initialize`) |
| `interfaces/mcp/tools.py` | `TOOLS` (name to `fn`, `description`, `schema`), `tool_list` |
| `interfaces/mcp/inspection.py` | the read-only tools: `ready`, `fit`, `state`, `models`, `model_folders`, `settings`, `drafting`, `deps` |
| `interfaces/mcp/lifecycle.py` | `load` and `unload`, on this Mac or through the page |
| `interfaces/mcp/page_client.py` | the page on this Mac over loopback (`page_get`, `page_post`), `models_across` |
| `interfaces/cli.py` | `knurlogic`: `COMMANDS` maps a subcommand to a module whose `main(argv)` runs it; no command starts the page |
| `interfaces/connect.py` | `knurlogic connect`: how to point a client at a server (env lines, Claude Code, OpenAI, curl, Codex, MCP JSON) |
| `interfaces/doctor.py` | `knurlogic doctor`: will this artifact run, and why not |
| `interfaces/drafting.py` | `knurlogic mtp`: which artifacts can draft |
| `interfaces/menubar.py` | the macOS menu-bar icon |

The MCP tools, in the order an agent uses them: `models`, `fit`,
`settings`, `ready`, `load`, `state`, `unload`, plus `drafting`,
`model_folders` and `deps`. Each is a plain function: the read-only ones in `mcp/inspection.py`,
`load` and `unload` in `mcp/lifecycle.py`. Tools that span machines ask
the page on this Mac (`mcp/page_client.py`: `page_get`, `page_post`;
address from `KNURLOGIC_PAGE`).

The CLI's commands: `ui`, `serve`, `doctor`, `smoke`, `vendor`,
`connect`, `mcp`, `loaded`, `mtp`, `models`, `deps`. Most live in the
package the command is about (`machine.discover`, `machine.loaded`,
`machine.deps`, `engine.smoke`, `engine.vendor`).

## Rules that keep it correct

- **The MCP never loads a model into its own process.** It reads the
  machine and starts servers; it imports nothing that could load one.
- **A refusal is an answer.** A tool that will not act returns its
  reason (`refused`), not an error; `isError` is set only when the
  result has `error`.
- **`ready` gates `load`.** `load` checks it first and nothing is evicted
  for the caller. The load lock (`machine/loadlock.py`) makes the check
  and the load atomic across processes.
- **Every load has a phase.** `state` reports loading, warming, serving,
  stalled or exited, with the log.
- **No tool deletes or writes model files.**
- **Parity with the page.** A feature a person can use on the page gets
  a tool or a tool argument; add both in the same change.

## Extending

- A new tool: a function in `mcp/inspection.py` (reads) or
  `mcp/lifecycle.py` (starts or stops), an entry in `mcp/tools.py`'s
  `TOOLS` with its description and `_schema`, and its name in the
  package `__init__`'s imports and `__all__`. If it acts on another Mac, go through the
  page (`page_post`), not straight to the peer.
- A new CLI command: one line in `cli.COMMANDS` naming the module, and a
  `main(argv) -> int` there.

## Notes

The MCP and the page start and stop model servers through the same
`interfaces/spawn.py` (`spawn`, `loading`, `children`, `stop`); the MCP
reshapes the page's `/loaded.json?peers=1` for cross-machine answers
(`models_across`).

## Tests

`tests/interfaces/test_mcp.py`, `tests/cluster/test_mcp_cluster.py`,
`tests/interfaces/test_bare_command.py`, `test_menubar.py`,
`tests/tuning/test_doctor_tune.py`.
