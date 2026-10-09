"""interfaces/ -- how a person or an agent talks to knurlogic.

Everything a person can see or do on the page, an agent can do through the
MCP, from the same functions.

  mcp/            the agent interface: stdio JSON-RPC, stdlib only
  http/           OpenAI, Anthropic Messages and Ollama on one server
  page/           `knurlogic ui`, its routes and assets
  cli.py          `knurlogic`: its COMMANDS table routes to the rest
  serve.py        `knurlogic serve`: settings resolved, then an endpoint
  spawn.py        the `knurlogic serve` children the page and MCP start
  load_checks.py  what a model must pass before it loads (known, runnable, fits)
  doctor.py       `knurlogic doctor`: will this artifact run, and why not
  drafting.py     `knurlogic mtp`: which artifacts on this Mac can draft
  connect.py      `knurlogic connect`: how to point a client at a server
  menubar.py      the macOS menu-bar icon

Depends on everything below it; nothing below depends on this folder.
"""
