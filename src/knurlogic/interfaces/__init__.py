"""interfaces/ -- how a person or an agent talks to knurlogic.

Two audiences, one set of answers. Everything a person can see or do on the
page, an agent can see or do through the MCP, from the same functions -- a
capability on one side only is a bug.

  mcp.py       the agent interface: stdio JSON-RPC, stdlib only
  web.py       the routes the page and /status.json are built on
  web/         the page itself
  ui.py        `knurlogic ui`: the page with nothing loaded, and loading
  serve.py     `knurlogic serve`: an OpenAI endpoint, settings resolved first
  messages.py  Anthropic Messages -> the engine's OpenAI endpoint
  connect.py   how to point a client at a running server
  doctor.py    will this artifact run, and why not
  cli.py       one entry point; its COMMANDS table routes to the rest

Depends on everything below it; nothing below depends on this folder.
"""
