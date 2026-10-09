"""interfaces/ -- how a person or an agent talks to knurlogic.

Everything a person can see or do on the page, an agent can do through the
MCP, from the same functions.

  mcp/        the agent interface: stdio JSON-RPC, stdlib only
  http/        OpenAI and Anthropic Messages on one server
  page/        `knurlogic ui`, its routes and assets
  serve.py     `knurlogic serve`: settings resolved, then an endpoint
  connect.py   how to point a client at a running server
  doctor.py, loading.py, drafting.py   checks before and around a load
  cli.py       one entry point; its COMMANDS table routes to the rest

Depends on everything below it; nothing below depends on this folder.
"""
