"""interfaces/ -- how a person or an agent talks to knurlogic.

Two audiences, one set of answers. Everything a person can see or do on the
page, an agent can see or do through the MCP, from the same functions -- a
capability on one side only is a bug.

  mcp.py       the agent interface: stdio JSON-RPC, stdlib only
  http/        the chat wire: OpenAI and Anthropic Messages on one server
               (server.py, openai.py, messages.py, residency.py)
  page/        the page: `knurlogic ui` (server.py), the routes it and
               /status.json are built on (documents.py), assets/index.html
  serve.py     `knurlogic serve`: an OpenAI endpoint, settings resolved first
  connect.py   how to point a client at a running server
  doctor.py    will this artifact run, and why not
  loading.py   what a model must pass before it is loaded
  drafting.py  `knurlogic mtp`: which artifacts have a drafting head
  cli.py       one entry point; its COMMANDS table routes to the rest

Multi-machine orchestration lives in cluster/ and history surgery in
context_management/; the page injects what cluster/ needs of it at
startup. Depends on everything below it; nothing below depends on this
folder.
"""
