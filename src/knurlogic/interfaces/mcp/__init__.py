"""knurlogic mcp -- the interface an agent gets to the local models.

    knurlogic mcp            # serve on stdio
    knurlogic mcp --list     # print the tool table

Stdio JSON-RPC, stdlib only: it reads the machine and starts servers, and
imports nothing that could load a model into this process. Every tool
answers deterministically, reports what it looked at, and refuses rather
than gambles: `ready` is a gate `load` checks first; `fit` counts free +
file-cache memory and refuses with the arithmetic; `settings` returns every
knob with its measurement; `load` always spawns a fresh server; nothing
deletes or writes an artifact.

This package __init__ is the MCP's public API: the tool functions, for
code that drives the tools in-process (`from knurlogic.interfaces import mcp
as K; K.load(...)`), and main for the CLI. Code inside knurlogic imports
from the defining module.

  server.py       the stdio JSON-RPC loop and main (`knurlogic mcp`)
  tools.py        TOOLS: name -> function, description, input schema
  inspection.py   read-only tools: ready fit state models model_folders
                  settings drafting deps
  lifecycle.py    load and unload, here or through the page
  page_client.py  the page on this Mac, over loopback; models_across
  placement.py    load's role words (here peers all fit) -> machines

Design: docs/design/mcp.md.
"""

from knurlogic.interfaces.mcp.inspection import (
                                                 deps,
                                                 drafting,
                                                 fit,
                                                 model_folders,
                                                 models,
                                                 ready,
                                                 settings,
                                                 state,
)
from knurlogic.interfaces.mcp.lifecycle import load, unload
from knurlogic.interfaces.mcp.server import main

__all__ = ["deps", "drafting", "fit", "load", "main", "model_folders",
           "models", "ready", "settings", "state", "unload"]
