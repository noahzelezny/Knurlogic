"""The MCP's stdio JSON-RPC loop (initialize, tools/list, tools/call,
ping) and `knurlogic mcp`'s main."""

from __future__ import annotations

import json
import sys
from typing import Any

from knurlogic.interfaces.mcp.tools import TOOLS, tool_list

SERVER_NAME = "knurlogic"
SERVER_VERSION = "0"
#: sent in the initialize result: how to drive these tools, for a model
#: that has nothing else to read
INSTRUCTIONS = (
    "knurlogic manages local MLX models on this Mac and the Macs its page "
    "sees. The loop: `models` (what is on disk and whether it fits) -> "
    "`fit` (will this artifact fit now, with the arithmetic) -> "
    "`settings` (the resolved knobs and why) -> `ready` (is it safe to "
    "load now) -> `load` -> `state` (poll until the phase is serving) -> "
    "`unload` when done. Never call `load` while `ready` is false: wait "
    "and call `ready` again, or unload something; nothing is evicted for "
    "you. A refusal is an answer with its reason, not an error to retry "
    "blindly. Never wait on silence: every load has a phase in `state` -- "
    "loading, warming, serving, stalled (stop waiting and read the log "
    "shown), or exited (exit code and log tail). Once serving, the model "
    "answers OpenAI and Anthropic Messages requests at "
    "http://<machine>:<port>/v1 (the port and leader `load` returned; "
    "127.0.0.1 for this Mac). `deps` says "
    "which mlx builds are installed. No tool deletes or modifies model "
    "files. The levers, when a load does not fit or runs short of memory: "
    "`load`'s tune (`default`, fastest; `lean`, 512-token prompt chunks, MTP "
    "off, 8-bit KV -- the most context in the least memory), or single "
    "settings in `load`'s sets: KNURLOGIC_CONTEXT_LENGTH (less context, "
    "less KV memory; past the native window the Qwen families use YaRN, "
    "up to 1,048,576), KNURLOGIC_KV_BITS=8, KNURLOGIC_MTP=off (frees the "
    "draft head's memory), KNURLOGIC_PREFILL_CHUNK=512 (a smaller spike "
    "while reading a prompt). `settings` shows what each resolves to."
)


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    t = TOOLS.get(name)
    if t is None:
        return {"error": f"unknown tool {name!r}",
                "available": sorted(TOOLS)}
    try:
        return t["fn"](**(args or {}))
    # a tool's failure is the tool call's error reply, never a dead server
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name}


def _reply(rid, result=None, error=None) -> None:
    msg = {"jsonrpc": "2.0", "id": rid}
    msg.update({"error": error} if error is not None else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _serve_stdio() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, method = req.get("id"), req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            result: dict = {"protocolVersion": "2024-11-05",
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": SERVER_NAME,
                                     "version": SERVER_VERSION},
                      "instructions": INSTRUCTIONS}
        elif method == "tools/list":
            result = {"tools": tool_list()}
        elif method == "tools/call":
            out = _call(params.get("name", ""), params.get("arguments") or {})
            # isError is how a client tells an answer from a failure. A
            # refusal (`refused`) is an ANSWER -- the tool did its job.
            result = {"content": [{"type": "text",
                                   "text": json.dumps(out, indent=1)}],
                      "isError": "error" in out}
        elif method == "ping":
            result = {}
        elif rid is None or method.startswith("notifications/") \
                or method == "initialized":
            continue
        else:
            _reply(rid, error={"code": -32601,
                               "message": f"method not found: {method}"})
            continue
        _reply(rid, result=result)
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="knurlogic mcp",
        description="the interface an agent gets to the local models")
    p.add_argument("--list", action="store_true",
                   help="print the tool table and exit")
    a = p.parse_args(argv)
    if a.list:
        for t in tool_list():
            print(f"{t['name']:<10} {t['description']}")
        return 0
    return _serve_stdio()
