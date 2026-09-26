"""How to point something at this server.

exo has a panel that writes the launch line for you, and it is the single
most useful thing in that UI: the endpoint is worthless until a client is
actually pointed at it, and the pointing is four environment variables that
nobody remembers.

TWO SHAPES, because they have different blast radius:

* a one-liner for a terminal, which affects that command and nothing else;
* a PROJECT settings file, which affects sessions started in one directory.

Deliberately not offered: writing the user's global `~/.claude/settings.json`.
That would silently route every session on the machine -- including the ones
they use for work that has nothing to do with a local model -- at a 3-bit
quantisation of a 30B. A config change nobody can see is how you end up
debugging the wrong thing for an afternoon.
"""

from __future__ import annotations

import json

#: Clients and what they need. Adding one is a row.
CLIENTS = ("claude", "openai", "settings")


def env_lines(base_url: str, model: str, sonnet: str = "",
              haiku: str = "") -> dict:
    """The variables a Claude-Messages harness reads. One model fills every
    tier unless others are named: pointed at a model server that is all it
    can answer; pointed at the page's router, each tier can be its own."""
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "x",          # unused locally, but must be set
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": sonnet or model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": haiku or model,
        # A local model is slower per token than the hosted one this harness
        # was tuned against, and the default timeout is the first thing to
        # bite on a long tool loop.
        "API_TIMEOUT_MS": "3000000",
    }


def claude_command(base_url: str, model: str, sonnet: str = "",
                   haiku: str = "") -> str:
    lines = [f"{k}={v} \\" for k, v in
             env_lines(base_url, model, sonnet, haiku).items()]
    return "\n".join(lines + ["claude"])


def project_settings(base_url: str, model: str, sonnet: str = "",
                     haiku: str = "") -> str:
    """A `.claude/settings.json` that scopes this to one directory.

    Scoped on purpose. Put it in a scratch directory and only sessions
    started there use the local model; everything else is untouched.
    """
    return json.dumps({"env": env_lines(base_url, model, sonnet, haiku)},
                      indent=2)


def openai_snippet(base_url: str, model: str) -> str:
    return (f"base_url  {base_url}/v1\n"
            f"api_key   anything\n"
            f"model     {model}")


def curl_snippet(base_url: str, model: str) -> str:
    """One request by hand -- the quickest proof the endpoint answers."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "hello"}]})
    return (f"curl {base_url}/v1/chat/completions \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{body}'")


#: The MCP server is stdio, not HTTP: a client starts `knurlogic mcp` itself,
#: so there is no base URL or model in it -- it reaches every model here.
MCP_ADD = "claude mcp add knurlogic -- knurlogic mcp"


#: Codex CLI registers MCP servers itself too, or reads them from its own
#: config file; either way it starts the same stdio command.
CODEX_MCP_ADD = "codex mcp add knurlogic -- knurlogic mcp"


def codex_toml() -> str:
    """The ~/.codex/config.toml entry. Codex has no per-directory config, so
    this one is global -- harmless here: an MCP server only adds tools, it
    does not reroute the model Codex talks to."""
    return ('[mcp_servers.knurlogic]\n'
            'command = "knurlogic"\n'
            'args = ["mcp"]')


def mcp_json() -> str:
    return json.dumps({"mcpServers": {"knurlogic": {
        "command": "knurlogic", "args": ["mcp"]}}}, indent=2)


def endpoints(base_url: str, model: str, router_url: str = "__ROUTER__",
              tiers=("__OPUS__", "__SONNET__", "__HAIKU__")) -> list:
    """Every way in, one entry each, for a page that shows one at a time.

    The same text `render` prints, split by client so a picker can list
    them; the page fills in the base URL and model it is pointed at. Claude
    Code's goes to the page's ROUTER (`router_url`), which hands each
    request to the server holding the model it names, so each tier
    (opus, sonnet, haiku) can be a different running model."""
    opus, sonnet, haiku = tiers
    return [
        {"id": "openai", "name": "OpenAI-compatible",
         "what": "anything that speaks OpenAI: Zed, Cline, Continue, "
                 "OpenWebUI, the openai SDKs",
         "needs_model": True,
         "blocks": [{"label": "settings",
                     "text": openai_snippet(base_url, model)},
                    {"label": "or through this page, which routes by "
                              "`model` to every running model",
                     "text": openai_snippet(router_url, model)}]},
        {"id": "claude", "name": "Claude Code",
         "what": "a Claude-Messages harness, over /v1/messages",
         "needs_model": True, "tiers": ["opus", "sonnet", "haiku"],
         "blocks": [{"label": "in a terminal",
                     "text": claude_command(router_url, opus, sonnet,
                                            haiku)},
                    {"label": "scoped to one directory, as "
                              ".claude/settings.json (not the global one: "
                              "that routes every session on the machine)",
                     "text": project_settings(router_url, opus, sonnet,
                                              haiku)}]},
        {"id": "mcp", "name": "MCP",
         "what": "the agent-facing tools (ready, fit, settings, load) over "
                 "stdio; the client starts it, so no address is needed",
         "needs_model": False,
         # `client` names whose line a block is, so the page can mark it
         # (Claude Code in its tier amber, Codex in white)
         "blocks": [{"label": "Claude Code", "client": "claude",
                     "text": MCP_ADD},
                    {"label": "Codex CLI", "client": "codex",
                     "text": CODEX_MCP_ADD},
                    {"label": "Codex CLI, as ~/.codex/config.toml",
                     "client": "codex", "text": codex_toml()},
                    {"label": "any MCP client's config", "text": mcp_json()}]},
        {"id": "curl", "name": "curl",
         "what": "one request by hand",
         "needs_model": True,
         "blocks": [{"label": "chat completion",
                     "text": curl_snippet(base_url, model)}]},
    ]


def render(base_url: str, model: str) -> str:
    return "\n".join([
        "a Claude-Messages harness, in a terminal:",
        "",
        "  " + claude_command(base_url, model).replace("\n", "\n  "),
        "",
        "the same thing scoped to one directory, as .claude/settings.json:",
        "",
        "  " + project_settings(base_url, model).replace("\n", "\n  "),
        "",
        "NOT your global ~/.claude/settings.json: that would route every",
        "session on this machine at a local model, including the ones that",
        "have nothing to do with it.",
        "",
        "anything that speaks OpenAI (Zed, Cline, Continue, OpenWebUI):",
        "",
        "  " + openai_snippet(base_url, model).replace("\n", "\n  "),
    ])


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="knurlogic connect",
        description="print how to point a client at a running server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="local",
                   help="the name clients should send; serve pins it anyway")
    a = p.parse_args(argv)
    print(render(f"http://{a.host}:{a.port}", a.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
