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


def env_lines(base_url: str, model: str) -> dict:
    """The variables a Claude-Messages harness reads."""
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "x",          # unused locally, but must be set
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        # A local model is slower per token than the hosted one this harness
        # was tuned against, and the default timeout is the first thing to
        # bite on a long tool loop.
        "API_TIMEOUT_MS": "3000000",
    }


def claude_command(base_url: str, model: str) -> str:
    lines = [f"{k}={v} \\" for k, v in env_lines(base_url, model).items()]
    return "\n".join(lines + ["claude"])


def project_settings(base_url: str, model: str) -> str:
    """A `.claude/settings.json` that scopes this to one directory.

    Scoped on purpose. Put it in a scratch directory and only sessions
    started there use the local model; everything else is untouched.
    """
    return json.dumps({"env": env_lines(base_url, model)}, indent=2)


def openai_snippet(base_url: str, model: str) -> str:
    return (f"base_url  {base_url}/v1\n"
            f"api_key   anything\n"
            f"model     {model}")


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
