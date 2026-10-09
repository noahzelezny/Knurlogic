"""interfaces/page/server.py `/peek` and page/documents.py `/connect.json`:
read-only views of other servers for the page. No network: every fetch is a stub."""

import json
from types import SimpleNamespace

from knurlogic.interfaces.page import documents
from knurlogic.interfaces.page import nodes as page_nodes
from knurlogic.interfaces.page import peek as page_peek
from knurlogic.interfaces.page import router as page_router


class Peers:
    def __init__(self, *ps):
        self.ps = ps

    def all(self):
        return list(self.ps)


def _peer(host, state="answering"):
    return SimpleNamespace(name="M4", host=host, port=8899, state=state,
                           key=f"{host}:8899")


def q(**kw):
    return {k: [v] for k, v in kw.items()}


def test_peek_reads_an_answering_peer_page(monkeypatch):
    monkeypatch.setattr(page_nodes, "PEERS", Peers(_peer("192.0.2.2")))
    monkeypatch.setattr(page_router, "chat_targets", lambda: set())
    seen = []

    def fetch(url, t):
        seen.append(url)
        return b'{"knobs": []}'
    code, body = page_peek.peek(q(where="http://192.0.2.2:8899",
                           path="/settings.json", tune="lean",
                           evil="x"), fetch=fetch)
    assert code == 200 and json.loads(body) == {"knobs": []}
    # only the whitelisted query keys travel
    assert seen == ["http://192.0.2.2:8899/settings.json?tune=lean"]


def test_peek_reads_a_running_models_sampling_defaults(monkeypatch):
    monkeypatch.setattr(page_nodes, "PEERS", None)
    monkeypatch.setattr(page_router, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})
    code, _ = page_peek.peek(q(where="http://127.0.0.1:8080/", path="/v1/models"),
                      fetch=lambda u, t: b'{"data": []}')
    assert code == 200


def test_peek_refuses_unknown_targets_paths_and_quiet_peers(monkeypatch):
    monkeypatch.setattr(page_nodes, "PEERS", Peers(_peer("192.0.2.3", "silent")))
    monkeypatch.setattr(page_router, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})

    def fetch(u, t):
        raise AssertionError("must not be fetched")
    for where, path in (("http://evil:80", "/settings.json"),
                        ("http://192.0.2.3:8899", "/settings.json"),
                        ("http://127.0.0.1:8080", "/loaded.json"),
                        ("http://127.0.0.1:8080", "/v1/chat/completions")):
        code, body = page_peek.peek(q(where=where, path=path), fetch=fetch)
        assert code == 403 and "error" in json.loads(body)


def test_peek_passes_on_json_only(monkeypatch):
    monkeypatch.setattr(page_nodes, "PEERS", None)
    monkeypatch.setattr(page_router, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})
    code, body = page_peek.peek(q(where="http://127.0.0.1:8080",
                           path="/v1/models"),
                         fetch=lambda u, t: b"<html>")
    assert code == 502 and "error" in json.loads(body)


def test_connect_json_lists_every_way_in():
    body, ctype = documents.routes()["/connect.json"]({}, 0)
    doc = json.loads(body)
    ids = [e["id"] for e in doc["endpoints"]]
    assert ids == ["openai", "ollama", "claude", "mcp", "curl"]
    by = {e["id"]: e for e in doc["endpoints"]}
    # OpenAI and curl: the router first, the model's own server second
    for k in ("openai", "curl"):
        assert by[k]["pick_model"]
        assert "__ROUTER__/v1" in by[k]["blocks"][0]["text"]
        assert by[k]["blocks"][1]["direct"]
        assert "__BASE__/v1" in by[k]["blocks"][1]["text"]
        assert "__MODEL__" in by[k]["blocks"][0]["text"]
    assert not by["mcp"]["needs_model"]
    assert "knurlogic mcp" in by["mcp"]["blocks"][0]["text"]
    # Claude Code and Codex each have a line, marked by client for the page
    mcp = {b.get("client"): b for b in by["mcp"]["blocks"][:2]}
    assert mcp["claude"]["text"] == "claude mcp add knurlogic -- knurlogic mcp"
    assert mcp["codex"]["text"] == "codex mcp add knurlogic -- knurlogic mcp"
    toml = by["mcp"]["blocks"][2]
    assert toml["client"] == "codex" and "config.toml" in toml["label"]
    assert toml["text"].splitlines() == ["[mcp_servers.knurlogic]",
                                         'command = "knurlogic"',
                                         'args = ["mcp"]']
    # Claude Code goes to the page's router, one model per tier
    assert by["claude"]["tiers"] == ["opus", "sonnet", "haiku"]
    cmd = by["claude"]["blocks"][0]["text"]
    for line in ("ANTHROPIC_BASE_URL=__ROUTER__",
                 "ANTHROPIC_DEFAULT_OPUS_MODEL=__OPUS__",
                 "ANTHROPIC_DEFAULT_SONNET_MODEL=__SONNET__",
                 "ANTHROPIC_DEFAULT_HAIKU_MODEL=__HAIKU__"):
        assert line in cmd
    # the scoped settings file, never the global one
    assert all("~/.claude" not in b["text"]
               for b in by["claude"]["blocks"])


def test_a_peers_preview_goes_by_identity_never_by_path(monkeypatch):
    """The picker with the other Mac picked showed no room line: the
    preview was this machine's. It asks the peer's page now, naming the
    model by identity -- an `artifact` path never travels."""
    monkeypatch.setattr(page_nodes, "PEERS", Peers(_peer("192.0.2.2")))
    monkeypatch.setattr(page_router, "chat_targets", lambda: set())
    seen = []

    def fetch(url, t):
        seen.append(url)
        return b'{"knobs": []}'
    code, _ = page_peek.peek(q(where="http://192.0.2.2:8899",
                                 path="/settings.json", identity="5e07",
                                 name="Qwen", kv_bits="8",
                                 artifact="/etc"), fetch=fetch)
    assert code == 200 and len(seen) == 1
    assert "identity=5e07" in seen[0] and "kv_bits=8" in seen[0]
    assert "artifact" not in seen[0]


def test_the_settings_preview_resolves_an_identity_here(monkeypatch):
    """The peer side: its own copy of the model, from its own stores."""
    got = {}
    monkeypatch.setattr(
        "knurlogic.machine.artifact.resolve_identity",
        lambda ident, paths=None, name="": "/models/q" if ident == "5e07"
        else None)
    monkeypatch.setattr(documents, "preview",
                        lambda art, tune, ws=None, **kw:
                        got.update(art=art) or {"knobs": []})
    h = documents.machine_settings()
    assert h(q(identity="5e07", name="q")) == {"knobs": []}
    assert got == {"art": "/models/q"}
    assert "does not have" in h(q(identity="ffff"))["error"]
