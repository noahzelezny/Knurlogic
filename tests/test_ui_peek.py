"""interfaces/ui.py `/peek` and web.py `/connect.json`: read-only views of
other servers for the page. No network: every fetch is a stub."""

import json
from types import SimpleNamespace

from knurlogic.interfaces import ui, web


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
    monkeypatch.setattr(ui, "PEERS", Peers(_peer("10.0.0.2")))
    monkeypatch.setattr(ui, "chat_targets", lambda: set())
    seen = []

    def fetch(url, t):
        seen.append(url)
        return b'{"knobs": []}'
    code, body = ui.peek(q(where="http://10.0.0.2:8899",
                           path="/settings.json", tune="fast",
                           evil="x"), fetch=fetch)
    assert code == 200 and json.loads(body) == {"knobs": []}
    # only the whitelisted query keys travel
    assert seen == ["http://10.0.0.2:8899/settings.json?tune=fast"]


def test_peek_reads_a_running_models_sampling_defaults(monkeypatch):
    monkeypatch.setattr(ui, "PEERS", None)
    monkeypatch.setattr(ui, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})
    code, _ = ui.peek(q(where="http://127.0.0.1:8080/", path="/v1/models"),
                      fetch=lambda u, t: b'{"data": []}')
    assert code == 200


def test_peek_refuses_unknown_targets_paths_and_quiet_peers(monkeypatch):
    monkeypatch.setattr(ui, "PEERS", Peers(_peer("10.0.0.3", "silent")))
    monkeypatch.setattr(ui, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})

    def fetch(u, t):
        raise AssertionError("must not be fetched")
    for where, path in (("http://evil:80", "/settings.json"),
                        ("http://10.0.0.3:8899", "/settings.json"),
                        ("http://127.0.0.1:8080", "/loaded.json"),
                        ("http://127.0.0.1:8080", "/v1/chat/completions")):
        code, body = ui.peek(q(where=where, path=path), fetch=fetch)
        assert code == 403 and "error" in json.loads(body)


def test_peek_passes_on_json_only(monkeypatch):
    monkeypatch.setattr(ui, "PEERS", None)
    monkeypatch.setattr(ui, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})
    code, body = ui.peek(q(where="http://127.0.0.1:8080",
                           path="/v1/models"),
                         fetch=lambda u, t: b"<html>")
    assert code == 502 and "error" in json.loads(body)


def test_connect_json_lists_every_way_in():
    body, ctype = web.routes()["/connect.json"]({}, 0)
    doc = json.loads(body)
    ids = [e["id"] for e in doc["endpoints"]]
    assert ids == ["openai", "claude", "mcp", "curl"]
    by = {e["id"]: e for e in doc["endpoints"]}
    assert "__BASE__/v1" in by["openai"]["blocks"][0]["text"]
    assert "__MODEL__" in by["curl"]["blocks"][0]["text"]
    assert not by["mcp"]["needs_model"]
    assert "knurlogic mcp" in by["mcp"]["blocks"][0]["text"]
    # Claude Code goes to the page's router, one model per tier
    assert by["claude"]["tiers"] == ["opus", "sonnet", "haiku"]
    cmd = by["claude"]["blocks"][0]["text"]
    for line in ("ANTHROPIC_BASE_URL=__ROUTER__",
                 "ANTHROPIC_DEFAULT_OPUS_MODEL=__OPUS__",
                 "ANTHROPIC_DEFAULT_SONNET_MODEL=__SONNET__",
                 "ANTHROPIC_DEFAULT_HAIKU_MODEL=__HAIKU__"):
        assert line in cmd
    assert "__ROUTER__/v1" in by["openai"]["blocks"][1]["text"]
    # the scoped settings file, never the global one
    assert all("~/.claude" not in b["text"]
               for b in by["claude"]["blocks"])
