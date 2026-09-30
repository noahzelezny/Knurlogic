"""Deleting a Hugging Face download: refused while a server runs the model,
and only for a well-formed repo id."""

import json

from knurlogic.interfaces.page import hub
from knurlogic.machine import servers


def test_a_repo_id_is_org_slash_name():
    for bad in ("x", "../x/y", "a/b/c", "a/../b", "a/b c", ""):
        assert "error" in hub.act(json.dumps(
            {"id": bad, "action": "delete"}).encode())


def test_delete_is_refused_while_a_server_runs_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_cache_dir",
                        lambda r: tmp_path / ("models--" + r.replace("/", "--")))
    snap = tmp_path / "models--o--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    monkeypatch.setattr(servers, "registry",
                        lambda: {8080: {"pid": 1, "artifact": str(snap)}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    out = hub.delete("o/m")
    assert "error" in out and "8080" in out["error"]
    assert snap.exists()
    monkeypatch.setattr(servers, "registry", lambda: {})
    monkeypatch.setattr(hub, "_refresh_models", lambda: None)
    monkeypatch.setattr(hub, "_save", lambda: None)
    assert hub.delete("o/m") == {"id": "o/m"} and not snap.exists()
