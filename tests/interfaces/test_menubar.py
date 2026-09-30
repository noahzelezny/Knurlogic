from knurlogic.interfaces import menubar as mb


def test_menu_model():
    doc = {"resident": [{"name": "Qwen", "state": "loaded"},
                        {"name": "Big", "state": "loading"},
                        {"name": "Peer", "machine": "m4"}]}
    m = mb.menu_model(doc, "mac")
    assert m["title"] == "Knurlogic — 2 models loaded"
    assert m["models"] == ["Qwen · mac", "Peer · m4"]
    assert mb.menu_model(None)["title"].endswith("0 models loaded")
    assert "1 model loaded" in mb.menu_model({"resident": [{"name": "a"}]})["title"]


def test_lock_is_exclusive(tmp_path):
    p = tmp_path / "l"
    a = mb.acquire_lock(p)
    assert a is not None and mb.acquire_lock(p) is None
    a.close()
    assert mb.acquire_lock(p) is not None


def test_no_gui_skips(monkeypatch):
    ok = dict(platform="darwin", uid=5, console_owner=lambda: 5, env={})
    assert mb.gui_available(**ok)
    assert not mb.gui_available(**{**ok, "env": {"SSH_CONNECTION": "x"}})
    assert not mb.gui_available(**{**ok, "console_owner": lambda: 0})
    assert not mb.gui_available(**{**ok, "platform": "linux"})
    called = []
    monkeypatch.setattr(mb.subprocess, "Popen", lambda *a, **k: called.append(1))
    assert not mb.spawn(1, enabled=False) and not called
