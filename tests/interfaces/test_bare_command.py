"""`knurlogic` alone starts the page; the page's default host is cluster."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.interfaces import cli
from knurlogic.interfaces.page import server


def test_bare_command_runs_ui(monkeypatch):
    seen = []
    monkeypatch.setattr(server, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main([]) == 0
    assert seen == [[]]
    assert cli.main(["--host", "127.0.0.1"]) == 0     # the page's options
    assert seen[-1] == ["--host", "127.0.0.1"]


def test_help_still_lists(capsys):
    assert cli.main(["help"]) == 0
    assert "usage: knurlogic" in capsys.readouterr().out
    assert cli.main(["--help"]) == 0


def test_default_host_cluster_port_8899(monkeypatch, tmp_path):
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path))
    seen = {}
    monkeypatch.setattr(server, "serve_ui", lambda host, port, *a, **k:
                        seen.update(host=host, port=port) or 0)
    assert server.main([]) == 0
    assert seen == {"host": "cluster", "port": 8899}
    server.main(["--host", "127.0.0.1"])
    assert seen["host"] == "127.0.0.1"
