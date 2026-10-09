"""interfaces/spawn.stop: a knurlogic server the page did not start (a
shell, another agent) is stopped by its port; anything else is refused."""
from knurlogic.interfaces import spawn
from knurlogic.machine import servers


def test_a_server_started_by_hand_is_found_by_its_port(monkeypatch):
    killed = []
    monkeypatch.setattr(spawn, "registry", lambda: {})
    monkeypatch.setattr(spawn, "save_registry", lambda reg: None)
    monkeypatch.setattr(servers, "listener_pid", lambda port: 4242)
    alive = {4242: True}
    monkeypatch.setattr(servers, "is_our_server", lambda pid: alive.get(pid, False))
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: alive.get(pid, False))

    def kill(pid, sig):
        killed.append((pid, sig))
        alive[pid] = False
    monkeypatch.setattr("os.kill", kill)
    monkeypatch.setattr(spawn, "_wait_exit", lambda pid, child: True)
    out = spawn.stop(8095)
    assert killed and killed[0][0] == 4242
    assert out["pid"] == 4242 and out["port"] == 8095


def test_a_port_with_no_knurlogic_server_is_refused(monkeypatch):
    monkeypatch.setattr(spawn, "registry", lambda: {})
    monkeypatch.setattr(servers, "listener_pid", lambda port: 777)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: False)
    assert "no knurlogic server" in spawn.stop(8095)["error"]
