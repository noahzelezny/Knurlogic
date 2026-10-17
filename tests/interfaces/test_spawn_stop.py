"""interfaces/spawn.stop: a knurlogic server the page did not start (a
shell, another agent) is stopped by its port; anything else is refused."""
from knurlogic.interfaces import spawn
from knurlogic.machine import processes, servers


def test_a_server_started_by_hand_is_found_by_its_port(monkeypatch):
    killed = []
    monkeypatch.setattr(spawn, "registry", dict)
    monkeypatch.setattr(spawn, "update_registry", lambda change: change({}))
    monkeypatch.setattr(processes, "listener", lambda port: 4242)
    alive = {4242: True}
    monkeypatch.setattr(servers, "is_our_server", lambda pid: alive.get(pid, False))
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: alive.get(pid, False))

    def kill(pid, sig):
        killed.append((pid, sig))
        alive[pid] = False
    monkeypatch.setattr("os.kill", kill)
    monkeypatch.setattr(spawn, "_wait_exit", lambda pid, child, wait_s: True)
    out = spawn.stop(8095)
    assert killed and killed[0][0] == 4242
    assert out["pid"] == 4242 and out["port"] == 8095


def test_a_port_with_no_knurlogic_server_is_refused(monkeypatch):
    monkeypatch.setattr(spawn, "registry", dict)
    monkeypatch.setattr(processes, "listener", lambda port: 777)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: False)
    assert "no knurlogic server" in spawn.stop(8095)["error"]


def test_a_stop_keeps_the_record_of_a_server_started_while_it_waited(
        tmp_path, monkeypatch):
    """stop waits up to STOP_SAVE_S for its server to go. It used to write
    back the registry it had read before the wait, wiping any server
    recorded meanwhile; now it drops its own port, and only while that
    port is still its pid's."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    servers.save_registry({8095: {"pid": 4242, "artifact": "/m/a"},
                           8096: {"pid": 1, "artifact": "/m/b"}})
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: None)

    def wait_while_another_starts(pid, child, wait_s):
        servers.update_registry(lambda r: r.update(
            {8097: {"pid": 3, "artifact": "/m/c"}}))
        return True
    monkeypatch.setattr(spawn, "_wait_exit", wait_while_another_starts)
    assert spawn.stop(8095)["pid"] == 4242
    assert sorted(servers.registry()) == [8096, 8097]
    # a port taken over by another pid meanwhile keeps that pid's record
    servers.update_registry(lambda r: r.update(
        {8095: {"pid": 5, "artifact": "/m/d"}}))
    spawn._forget(8095, 4242)
    assert servers.registry()[8095]["pid"] == 5

