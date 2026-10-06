"""A non-streamed request writes nothing until its reply is complete, so a
client that hangs up is never seen by a failed write: its connection is
watched instead (server._watch_hangup). A client that timed out and resent
left every copy prefilling 170k tokens to the end (2026-10-06)."""
import socket
import threading

from knurlogic.interfaces.http import server as S


class Job:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def _watch(sock, job, monkeypatch):
    monkeypatch.setattr(S, "HANGUP_POLL_S", 0.02)
    done = threading.Event()
    t = threading.Thread(target=S._watch_hangup, args=(job, sock, done))
    t.start()
    return done, t


def test_a_client_that_hangs_up_cancels_its_job(monkeypatch):
    srv, cli = socket.socketpair()
    job = Job()
    done, t = _watch(srv, job, monkeypatch)
    cli.close()
    t.join(2)
    assert job.cancelled and not t.is_alive()
    srv.close()


def test_a_client_still_waiting_is_left_alone(monkeypatch):
    srv, cli = socket.socketpair()
    cli.sendall(b"GET / HTTP/1.1\r\n")      # pipelined bytes: not a hang-up
    job = Job()
    done, t = _watch(srv, job, monkeypatch)
    t.join(0.2)
    assert not job.cancelled and t.is_alive()
    done.set()
    t.join(2)
    assert not t.is_alive() and not job.cancelled
    assert srv.recv(64) == b"GET / HTTP/1.1\r\n"    # nothing was consumed
    srv.close()
    cli.close()


def test_a_finished_reply_stops_the_watch(monkeypatch):
    srv, cli = socket.socketpair()
    job = Job()
    done, t = _watch(srv, job, monkeypatch)
    done.set()
    t.join(2)
    cli.close()
    assert not t.is_alive() and not job.cancelled
    srv.close()
