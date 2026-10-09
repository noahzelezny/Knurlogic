"""A streamed request still queued is kept alive, and cancelled when its
client goes away (interfaces/http/server.py queued).

Before: nothing was written until the job's first event, so a client that
hung up or timed out while its request waited in the queue could not be
noticed, and the job prefilled for nobody ahead of live requests.
"""
import queue

from knurlogic.interfaces.http import server as S


class _Job:
    def __init__(self):
        self.outbox = queue.Queue()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Reply:
    def __init__(self, job):
        self.job = job

    def first(self, timeout=None):
        return self.job.outbox.get(timeout=timeout)

    def events(self, first):
        yield b"data: " + first[1] + b"\n\n"
        yield b"data: [DONE]\n\n"


def test_keepalives_while_queued_then_the_reply(monkeypatch):
    monkeypatch.setattr(S, "QUEUED_KEEPALIVE_S", 0.01)
    job = _Job()
    gen = S.queued(job, _Reply(job))
    assert next(gen) == b": keepalive queued\n\n"
    assert next(gen) == b": keepalive queued\n\n"
    job.outbox.put(("delta", b"hi"))
    rest = list(gen)
    assert rest[-2:] == [b"data: hi\n\n", b"data: [DONE]\n\n"]
    assert all(c.startswith(b": keepalive") for c in rest[:-2])


def test_a_client_that_left_cancels_the_queued_job(monkeypatch):
    monkeypatch.setattr(S, "QUEUED_KEEPALIVE_S", 0.01)
    job = _Job()
    gen = S.queued(job, _Reply(job))
    next(gen)
    # the handler's write failed: it closes the iterator
    gen.close()
    assert job.cancelled


def test_a_refusal_after_the_200_is_an_error_event(monkeypatch):
    monkeypatch.setattr(S, "QUEUED_KEEPALIVE_S", 0.01)
    job = _Job()
    job.outbox.put(("error", ValueError("prompt too long")))
    out = list(S.queued(job, _Reply(job)))
    assert out[-1] == b"data: [DONE]\n\n"
    assert b"error" in out[-2]
    assert job.cancelled
