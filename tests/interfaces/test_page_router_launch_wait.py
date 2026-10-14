"""A request for a model whose split job is still starting waits for it
(the page's router), instead of a 404 while rank 0 has no port yet."""
from knurlogic.interfaces.page import router


class _H:
    connection = None


def _setup(monkeypatch, phases, tables):
    calls = {"n": 0}

    def routable(fetch=None, ttl=5.0):
        i = min(calls["n"], len(tables) - 1)
        calls["n"] += 1
        return tables[i]
    seq = iter(phases)
    last = {"v": phases[-1]}

    def starting():
        try:
            last["v"] = next(seq)
        except StopIteration:
            pass
        return last["v"]
    monkeypatch.setattr(router, "routable", routable)
    monkeypatch.setattr(router, "refresh_targets", lambda: None)
    monkeypatch.setattr(router, "_starting_jobs", starting)
    monkeypatch.setattr(router, "hung_up", lambda c: False)


def test_waits_while_a_split_job_loads(monkeypatch):
    _setup(monkeypatch, [(True, False)] * 3 + [(False, True)],
           [{}, {}, {}, {"m": "http://b"}])
    base, _ = router._await_launch(_H(), "m", poll=0)
    assert base == "http://b"


def test_no_launch_under_way_refuses_at_once(monkeypatch):
    _setup(monkeypatch, [(False, False)], [{}])
    assert router._await_launch(_H(), "m", poll=0)[0] is None


def test_a_loaded_job_gets_a_bounded_grace(monkeypatch):
    _setup(monkeypatch, [(False, True)], [{}])
    monkeypatch.setattr(router, "READY_GRACE_S", 0.0)
    assert router._await_launch(_H(), "m", poll=0)[0] is None


def test_a_client_that_hangs_up_stops_the_wait(monkeypatch):
    _setup(monkeypatch, [(True, False)], [{}])
    monkeypatch.setattr(router, "hung_up", lambda c: True)
    assert router._await_launch(_H(), "m", poll=0)[0] is None
