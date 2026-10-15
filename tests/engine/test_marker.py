"""engine/split/marker: a follower's step watchdog and chunk trace."""
import logging

from knurlogic.engine.split import marker


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_step_watch_dumps_once_when_a_step_stalls(monkeypatch):
    said = []
    monkeypatch.setattr(marker, "progress", lambda **f: said.append(f))
    dumps, clock = [], Clock()
    w = marker.StepWatch(stall_s=10, dump=lambda: dumps.append(1),
                         clock=clock)
    assert not w.check()                    # idle: nothing to judge
    w.begin()
    clock.t = 9
    assert not w.check()
    w.moved()                               # a prefill chunk is progress
    clock.t = 18
    assert not w.check()
    clock.t = 30
    assert w.check() and dumps == [1]
    clock.t = 300
    assert not w.check() and dumps == [1]   # once, and nothing killed
    w.end(4)
    assert said == [{"busy": True}, {"busy": False, "step": 4}]
    w.begin()
    clock.t = 400
    assert w.check() and dumps == [1, 1]    # a new step stalls anew


def test_the_step_watch_starts_no_thread_outside_a_job(monkeypatch):
    monkeypatch.setitem(marker.CURRENT, "marker", None)
    import threading
    before = threading.active_count()
    marker.StepWatch().start(every=0.01)
    assert threading.active_count() == before


def test_chunks_are_logged_only_on_a_pipeline_follower(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=marker.__name__)
    monkeypatch.setitem(marker.CURRENT, "trace_chunks", False)
    marker.chunk_start(0, 512, 4096)
    assert not caplog.records
    monkeypatch.setitem(marker.CURRENT, "trace_chunks", True)
    marker.chunk_start(512, 1024, 4096)
    assert "prefill chunk 512-1024 of 4096" in caplog.text


def test_a_chunk_done_moves_the_watch(monkeypatch):
    clock = Clock()
    w = marker.StepWatch(stall_s=10, dump=lambda: None, clock=clock)
    monkeypatch.setattr(marker, "progress", lambda **f: None)
    monkeypatch.setitem(marker.CURRENT, "marker", None)
    monkeypatch.setitem(marker.CURRENT, "watch", w)
    w.begin()
    clock.t = 8
    marker.chunk_done()
    clock.t = 15
    assert not w.check()
