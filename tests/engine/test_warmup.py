"""The load-time warm-up: one tiny generation before the host says ready."""
import importlib

import mlx.nn as nn
import pytest

from knurlogic.engine.runtime import host as H


@pytest.fixture
def host(tmp_path, monkeypatch):
    L = importlib.import_module("knurlogic.engine.serve.load")
    monkeypatch.setenv("KNURLOGIC_LOADLOCK", str(tmp_path / "load.lock"))
    monkeypatch.setattr(L, "load_unlocked",
                        lambda path, code, lazy=False: (nn.Linear(2, 2),
                                                        object()))
    monkeypatch.setattr(H.ModelHost, "_bind_head", lambda self, path: None)
    monkeypatch.delenv("KNURLOGIC_WARMUP", raising=False)
    return H.ModelHost(vision=False), tmp_path


def test_warm_up_runs_once_while_warming_then_ready(host):
    h, path = host
    seen = []
    h.warm = lambda: seen.append(h.state)
    h.load(str(path))
    assert seen == ["warming"]
    assert h.state == "ready"


def test_warm_up_skipped_when_off(host, monkeypatch):
    h, path = host
    monkeypatch.setenv("KNURLOGIC_WARMUP", "off")
    seen = []
    h.warm = lambda: seen.append(1)
    h.load(str(path))
    assert seen == [] and h.state == "ready"


def test_a_failed_warm_up_still_serves(host):
    h, path = host

    def boom():
        raise RuntimeError("no kernel")
    h.warm = boom
    h.load(str(path))
    assert h.state == "ready"


def test_every_parameter_is_evaluated_inside_the_load(host, monkeypatch):
    """A model run from its own bundled model.py may skip mlx-lm's own
    evaluation; weights left lazy were then read by the first request, with
    the page already saying ready."""
    import mlx.core as mx
    h, path = host
    seen = []
    real = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *a: seen.append(a) or real(*a))
    h._weights(str(path))
    assert seen


def test_the_walker_finds_arrays_parameters_does_not_see():
    """A bundled model.py may keep codebooks in "_"-named attributes or on
    plain helper objects: nn.Module.parameters() skips them, so they stayed
    lazy until the first request."""
    import mlx.core as mx

    class Helper:
        def __init__(self):
            self.table = mx.zeros((2,))
            self.more = [mx.ones((1,)), {"k": mx.ones((3,))}]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.ones((2,))
            self._codebook = mx.ones((4,))
            self.helper = Helper()
            self.loop = self          # a cycle must not hang

    m = M()
    assert len(H.held_arrays(m)) == 5
    assert H.evaluate_everything(m) == 5


def _warm_prompt(monkeypatch, chunk, cap, native=0):
    """The prompt _warm_up hands _insert for this chunk and window."""
    from types import SimpleNamespace
    from knurlogic.engine.runtime import scheduler as S
    from knurlogic.machine import artifact as A
    monkeypatch.setenv("KNURLOGIC_CONTEXT_LENGTH", str(cap))
    monkeypatch.setattr(A, "context_length", lambda path: native)
    s = object.__new__(S.Scheduler)
    s.prefill_step_size = chunk
    s.host = SimpleNamespace(path="/m")
    s._rows = []
    seen = []
    s._insert = lambda job: seen.append(job.request.prompt)
    s._warm_up()
    return seen[0]


def test_warm_up_prompt_fits_a_small_context_cap(monkeypatch):
    """chunk/3 repeats of "Hello, world. " is ~1.33 chunks of tokens: at
    a 4096 chunk with a 4k cap it raised PromptError every launch."""
    for cap, native in ((4096, 0), (0, 4096), (512, 0), (100, 0)):
        p = _warm_prompt(monkeypatch, 4096, cap, native)
        n = p.count("Hello")
        assert n >= 1
        assert n * 5 + 128 <= max(cap or native, 133)


def test_warm_up_prompt_stays_chunk_wide_in_a_large_window(monkeypatch):
    assert _warm_prompt(monkeypatch, 4096, 0, 262144).count("Hello") == 1366
    assert _warm_prompt(monkeypatch, 4096, 0, 0).count("Hello") == 1366
