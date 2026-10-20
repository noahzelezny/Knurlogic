"""engine/embeddings.py: the embedder's pooling, normalization, one pass
per text, truncation and idle drop, over a fake model (no download)."""

import math

import mlx.core as mx
import pytest

from knurlogic.engine import embeddings as E


class _Trunk:
    """Hidden state of token t at every position: [t, 1, 0, ...] scaled by
    the position, so the last token's own row is recognisable."""

    def __init__(self):
        self.calls = []

    def __call__(self, ids):
        self.calls.append(ids.tolist())
        rows = [[float(t) * (i + 1), 1.0, 0.0]
                for i, t in enumerate(ids.tolist()[0])]
        return mx.array([rows])


class _Model:
    def __init__(self):
        self.model = _Trunk()


class _Tok:
    EOS = 9

    def encode(self, text):
        return [len(w) for w in text.split()] + [self.EOS]


def _embedder(**kw):
    loads = []

    def load(path):
        loads.append(path)
        return _Model(), _Tok()
    e = E.Embedder("/models/fake-Embedding", load=load, **kw)
    return e, loads


def test_the_last_tokens_hidden_state_unit_length():
    e, _ = _embedder()
    got = e.embed(["ab c"])
    (v,) = got.vectors
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-6)
    # ids [2, 1, 9]: the last row is [9 * 3, 1, 0]
    n = math.hypot(27.0, 1.0)
    assert v == pytest.approx([27.0 / n, 1.0 / n, 0.0])
    assert got.tokens == 3


def test_one_forward_pass_per_text_and_no_padding():
    e, _ = _embedder()
    got = e.embed(["a", "bbb cc", "dddd"])
    assert len(got.vectors) == 3
    assert e.model.model.calls == [[[1, 9]], [[3, 2, 9]], [[4, 9]]]
    assert got.tokens == 2 + 3 + 2


def test_a_long_text_keeps_its_closing_token():
    assert E.clip([1, 2, 3, 4, 9], 3) == [1, 2, 9]
    assert E.clip([1, 9], 3) == [1, 9]


def test_loaded_on_first_use_and_dropped_when_idle():
    now = [0.0]
    e, loads = _embedder(idle_s=10.0, clock=lambda: now[0])
    assert e.status()["state"] == "empty" and loads == []
    e.embed(["x"])
    assert e.status()["state"] == "ready" and len(loads) == 1
    now[0] = 5.0
    assert not e.drop_if_idle()
    now[0] = 11.0
    assert e.drop_if_idle()
    assert e.status()["state"] == "empty" and e.model is None
    e.embed(["x"])
    assert len(loads) == 2


def test_a_failed_load_is_said():
    def load(_path):
        raise OSError("gone")
    e = E.Embedder("/models/x", load=load)
    with pytest.raises(OSError):
        e.embed(["x"])
    assert e.status() == {"state": "failed", "error": "OSError: gone"}


def test_only_a_known_recipe_is_served():
    assert E.refusal("qwen3") is None
    assert "qwen3" in E.refusal("bert")


def test_warm_loads_at_launch_and_a_failure_is_only_said():
    e, loads = _embedder()
    e.warm()
    assert e.status()["state"] == "ready" and len(loads) == 1

    def load(_path):
        raise OSError("gone")
    bad = E.Embedder("/models/x", load=load)
    bad.warm()
    assert bad.status()["state"] == "failed"


def test_the_load_bounds_mlxs_buffer_pool(monkeypatch):
    from knurlogic.engine import model as M
    seen = []
    monkeypatch.setattr(M, "set_cache_limit", seen.append)
    monkeypatch.setattr(M, "load", lambda p: ("model", "tok"))
    assert E._load("/models/x") == ("model", "tok")
    assert seen == [E.CACHE_LIMIT_GIB]
