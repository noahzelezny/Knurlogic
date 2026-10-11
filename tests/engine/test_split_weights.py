"""engine/split/weights.py: a split rank's share filled from a fake peer
instead of the mount, the load order over the bell, and the fallback."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from knurlogic.engine.split import tensor as T
from knurlogic.engine.split import weights as W


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(8, 16), nn.Linear(16, 8), nn.Linear(8, 8)]
        self.norm = nn.RMSNorm(8)


@pytest.fixture
def shard(tmp_path):
    """One safetensors shard of a Tiny with known weights, bf16 and f32."""
    m = Tiny()
    m.layers[0].weight = mx.random.normal((16, 8)).astype(mx.bfloat16)
    path = tmp_path / "model-00001-of-00001.safetensors"
    from mlx.utils import tree_flatten
    mx.save_safetensors(str(path), dict(tree_flatten(m.parameters())))
    return path


class Peer:
    """A fake peer page: serves byte ranges of the shard by file name."""

    def __init__(self, root, fail_at=None):
        self.root, self.calls, self.fail_at = root, [], fail_at

    def __call__(self, file, offset, length, into):
        assert "/" not in file
        if self.fail_at is not None and \
                offset <= self.fail_at < offset + length:
            raise ConnectionError("the peer went away")
        self.calls.append((file, offset, length))
        assert length <= W.PIECE
        with open(self.root / file, "rb") as f:
            f.seek(offset)
            into[:] = f.read(length)


@pytest.fixture
def plan():
    yield
    W.CURRENT.update(plan=None, link=None)
    W.forget()


def lazy_model(path):
    with W.recording():
        m = Tiny()
        m.load_weights(list(mx.load(str(path)).items()))
    return m


def flat(m):
    from mlx.utils import tree_flatten
    return {k: v for k, v in tree_flatten(m.parameters())}


def test_a_pipeline_share_comes_from_the_peer(shard, plan):
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer, source="holder")
    m = lazy_model(shard)
    m.layers = m.layers[1:]                  # this rank's stage: layers 1..2
    out = W.fill(m, peer, "holder")
    assert out["left"] == 0 and not out["failed"]
    # only the share's tensors were asked for, and nothing of layer 0
    # neighbours in the file go as one request
    assert 0 < len(peer.calls) <= len(flat(m))
    ref = mx.load(str(shard))
    for k, v in flat(m).items():
        kk = k.replace("layers.0.", "layers.1.", 1) if k.startswith(
            "layers.0.") else k.replace("layers.1.", "layers.2.", 1) \
            if k.startswith("layers.1.") else k
        assert mx.array_equal(v, ref[kk]).item(), k


def test_a_tensor_cut_is_rebuilt_from_the_fetched_source(shard, plan):
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    m = lazy_model(shard)
    lay = m.layers[0]
    lay.update(T.split_params(lay.parameters(),
                              lambda p, w: 0 if w.ndim == 2 else None, 1, 2))
    out = W.fill(m, peer)
    assert out["left"] == 0
    ref = mx.load(str(shard))["layers.0.weight"]
    got = m.layers[0].weight
    assert got.dtype == mx.bfloat16 and got.shape == (8, 8)
    assert mx.array_equal(got, ref[8:]).item()


def test_a_failing_peer_leaves_the_rest_to_the_mount(shard, plan):
    last = max(W.header(str(shard)).values())    # the shard's last tensor
    peer = Peer(shard.parent, fail_at=last)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    m = lazy_model(shard)
    out = W.fill(m, peer)
    assert out["failed"] and out["left"] > 0
    mx.eval(m.parameters())                  # the mount reads the rest
    ref = mx.load(str(shard))
    for k, v in flat(m).items():
        assert mx.array_equal(v, ref[k]).item(), k


class Bell:
    def __init__(self, log):
        self.log = log

    def align(self, timeout=None):
        self.log.append("align")


@pytest.mark.parametrize("turn,want", [
    (0, ["fill", "eval", "align"]),          # reads first, then lets go
    (1, ["align", "eval"]),                  # waits, then reads its own
])
def test_the_turn_is_kept_over_the_bell(turn, want, monkeypatch, plan):
    log, marks = [], []
    monkeypatch.setattr(W, "progress", lambda **f: marks.append(f))
    monkeypatch.setattr(W, "fill", lambda *a, **k: log.append("fill") or {
        "fetched": 0, "left": 0, "failed": ""})
    W.CURRENT.update(link=Bell(log), plan=W.ReadPlan(
        read=(lambda *a: None) if turn == 0 else None, turn=turn,
        ordered=True))
    W.read_share(object(), lambda m: log.append("eval"))
    assert log == want
    if turn == 1:
        assert marks[0] == {"turn": "waiting"}


def test_no_order_no_waiting(plan):
    log = []
    W.CURRENT.update(link=Bell(log), plan=W.ReadPlan(turn=1, ordered=False))
    W.read_share(object(), lambda m: log.append("eval"))
    assert log == ["eval"]


def test_the_plan_comes_from_the_environment():
    got = W.plan_from_env(
        {"KNURLOGIC_WEIGHTS": '{"from": "198.51.100.1:8899", "source": '
                              '"holder", "turn": 0, "ordered": true}'},
        lambda page: ("reader", page))
    assert got.read == ("reader", "198.51.100.1:8899") and got.ordered
    assert W.plan_from_env({}, lambda p: None) is None
    own = W.plan_from_env({"KNURLOGIC_WEIGHTS": '{"turn": 1, "ordered": true}'},
                          lambda p: 1 / 0)
    assert own.read is None and own.turn == 1


def test_a_large_tensor_comes_in_parallel_pieces(shard, plan, monkeypatch):
    monkeypatch.setattr(W, "PIECE", 64)          # bytes: many pieces each
    monkeypatch.setattr(W, "BUDGET", 256)
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    m = lazy_model(shard)
    out = W.fill(m, peer)
    assert out["left"] == 0 and len(peer.calls) > len(flat(m))
    ref = mx.load(str(shard))
    for k, v in flat(m).items():
        assert mx.array_equal(v, ref[k]).item(), k


def test_neighbouring_tensors_go_as_one_request_in_file_order():
    items = [("b", 0, 10), ("a", 100, 5), ("a", 0, 100), ("a", 105, 1 << 40)]
    got = W._groups(items)
    assert [(f, o, n) for f, o, n, _ in got] == [
        ("a", 0, 105), ("a", 105, 1 << 40), ("b", 0, 10)]
    assert got[0][3] == [(2, 0, 100), (1, 100, 5)]


def test_what_is_bound_after_the_weights_comes_from_the_peer(shard, plan):
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    with W.recording():                      # a head bound beside the model
        head = {"w": mx.load(str(shard))["layers.2.weight"]}
    W.fill_bound([head, None])
    assert peer.calls and not W._SOURCES
    assert mx.array_equal(head["w"], mx.load(str(shard))["layers.2.weight"])


class Raising:
    """A transformers-like object whose attributes import (and raise)."""
    __module__ = "transformers.models.auto"

    def __init__(self):
        self.mapping = {"k": 1}

    @property
    def __dict__(self):
        raise ValueError("Could not find a model class")


class Holder:
    def __init__(self, w):
        self.w = w
        self.processor = Raising()


def test_a_walk_skips_what_it_cannot_read(shard, plan):
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    with W.recording():
        h = Holder(mx.load(str(shard))["layers.2.weight"])
    W.fill_bound([h])
    assert peer.calls
    assert mx.array_equal(h.w, mx.load(str(shard))["layers.2.weight"])


def test_a_failing_bound_fill_never_fails_the_load(shard, plan, monkeypatch):
    W.CURRENT["plan"] = W.ReadPlan(read=Peer(shard.parent))
    with W.recording():
        head = {"w": mx.load(str(shard))["layers.2.weight"]}
    monkeypatch.setattr(W, "fill", lambda *a, **k: 1 / 0)
    W.fill_bound([head])                     # logged, not raised
    assert not W._SOURCES


def test_a_failing_fill_leaves_the_share_to_the_mount(monkeypatch, plan):
    log = []
    monkeypatch.setattr(W, "fill", lambda *a, **k: 1 / 0)
    W.CURRENT.update(plan=W.ReadPlan(read=lambda *a: None))
    W.read_share(object(), lambda m: log.append("eval"))
    assert log == ["eval"]


def test_a_family_takes_its_tensors_from_the_peer_before_using_them(
        shard, plan):
    peer = Peer(shard.parent)
    W.CURRENT["plan"] = W.ReadPlan(read=peer)
    with W.recording():
        arrays = mx.load(str(shard))
        got = W.fetch_now({k: arrays[k] for k in ("norm.weight",
                                                  "layers.1.weight")})
    assert peer.calls
    for k, v in got.items():
        assert mx.array_equal(v, mx.load(str(shard))[k]).item()


def test_without_a_plan_a_family_reads_as_before(shard, plan):
    d = {"w": mx.load(str(shard))["norm.weight"]}
    assert W.fetch_now(d) is d
