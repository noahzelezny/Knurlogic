"""engine/split/weights.py: a split rank's share filled from a fake peer
instead of its files (model.parameters(), model.update), a tensor split's
cut redone on the fetched bytes, what the model's sanitize made made again,
the load order over the bell, and the fallback."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from knurlogic.engine.model import weight_files as WF
from knurlogic.engine.split import tensor as T
from knurlogic.engine.split import weights as W


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(8, 16), nn.Linear(16, 8), nn.Linear(8, 8)]
        self.norm = nn.RMSNorm(8)


@pytest.fixture
def art(tmp_path):
    """An artifact folder: one shard of a Tiny with known weights."""
    d = tmp_path / "Tiny"
    d.mkdir()
    m = Tiny()
    m.layers[0].weight = mx.random.normal((16, 8)).astype(mx.bfloat16)
    mx.save_safetensors(str(d / "model.safetensors"),
                        dict(tree_flatten(m.parameters())))
    return d


class Page:
    def __init__(self, root, fail=False):
        self.root, self.fail, self.calls = root, fail, []

    def __call__(self, file, offset, length, into):
        if self.fail:
            raise ConnectionError("the peer went away")
        self.calls.append((file, offset, length))
        with open(self.root / file, "rb") as f:
            f.seek(offset)
            into[:] = f.read(length)


@pytest.fixture
def peer():
    def use(root, **kw):
        page = Page(root, **kw)
        WF.use_peer(WF.Peer(read=page, source="holder", root=root))
        return page
    yield use
    WF.use_peer(None)
    W.CURRENT.update(plan=None, link=None)


def lazy_model(d):
    m = Tiny()
    m.load_weights(list(WF.load(d / "model.safetensors",
                                deferred=True).items()))
    return m


def flat(m):
    return dict(tree_flatten(m.parameters()))


def test_a_pipeline_share_comes_from_the_peer(art, peer):
    page = peer(art)
    m = lazy_model(art)
    m.layers = m.layers[:2]                   # this rank's layers only
    out = W.fill(m)
    assert out["left"] == 0 and page.calls
    ref = mx.load(str(art / "model.safetensors"))
    got = flat(m)
    assert "layers.2.weight" not in got
    for k, v in got.items():
        assert mx.array_equal(v, ref[k]).item(), k
    # only the share's bytes moved
    head = WF.header(art / "model.safetensors")
    assert sum(n for _, _, n in page.calls) == sum(head[k][1] for k in got)


def test_a_tensor_cut_is_rebuilt_from_the_fetched_source(art, peer):
    peer(art)
    m = lazy_model(art)
    w = m.layers[0].weight
    cut = T.split_params({"w": w}, lambda p, a: 0, rank=1, n=2)["w"]
    m.layers[0].weight = cut
    W.fill(m)
    ref = mx.load(str(art / "model.safetensors"))["layers.0.weight"]
    assert mx.array_equal(m.layers[0].weight, ref[8:]).item()


def test_a_parameter_that_is_no_files_tensor_is_left(art, peer):
    peer(art)
    m = lazy_model(art)
    m.norm.weight = m.norm.weight * 2         # made, not read
    out = W.fill(m)
    assert out["left"] == 1


class Fused(nn.Module):
    """A model whose sanitize glues the files' two first layers into one
    (as DeepSeek's fuses wq_a and wkv) and drops the third."""

    def __init__(self, partial_ok=True):
        super().__init__()
        self.fused = nn.Linear(8, 32)
        self.norm = nn.RMSNorm(8)
        self.partial_ok = partial_ok

    def sanitize(self, w):
        if not self.partial_ok and "norm.weight" not in w:
            raise KeyError("norm.weight")
        out = {k: v for k, v in w.items() if not k.startswith("layers.")}
        if "layers.0.weight" in w:
            out["fused.weight"] = mx.concatenate(
                [w["layers.0.weight"].astype(mx.float32),
                 w["layers.1.weight"].T], axis=0)
            out["fused.bias"] = mx.concatenate(
                [w["layers.0.bias"], w["layers.1.bias"],
                 w["layers.1.bias"]], axis=0)
        return out


def fused_model(d, **kw):
    m = Fused(**kw)
    w = WF.sanitize(m, WF.load(d / "model.safetensors", deferred=True))
    m.load_weights(list(w.items()))
    return m


def reference(d):
    m = Fused()
    m.load_weights(list(m.sanitize(
        mx.load(str(d / "model.safetensors"))).items()))
    return flat(m)


def test_what_sanitize_made_is_made_again_from_the_peer(art, peer):
    page = peer(art)
    m = fused_model(art)
    out = W.fill(m)
    assert out["made"] == 2 and out["left"] == 0
    ref = reference(art)
    for k, v in flat(m).items():
        assert mx.array_equal(v, ref[k]).item(), k
    # the norm, and every tensor sanitize consumed -- the dropped third
    # layer too: which inputs feed which output is not knowable
    head = WF.header(art / "model.safetensors")
    assert out["fetched"] == sum(n for _, _, n in page.calls) == sum(
        head[k][1] for k in head if k != "__metadata__")


def test_a_tensor_cut_of_what_sanitize_made_is_redone(art, peer):
    peer(art)
    m = fused_model(art)
    w = m.fused.weight
    m.fused.weight = T.split_params({"w": w}, lambda p, a: 0,
                                    rank=0, n=2)["w"]
    W.fill(m)
    assert mx.array_equal(m.fused.weight,
                          reference(art)["fused.weight"][:16]).item()


def test_nothing_made_needed_nothing_remade(art, peer, monkeypatch):
    peer(art)
    m = fused_model(art)
    m.fused.weight = m.fused.weight * 1      # no longer sanitize's own
    m.fused.bias = m.fused.bias * 1
    monkeypatch.setattr(WF, "remake", lambda **k: pytest.fail("remade"))
    out = W.fill(m)
    assert out["made"] == 0 and out["left"] == 2


def test_a_sanitize_that_needs_the_whole_dict_leaves_it_to_the_files(
        art, peer):
    peer(art)
    m = fused_model(art, partial_ok=False)
    out = W.fill(m)
    assert out["made"] == 0 and out["left"] == 2


def test_a_failing_peer_leaves_the_share_to_the_files(art, peer):
    peer(art, fail=True)
    W.CURRENT["plan"] = W.ReadPlan()
    m = lazy_model(art)
    evaluated = []
    W.read_share(m, lambda mm: evaluated.append(mm) or mx.eval(
        mm.parameters()))
    assert evaluated == [m]
    ref = mx.load(str(art / "model.safetensors"))
    for k, v in flat(m).items():
        assert mx.array_equal(v, ref[k]).item(), k


def test_ctrl_c_stops_the_read(art, peer, monkeypatch):
    peer(art)
    W.CURRENT["plan"] = W.ReadPlan()

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(W, "fill", interrupted)
    with pytest.raises(KeyboardInterrupt):
        W.read_share(lazy_model(art), lambda m: None)


class Bell:
    def __init__(self, log):
        self.log = log

    def align(self):
        self.log.append("align")


@pytest.mark.parametrize("turn,want", [
    (0, ["fill", "eval", "align"]),          # reads first, then lets go
    (1, ["align", "eval"]),                  # waits, then reads its own
])
def test_the_turn_is_kept_over_the_bell(turn, want, monkeypatch, peer,
                                        tmp_path):
    log, marks = [], []
    monkeypatch.setattr(W, "progress", lambda **f: marks.append(f))
    monkeypatch.setattr(W, "fill", lambda *a, **k: log.append("fill") or {
        "fetched": 0, "left": 0, "seconds": 0.0})
    if turn == 0:
        peer(tmp_path)
    W.CURRENT.update(link=Bell(log), plan=W.ReadPlan(turn=turn,
                                                     ordered=True))
    W.read_share(object(), lambda m: log.append("eval"))
    W.done()                                 # after the head and the tower
    assert log == want
    if turn == 1:
        assert marks[0] == {"turn": "waiting"}
    assert WF.peer() is None                 # done stops reading the peer


@pytest.mark.parametrize("turn", [0, 1])
def test_no_order_no_waiting(peer, turn):
    log = []
    W.CURRENT.update(link=Bell(log), plan=W.ReadPlan(turn=turn,
                                                     ordered=False))
    W.read_share(object(), lambda m: log.append("eval"))
    W.done()
    assert log == ["eval"]


def test_without_a_plan_it_reads_as_before():
    log = []
    W.read_share(object(), lambda m: log.append("eval"))
    assert log == ["eval"]


def test_the_plan_comes_from_the_environment():
    plan, page, source = W.plan_from_env(
        {"KNURLOGIC_WEIGHTS": '{"from": "198.51.100.1:8899", "source": '
                              '"holder", "turn": 0, "ordered": true}'})
    assert (page, source, plan.turn, plan.ordered) == (
        "198.51.100.1:8899", "holder", 0, True)
    assert W.plan_from_env({}) == (None, "", "")
    own, page, _ = W.plan_from_env(
        {"KNURLOGIC_WEIGHTS": '{"turn": 1, "ordered": true}'})
    assert page == "" and own.turn == 1
