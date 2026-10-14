"""The prompt cache under a tensor split (phase 3 of
docs/design/prompt-cache-shards.md): each rank saves its head range of
every layer (`L{i:03d}.h{a}-{b}.safetensors`), a replicated cache whole
once (rank 0); any split reads what it needs -- one exact slice, slices
joined, or a whole layer cut -- bit for bit. A range no file covers is a
miss on every rank; an old per-split tensor entry still restores. Tiny
fixtures in one process (each rank's share cut from the whole model's
cache, as engine/split/tensor.shard leaves it); the two-process ring is
test_a_tensor_ring_saves_and_restores_its_slices."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_prompt_cache_layers import LINK, _gather_as, _key, _lru_of  # noqa: E402
from test_prompt_cache_transfer import _Peer  # noqa: E402
from test_prompt_disk import _family, _forced  # noqa: E402

from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import layers as LY  # noqa: E402
from knurlogic.engine.prompt_cache import slices as SL  # noqa: E402
from knurlogic.engine.prompt_cache import transfer as TR  # noqa: E402
from knurlogic.engine.split import tensor_rules as TRL  # noqa: E402


def _host(model, layout):
    return NS(model=model, cache_layout=layout)


def _tensor_parts(model, n):
    """Each rank's Part of a tensor split `n` ways (the cut tensor.shard
    records before it splits)."""
    object.__setattr__(model, "cache_cuts", TRL.cache_cuts(model))
    return [LY.part_of(_host(model, {"split": "tensor", "world": n,
                                     "rank": r})) for r in range(n)]


def _layout_parts(model, how):
    """"single", "tensor<n>" or a pipeline's layer counts (rank order)."""
    if how == "single":
        return [LY.part_of(_host(model, {"split": "none"}))]
    if isinstance(how, str) and how.startswith("tensor"):
        return _tensor_parts(model, int(how[6:]))
    from knurlogic.engine.split.pipeline import bounds_of
    b = bounds_of(how)
    cuts = TRL.cache_cuts(model)
    total = max(e for _, e in b)
    return [LY.Part(s, e - s, total, r == 0, r, len(b),
                    cuts=tuple(cuts[s:e]) if cuts else ())
            for r, (s, e) in enumerate(b)]


def _share(p: LY.Part, cache: list) -> list:
    """What rank `p` holds of a whole model's cache list."""
    out = []
    for i in p.slots:
        c, cut, rng = cache[i], p.cut_of(i), p.heads_of(i)
        out.append(c if cut is None or rng is None
                   else SL.cut(c, cut, *rng))
    return out


def _save(monkeypatch, roots, key, parts, tokens, cache):
    """Every rank saves its share (rank r into roots[r], or one root for
    all), then the commit's vote."""
    stores = [LY.Store(roots[p.rank % len(roots)], key, p) for p in parts]
    entries = None
    for st in stores:
        got = st.save(_lru_of(tokens, _share(st.part, cache)),
                      commit=len(parts) == 1, sweep_after=False)
        if st.part.rank == 0:
            entries = got["pending"]
    if len(parts) > 1:
        _gather_as(monkeypatch, [[1 if st.ready(e) else 0 for e in entries]
                                 for st in stores])
        for st in stores:
            assert LY.commit(LINK, st, entries) == \
                [e["tokens_hash"] for e in entries]
    return stores


def _restore(monkeypatch, roots, key, parts, peers=None):
    gots = []
    for p in parts:
        st = LY.Store(roots[p.rank % len(roots)], key, p)
        if peers is not None:
            monkeypatch.setattr(TR, "_FETCHER", [peers[p.rank]])
        gots.append(LY.restore(8, st, sweep_first=False))
    monkeypatch.setattr(TR, "_FETCHER", [None])
    if len(parts) > 1:
        _gather_as(monkeypatch, [LY.vote_row(g) for g in gots])
        gots = [LY.agree(LINK, g) for g in gots]
    return gots


def _whole(parts, caches: list) -> list:
    """The whole model's cache list from every rank's restored share: a
    pipeline's stages in order, a tensor's slices joined."""
    if parts[0].heads:
        out = []
        for i in parts[0].slots:
            cut = parts[0].cut_of(i)
            if cut is None:
                out.append(caches[0][i])
                continue
            seen, mine = set(), []
            for p, cs in zip(parts, caches):
                rng = p.heads_of(i)
                if rng not in seen:
                    seen.add(rng)
                    mine.append(cs[i])
            out.append(SL.join(mine, cut))
        return out
    out = []
    for p, cs in sorted(zip(parts, caches), key=lambda x: x[0].start):
        out += list(cs)[:p.n]
    return out


def _arrays_equal(a, b) -> bool:
    xs, ys = list(D._arrays_of(a)), list(D._arrays_of(b))
    return len(xs) == len(ys) and all(
        x.shape == y.shape and x.dtype == y.dtype and mx.array_equal(x, y)
        for x, y in zip(xs, ys))


def _entry(family, bits=None, seed=3):
    model = _family(family, bits)
    mx.random.seed(seed)
    prompt = (mx.random.randint(0, 60, (40,)) % 60).tolist()
    cache, cont = _forced(model, prompt, [7, 3, 11, 5])
    return model, prompt, cache, copy.deepcopy(cache), cont


@pytest.mark.parametrize("family,bits", [
    ("qwen3_5_moe", None),    # K/V by kv heads, deltanet by key heads
    ("qwen3_5_moe", 8),       # quantized K/V: each array by kv heads
    ("qwen4_exp", None),      # its indexer keys and PLE slots replicated
    ("deepseek_v4", None),    # every cache replicated: whole, rank 0
])
@pytest.mark.parametrize("save,load", [
    ("single", "tensor2"),
    ("tensor2", "single"),
    ("tensor2", [1, 3]),      # tensor -> pipeline
    ([1, 3], "tensor2"),      # pipeline -> tensor
])
def test_an_entry_crosses_between_tensor_and_other_splits(
        tmp_path, monkeypatch, family, bits, save, load):
    """Saved under one split, restored under another: each rank's share
    is exactly the original's cut to its range, and the next tokens'
    logits from the rejoined cache equal the original's, bit for bit."""
    model, prompt, cache, keep, cont = _entry(family, bits)
    key = _key()
    sp = _layout_parts(model, save)
    _save(monkeypatch, [tmp_path], key, sp, prompt, cache)
    d = tmp_path / D.key_id(key) / D.tokens_hash(prompt)
    names = sorted(f.name for f in d.iterdir())
    meta = json.loads((d / LY.ENTRY).read_text())
    if save == "tensor2" and family != "deepseek_v4":
        assert "L000.h0-1.safetensors" in names and \
            "L000.h1-2.safetensors" in names
        assert meta["stored"]["0"] == [[0, 1], [1, 2]]
    else:
        assert names == [f"L00{i}.safetensors" for i in range(4)] + \
            ["entry.json"]
        assert meta["stored"] == {}
    lp = _layout_parts(model, load)
    gots = _restore(monkeypatch, [tmp_path], key, lp)
    assert all(len(g) == 1 and g[0][1] == prompt for g in gots)
    shares = [g[0][2] for g in gots]
    for p, mine in zip(lp, shares):
        if p.heads:
            want = _share(p, keep)
            assert all(_arrays_equal(a, b) for a, b in zip(mine, want))
    back = _whole(lp, shares)
    assert [type(c) for c in back] == [type(c) for c in keep]
    assert mx.array_equal(cont(keep), cont(back))


def test_rank_0_alone_owns_the_drafting_heads_cache():
    # rank 0 alone loads and runs the head, whole (a follower never loads
    # it), so it writes and reads draft.safetensors; without it a restored
    # entry has no aligned head cache and a drafting row re-prefills
    model, _, _, _, _ = _entry("qwen3_5_moe")
    parts = _tensor_parts(model, 2)
    assert [p.extras for p in parts] == [True, False]


def test_a_replicated_cache_is_written_once_by_rank_0(tmp_path, monkeypatch):
    """deepseek_v4 keeps its caches whole on every rank: rank 0 writes the
    whole layers, rank 1 nothing, and both read them."""
    model, prompt, cache, keep, _ = _entry("deepseek_v4")
    parts = _tensor_parts(model, 2)
    assert parts[0].writes == ("whole",) * 4 and parts[1].writes == ("",) * 4
    roots = [tmp_path / "a", tmp_path / "b"]
    _save(monkeypatch, roots, _key(), parts, prompt, cache)
    h = D.tokens_hash(prompt)
    held = [sorted(f.name for f in (r / D.key_id(_key()) / h).iterdir())
            for r in roots]
    assert held == [[f"L00{i}.safetensors" for i in range(4)] +
                    ["entry.json"], ["entry.json"]]
    # rank 1 on its own machine fetches them from rank 0's page
    gots = _restore(monkeypatch, roots, _key(), parts,
                    peers=[_Peer([roots[1]]), _Peer([roots[0]])])
    assert all(len(g) == 1 for g in gots)
    assert all(_arrays_equal(a, b) for a, b in zip(gots[1][0][2], keep))


def test_slices_on_separate_machines_are_fetched_from_the_peers(
        tmp_path, monkeypatch):
    """Saved by a tensor split whose ranks keep their own roots: one
    machine alone restores the whole entry, both slices of each layer
    fetched from the other machine's page and joined, bit for bit."""
    model, prompt, cache, keep, cont = _entry("qwen3_5_moe")
    roots = [tmp_path / "a", tmp_path / "b"]
    _save(monkeypatch, roots, _key(), _tensor_parts(model, 2), prompt, cache)
    one = _layout_parts(model, "single")
    peer = _Peer([roots[1]])
    gots = _restore(monkeypatch, roots[:1], _key(), one, peers=[peer])
    assert len(gots[0]) == 1
    assert ("get", "0.h1-2") in peer.calls
    assert mx.array_equal(cont(keep), cont(list(gots[0][0][2])))


@pytest.mark.parametrize("load", ["tensor2", "single"])
def test_a_slice_missing_is_a_miss_on_every_rank(tmp_path, monkeypatch,
                                                 load):
    """A head range no file covers (rank 1's slice of layer 1 gone): the
    entry is a miss on every rank (and on one machine); the other entry
    restores."""
    model, prompt, cache, _, _ = _entry("qwen3_5_moe")
    key = _key()
    tp = _tensor_parts(model, 2)
    _save(monkeypatch, [tmp_path], key, tp, prompt, cache)
    other = [9] + prompt[1:]
    _save(monkeypatch, [tmp_path], key, tp, other, cache)
    d = tmp_path / D.key_id(key) / D.tokens_hash(prompt)
    (d / SL.slice_name(1, 1, 2)).unlink()
    lp = _layout_parts(model, load)
    gots = _restore(monkeypatch, [tmp_path], key, lp)
    assert [[g[1] for g in got] for got in gots] == [[other]] * len(lp)


def test_a_reader_cuts_a_whole_layer_or_joins_smaller_slices():
    """The plan: the whole layer when there; else a chain of slices from
    the range's start, each reaching furthest; None when a gap remains."""
    spec = ("kv", 4, (("keys", 1, (1,)), ("values", 1, (1,))))
    assert SL.plan({None, (0, 2)}, spec, (2, 4)) == [None]
    assert SL.plan({(0, 2), (2, 4)}, spec, (2, 4)) == [(2, 4)]
    assert SL.plan({(0, 1), (1, 2), (0, 4)}, spec, (1, 3)) == [(0, 4)]
    assert SL.plan({(0, 1), (1, 2), (2, 4)}, spec, None) == \
        [(0, 1), (1, 2), (2, 4)]
    assert SL.plan({(0, 1), (2, 4)}, spec, (0, 4)) is None
    assert SL.plan({(0, 2)}, None, None) is None


def test_fewer_kv_heads_than_ranks_one_holder_writes_each(tmp_path,
                                                         monkeypatch):
    """Two kv heads on four ranks: ranks 0 and 2 write h0-1 and h1-2, ranks
    1 and 3 hold copies and write nothing; every rank reads its head."""
    from mlx_lm.models.cache import KVCache
    assert [TRL.head_range(2, 4, r, True) for r in range(4)] == \
        [(0, 1, True), (0, 1, False), (1, 2, True), (1, 2, False)]
    assert TRL.head_range(2, 4, 0) is None          # deltanet: no copies
    spec = ("kv", 2, (("keys", 1, (1,)), ("values", 1, (1,))))
    whole = KVCache()
    k = mx.random.normal((1, 2, 6, 8))
    whole.update_and_fetch(k, k + 1)
    parts = []
    for r in range(4):
        a, b, w = TRL.head_range(2, 4, r, True)
        parts.append(LY.Part(0, 1, 1, False, r, 4, (spec,), ((a, b),),
                             ("slice" if w else "",)))
    tokens = list(range(6))
    _save(monkeypatch, [tmp_path], _key(), parts, tokens, [whole])
    d = tmp_path / D.key_id(_key()) / D.tokens_hash(tokens)
    assert sorted(f.name for f in d.iterdir()) == \
        ["L000.h0-1.safetensors", "L000.h1-2.safetensors", "entry.json"]
    gots = _restore(monkeypatch, [tmp_path], _key(), parts)
    for p, g in zip(parts, gots):
        assert _arrays_equal(g[0][2][0], SL.cut(whole, spec, *p.heads[0]))


def test_a_family_without_a_spec_keeps_the_per_split_format():
    """No cache spec (or units that do not split): part_of is None and the
    per-split format is used; uses_layers says so on every rank."""
    model = NS(layers=[], make_cache=lambda: [])
    host = _host(model, {"split": "tensor", "world": 2, "rank": 0})
    assert TRL.cache_cuts(NS(model_type="gemma4", layers=[])) is None
    assert not LY.uses_layers(host)
    model2 = _family("qwen3_5_moe", None)
    object.__setattr__(model2, "cache_cuts", TRL.cache_cuts(model2))
    h3 = _host(model2, {"split": "tensor", "world": 3, "rank": 0})
    assert LY.uses_layers(h3) and LY.part_of(h3) is None


def test_an_old_per_split_tensor_entry_still_restores(tmp_path):
    """What a tensor rank wrote before phase 3 (disk.py's format, keyed by
    its layout) is still read under that exact key."""
    from mlx_lm.models.cache import KVCache
    c = KVCache()
    k = mx.random.normal((1, 1, 5, 8))
    c.update_and_fetch(k, k)
    tokens = [1, 2, 3, 4, 5]
    layout = {"split": "tensor", "world": 2, "rank": 1}
    key = D.identity("/nonexistent/tiny", layout=layout)
    got = D.save(_lru_of(tokens, [c]), key, tmp_path, owners=None,
                 sweep_after=False)
    assert got["saved"] == 1
    back = D.read(key, D.candidates(key, 8, base=tmp_path))
    assert len(back) == 1 and back[0][1] == tokens
    assert _arrays_equal(back[0][2][0], c)


@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen4_exp",
                                    "deepseek_v4"])
def test_a_tensor_ring_saves_and_restores_its_slices(tmp_path, family):
    """Two processes on 127.0.0.1, the tiny model split in two: each rank
    saves its slices and the commit makes the entry whole; restored over
    the ring's vote, a rank's continuation equals its own cache's, bit for
    bit. The whole model's entry cut to each rank's range, and the split's
    entry joined on one machine, continue as the original does (to the
    split's float32 rounding: the ranks compute the K/V in another
    order)."""
    import os
    import socket
    import subprocess

    import numpy as np

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{free_port()}"],
                                 [f"127.0.0.1:{free_port()}"]]))
    here = Path(__file__).resolve().parents[1] / "support"
    out = tmp_path / "res.json"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               KNURLOGIC_PROMPT_CACHE_DISK="on",
               PYTHONPATH=os.pathsep.join(
                   [str(here.parents[1] / "src"), str(here)] + sys.path))
    procs = [subprocess.Popen(
        [sys.executable, str(here / "tensor_ring_worker.py"), str(out),
         family, "cache"],
        env=dict(env, MLX_RANK=str(r)), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT) for r in range(2)]
    try:
        logs = [p.communicate(timeout=120)[0].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert all(p.returncode == 0 for p in procs), "\n".join(logs)
    rs = [json.loads(Path(f"{out}.{r}").read_text()) for r in range(2)]
    for r in rs:
        assert np.array_equal(np.array(r["split_own"]),
                              np.array(r["split_back"]))
        assert np.abs(np.array(r["split_own"]) -
                      np.array(r["split_from_whole"])).max() < 1e-3
    w = np.array(rs[0]["whole_own"])
    assert np.abs(w - np.array(rs[0]["whole_from_split"])).max() < 1e-3
    if family == "deepseek_v4":
        assert "L000.safetensors" in rs[0]["files"]
    else:
        assert {"L000.h0-1.safetensors", "L000.h1-2.safetensors"} <= \
            set(rs[0]["files"])
