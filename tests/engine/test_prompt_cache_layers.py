"""The per-layer prompt cache (engine/prompt_cache/layers.py, location.py):
an entry saved under one split restores under another, bit for bit;
incomplete entries are a miss on every rank; the old per-split files still
restore; where the cache lives; drop, sweep, pins and TTL on entry
directories. Tiny fixtures only; the two-process ring is in
test_pipeline.py."""
import copy
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_prompt_disk import Host, _forced  # noqa: E402
from test_scheduler import Tok, _collect, _job  # noqa: E402

from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import layers as LY  # noqa: E402
from knurlogic.engine.prompt_cache import location as LOC  # noqa: E402


def _key(**kw):
    return LY.shard_key("/nonexistent/tiny", **kw)


def _parts(counts):
    """Each rank's Part of a pipeline with `counts` layers per rank (rank
    order; rank 0 holds the last layers), or a whole model for one count."""
    from knurlogic.engine.split.pipeline import bounds_of
    b = bounds_of(counts)
    total = max(e for _, e in b)
    return [LY.Part(s, e - s, total, r == 0, r, len(b))
            for r, (s, e) in enumerate(b)]


def _lru_of(tokens, cache, owner=None):
    from mlx_lm.models.cache import LRUPromptCache
    lru = LRUPromptCache(max_size=8)
    lru.insert_cache("m", list(tokens), cache, cache_type="user")
    return lru


def _gather_as(monkeypatch, rows):
    """Every rank's row of the next collective, as if gathered."""
    monkeypatch.setattr(LY, "_gather", lambda link, row: rows)


def _save_on(monkeypatch, base, key, parts, tokens, cache, owners=None):
    """Save `cache` (a whole model's, after a prefill) as the ranks of
    `parts` would: each its own slots, then the commit's vote."""
    stores = [LY.Store(base, key, p) for p in parts]
    entries = None
    for st in stores:
        p = st.part
        mine = list(cache[p.start:p.start + p.n]) + \
            (list(cache[p.total:]) if p.extras else [])
        got = st.save(_lru_of(tokens, mine), owners=owners,
                      commit=len(parts) == 1, sweep_after=False)
        if p.rank == 0:
            entries = got["pending"]
    if len(parts) > 1:
        _gather_as(monkeypatch, [[1 if st.ready(e) else 0 for e in entries]
                                 for st in stores])
        for st in stores:
            assert LY.commit(object(), st, entries) == \
                [e["tokens_hash"] for e in entries]
    return stores


def _restore_on(monkeypatch, base, key, parts):
    """Each rank's read and the vote: the caches every rank restored,
    assembled in global slot order (the whole model's list)."""
    stores = [LY.Store(base, key, p) for p in parts]
    gots = [st.read(st.candidates(8)) for st in stores]
    if len(parts) > 1:
        _gather_as(monkeypatch, [LY.vote_row(g) for g in gots])
        gots = [LY.agree(object(), g) for g in gots]
    return stores, gots


def _whole(gots, parts):
    """One entry's caches from every rank, in global layer order."""
    out = []
    for p, g in sorted(zip(parts, gots), key=lambda x: x[0].start):
        out += list(g[0][2])[:p.n]
    return out


def _model(family):
    import pipeline_ring_worker as W
    return W.build(family)


@pytest.mark.parametrize("family", ["qwen3_5_moe", "glm5_next",
                                    "deepseek_v4"])
@pytest.mark.parametrize("save,load", [
    ([4], [2, 2]),            # one machine -> a pipeline
    ([1, 3], [4]),            # a pipeline -> one machine
    ([1, 3], [3, 1]),         # one cut -> another
])
def test_an_entry_saved_under_one_split_restores_under_another(
        tmp_path, monkeypatch, family, save, load):
    """The next tokens' logits from the restored cache equal the
    original's, bit for bit, whichever ranks wrote and read the layers."""
    model = _model(family)
    mx.random.seed(3)
    prompt = (mx.random.randint(0, 60, (40,)) % 60).tolist()
    toks = [7, 3, 11, 5]
    cache, cont = _forced(model, prompt, toks)
    keep = copy.deepcopy(cache)
    key = _key()
    _save_on(monkeypatch, tmp_path, key, _parts(save), prompt, cache)
    d = tmp_path / D.key_id(key) / D.tokens_hash(prompt)
    names = sorted(f.name for f in d.iterdir())
    assert names == ["L000.safetensors", "L001.safetensors",
                     "L002.safetensors", "L003.safetensors", "entry.json"]
    meta = json.loads((d / "entry.json").read_text())
    assert meta["tokens"] == prompt and meta["layers"] == [0, 1, 2, 3]
    assert "layout" not in meta["key"] and "knurlogic" not in meta["key"]
    parts = _parts(load)
    _, gots = _restore_on(monkeypatch, tmp_path, key, parts)
    assert all(len(g) == 1 and g[0][1] == prompt for g in gots)
    back = _whole(gots, parts)
    assert [type(c) for c in back] == [type(c) for c in keep]
    assert mx.array_equal(cont(keep), cont(back))


def test_a_drafting_heads_cache_goes_to_rank_0_only(tmp_path, monkeypatch):
    """The entry's tail past the layers (a drafting head's cache) is
    draft.safetensors: written by the rank holding the head (rank 0 or a
    single machine), read back only by a rank running the same head."""
    from mlx_lm.models.cache import KVCache

    from knurlogic.engine.families.deepseek.heads.deepseek_v4_dspark import (
        DSparkCache,
    )
    caches = []
    for _ in range(3):
        c = KVCache()
        k = mx.random.normal((1, 2, 6, 32))
        c.update_and_fetch(k, k)
        caches.append(c)
    h = DSparkCache(2, 4)
    h.append([mx.random.normal((1, 6, 16)), mx.random.normal((1, 6, 16))], 6)
    head = {"head": "x:Head", "block": 4}
    tokens = list(range(6))
    st = LY.Store(tmp_path, _key(), LY.Part(0, 3, 3), draft=head)
    st.save(_lru_of(tokens, caches + [h]), sweep_after=False)
    d = st.dir / D.tokens_hash(tokens)
    assert (d / LY.DRAFT).is_file()
    (_, _, got, *_), = st.read([d])
    assert len(got) == 4 and type(got[3]) is DSparkCache
    assert all(mx.array_equal(a, b) for a, b in zip(h.keys, got[3].keys))
    # a pipeline: rank 0 (layers 2..3) gets the head's cache, rank 1 not
    r0 = LY.Store(tmp_path, _key(), LY.Part(2, 1, 3, True, 0, 2), draft=head)
    r1 = LY.Store(tmp_path, _key(), LY.Part(0, 2, 3, False, 1, 2))
    assert len(r0.read([d])[0][2]) == 2 and len(r1.read([d])[0][2]) == 2
    # another head (or none): the layers, without it
    other = LY.Store(tmp_path, _key(), LY.Part(0, 3, 3), draft=None)
    assert len(other.read([d])[0][2]) == 3


# --- incomplete entries -------------------------------------------------------

def _tiny_entry(base, key, parts, monkeypatch, n=2, length=12):
    from mlx_lm.models.cache import KVCache
    out = []
    for i in range(n):
        caches = []
        for _ in range(parts[0].total):
            c = KVCache()
            k = mx.random.normal((1, 2, length, 32))
            c.update_and_fetch(k, k)
            caches.append(c)
        toks = [200 + i] + list(range(length - 1))
        _save_on(monkeypatch, base, key, parts, toks, caches)
        out.append(toks)
    return out


@pytest.mark.parametrize("missing", ["L000.safetensors", "entry.json"])
def test_an_incomplete_entry_is_a_miss_on_every_rank(tmp_path, monkeypatch,
                                                     missing):
    """A layer file one rank needs is gone (or entry.json was never
    written): that rank cannot read the entry, and the vote leaves it out
    on every rank -- the other entry restores on both."""
    key = _key()
    a, b = _tiny_entry(tmp_path, key, _parts([2, 2]), monkeypatch)
    (tmp_path / D.key_id(key) / D.tokens_hash(a) / missing).unlink()
    parts = _parts([2, 2])
    _, gots = _restore_on(monkeypatch, tmp_path, key, parts)
    assert [[g[1] for g in got] for got in gots] == [[b], [b]]


def test_a_commit_a_rank_did_not_write_makes_no_entry(tmp_path, monkeypatch):
    """The commit's vote: a rank whose part is missing says so; no rank
    writes entry.json for that entry, so it never appears half."""
    from mlx_lm.models.cache import KVCache
    key = _key()
    parts = _parts([2, 2])
    caches = []
    for _ in range(4):
        c = KVCache()
        k = mx.random.normal((1, 2, 5, 32))
        c.update_and_fetch(k, k)
        caches.append(c)
    toks = [1, 2, 3, 4, 5]
    stores = [LY.Store(tmp_path, key, p) for p in parts]
    got = stores[0].save(_lru_of(toks, caches[2:]), commit=False,
                         sweep_after=False)
    entries = got["pending"]
    # rank 1 never saved its layers
    _gather_as(monkeypatch, [[1 if s.ready(e) else 0 for e in entries]
                             for s in stores])
    for s in stores:
        assert LY.commit(object(), s, entries) == []
    d = tmp_path / D.key_id(key) / D.tokens_hash(toks)
    assert not (d / LY.ENTRY).exists() and not stores[0].pending
    assert stores[0].entries() == []


def test_the_old_per_split_files_still_restore_at_load(tmp_path):
    """An entry an earlier build wrote (disk.py's format, under its exact
    key) restores at load beside the new format; a new save writes only
    the new one."""
    from test_batch_drafting import _tiny

    from knurlogic.engine.runtime.scheduler import Scheduler
    model, _head, prompts = _tiny(512)
    host = Host(model, Tok(prompts))
    s = Scheduler(host, prefill_step_size=16).start()
    path = "/nonexistent/tiny-old"
    s.load(path).done.wait(60)
    p = prompts[2]
    _collect(s.submit(_job(p, max_tokens=4, session="old")))
    # what an earlier build's save wrote: the per-split file
    got = D.save(s.cache.lru, D.host_key(host), owners=s.cache.owners)
    assert got["saved"] >= 1
    s.unload().done.wait(60)
    s.load(path).done.wait(60)
    assert s._restored and all(Path(v["file"]).is_file()
                               for v in s._restored.values())
    _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4)))
    assert warm["knurlogic"]["cache"]["disk"]["tokens"] >= len(p) - 1
    # a new save: the per-layer format only
    olds = len(D._all_files(D.root()))
    assert s.save_prompt_cache().done.wait(60)
    assert len(D._all_files(D.root())) == olds
    assert LY.units(D.root())


# --- where it lives ------------------------------------------------------------

def test_the_cache_lives_beside_the_models(tmp_path):
    models = tmp_path / "Models"
    (models / "tiny-4bit").mkdir(parents=True)
    root, kind = LOC.root_for(models / "tiny-4bit")
    assert (root, kind) == (models / "Prompt cache", "models")
    assert "safe to delete" in (root / LOC.README).read_text()
    assert root in LOC.known()
    # a Hugging Face cache: the hub folder, not a snapshot
    snap = tmp_path / "hub" / "models--org--tiny" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert LOC.models_folder(snap) == tmp_path / "hub"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root writes into a read-only folder")
def test_a_read_only_models_folder_falls_back(tmp_path):
    models = tmp_path / "ro"
    (models / "tiny").mkdir(parents=True)
    os.chmod(models, 0o555)
    try:
        root, kind = LOC.root_for(models / "tiny")
    finally:
        os.chmod(models, 0o755)
    assert (root, kind) == (LOC.fallback(), "fallback")
    assert not (models / "Prompt cache").exists()
    assert LOC.root_for("/nonexistent/tiny")[1] == "fallback"


def test_a_scheduler_saves_beside_the_models_and_drops_there(tmp_path):
    """Through the scheduler: a save lands in <models>/Prompt cache/, the
    registry's disk side says where, a drop by session removes the entry
    directory, and the next load restores what is left."""
    from test_batch_drafting import _tiny

    from knurlogic.engine.runtime.scheduler import Scheduler
    model, _head, prompts = _tiny(512)
    path = tmp_path / "Models" / "tiny"
    path.mkdir(parents=True)
    s = Scheduler(Host(model, Tok(prompts)), prefill_step_size=16).start()
    s.load(str(path)).done.wait(60)
    _collect(s.submit(_job(prompts[0], max_tokens=3, session="a")))
    _collect(s.submit(_job(prompts[2], max_tokens=3, session="b")))
    cmd = s.save_prompt_cache()
    assert cmd.done.wait(60) and not cmd.error
    assert cmd.result["location"] == str(tmp_path / "Models" / "Prompt cache")
    rows = [r for r in D.list_disk() if r["model"] == "tiny"]
    assert {r["session"] for r in rows} == {"a", "b"}
    assert all(r["location"] == cmd.result["location"] for r in rows)
    listing = s.list_prompt_cache()
    assert listing.done.wait(60)
    assert listing.result["location"] == cmd.result["location"]
    drop = s.drop_prompt_cache("a")
    assert drop.done.wait(60) and drop.result["disk"] >= 1
    left = [r for r in D.list_disk() if r["model"] == "tiny"]
    assert {r["session"] for r in left} == {"b"}
    assert all(Path(r["file"]).is_dir() for r in left)
    s.unload().done.wait(60)
    s.load(str(path)).done.wait(60)
    assert {(v["owner"] or {}).get("session")
            for v in s._restored.values()} == {"b"}


# --- sweep, pins, TTL ---------------------------------------------------------

def test_ttl_budget_and_pins_on_entry_directories(tmp_path, monkeypatch):
    key = _key()
    owners = {}
    parts = _parts([3])
    from mlx_lm.models.cache import KVCache
    dirs = []
    for i, sess in enumerate(["p", "u1", "u2"]):
        caches = []
        for _ in range(3):
            c = KVCache()
            k = mx.random.normal((1, 2, 10, 32))
            c.update_and_fetch(k, k)
            caches.append(c)
        toks = [300 + i] + list(range(9))
        owners[tuple(toks)] = {"session": sess, "role": None, "run": None}
        _save_on(monkeypatch, tmp_path, key, parts, toks, caches,
                 owners=owners)
        d = tmp_path / D.key_id(key) / D.tokens_hash(toks)
        os.utime(d / LY.ENTRY, (time.time() - 100 + i,) * 2)
        dirs.append(d)
    D.set_pin("p", True, tmp_path)
    size = LY.units(tmp_path)[0][1]
    # the budget: the least recently used unpinned entry goes first
    monkeypatch.setenv(D.ENV_GB, str(1.5 * size / D.GIB))
    out = D.sweep(tmp_path)
    assert out["budget"] == 1 and out["pinned"] == 1
    assert dirs[0].exists() and not dirs[1].exists() and dirs[2].exists()
    # the TTL: everything unpinned, never the pinned one
    monkeypatch.delenv(D.ENV_GB)
    monkeypatch.setenv(D.ENV_TTL, "1")
    assert D.sweep(tmp_path, now=time.time() + 7200)["ttl"] == 1
    assert dirs[0].exists() and not dirs[2].exists()
    # only a drop removes a pinned session's entry
    assert D.drop_files("p", tmp_path) == 1
    assert not dirs[0].exists() and "p" not in D.pins(tmp_path)


def test_a_stale_incomplete_entry_is_swept(tmp_path, monkeypatch):
    key = _key()
    (toks,) = _tiny_entry(tmp_path, key, _parts([2, 2]), monkeypatch, n=1)
    d = tmp_path / D.key_id(key) / D.tokens_hash(toks)
    (d / LY.ENTRY).unlink()
    D.sweep(tmp_path)
    assert d.exists()                      # a writer may still be at it
    old = time.time() - 2 * D._TMP_STALE_S
    for f in d.iterdir():
        os.utime(f, (old, old))
    D.sweep(tmp_path)
    assert not d.exists()


def test_a_follower_saves_commits_reads_back_and_drops_by_name(
        tmp_path, monkeypatch):
    """Rank 0 and a follower, each with its own folder (no shared storage):
    rank 0 saves its layers and journals the commit after the save op; the
    follower applies save_cache (its layers), then commit_entries (the
    vote: each folder gets the same entry.json), then read_back of its own
    layers by the entry's name, then drop_files by that name."""
    from mlx_lm.models.cache import KVCache

    from knurlogic.engine.prompt_cache.memory import PromptCache
    from knurlogic.engine.prompt_cache.ring import apply_cache_op
    key = _key()
    p0, p1 = _parts([2, 2])
    r0 = LY.Store(tmp_path / "a", key, p0)
    r1 = LY.Store(tmp_path / "b", key, p1)
    toks = list(range(9))
    caches = []
    for _ in range(4):
        c = KVCache()
        k = mx.random.normal((1, 2, 9, 32))
        c.update_and_fetch(k, k)
        caches.append(c)
    own = {"session": "s", "role": None, "run": None}
    pc0, pc1 = PromptCache(4), PromptCache(4)
    pc0.insert("m", toks, caches[2:], "user", owner=dict(own))
    pc1.insert("m", toks, caches[:2], "user", owner=dict(own))
    got = r0.save(pc0.lru, owners=pc0.owners, commit=False,
                  sweep_after=False)
    r0.outbox += got.pop("pending")
    op = {"op": "commit_entries", "entries": r0.outbox}

    def save_disk(only_new=False, select=None):
        r1.save(pc1.lru, owners=pc1.owners, shared=pc1.shared,
                only_new=only_new, select=select, commit=False,
                sweep_after=False)
    assert apply_cache_op({"op": "save_cache"}, pc1, "m", {}, save_disk,
                          store=r1)
    rows = [[int(r0.ready(e)) for e in op["entries"]],
            [int(r1.ready(e)) for e in op["entries"]]]
    assert rows == [[1], [1]]
    _gather_as(monkeypatch, rows)
    assert apply_cache_op(op, pc1, "m", {}, save_disk, store=r1,
                          link=object())
    LY.commit(object(), r0, op["entries"])
    h = D.tokens_hash(toks)
    a, b = r0.dir / h / LY.ENTRY, r1.dir / h / LY.ENTRY
    assert a.read_bytes() == b.read_bytes()
    assert sorted(f.name for f in (r1.dir / h).iterdir()) == \
        ["L000.safetensors", "L001.safetensors", "entry.json"]
    # read back by name: the follower's own two layers
    pc1.remove("m", toks)
    assert apply_cache_op({"op": "read_back", "name": h}, pc1, "m", {},
                          save_disk, store=r1)
    assert [tuple(t) for _, t in pc1.of_session("s")] == [tuple(toks)]
    assert apply_cache_op({"op": "drop_files", "names": [h]}, pc1, "m", {},
                          save_disk, store=r1)
    assert not (r1.dir / h).exists() and (r0.dir / h).exists()


def test_the_commit_op_is_in_the_plan_schema():
    from knurlogic.engine.split import plan as P
    ok = {"ops": [{"op": "commit_entries",
                   "entries": [{"tokens_hash": "a" * 24, "gen": 1}]}]}
    P.check(ok)
    with pytest.raises(P.PlanError):
        P.check({"ops": [{"op": "commit_entries",
                          "entries": [{"tokens_hash": "../x"}]}]})


def test_rounding_on_is_one_key_on_any_machines():
    # a single Mac and a two-Mac split with rounding off read the same
    # entries; the split's descriptor lists its chips even when off
    one = {"on": False, "setting": "off", "chips": [], "why": "off",
           "label": ""}
    two = {"on": False, "setting": "off", "chips": ["chip A", "chip B"],
           "why": "off", "label": "chip A + chip B"}
    assert LY.rounding_of(one) == LY.rounding_of(two) == "off"
    on_ab = dict(two, on=True, setting="auto")
    on_ba = dict(on_ab, chips=["chip B", "chip A"])
    assert LY.rounding_of(on_ab) == LY.rounding_of(on_ba) != "off"


# the cache key's own field for per-chip rounding (KNURLOGIC_PER_CHIP_ROUNDING)
# keeps its name and values, so these ids -- the ones saved entries carry,
# computed before the setting was renamed -- must hold
_M3 = {"name": "M3 Ultra", "arch": "applegpu_g15d"}
_M4 = {"name": "M4 Max", "arch": "applegpu_g16s"}


@pytest.mark.parametrize("setting,chips,field,kid", [
    # the default: per-chip rounding on
    ("on", None, "off", "25e9a271a6701d86"),
    # rounding off on one machine
    ("off", None, repr({"setting": "on", "chips": []}), "45d63a889b85a83e"),
    # rounding off across chips (an M3 and an M4)
    ("off", [_M3, _M4],
     repr({"setting": "on", "chips": ["M3 Ultra", "M4 Max"]}),
     "8a2d7e49284719da"),
])
def test_the_rounding_key_is_the_one_saved_before_the_rename(
        setting, chips, field, kid):
    from knurlogic.engine import matmul_padding
    from knurlogic.engine.prompt_cache import disk
    r = matmul_padding.resolve(setting, chips)
    key = LY.shard_key(None, rounding=r)
    assert key["cross_chip"] == field
    assert disk.key_id(key) == kid
    # the whole-model entry's key: the full resolution, as before
    ident = disk.identity(None, rounding=r)["cross_chip"]
    assert ident == repr({"on": r["on"], "setting": r["setting"],
                          "chips": r["chips"], "why": r["why"],
                          "label": r["label"]})
