"""Saved entries converted between KV bit widths
(engine/prompt_cache/convert.py): a 16-bit entry restored by the model
running 8-bit KV is quantized exactly as the 8-bit cache quantizes, its
next tokens' logits close to a native 8-bit prefill's; the old width's
entry is replaced, never kept beside it; the setting's three values; the
other direction; a pipeline's and a tensor split's ranks (a sliced
convert equals the whole convert cut); recurrent state untouched; the
registry's bits and usage's `converted`. Tiny fixtures in one process."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_prompt_cache_layers import LINK, _gather_as, _key, _lru_of  # noqa: E402
from test_prompt_cache_tensor import _layout_parts, _share  # noqa: E402
from test_prompt_disk import _family, _forced  # noqa: E402

from knurlogic.engine import kvquant as KQ  # noqa: E402
from knurlogic.engine.arrays import arrays_of  # noqa: E402
from knurlogic.engine.prompt_cache import convert as CV  # noqa: E402
from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import layers as LY  # noqa: E402
from knurlogic.engine.prompt_cache import slices as SL  # noqa: E402

#: the largest |logit| difference allowed between a converted 16 -> 8
#: entry's next tokens and a native 8-bit prefill's (tiny qwen3_5_moe,
#: 48-token prompt; measured 5.2e-2, where the unconverted 16-bit cache in
#: the same 8-bit model is 4.4 away)
TOL = 0.1


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.setenv(CV.ENV, "down")
    monkeypatch.setenv(D.ENV_ON, "on")


def _setup(bits_from, family="qwen3_5_moe"):
    """A tiny model, a prompt, and its cache prefilled at `bits_from`; the
    model's make_cache at 16 and at 8 bits (same weights)."""
    model = _family(family, None)
    make16 = model.make_cache
    KQ.install(model, 8)
    make8 = model.make_cache
    model.make_cache = make8 if bits_from == 8 else make16
    mx.random.seed(7)
    prompt = mx.random.randint(0, 60, (48,)).tolist()
    toks = [7, 3, 11, 5]
    cache, _ = _forced(model, prompt, toks)
    return model, make16, make8, prompt, toks, cache


def _save(base, bits, parts, prompt, cache, monkeypatch):
    key = _key(kv_bits=bits)
    stores = [LY.Store(base, key, p) for p in parts]
    entries = None
    for st in stores:
        got = st.save(_lru_of(prompt, _share(st.part, cache)),
                      commit=len(parts) == 1, sweep_after=False)
        if st.part.rank == 0:
            entries = got["pending"]
    if len(parts) > 1:
        _gather_as(monkeypatch, [[1 if st.ready(e) else 0 for e in entries]
                                 for st in stores])
        for st in stores:
            LY.commit(LINK, st, entries)
    return key


def _store(base, bits, part, make):
    st = LY.Store(base, _key(kv_bits=bits), part)
    st.template = lambda: make()[part.start:part.start + part.n]
    return st


def _kv_arrays(c):
    return [c.keys, c.values] if not isinstance(c.keys, tuple) \
        else list(c.keys) + list(c.values)


def _quant_of(c16):
    """What the 8-bit cache stores for `c16`'s K/V."""
    n = c16.offset
    k, v = c16.keys[..., :n, :], c16.values[..., :n, :]
    g = KQ._group(k.shape[-1], v.shape[-1])
    return list(KQ._q(k, g, 8)) + list(KQ._q(v, g, 8))


def _equal(a, b):
    return all(mx.array_equal(x, y) for x, y in zip(a, b)) and \
        len(a) == len(b)


def _dirs(base, bits):
    d = base / D.key_id(_key(kv_bits=bits))
    return sorted(p.name for p in d.iterdir() if (p / "entry.json").exists()) \
        if d.is_dir() else []


def test_a_16_bit_entry_restores_into_an_8_bit_model_converted(
        tmp_path, monkeypatch):
    model, make16, make8, prompt, toks, c16 = _setup(None)
    keep = copy.deepcopy(c16)
    part = LY.part_of(type("H", (), {"model": model,
                                     "cache_layout": None})())
    _save(tmp_path, None, [part], prompt, c16, monkeypatch)
    h = D.tokens_hash(prompt)
    assert _dirs(tmp_path, None) == [h]
    model.make_cache = make8
    st = _store(tmp_path, 8, part, make8)
    got = LY.restore(8, st, sweep_first=False)
    assert len(got) == 1
    d, toks_back, back, _kind, _ms, meta = got[0]
    assert toks_back == prompt
    assert meta["converted"] == {"from_bits": None, "to_bits": 8}
    # K/V == quantize(original), exactly; recurrent state as it was
    nq = 0
    for a, b in zip(keep, back):
        if isinstance(b, KQ.QuantKVCache):
            nq += 1
            assert b.offset == a.offset and b.kv_bits == 8
            assert _equal(_kv_arrays(b), _quant_of(a))
        else:
            assert type(a) is type(b)
            assert _equal([x for x in arrays_of(a)],
                          [x for x in arrays_of(b)])
    assert nq >= 1
    # the old width's entry is gone; the converted one is there once
    assert _dirs(tmp_path, None) == [] and _dirs(tmp_path, 8) == [h]
    assert Path(d) == tmp_path / st.key_id / h
    row = LY.row(Path(d), 0, tmp_path)
    assert row["kv_bits"] == 8
    m = json.loads((Path(d) / "entry.json").read_text())
    assert m["key"]["kv_bits"] == 8 and m["converted_from"] == 16
    # it reads back natively next time (no conversion)
    again = LY.restore(8, _store(tmp_path, 8, part, make8),
                       sweep_first=False)
    assert len(again) == 1 and "converted" not in again[0][5]
    # logits close to a native 8-bit prefill's
    native, cont = _forced(model, prompt, toks)
    diff = mx.abs(cont(back) - cont(native)).max().item()
    gap = mx.abs(cont(keep) - cont(native)).max().item()
    print(f"16->8 converted vs native 8-bit: max |logit diff| {diff:.2e} "
          f"(16-bit cache vs native 8-bit: {gap:.2e})")
    assert diff <= TOL


def test_off_is_a_miss_and_keeps_the_old_entry(tmp_path, monkeypatch):
    monkeypatch.setenv(CV.ENV, "off")
    model, make16, make8, prompt, toks, c16 = _setup(None)
    part = LY.Part(0, len(c16), len(c16))
    _save(tmp_path, None, [part], prompt, c16, monkeypatch)
    assert LY.restore(8, _store(tmp_path, 8, part, make8),
                      sweep_first=False) == []
    assert _dirs(tmp_path, None) == [D.tokens_hash(prompt)]


def test_down_does_not_convert_8_to_16(tmp_path, monkeypatch):
    model, make16, make8, prompt, toks, c8 = _setup(8)
    part = LY.Part(0, len(c8), len(c8))
    _save(tmp_path, 8, [part], prompt, c8, monkeypatch)
    assert LY.restore(8, _store(tmp_path, None, part, make16),
                      sweep_first=False) == []
    assert _dirs(tmp_path, 8) == [D.tokens_hash(prompt)]


def test_both_converts_8_to_16_as_dequantized(tmp_path, monkeypatch):
    monkeypatch.setenv(CV.ENV, "both")
    model, make16, make8, prompt, toks, c8 = _setup(8)
    keep = copy.deepcopy(c8)
    part = LY.Part(0, len(c8), len(c8))
    _save(tmp_path, 8, [part], prompt, c8, monkeypatch)
    got = LY.restore(8, _store(tmp_path, None, part, make16),
                     sweep_first=False)
    assert len(got) == 1
    assert got[0][5]["converted"] == {"from_bits": 8, "to_bits": None}
    n = 0
    for a, b in zip(keep, got[0][2]):
        if isinstance(a, KQ.QuantKVCache):
            n += 1
            assert not isinstance(b, KQ.QuantKVCache)
            k = KQ._dq(tuple(p[..., :a.offset, :] for p in a.keys),
                       a.group, 8, a.keys[1].dtype)
            assert b.offset == a.offset and mx.array_equal(b.keys, k)
        else:
            assert _equal(list(arrays_of(a)), list(arrays_of(b)))
    assert n >= 1
    assert _dirs(tmp_path, 8) == [] and \
        _dirs(tmp_path, None) == [D.tokens_hash(prompt)]


def test_a_read_back_converts_and_replaces(tmp_path, monkeypatch):
    """A single machine's read-back index lists the other width's entry;
    reading it back converts it and moves it under this key."""
    model, make16, make8, prompt, toks, c16 = _setup(None)
    part = LY.Part(0, len(c16), len(c16))
    _save(tmp_path, None, [part], prompt, c16, monkeypatch)
    st = _store(tmp_path, 8, part, make8)
    idx = CV.index(st, set())
    assert list(idx) == [tuple(prompt)] and idx[tuple(prompt)]["convert"] \
        is None
    assert CV.index(st, {tuple(prompt)}) == {}
    got = CV.read_back(st, idx[tuple(prompt)]["file"])
    assert len(got) == 1 and got[0][5]["converted"]["to_bits"] == 8
    assert _dirs(tmp_path, None) == [] and \
        _dirs(tmp_path, 8) == [D.tokens_hash(prompt)]


def test_the_setting_and_its_directions():
    assert CV.allowed(None, 8, "down") and CV.allowed(None, 4, "down")
    assert not CV.allowed(8, None, "down") and CV.allowed(8, None, "both")
    assert not CV.allowed(None, 8, "off") and not CV.allowed(8, 4, "both")
    assert CV.mode({}) == "down" and CV.mode({CV.ENV: "junk"}) == "down"
    from knurlogic.tuning import groups
    from knurlogic.tuning.checks import check_knob
    assert CV.ENV in groups.PROMPT_CACHE_KNOBS
    assert groups.PROMPT_CACHE_KNOBS[CV.ENV][0] == "down"
    assert check_knob(CV.ENV, "both") is None and check_knob(CV.ENV, "up")


def test_the_page_shows_the_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from knurlogic.interfaces.page import documents
    doc = documents.prompt_cache_document({})
    row = next(k for k in doc["knobs"] if k["name"] == CV.ENV)
    assert row["values"] == ["down", "both", "off"] and row["help"]


def _ring_restore(monkeypatch, base, parts, make):
    """Each rank's restore, its two votes as if gathered: what each rank
    read (converted), the agreed entries, then each rank's write and the
    commit of the converted ones."""
    stores = [_store(base, 8, p, make) for p in parts]
    mine = []
    for st in stores:
        c = CV.candidates(st, 8, None)
        mine.append(CV.read(st, c))
    _gather_as(monkeypatch, [LY.vote_row(g) for g in mine])
    agreed = [LY.agree(LINK, g) for g in mine]
    out = []
    for st, g in zip(stores, agreed):
        _gather_as(monkeypatch, lambda row: [row] * len(parts))
        out.append(CV.persist_ring(LINK, st, g))
    return stores, out


@pytest.mark.parametrize("how", [[2, 2], "tensor2"])
def test_a_rings_ranks_convert_their_parts(tmp_path, monkeypatch, how):
    """Pipeline and tensor: each rank converts its slots (a tensor rank its
    head range), equal to the whole entry converted and cut; the entry is
    then whole under the 8-bit key, the 16-bit one gone."""
    model, make16, make8, prompt, toks, c16 = _setup(None)
    parts = _layout_parts(model, how)
    _save(tmp_path, None, parts, prompt, c16, monkeypatch)
    whole = CV.convert_caches(list(c16), make8())
    stores, out = _ring_restore(monkeypatch, tmp_path, parts, make8)
    for p, got in zip(parts, out):
        assert len(got) == 1 and got[0][5]["converted"]["to_bits"] == 8
        want = _share(p, whole)
        for a, b in zip(want, got[0][2][:p.n]):
            assert type(a) is type(b)
            assert _equal(list(arrays_of(a)), list(arrays_of(b)))
    h = D.tokens_hash(prompt)
    assert _dirs(tmp_path, None) == [] and _dirs(tmp_path, 8) == [h]
    # and the 8-bit key's entry now restores natively on every rank
    for st in stores:
        assert st.holds(tmp_path / st.key_id / h,
                        LY.entry_meta(tmp_path / st.key_id / h))


def test_a_sliced_convert_is_the_whole_convert_cut():
    """Quantization groups run along the head dim: converting a head range
    equals converting the layer and cutting the range."""
    from mlx_lm.models.cache import KVCache
    c = KVCache()
    mx.random.seed(1)
    c.update_and_fetch(mx.random.normal((1, 4, 10, 64)).astype(mx.bfloat16),
                       mx.random.normal((1, 4, 10, 64)).astype(mx.bfloat16))
    spec = ("kv", 4, (("keys", 1, (1,)), ("values", 1, (1,))))
    qspec = ("kv", 4, (("keys", 1, (1,)), ("values", 1, (1,))))
    whole = CV.convert_one(c, KQ.QuantKVCache(8))
    for a, b in ((0, 2), (2, 4), (1, 3)):
        part = CV.convert_one(SL.cut(c, spec, a, b), KQ.QuantKVCache(8))
        cut = SL.cut(whole, qspec, a, b)
        assert _equal(_kv_arrays(part), _kv_arrays(cut))


def test_the_scheduler_serves_a_converted_entry_and_usage_says_so(
        tmp_path, monkeypatch):
    """Saved by a 16-bit load, loaded again with 8-bit KV: restored,
    converted, and the request's usage names the conversion."""
    from test_batch_drafting import _tiny
    from test_prompt_disk import Host
    from test_scheduler import Tok, _collect, _job

    from knurlogic.engine.prompt_cache import location as LOC
    from knurlogic.engine.runtime.scheduler import Scheduler
    monkeypatch.setattr(LOC, "fallback", lambda: tmp_path)
    monkeypatch.setattr(LOC, "root_for", lambda path: (tmp_path, "test"))
    model, _head, prompts = _tiny(512)
    make16 = model.make_cache
    host = Host(model, Tok(prompts))
    s = Scheduler(host, prefill_step_size=16).start()
    path = "/nonexistent/tiny-convert"
    s.load(path).done.wait(60)
    p = prompts[2]
    _collect(s.submit(_job(p, max_tokens=4, session="c1")))
    assert s.save_prompt_cache().done.wait(60)
    s.unload().done.wait(60)
    KQ.install(model, 8)
    host.kv_bits = 8
    try:
        s.load(path).done.wait(60)
        _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4)))
        c = warm["knurlogic"]["cache"]
        assert c["disk"]["tokens"] > 0
        assert c["converted"] == {"from_bits": None, "to_bits": 8}
        rows = [r for r in D.list_disk(tmp_path)
                if r.get("model") and r.get("kv_bits") is not None]
        assert rows and all(r["kv_bits"] == 8 for r in rows)
    finally:
        s.unload().done.wait(60)
        model.make_cache = make16
        host.kv_bits = None
