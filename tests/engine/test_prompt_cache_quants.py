"""Saved entries borrowed from another quant of the same model
(engine/prompt_cache/quants.py): the fingerprint (architecture and
tokenizer) matches across quants and nothing else; a restore borrows only
after its own entries and its own converted widths, the same width before
a converted one; a borrowed entry is never deleted or moved; one of
another width is converted; the setting off borrows nothing; usage's
`borrowed_from`; a ring's ranks each read their part. Tiny fixtures in one
process, mirroring test_prompt_cache_convert."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_prompt_cache_convert import _equal, _setup  # noqa: E402
from test_prompt_cache_layers import _gather_as, _lru_of  # noqa: E402
from test_prompt_cache_tensor import _layout_parts, _share  # noqa: E402

from knurlogic.engine.prompt_cache import convert as CV  # noqa: E402
from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import layers as LY  # noqa: E402
from knurlogic.engine.prompt_cache import quants as QT  # noqa: E402

FP = "0123456789abcdef"


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.setenv(CV.ENV, "down")
    monkeypatch.setenv(QT.ENV, "on")
    monkeypatch.setenv(D.ENV_ON, "on")


def _key(quant, bits=None):
    return LY.shard_key(f"/nonexistent/tiny-{quant}", kv_bits=bits)


def _save(base, key, parts, prompt, cache, monkeypatch, fp=FP):
    stores = [LY.Store(base, key, p) for p in parts]
    entries = None
    for st in stores:
        st.fingerprint = fp
        got = st.save(_lru_of(prompt, _share(st.part, cache)),
                      commit=len(parts) == 1, sweep_after=False)
        if st.part.rank == 0:
            entries = got["pending"]
    if len(parts) > 1:
        _gather_as(monkeypatch, [[1 if st.ready(e) else 0 for e in entries]
                                 for st in stores])
        for st in stores:
            LY.commit(object(), st, entries)


def _store(base, key, part, make, fp=FP):
    st = LY.Store(base, key, part)
    st.template = lambda: make()[part.start:part.start + part.n]
    st.fingerprint = fp
    return st


def _files(base):
    """Every file under `base`, with its bytes (what must not change)."""
    return {str(p.relative_to(base)): p.read_bytes()
            for p in sorted(base.rglob("*")) if p.is_file()}


# -------------------------------------------------------- the fingerprint

def _artifact(d, **over):
    d.mkdir(parents=True)
    cfg = {"model_type": "qwen3_5_moe", "quantization": {"bits": 4},
           "text_config": {"model_type": "qwen3_5_moe_text",
                           "num_hidden_layers": 40, "hidden_size": 2048,
                           "num_attention_heads": 16,
                           "num_key_value_heads": 2, "head_dim": 256,
                           "full_attention_interval": 4,
                           "vocab_size": 248320}}
    tok = over.pop("tokenizer", '{"model": {"vocab": {"a": 0}}}')
    for k, v in over.items():
        if k in cfg["text_config"]:
            cfg["text_config"][k] = v
        else:
            cfg[k] = v
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text(tok)
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": str(over)}))
    return d


def test_the_same_model_at_another_quant_matches(tmp_path):
    a = QT.fingerprint(_artifact(tmp_path / "a"))
    b = QT.fingerprint(_artifact(tmp_path / "b",
                                 quantization={"bits": 8, "mode": "vq"}))
    assert a and a == b
    assert QT.fingerprint(tmp_path / "nothing") is None


@pytest.mark.parametrize("over", [
    {"num_hidden_layers": 48}, {"num_attention_heads": 32},
    {"num_key_value_heads": 4}, {"tokenizer": '{"model": {"vocab": {}}}'}])
def test_another_shape_or_tokenizer_does_not(tmp_path, over):
    a = QT.fingerprint(_artifact(tmp_path / "a"))
    assert QT.fingerprint(_artifact(tmp_path / "b", **over)) != a


def test_formatting_and_side_files_do_not_split_one_model(tmp_path):
    """Measured: Qwen3.6-35B's VQ and mlx-community 8-bit copies ship one
    vocabulary formatted differently, one with vocab.json and merges.txt
    beside tokenizer.json, and rope blocks naming the type `type` or
    `rope_type`."""
    tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1},
                     "merges": [["a", "b"]]},
           "added_tokens": [{"id": 2, "content": "<x>", "special": True}],
           "pre_tokenizer": {"type": "ByteLevel"}}
    a = _artifact(tmp_path / "a", tokenizer=json.dumps(tok),
                  rope_parameters={"type": "default", "rope_theta": 1e7})
    tok2 = dict(tok, model=dict(tok["model"], merges=["a b"]))
    b = _artifact(tmp_path / "b", tokenizer=json.dumps(tok2, indent=2),
                  rope_parameters={"rope_type": "default", "rope_theta": 1e7})
    (b / "vocab.json").write_text("{}")
    (b / "merges.txt").write_text("a b\n")
    assert QT.fingerprint(a) == QT.fingerprint(b)


def test_a_different_splitting_rule_is_another_tokenizer(tmp_path):
    """Measured: the 397B VQ copies split text without combining marks
    (\\p{M}), Qwen's own with: some text tokenizes differently."""
    tok = {"model": {"vocab": {"a": 0}},
           "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\p{L}+"}}}
    a = _artifact(tmp_path / "a", tokenizer=json.dumps(tok))
    tok["pre_tokenizer"]["pattern"]["Regex"] = "[\\p{L}\\p{M}]+"
    b = _artifact(tmp_path / "b", tokenizer=json.dumps(tok))
    assert QT.fingerprint(a) != QT.fingerprint(b)


def test_an_older_entry_takes_its_fingerprint_from_its_artifact(tmp_path):
    a = _artifact(tmp_path / "a")
    meta = {"key": {"artifact": f"path:{a}"}}
    assert QT.of_entry(meta) == QT.fingerprint(a)
    assert QT.of_entry({"fingerprint": FP, "key": meta["key"]}) == FP
    assert QT.of_entry({"key": {"artifact": "feedfacefeedface"}}) is None


def test_the_setting(monkeypatch):
    from knurlogic.tuning import groups
    from knurlogic.tuning.checks import check_knob
    assert QT.on({}) and not QT.on({QT.ENV: "off"})
    assert groups.PROMPT_CACHE_KNOBS[QT.ENV][0] == "on"
    assert check_knob(QT.ENV, "off") is None and check_knob(QT.ENV, "junk")


def test_the_page_shows_the_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from knurlogic.interfaces import documents
    doc = documents.prompt_cache_document({})
    row = next(k for k in doc["knobs"] if k["name"] == QT.ENV)
    assert row["values"] == ["on", "off"] and row["help"]


# ------------------------------------------------------------ the restore

def test_a_restore_borrows_another_quants_entry_and_leaves_it(
        tmp_path, monkeypatch):
    model, make16, make8, prompt, toks, c = _setup(None)
    keep = copy.deepcopy(c)
    part = LY.Part(0, len(c), len(c))
    _save(tmp_path, _key("a"), [part], prompt, c, monkeypatch)
    before = _files(tmp_path)
    st = _store(tmp_path, _key("b"), part, make16)
    got = LY.restore(8, st, sweep_first=False)
    assert len(got) == 1
    d, back_toks, back, _k, _ms, meta = got[0]
    assert back_toks == prompt and "converted" not in meta
    assert meta["borrowed_from"] == "path:/nonexistent/tiny-a"
    assert Path(d).parent.name == D.key_id(_key("a"))
    for a, b in zip(keep, back):
        assert _equal(list(D._arrays_of(a)), list(D._arrays_of(b)))
    # nothing written, moved or deleted
    assert _files(tmp_path) == before and not st.dir.exists()
    # inserted: says so, and the owner map holds no file of it
    lru = _lru_of([1], [])
    ins = D.insert(lru, "m", got)
    assert ins[tuple(prompt)]["borrowed_from"] == meta["borrowed_from"]
    owners = {}
    ins[tuple(prompt)]["owner"] = {"session": "s"}
    D.adopt(owners, set(), ins, base=tmp_path)
    assert owners[tuple(prompt)]["file"] is None


def test_off_or_another_model_borrows_nothing(tmp_path, monkeypatch):
    model, make16, make8, prompt, toks, c = _setup(None)
    part = LY.Part(0, len(c), len(c))
    _save(tmp_path, _key("a"), [part], prompt, c, monkeypatch)
    other = _store(tmp_path, _key("b"), part, make16, fp="f" * 16)
    assert LY.restore(8, other, sweep_first=False) == []
    assert LY.restore(8, _store(tmp_path, _key("b"), part, make16,
                                fp=None), sweep_first=False) == []
    monkeypatch.setenv(QT.ENV, "off")
    st = _store(tmp_path, _key("b"), part, make16)
    assert LY.restore(8, st, sweep_first=False) == []
    assert QT.index(st, set()) == {}


def test_own_entries_then_own_widths_then_borrowed(tmp_path, monkeypatch):
    """Own first; a borrowed entry fills only what is left, and comes
    before (is older than) the own ones in the restore's order; borrowed
    at this width before borrowed and converted."""
    model, make16, make8, prompt, toks, c = _setup(None)
    part = LY.Part(0, len(c), len(c))
    p2, p3, p4 = prompt[:40], prompt[:32], prompt[:24]
    _save(tmp_path, _key("b", 8), [part], prompt, c, monkeypatch)
    _save(tmp_path, _key("b"), [part], p2, c, monkeypatch)       # 16: convert
    _save(tmp_path, _key("a", 8), [part], p3, c, monkeypatch)    # borrow
    _save(tmp_path, _key("a"), [part], p4, c, monkeypatch)       # borrow+conv
    model.make_cache = make8
    st = _store(tmp_path, _key("b", 8), part, make8)
    got = LY.restore(1, st, sweep_first=False)
    assert [g[1] for g in got] == [prompt]
    names = [Path(d).name for d in QT.candidates(st, 8)]
    assert names == [D.tokens_hash(p4), D.tokens_hash(p3)]
    got = LY.restore(8, _store(tmp_path, _key("b", 8), part, make8),
                     sweep_first=False)
    assert [len(g[1]) for g in got] == [24, 32, 40, 48]
    m = [g[5] for g in got]
    assert m[0]["borrowed_from"] and m[0]["converted"]["to_bits"] == 8
    assert m[1]["borrowed_from"] and "converted" not in m[1]
    assert "borrowed_from" not in m[2] and m[2]["converted"]
    assert "borrowed_from" not in m[3] and "converted" not in m[3]


def test_another_width_is_converted_and_the_source_kept(tmp_path,
                                                         monkeypatch):
    """Another quant's 16-bit entry into an 8-bit model: converted as
    convert.py converts, but the source stays where it was (convert.py's
    own widths are replaced; a borrowed entry is the other quant's)."""
    model, make16, make8, prompt, toks, c = _setup(None)
    part = LY.Part(0, len(c), len(c))
    _save(tmp_path, _key("a"), [part], prompt, c, monkeypatch)
    before = _files(tmp_path)
    whole = CV.convert_caches(list(c), make8())
    st = _store(tmp_path, _key("b", 8), part, make8)
    got = LY.restore(8, st, sweep_first=False)
    assert len(got) == 1
    assert got[0][5]["converted"] == {"from_bits": None, "to_bits": 8}
    for a, b in zip(whole, got[0][2]):
        assert type(a) is type(b)
        assert _equal(list(D._arrays_of(a)), list(D._arrays_of(b)))
    assert _files(tmp_path) == before


def test_a_read_back_borrows_in_place(tmp_path, monkeypatch):
    model, make16, make8, prompt, toks, c = _setup(None)
    part = LY.Part(0, len(c), len(c))
    _save(tmp_path, _key("a"), [part], prompt, c, monkeypatch)
    before = _files(tmp_path)
    st = _store(tmp_path, _key("b"), part, make16)
    idx = QT.index(st, set())
    assert list(idx) == [tuple(prompt)] and idx[tuple(prompt)]["borrow"]
    assert QT.index(st, {tuple(prompt)}) == {}
    got = QT.read_back(st, idx[tuple(prompt)]["file"])
    assert len(got) == 1 and got[0][5]["borrowed_from"]
    assert _files(tmp_path) == before


@pytest.mark.parametrize("how", [[2, 2], "tensor2"])
def test_a_rings_ranks_borrow_their_parts(tmp_path, monkeypatch, how):
    """Pipeline and tensor: each rank reads its part of the other quant's
    entry, the vote keeps it, and nothing is written or deleted."""
    model, make16, make8, prompt, toks, c = _setup(None)
    parts = _layout_parts(model, how)
    _save(tmp_path, _key("a"), parts, prompt, c, monkeypatch)
    before = _files(tmp_path)
    stores = [_store(tmp_path, _key("b"), p, make16) for p in parts]
    mine = [QT.read(st, QT.candidates(st, 8)) for st in stores]
    _gather_as(monkeypatch, [LY.vote_row(g) for g in mine])
    agreed = [LY.agree(object(), g) for g in mine]
    monkeypatch.setattr(LY, "_gather", lambda link, row: [row] * len(parts))
    out = [CV.persist_ring(object(), st, g) for st, g in zip(stores, agreed)]
    for p, got in zip(parts, out):
        assert len(got) == 1 and got[0][5]["borrowed_from"]
        for a, b in zip(_share(p, c), got[0][2][:p.n]):
            assert _equal(list(D._arrays_of(a)), list(D._arrays_of(b)))
    assert _files(tmp_path) == before


def test_the_scheduler_serves_a_borrowed_entry_and_usage_says_so(
        tmp_path, monkeypatch):
    """Saved by one quant, the other loaded: restored from the first's
    entry, the request's usage and the registry name it, and the first's
    entry is still there after the second's save."""
    from test_batch_drafting import _tiny
    from test_prompt_disk import Host
    from test_scheduler import Tok, _collect, _job

    from knurlogic.engine.prompt_cache import location as LOC
    from knurlogic.engine.runtime.scheduler import Scheduler
    monkeypatch.setattr(LOC, "fallback", lambda: tmp_path)
    monkeypatch.setattr(LOC, "root_for", lambda path: (tmp_path, "test"))
    monkeypatch.setattr(QT, "fingerprint", lambda path: FP)
    model, _head, prompts = _tiny(512)
    host = Host(model, Tok(prompts))
    s = Scheduler(host, prefill_step_size=16).start()
    s.load("/nonexistent/tiny-q4").done.wait(60)
    p = prompts[2]
    _collect(s.submit(_job(p, max_tokens=4, session="c1")))
    assert s.save_prompt_cache().done.wait(60)
    s.unload().done.wait(60)
    src = _files(tmp_path)
    try:
        s.load("/nonexistent/tiny-q8").done.wait(60)
        _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4)))
        c = warm["knurlogic"]["cache"]
        assert c["disk"]["tokens"] > 0
        assert c["borrowed_from"] == "tiny-q4" and "converted" not in c
        rows = s.list_prompt_cache()
        assert rows.done.wait(60)
        assert any(r.get("borrowed_from") == "tiny-q4"
                   for r in rows.result["entries"])
        assert s.save_prompt_cache().done.wait(60)
        now = _files(tmp_path)
        assert all(now.get(k) == v for k, v in src.items()
                   if not k.endswith("pins.json"))
    finally:
        s.unload().done.wait(60)
