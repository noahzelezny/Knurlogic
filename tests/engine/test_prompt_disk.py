"""The prompt cache on disk (engine/prompt_cache/disk.py): saved at unload,
restored at load, keyed to the model, budgeted, expired, prefix-served,
reported. Tiny fixtures only."""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_scheduler import Tok, _collect, _job  # noqa: E402

from knurlogic.engine.prompt_cache import disk as D  # noqa: E402


def _key(**kw):
    return D.identity("/nonexistent/tiny", **kw)


# --- every cache type, round-tripped -----------------------------------------

def _forced(model, prompt, toks):
    """Prefill `prompt` into a fresh cache; then the logits of `toks` fed
    one at a time from that cache."""
    c = model.make_cache()

    def call(x, cache):
        y = model(mx.array([x]), cache=cache)
        return (y if isinstance(y, mx.array) else y.logits)[0]
    call(prompt, c)
    mx.eval([a for a in D._arrays_of(c)])
    return c, lambda cache: mx.concatenate([call([t], cache)[-1:]
                                            for t in toks])


def _family(name, bits):
    if name == "gemma4":
        from fixtures_vision_gemma4 import tiny_gemma4_config, tiny_text_model
        tc = dict(tiny_gemma4_config()["text_config"], num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                  global_head_dim=64, intermediate_size=256,
                  sliding_window=16, num_kv_shared_layers=2,
                  layer_types=["sliding_attention", "full_attention"] * 3)
        model, _ = tiny_text_model(tc)
    elif name == "qwen3_5":
        from test_batch_drafting import _tiny
        model = _tiny(512)[0]
    else:
        import pipeline_ring_worker as W
        model = W.build(name)
    if bits:
        # a test before this one may have registered qwen4_exp again: the
        # factory's classes must be the module now registered (a server
        # registers once)
        from knurlogic.engine.families.qwen import kvcache
        kvcache._CLASSES.clear()
        from knurlogic.engine.kvquant import install
        assert install(model, bits) >= 1
    return model


@pytest.mark.parametrize("family,bits", [
    ("qwen3_5", None),        # KVCache + deltanet ArraysCache
    ("qwen3_5", 8),           # QuantKVCache + deltanet
    ("gemma4", None),         # RotatingKVCache windows + KVCache
    ("gemma4", 8),
    ("qwen4_exp", None),      # attention + indexer caches
    ("qwen4_exp", 8),
    ("glm5_next", None),      # MLA latent + DSA indexer (CacheList)
    ("glm5_next", 8),
    ("deepseek_v4", None),    # DeepseekV4Cache: windows + compressor pools
])
def test_every_cache_type_comes_back_exactly(tmp_path, family, bits):
    """Saved after a prefill, read back: the next tokens' logits from the
    restored cache equal the original's, bit for bit."""
    import copy
    model = _family(family, bits)
    mx.random.seed(5)
    prompt = mx.random.randint(0, 60, (70,)).tolist()
    toks = mx.random.randint(0, 60, (5,)).tolist()
    cache, cont = _forced(model, prompt, toks)
    keep = copy.deepcopy(cache)
    key = _key(kv_bits=bits)
    f = D.save_entry(tmp_path, key, prompt, cache, "user", 1, 0)
    toks_back, back, kind = D.load_entry(f, key)
    assert toks_back == prompt and kind == "user"
    assert [type(c) for c in back] == [type(c) for c in keep]
    assert mx.array_equal(cont(keep), cont(back))


def test_an_entry_the_format_cannot_carry_is_skipped_not_written(tmp_path):
    from mlx_lm.models.cache import KVCache, LRUPromptCache
    c = KVCache()
    k = mx.ones((1, 2, 4, 32))
    c.update_and_fetch(k, k)
    odd = KVCache()
    odd.update_and_fetch(k, k)
    odd.hook = lambda: None             # a function: no file can hold it
    lru = LRUPromptCache(max_size=4)
    lru.insert_cache("m", [1, 2, 3], [c])
    lru.insert_cache("m", [7, 8, 9], [odd])
    got = D.save(lru, _key(), base=tmp_path)
    assert got["saved"] == 1 and got["skipped"] == 1
    assert len(D.entries(tmp_path / D.key_id(_key()))) == 1


# --- the key -------------------------------------------------------------------

def _lru(n=3, length=20):
    from mlx_lm.models.cache import KVCache, LRUPromptCache
    lru = LRUPromptCache(max_size=10)
    for i in range(n):
        c = KVCache()
        k = mx.random.normal((1, 2, length, 32))
        c.update_and_fetch(k, k)
        lru.insert_cache("m", [100 + i] + list(range(length - 1)), [c])
    return lru


@pytest.mark.parametrize("other", [
    dict(path="/nonexistent/other"),                       # another model
    dict(kv_bits=8),                                       # other KV bits
    dict(layout={"split": "pipeline", "world": 2, "rank": 0,
                 "bounds": [[2, 4], [0, 2]]}),             # another split
    dict(draft={"head": "x:Head", "block": 0}),            # a drafting head
])
def test_a_different_key_is_a_miss(tmp_path, other):
    from mlx_lm.models.cache import LRUPromptCache
    D.save(_lru(), _key(), base=tmp_path)
    path = other.pop("path", "/nonexistent/tiny")
    k2 = D.identity(path, **other)
    assert D.key_id(k2) != D.key_id(_key())
    lru = LRUPromptCache(max_size=10)
    assert D.restore(lru, "m", k2, base=tmp_path) == {}
    assert len(lru) == 0
    # and the right key restores all three
    assert len(D.restore(lru, "m", _key(), base=tmp_path)) == 3


def test_a_file_under_the_wrong_key_is_never_read_in(tmp_path):
    D.save(_lru(1), _key(), base=tmp_path)
    (f,) = [e[3] for e in D.entries(tmp_path / D.key_id(_key()))]
    assert D.load_entry(f, _key(kv_bits=8)) is None
    assert not f.exists()


def test_the_artifact_identity_covers_model_py(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.py").write_text("A = 1\n")
    a = D.key_id(D.identity(tmp_path))
    (tmp_path / "model.py").write_text("A = 2\n")
    from knurlogic.machine import artifact
    artifact._IDENT.clear()
    assert D.key_id(D.identity(tmp_path)) != a


# --- kept, budgeted, expired ---------------------------------------------------

def test_a_second_save_keeps_files_and_follows_the_lru_order(tmp_path):
    lru = _lru()
    D.save(lru, _key(), base=tmp_path)
    got = D.save(lru, _key(), base=tmp_path)
    assert got["kept"] == 3 and got["saved"] == 0
    es = D.entries(tmp_path / D.key_id(_key()))
    assert [e[0] for e in es] == [2, 2, 2] and [e[1] for e in es] == [0, 1, 2]


def test_lru_eviction_by_the_budget(tmp_path, monkeypatch):
    D.save(_lru(3), _key(), base=tmp_path)
    files = sorted(D._all_files(tmp_path), key=lambda x: x[0].name)
    size = files[0][1]
    for i, (f, _, _) in enumerate(files):      # oldest use first
        os.utime(f, (time.time() - 100 + i, time.time() - 100 + i))
    monkeypatch.setenv(D.ENV_GB, str(2.5 * size / D.GIB))
    out = D.sweep(tmp_path)
    assert out["budget"] == 1
    left = {f.name for f, _, _ in D._all_files(tmp_path)}
    assert files[0][0].name not in left and len(left) == 2


def test_the_default_budget_is_a_share_of_free_space(tmp_path, monkeypatch):
    monkeypatch.delenv(D.ENV_GB, raising=False)
    b = D.budget_bytes(tmp_path)
    assert 0 < b <= D.DEFAULT_CAP_GIB * D.GIB


def test_ttl_expiry(tmp_path, monkeypatch):
    D.save(_lru(2), _key(), base=tmp_path)
    monkeypatch.setenv(D.ENV_TTL, "1")
    assert D.sweep(tmp_path, now=time.time() + 1800)["ttl"] == 0
    assert D.sweep(tmp_path, now=time.time() + 7200)["ttl"] == 2
    assert not (tmp_path / D.key_id(_key())).exists()


def test_off_saves_and_restores_nothing(tmp_path, monkeypatch):
    from mlx_lm.models.cache import LRUPromptCache
    D.save(_lru(1), _key(), base=tmp_path)
    monkeypatch.setenv(D.ENV_ON, "off")
    assert D.save(_lru(1), _key(), base=tmp_path)["saved"] == 0
    assert D.restore(LRUPromptCache(), "m", _key(), base=tmp_path) == {}


def test_the_saved_setting_beats_the_environment(tmp_path, monkeypatch):
    from knurlogic.machine import preferences
    monkeypatch.setenv(D.ENV_TTL, "5")
    preferences.path().write_text('{"KNURLOGIC_PROMPT_CACHE_TTL_H": "2"}')
    assert D.ttl_s() == 7200
    from knurlogic.tuning.settings import check_knob
    assert check_knob("KNURLOGIC_PROMPT_CACHE_TTL_H", "0")
    assert check_knob("KNURLOGIC_PROMPT_CACHE_DISK", "maybe")
    assert check_knob("KNURLOGIC_PROMPT_CACHE_DISK_GB", "") is None


# --- read back -----------------------------------------------------------------

def test_a_corrupt_file_is_a_miss_and_deleted(tmp_path):
    from mlx_lm.models.cache import LRUPromptCache
    D.save(_lru(2), _key(), base=tmp_path)
    es = D.entries(tmp_path / D.key_id(_key()))
    f = es[0][3]
    data = f.read_bytes()
    f.write_bytes(data[:len(data) // 2])            # truncated
    lru = LRUPromptCache(max_size=10)
    got = D.restore(lru, "m", _key(), base=tmp_path)
    assert len(got) == 1 and not f.exists() and es[1][3].exists()


def test_a_restored_entry_serves_any_prompt_it_prefixes(tmp_path):
    """Restored into the in-memory trie, the entry is found by its own
    prefix rule: a longer prompt is served the cached tokens."""
    from knurlogic.engine.prompt_cache.memory import PromptCache
    lru = _lru(1, length=30)
    D.save(lru, _key(), base=tmp_path)
    (toks,) = [t for _, t, _ in D._lru_entries(lru)]
    pc = PromptCache(10)
    got = D.restore(pc.lru, "m", _key(), base=tmp_path)
    assert list(got) == [tuple(toks)]
    prompt = list(toks) + [5, 6, 7]
    cache, rest = pc.fetch("m", prompt)
    assert cache is not None and rest == [5, 6, 7]
    assert cache[0].offset == len(toks)
    assert D.source_of(pc.lru, "m", prompt, len(toks)) == tuple(toks)


def test_a_vote_names_the_entries_in_order():
    a = [Path("00000001-000000-" + "a" * 24 + ".safetensors"),
         Path("00000001-000001-" + "b" * 24 + ".safetensors")]
    assert D.vote(a) == D.vote(list(a))
    assert D.vote(a) != D.vote(a[::-1]) and D.vote(a) != D.vote(a[:1])
    assert all(0 <= v < (1 << 28) for v in D.vote(a)[1:])


class _Link:
    """A ring's link to the agreement: what the other ranks voted."""
    def __init__(self, others):
        self.others = others

    def align(self):
        pass


def test_the_ranks_restore_all_or_none(monkeypatch, tmp_path):
    files = [Path("00000001-000000-" + "a" * 24 + ".safetensors")]

    def gather(x, group=None, stream=None):
        mine = x.tolist()
        return mx.array(mine + [v for o in group for v in o])
    monkeypatch.setattr(mx.distributed, "all_gather", gather)
    same = _Link(None)
    same.group = [D.vote(files)]
    assert D.agree(same, files) == files
    other = _Link(None)
    other.group = [D.vote([])]
    assert D.agree(other, files) == []


# --- through the scheduler -----------------------------------------------------

class Host:
    """A host the scheduler loads and unloads: the tiny model stays the
    test's (MLX frees on the thread that made it)."""
    state = "empty"
    error = ""
    kv_bits = None
    cross_chip = None

    def __init__(self, model, tok):
        self._m, self._t = model, tok
        self.model = self.tokenizer = self.model_key = None
        self.path = None
        self.after_bind = None
        self.cache_layout = None

    def expect(self, path):
        pass

    def load(self, path, executes_artifact_code=False):
        self.path, self.model, self.tokenizer = path, self._m, self._t
        self.model_key = (path, None, None)
        if self.after_bind is not None:
            self.after_bind()
        self.state = "ready"

    def unload(self):
        self.model = self.tokenizer = self.model_key = None
        self.state = "empty"


@pytest.fixture(scope="module")
def disk_sched():
    from test_batch_drafting import _tiny

    from knurlogic.engine.runtime.scheduler import Scheduler
    model, _head, prompts = _tiny(512)
    s = Scheduler(Host(model, Tok(prompts)), prefill_step_size=16).start()
    s.prompts = prompts
    yield s


def test_a_save_asked_for_restores_at_load_and_usage_says_disk(
        disk_sched, monkeypatch):
    s = disk_sched
    s.load("/nonexistent/tiny-a").done.wait(60)
    p = s.prompts[2]
    _, cold = _collect(s.submit(_job(p, max_tokens=4, session="t1")))
    assert cold["knurlogic"]["cache"]["disk"] == {"tokens": 0,
                                                  "read_ms": 0.0}
    before = set(D._all_files(D.root()))
    s.unload().done.wait(60)
    # an unload saves nothing: only a client's ask does (the maintainer: nobody is
    # surprised by cache files)
    assert set(D._all_files(D.root())) == before
    s.load("/nonexistent/tiny-a").done.wait(60)
    _collect(s.submit(_job(p, max_tokens=4, session="t1")))
    assert s.save_prompt_cache().done.wait(60)
    s.unload().done.wait(60)
    assert D._all_files(D.root())
    s.load("/nonexistent/tiny-a").done.wait(60)
    _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4)))
    c = warm["knurlogic"]["cache"]
    assert c["used"] >= len(p) - 1 and c["disk"]["tokens"] == c["used"]
    assert c["disk"]["read_ms"] > 0
    # the next turn's entry was made in memory: a memory hit
    _, again = _collect(s.submit(_job(p + [5, 6, 7, 9], max_tokens=4)))
    assert again["knurlogic"]["cache"]["used"] > 0
    assert again["knurlogic"]["cache"]["disk"]["tokens"] == 0
    # another model finds nothing of this one's
    s.load("/nonexistent/tiny-b").done.wait(60)
    _, other = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4)))
    assert other["knurlogic"]["cache"]["disk"]["tokens"] == 0


def test_a_save_on_request(disk_sched):
    s = disk_sched
    s.load("/nonexistent/tiny-c").done.wait(60)
    _collect(s.submit(_job(s.prompts[0], max_tokens=3, session="t1")))
    cmd = s.save_prompt_cache()
    assert cmd.done.wait(60) and not cmd.error
    assert cmd.result["saved"] + cmd.result["kept"] >= 1


def test_a_drafting_heads_cache_rides_in_the_entry(tmp_path):
    """DSpark's head cache (a pool entry's tail past the trunk) round-trips
    beside a trunk cache."""
    from mlx_lm.models.cache import KVCache

    from knurlogic.engine.families.deepseek.heads.deepseek_v4_dspark import (
        DSparkCache,
    )
    t = KVCache()
    k = mx.random.normal((1, 2, 6, 32))
    t.update_and_fetch(k, k)
    h = DSparkCache(2, 4)
    h.append([mx.random.normal((1, 6, 16)), mx.random.normal((1, 6, 16))], 6)
    f = D.save_entry(tmp_path, _key(), list(range(6)), [t, h], "assistant",
                     1, 0)
    _, (t2, h2), _ = D.load_entry(f, _key())
    assert type(h2) is DSparkCache and h2.lengths == [6] and h2.window == 4
    assert all(mx.array_equal(a, b) for a, b in zip(h.keys, h2.keys))
    assert mx.array_equal(t.state[0], t2.state[0]) and t2.offset == 6
