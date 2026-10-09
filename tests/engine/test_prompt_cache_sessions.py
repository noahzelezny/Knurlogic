"""Session-owned prompt-cache entries (docs/design/prompt-cache-disk.md,
"Sessions"): ownership in the side map, drop, pins the sweep spares, the
incremental save, the on-demand read-back, a follower's journal ops. Tiny
fake caches only."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_scheduler import Tok, _collect, _job  # noqa: E402

from knurlogic.engine.runtime.scheduler import PromptCache  # noqa: E402
from knurlogic.engine.serve import prompt_disk as D  # noqa: E402


def _key(**kw):
    return D.identity("/nonexistent/tiny", **kw)


def _kv(n=8):
    from mlx_lm.models.cache import KVCache
    c = KVCache()
    k = mx.random.normal((1, 2, n, 32))
    c.update_and_fetch(k, k)
    return [c]


def _own(s, role="main", run="r1"):
    return {"session": s, "role": role, "run": run}


def _pc(size=10):
    """s1 owns two entries, s2 one, one is anonymous."""
    pc = PromptCache(size)
    pc.insert("m", [1, 2, 3, 4, 5, 6, 7, 8], _kv(), "user", owner=_own("s1"))
    pc.insert("m", [9, 2, 3, 4, 5, 6, 7, 8], _kv(), "user", owner=_own("s1"))
    pc.insert("m", [10, 2, 3, 4, 5, 6, 7, 8], _kv(), "user",
              owner=_own("s2"))
    pc.insert("m", [11, 2, 3, 4, 5, 6, 7, 8], _kv(), "user")
    return pc


# --- ownership -----------------------------------------------------------------

def test_an_entry_records_its_owner_and_anonymous_owns_nothing():
    pc = _pc()
    assert pc.owners[(1, 2, 3, 4, 5, 6, 7, 8)]["session"] == "s1"
    assert pc.owners[(1, 2, 3, 4, 5, 6, 7, 8)]["role"] == "main"
    assert (11, 2, 3, 4, 5, 6, 7, 8) not in pc.owners
    assert len(pc.of_session("s1")) == 2 and len(pc.of_session("s2")) == 1


def test_the_side_map_is_pruned_once_the_lru_evicted_silently():
    pc = _pc(size=2)        # mlx-lm evicted the two oldest on insert
    assert len(pc.lru) == 2
    pc.live()
    assert set(pc.owners) == {(10, 2, 3, 4, 5, 6, 7, 8)}


# --- drop ----------------------------------------------------------------------

def test_drop_takes_a_session_out_of_memory_and_off_disk(tmp_path):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners)
    # another model's directory holds one of s1's too
    D.save(pc.lru, _key(kv_bits=8), base=tmp_path, owners=pc.owners)
    assert len(D._all_files(tmp_path)) == 6     # 3 sessioned x 2 keys
    nbytes = pc.lru.nbytes
    assert pc.drop("s1") == 2
    assert len(pc.lru) == 2 and pc.of_session("s1") == []
    assert pc.lru.nbytes < nbytes
    assert D.drop_files("s1", tmp_path) == 4
    left = [D._header(f)["owner"]["session"]
            for f, _, _ in D._all_files(tmp_path)]
    assert left == ["s2", "s2"]


def test_remove_entry_keeps_mlx_lm_s_books():
    pc = _pc()
    before = pc.lru.nbytes
    one = pc.lru._trie.get("m", [9, 2, 3, 4, 5, 6, 7, 8]).nbytes
    assert D.remove_entry(pc.lru, "m", (9, 2, 3, 4, 5, 6, 7, 8))
    assert pc.lru.nbytes == before - one and len(pc.lru) == 3
    assert not D.remove_entry(pc.lru, "m", (9, 2, 3, 4, 5, 6, 7, 8))
    cache, rest = pc.fetch("m", [9, 2, 3, 4, 5])
    assert cache is None


# --- the save -------------------------------------------------------------------

def test_only_sessions_are_saved_and_files_carry_the_owner(tmp_path):
    pc = _pc()
    got = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                 model="tiny")
    assert got["saved"] == 3 and got["entries"] == 3
    heads = [D._header(f) for f, _, _ in D._all_files(tmp_path)]
    assert {h["owner"]["session"] for h in heads} == {"s1", "s2"}
    assert all(h["saved_at"] and h["model"] == "tiny" and
               h["pinned"] is False for h in heads)
    assert all(m["file"] for m in pc.owners.values())


def test_the_incremental_save_writes_only_what_is_new(tmp_path):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners, only_new=True)
    d = tmp_path / D.key_id(_key())
    first = {e[3].name for e in D.entries(d)}
    mtimes = {f: f.stat().st_mtime_ns for _, _, _, f in D.entries(d)}
    again = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                   only_new=True)
    assert again["saved"] == 0 and again["kept"] == 3
    pc.insert("m", [12, 2, 3, 4, 5, 6, 7, 8], _kv(), "user",
              owner=_own("s2"))
    third = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                   only_new=True)
    assert third["saved"] == 1
    es = D.entries(d)
    # nothing renamed or rewritten; the new one in a later save
    assert first <= {e[3].name for e in es}
    assert all(f.stat().st_mtime_ns == t for f, t in mtimes.items())
    assert es[-1][0] == max(e[0] for e in es) > es[0][0]
    # restore order: oldest save first, the new entry last
    from mlx_lm.models.cache import LRUPromptCache
    lru = LRUPromptCache(max_size=10)
    back = D.restore(lru, "m", _key(), base=tmp_path)
    assert list(back)[-1] == (12, 2, 3, 4, 5, 6, 7, 8)


def test_break_even_skips_an_entry_quicker_to_recompute(tmp_path):
    """Recompute (tokens / prefill rate) under read-back (bytes /
    READ_BPS): not written, counted not_worth. No rate yet: written."""
    pc = _pc()
    e = pc.lru._trie.get("m", [1, 2, 3, 4, 5, 6, 7, 8])
    read_s = e.nbytes / D.READ_BPS
    fast = 8 / read_s * 10          # recompute 10x quicker than the read
    slow = 8 / read_s / 10
    assert D.not_worth(8, e.nbytes, fast)
    assert not D.not_worth(8, e.nbytes, slow)
    assert not D.not_worth(8, e.nbytes, None)
    got = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                 prefill_tps=fast)
    assert got["not_worth"] == 3 and got["saved"] == 0
    assert D._all_files(tmp_path) == []
    got = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                 prefill_tps=slow)
    assert got["not_worth"] == 0 and got["saved"] == 3


def test_select_saves_only_the_named_entries(tmp_path):
    pc = _pc()
    got = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                 only_new=True, select={(9, 2, 3, 4, 5, 6, 7, 8)})
    assert got["saved"] == 1 and got["entries"] == 1


def test_a_reinserted_entry_is_written_again_and_its_old_file_goes(tmp_path):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners, only_new=True)
    pc.insert("m", [10, 2, 3, 4, 5, 6, 7, 8], _kv(), "user",
              owner=_own("s3"))           # another session made it now
    got = D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners,
                 only_new=True)
    assert got["saved"] == 1
    owners = [D._header(f)["owner"]["session"]
              for f, _, _ in D._all_files(tmp_path)]
    assert sorted(owners) == ["s1", "s1", "s3"]


def test_restore_puts_the_owners_back(tmp_path):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners)
    D.set_pin("s2", True, tmp_path)
    fresh = PromptCache(10)
    back = D.restore(fresh.lru, "m", _key(), base=tmp_path)
    D.adopt(fresh.owners, fresh.pinned, back, base=tmp_path)
    assert {m["session"] for m in fresh.owners.values()} == {"s1", "s2"}
    assert fresh.pinned == {"s2"}
    assert all(m["file"] for m in fresh.owners.values())


# --- pins and the sweep ----------------------------------------------------------

def test_the_sweep_spares_pinned_files_from_ttl_and_budget(tmp_path,
                                                          monkeypatch):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners)
    D.set_pin("s1", True, tmp_path)
    assert D.pins(tmp_path) == {"s1": True}
    monkeypatch.setenv(D.ENV_TTL, "1")
    out = D.sweep(tmp_path, now=time.time() + 7200)
    assert out["ttl"] == 1 and out["pinned"] == 2
    assert {D._header(f)["owner"]["session"]
            for f, _, _ in D._all_files(tmp_path)} == {"s1"}
    # the budget counts unpinned files only: a zero budget keeps them
    monkeypatch.setenv(D.ENV_TTL, "24")
    monkeypatch.setenv(D.ENV_GB, "0")
    assert D.sweep(tmp_path)["budget"] == 0
    assert len(D._all_files(tmp_path)) == 2
    # unpinned, they are swept like any other
    D.set_pin("s1", False, tmp_path)
    assert D.sweep(tmp_path)["budget"] == 2


def test_a_drop_forgets_the_pin(tmp_path):
    D.set_pin("s1", True, tmp_path)
    D.set_pin("s2", True, tmp_path)
    D.drop_files("s1", tmp_path)
    assert D.pins(tmp_path) == {"s2": True}


def test_a_pin_is_sticky_for_the_sessions_later_entries():
    pc = PromptCache(10)
    pc.set_pinned("s1", True)
    pc.insert("m", [1, 2, 3], _kv(3), "user", owner=_own("s1"))
    assert pc.owners[(1, 2, 3)]["pinned"] is True
    pc.set_pinned("s1", False)
    assert pc.owners[(1, 2, 3)]["pinned"] is False


# --- the read-back index ----------------------------------------------------------

def test_the_index_holds_each_files_tokens_without_reading_arrays(tmp_path):
    pc = _pc()
    D.save(pc.lru, _key(), base=tmp_path, owners=pc.owners)
    idx = D.index(tmp_path / D.key_id(_key()))
    assert set(idx) == {(1, 2, 3, 4, 5, 6, 7, 8), (9, 2, 3, 4, 5, 6, 7, 8),
                        (10, 2, 3, 4, 5, 6, 7, 8)}
    assert idx[(10, 2, 3, 4, 5, 6, 7, 8)]["owner"]["session"] == "s2"
    f = idx[(1, 2, 3, 4, 5, 6, 7, 8)]["file"]
    assert D.tokens_of(f) == [1, 2, 3, 4, 5, 6, 7, 8]


# --- a ring's followers ------------------------------------------------------------

class _Journal:
    def __init__(self):
        self.ops = []

    def add(self, op, **kw):
        self.ops.append(dict(op=op, **kw))


def test_rank_0_journals_owner_drop_and_pin_and_a_follower_applies_them(
        tmp_path, monkeypatch):
    """Rank 0's JournalPromptCache records each change; a follower stub
    applies the ops (tensor.apply_cache_op) to its own cache and its own
    files, and ends with the same side map."""
    from knurlogic.engine.runtime.tensor import (
        JournalPromptCache,
        apply_cache_op,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    j = _Journal()
    r0 = JournalPromptCache(PromptCache(10), j)
    follower = PromptCache(10)
    saves = []
    last = {}
    for uid, (toks, s) in enumerate([([1, 2, 3, 4], "s1"),
                                     ([5, 2, 3, 4], "s2")]):
        c = _kv(4)
        r0.insert("m", toks, c, "assistant", origin=("finished", uid),
                  owner=_own(s))
        last[("finished", uid)] = (toks, _kv(4))
    r0.set_pinned("s2", True)
    D.save(follower.lru, _key(), owners=follower.owners)   # nothing yet
    for op in j.ops:
        assert apply_cache_op(op, follower, "m", last,
                              lambda only_new=False, select=None:
                              saves.append((only_new, select)))
    assert follower.owners.keys() == r0.owners.keys()
    assert follower.pinned == {"s2"} and D.pins() == {"s2": True}
    # the follower's own file of s1, then a drop op removes it too
    D.save(follower.lru, _key(), owners=follower.owners)
    j.ops.clear()
    assert r0.drop("s1") == 1
    assert [o["op"] for o in j.ops] == ["drop"]
    for op in j.ops + [{"op": "save_cache", "session": "s2"},
                       {"op": "save_cache"}]:
        apply_cache_op(op, follower, "m", last,
                       lambda only_new=False, select=None:
                       saves.append((only_new, select)))
    assert follower.of_session("s1") == [] and len(follower.lru) == 1
    assert [D._header(f)["owner"]["session"]
            for f, _, _ in D._all_files(D.root())] == ["s2"]
    assert saves == [(True, {(5, 2, 3, 4)}), (False, None)]
    assert not apply_cache_op({"op": "set"}, follower, "m", last, None)


# --- through the scheduler ---------------------------------------------------------

class Host:
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
def sched():
    from test_batch_drafting import _tiny

    from knurlogic.engine.runtime.scheduler import Scheduler
    model, _head, prompts = _tiny(512)
    s = Scheduler(Host(model, Tok(prompts)), prefill_step_size=16).start()
    s.prompts = prompts
    yield s


def test_an_entry_on_disk_is_read_back_on_demand(sched):
    """A session's entries saved, then freed from memory: its next request
    reads the longest on-disk prefix back first, and usage says disk."""
    s = sched
    s.load("/nonexistent/tiny-rb").done.wait(60)
    p = s.prompts[2]
    _collect(s.submit(_job(p, max_tokens=4, session="a1")))
    _collect(s.submit(_job(s.prompts[0], max_tokens=3)))      # anonymous
    cmd = s.save_prompt_cache()
    assert cmd.done.wait(60) and not cmd.error
    assert cmd.result["entries"] >= 1 and cmd.result["saved"] >= 1
    assert all(m["session"] == "a1" for m in s.cache.owners.values())
    # out of memory, on disk: the index has it
    for m, t in s.cache.of_session("a1"):
        s.cache.remove(m, t)
    assert s._disk_index
    _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4,
                                     session="a1")))
    c = warm["knurlogic"]["cache"]
    assert c["used"] >= len(p) - 1 and c["disk"]["tokens"] == c["used"]
    assert c["disk"]["read_ms"] > 0


def test_drop_pin_and_list_commands(sched):
    s = sched
    s.load("/nonexistent/tiny-cmd").done.wait(60)
    _collect(s.submit(_job(s.prompts[1], max_tokens=3, session="b1",
                           pin=True)))
    assert "b1" in s.cache.pinned and D.pins().get("b1") is True
    ls = s.list_prompt_cache()
    assert ls.done.wait(60) and not ls.error
    mine = [e for e in ls.result["entries"] if e["session"] == "b1"]
    assert mine and all(e["pinned"] and e["in_memory"] for e in mine)
    assert ls.result["model"] == "tiny-cmd" and ls.result["key_id"]
    s.save_prompt_cache().done.wait(60)
    un = s.pin_prompt_cache("b1", False)
    assert un.done.wait(60) and un.result["pinned"] is False
    assert D.pins().get("b1") is False
    dr = s.drop_prompt_cache("b1")
    assert dr.done.wait(60) and not dr.error
    assert dr.result["memory"] >= 1 and dr.result["disk"] >= 1
    assert s.cache.of_session("b1") == [] and "b1" not in D.pins()


def test_a_sessions_save_writes_its_longest_entry_only(sched):
    s = sched
    s.load("/nonexistent/tiny-ss").done.wait(60)
    s._prefill_tps = None           # no break-even skip here
    _collect(s.submit(_job(s.prompts[1], max_tokens=3, session="d1")))
    _collect(s.submit(_job(s.prompts[0], max_tokens=3, session="d2")))
    mine = [tuple(t) for _, t in s.cache.of_session("d1")]
    cmd = s.save_prompt_cache("d1")
    assert cmd.done.wait(60) and not cmd.error
    assert cmd.result["saved"] == 1 and cmd.result["entries"] == 1
    (f,) = [r for r in D.list_disk()]
    assert f["session"] == "d1" and f["tokens"] == max(len(t) for t in mine)
    again = s.save_prompt_cache("d1")
    assert again.done.wait(60) and again.result["saved"] == 0
    assert again.result["kept"] == 1
    none = s.save_prompt_cache("nobody")
    assert none.done.wait(60) and none.result["entries"] == 0


def test_a_pin_never_moves_a_session_out_of_memory(sched):
    """Pin means never deleted automatically, nothing else: when a session
    leaves memory is the client's call (POST .../park), never a clock's."""
    s = sched
    s.load("/nonexistent/tiny-pin-stays").done.wait(60)
    s._prefill_tps = None
    _collect(s.submit(_job(s.prompts[1], max_tokens=3, session="c1",
                           pin=True)))
    assert "c1" in s.cache.pinned
    for _ in range(5):
        s._wake.set()
        time.sleep(0.05)
    assert s.cache.of_session("c1")


def _chat(system, user, **kw):
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime.scheduler import Job
    say = lambda ids: " ".join(str(i) for i in ids)     # noqa: E731
    return Job(P.ChatRequest(messages=[
        {"role": "system", "content": say(system)},
        {"role": "user", "content": say(user)}]), P.PromptArgs(), **kw)


def test_keep_latest_replaces_the_sessions_earlier_steps(sched):
    """X-Cache-Keep: latest (the harness: a worker only appends, so only its
    latest step is ever reused): each step's entries replace the
    session's earlier ones in memory and on disk; the system prompt's
    checkpoint is nobody's, one copy every session shares."""
    s = sched
    s.load("/nonexistent/tiny-latest").done.wait(60)
    s._prefill_tps = None
    sys_p = list(s.prompts[0][:20])
    a = list(s.prompts[1])
    b = a + list(s.prompts[2])                  # the worker appended
    _collect(s.submit(_chat(sys_p, a, max_tokens=3, session="k1",
                            keep_latest=True)))
    first = {tuple(t) for _, t in s.cache.of_session("k1")}
    assert first
    cmd = s.save_prompt_cache()                 # the first step on disk
    assert cmd.done.wait(60) and cmd.result["saved"] == len(first) + 1  # + shared
    assert {f["session"] for f in D.list_disk()
            if f["model"] == "tiny-latest"} == {"k1", None}    # None: shared
    _collect(s.submit(_chat(sys_p, b, max_tokens=3, session="k1",
                            keep_latest=True)))
    now = {tuple(t) for _, t in s.cache.of_session("k1")}
    assert now and not (now & first)            # replaced, not stacked
    assert len(now) <= 2                        # its checkpoint + answer
    assert not [f for f in D.list_disk() if f["session"] == "k1"]
    shared = [t for _, t, _, m in s.cache.live()
              if m is None and list(t) == sys_p]
    assert len(shared) == 1                     # the system prompt's, unowned
    # a session that does not ask keeps every step, as before
    _collect(s.submit(_chat(sys_p, a, max_tokens=3, session="k2")))
    _collect(s.submit(_chat(sys_p, b, max_tokens=3, session="k2")))
    assert len(s.cache.of_session("k2")) > 2


def test_the_shared_system_checkpoint_is_saved_and_restored(sched):
    """Nobody's, yet kept across a reload: the first worker after it
    starts from the system prompt and tools, not from nothing."""
    s = sched
    s.load("/nonexistent/tiny-shared").done.wait(60)
    s._prefill_tps = None
    sys_p = list(s.prompts[0][:20])
    _collect(s.submit(_chat(sys_p, list(s.prompts[1]), max_tokens=3,
                            session="w1", keep_latest=True)))
    assert tuple(sys_p) in s.cache.shared
    cmd = s.save_prompt_cache()
    assert cmd.done.wait(60) and not cmd.error
    on = [f for f in D.list_disk()
          if f["model"] == "tiny-shared" and f["tokens"] == len(sys_p)]
    assert len(on) == 1 and on[0]["session"] is None
    s.unload().done.wait(60)
    s.load("/nonexistent/tiny-shared").done.wait(60)
    assert tuple(sys_p) in s.cache.shared       # restored as shared
    assert s.cache.hit_length(s.host.model_key, sys_p + [1, 2]) >= len(sys_p)


def test_a_session_parked_on_request_is_read_back(sched):
    """POST .../park (the harness parks its PM while sub-agents work): saved,
    out of memory, the registry says disk only, and the next request reads
    it back from disk."""
    s = sched
    s.load("/nonexistent/tiny-park-req").done.wait(60)
    s._prefill_tps = None
    p = s.prompts[2]
    _collect(s.submit(_job(p, max_tokens=4, session="pm")))
    _collect(s.submit(_job(s.prompts[0], max_tokens=3, session="w1")))
    cmd = s.park_prompt_cache("pm")
    assert cmd.done.wait(60) and not cmd.error, cmd.error
    r = cmd.result
    assert r["session"] == "pm" and r["freed"] >= 1 and r["in_memory"] == 0
    assert r["saved"] >= 1 and r["bytes"] > 0
    assert s.cache.of_session("pm") == [] and s.cache.of_session("w1")
    lst = s.list_prompt_cache()
    assert lst.done.wait(60)
    assert not [e for e in lst.result["entries"] if e["session"] == "pm"]
    on = [f for f in D.list_disk()
          if f["model"] == "tiny-park-req" and f["session"] == "pm"]
    assert on
    _, warm = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4,
                                     session="pm")))
    assert warm["knurlogic"]["cache"]["disk"]["tokens"] >= len(p) - 1
    again = s.park_prompt_cache("nobody")
    assert again.done.wait(60) and again.result["freed"] == 0


def test_entries_no_session_owns_can_be_dropped(sched):
    """the harness's eval batches leave session-less entries (calls with no
    session, shared system-prompt copies) that drop {"session"} cannot
    name: drop {"sessionless": true} clears them, memory and disk; with an
    age, only files unused that long."""
    s = sched
    s.load("/nonexistent/tiny-sessionless").done.wait(60)
    s._prefill_tps = None
    sys_p = list(s.prompts[0][:20])
    _collect(s.submit(_chat(sys_p, list(s.prompts[1]), max_tokens=3,
                            session="e1", keep_latest=True)))
    _collect(s.submit(_job(s.prompts[2], max_tokens=3)))        # nobody's
    assert s.save_prompt_cache().done.wait(60)
    mine = lambda: [f for f in D.list_disk()                   # noqa: E731
                    if f["model"] == "tiny-sessionless"]
    assert any(f["session"] is None for f in mine())           # the shared
    young = s.drop_sessionless(older_than_s=3600)
    assert young.done.wait(60) and young.result["disk"] == 0
    cmd = s.drop_sessionless()
    assert cmd.done.wait(60) and not cmd.error
    assert cmd.result["memory"] >= 2 and cmd.result["disk"] >= 1
    assert not [f for f in mine() if f["session"] is None]
    assert [f for f in mine() if f["session"] == "e1"]          # untouched
    assert s.cache.of_session("e1") and not s.cache.shared


def test_a_follower_drops_the_same_files_by_name(tmp_path):
    """No half entries on a split model: rank 0's drop by age names the
    files it deleted, and each other rank deletes its own of those names
    -- nothing else, and nothing outside its key directory."""
    from knurlogic.engine.runtime.tensor import apply_cache_op
    d = tmp_path / "key"
    d.mkdir()
    a, b = d / "00000001-000000-aaa.safetensors", \
        d / "00000001-000001-bbb.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    (tmp_path / "outside.safetensors").write_bytes(b"x")
    op = {"op": "drop_files", "names": [a.name, "../outside.safetensors"]}
    assert apply_cache_op(op, None, "m", {}, None, d)
    assert not a.exists() and b.exists()
    assert (tmp_path / "outside.safetensors").exists()


def test_a_rings_prompt_cache_has_what_the_scheduler_reads():
    """A split model's cache is the journaled wrapper: every attribute the
    scheduler reads on `self.cache` must be there (cache.shared was not,
    and a 397B over two Macs failed at its load's restore)."""
    import re
    from pathlib import Path

    from knurlogic.engine.runtime import scheduler as SC
    from knurlogic.engine.runtime.tensor import JournalPromptCache
    src = Path(SC.__file__).read_text()
    used = set(re.findall(r"self\.cache\.(\w+)", src))
    # park is refused on a ring before it would reach .remove
    missing = {a for a in used - {"remove"}
               if not hasattr(JournalPromptCache, a)}
    assert not missing, missing


def test_every_journaled_cache_op_passes_the_plan_check():
    """A split model's plan is checked on every rank; the cache ops added
    for sessions were never in its schema, so the 397B over two Macs died
    on its first request that named a session (insert ... 'owner')."""
    import re
    from pathlib import Path

    from knurlogic.engine.runtime import plan as P
    from knurlogic.engine.runtime import scheduler as SC
    from knurlogic.engine.runtime import tensor as T
    src = Path(SC.__file__).read_text() + Path(T.__file__).read_text()
    added = set(re.findall(r'journal\.add\(\s*"(\w+)"', src))
    assert added and not added - set(P.OPS), added - set(P.OPS)
    ops = [{"op": "insert", "uid": 1, "event": "finished", "kind": "assistant",
            "owner": {"session": "pm", "role": None, "run": None,
                      "step": 1, "latest": True}},
           {"op": "insert", "uid": 2, "event": "checkpoint", "kind": "system"},
           {"op": "save_cache", "session": "pm"}, {"op": "save_cache"},
           {"op": "drop", "session": "pm"},
           {"op": "pin", "session": "pm", "pinned": True},
           {"op": "drop_sessionless"},
           {"op": "drop_files", "names": ["00000001-000000-abc.safetensors"]}]
    assert P.decode(P.encode({"ops": ops}))["ops"] == ops
    for bad in ({"op": "insert", "uid": 1, "event": "finished",
                 "kind": "assistant", "who": 1},
                {"op": "drop", "session": ""},
                {"op": "pin", "session": "a", "pinned": "yes"}):
        with pytest.raises(P.PlanError):
            P.check({"ops": [bad]})
