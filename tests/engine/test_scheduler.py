"""The scheduler on the tiny model: requests in, deltas and usage out, on
its own thread. A stand-in tokenizer renders messages as token ids, so the
test is about scheduling, not templates."""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

mx = pytest.importorskip("mlx.core")

from test_batch_drafting import _tiny  # noqa: E402

EOT = 3


class Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens, self.text, self.offset = [], "", 0

    def add_token(self, t):
        self.tokens.append(t)
        self.text += f"<{t}>"

    def finalize(self):
        pass

    @property
    def last_segment(self):
        seg, self.offset = self.text[self.offset:], len(self.text)
        return seg


class Tok:
    """Messages are strings of space-separated ids; the 'template' joins
    them. 'FAIL' in a message cannot render."""
    eos_token_ids = [EOT]
    has_thinking = False
    has_tool_calling = False
    has_chat_template = True
    tool_parser = None

    def __init__(self, prompts):
        self.prompts = prompts

    @property
    def detokenizer(self):
        return Detok()

    def convert_ids_to_tokens(self, t):
        return f"<{t}>"

    def encode(self, s):
        return [int(x) for x in s.split()]

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True, **kw):
        out = []
        for m in messages:
            if "FAIL" in m["content"]:
                raise ValueError("cannot render")
            out += self.encode(m["content"])
        return out


class Host:
    state = "ready"
    error = ""

    def __init__(self, model, tok):
        self.model, self.tokenizer = model, tok
        self.model_key = ("tiny", None, None)

    def expect(self, path):
        pass


@pytest.fixture(scope="module")
def sched():
    """One scheduler for the module, never stopped: the test owns the tiny
    model, and arrays the scheduler thread made inside it (rope tables,
    caches) must not be freed after that thread has ended -- MLX segfaults.
    A real server's stop() unloads on the thread instead."""
    from knurlogic.engine.serve import state
    from knurlogic.engine.runtime.scheduler import Scheduler
    model, head, prompts = _tiny(512)
    # everything evaluated here: MLX streams are per thread, and a lazy
    # array made on this one cannot be evaluated on the scheduler's
    import mlx.nn as nn
    mx.eval([v if isinstance(v, mx.array) else v.parameters()
             for v in vars(head).values()
             if isinstance(v, (mx.array, nn.Module))])
    state.DRAFT.update(head=head, on=True)
    s = Scheduler(Host(model, Tok(prompts)), prefill_step_size=16).start()
    s.prompts = prompts
    yield s
    state.DRAFT.update(head=None, on=False)


def _job(ids, **kw):
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime.scheduler import Job
    msg = " ".join(str(i) for i in ids) if not isinstance(ids, str) else ids
    return Job(P.ChatRequest(messages=[{"role": "user", "content": msg}]),
               P.PromptArgs(), **kw)


def _collect(job, timeout=60):
    text, usage, end = "", None, time.time() + timeout
    while time.time() < end:
        kind, val = job.outbox.get(timeout=timeout)
        if kind == "delta":
            text += val.content
        elif kind == "done":
            return text, val
        elif kind == "error":
            return None, val
    raise TimeoutError


def test_requests_run_concurrently_and_report_usage(sched):
    jobs = [sched.submit(_job(p, max_tokens=12)) for p in sched.prompts]
    for j, p in zip(jobs, sched.prompts):
        text, usage = _collect(j)
        assert text and usage["prompt_tokens"] == len(p)
        assert usage["completion_tokens"] <= 12
        assert usage["total_tokens"] == len(p) + usage["completion_tokens"]


def test_a_seed_under_concurrent_load_equals_it_alone(sched):
    s = {"temp": 1.0, "seed": 11}
    alone, _ = _collect(sched.submit(_job(sched.prompts[0], max_tokens=16,
                                          sampling=s)))
    others = [sched.submit(_job(p, max_tokens=16, sampling={"temp": 1.0}))
              for p in sched.prompts[1:]]
    again = sched.submit(_job(sched.prompts[0], max_tokens=16, sampling=s))
    got, _ = _collect(again)
    for o in others:
        _collect(o)
    assert got == alone


def test_a_failing_request_beside_a_working_one(sched):
    bad = sched.submit(_job("FAIL"))
    good = sched.submit(_job(sched.prompts[1], max_tokens=6))
    _, err = _collect(bad)
    assert "render" in str(err)
    text, usage = _collect(good)
    assert text and usage["completion_tokens"] == 6


def test_a_stop_string_ends_the_answer_and_frees_the_row(sched):
    free, _ = _collect(sched.submit(_job(sched.prompts[0], max_tokens=10)))
    cut = free.index(">", free.index(">") + 1) + 1       # after 2 tokens
    stop = free[cut:cut + 4]
    text, usage = _collect(sched.submit(_job(sched.prompts[0], max_tokens=10,
                                             stops=[stop])))
    assert text == free[:cut] and stop not in text
    assert sched.width == 0


def test_a_cancelled_request_frees_its_row(sched):
    j = sched.submit(_job(sched.prompts[2], max_tokens=5000))
    kind, _ = j.outbox.get(timeout=60)
    j.cancel()
    for _ in range(100):
        if sched.width == 0:
            break
        time.sleep(0.05)
    assert sched.width == 0


def test_a_shared_prefix_is_reused_from_the_prompt_cache(sched):
    from knurlogic.engine.serve import cache_report
    p = sched.prompts[2]
    first = sched.submit(_job(p, max_tokens=4))
    _collect(first)
    second = sched.submit(_job(p + [5, 6, 7], max_tokens=4))
    _, usage = _collect(second)
    # the checkpoint at the prompt less its last token (fed as its own
    # segment) is the nearest entry
    assert usage["prompt_tokens_details"]["cached_tokens"] == len(p) - 1


def test_memory_past_the_limit_empties_the_prompt_cache_then_stops_the_newest():
    """A step that outgrows the working set aborts the process in Metal, so
    the scheduler gives up the prompt cache first, then the newest rows,
    each with an OutOfMemory -- and keeps new requests waiting."""
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime.scheduler import (GIB, Job, OutOfMemory,
                                                    Scheduler, _Row)

    class Cache:
        nbytes = 6 * GIB

        def trim_to(self, n):
            freed = self.nbytes - n
            self.nbytes = n
            mem["active"] -= freed

    class Ex:
        removed = []

        def remove(self, uids):
            self.removed += uids
            mem["active"] -= 5 * GIB * len(uids)

    mem = {"active": 110 * GIB}
    s = Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)
    s._spike = 4 * GIB          # measured: margin 5, limit 100
    s.cache, s._ex = Cache(), Ex()
    s._active = lambda: mem["active"]
    s._release = lambda: None
    jobs = {uid: Job(P.ChatRequest(), P.PromptArgs()) for uid in (1, 2, 3)}
    s._rows = {uid: _Row(j, None, []) for uid, j in jobs.items()}
    assert not s._room_to_admit()
    s._guard_memory()
    # 10 GiB over: the whole 6 GiB cache goes (over + a margin), then the
    # newest row; the older two keep running
    assert s.cache.nbytes == 0
    assert s._ex.removed == [3] and sorted(s._rows) == [1, 2]
    kind, err = jobs[3].outbox.get_nowait()
    assert kind == "error" and isinstance(err, OutOfMemory)
    assert jobs[1].outbox.empty() and jobs[2].outbox.empty()
    assert mem["active"] <= 100 * GIB


def test_the_guard_trims_the_cache_by_the_overage_not_a_margin_more():
    """The limit already leaves a step's margin: 10 GiB over with a 20 GiB
    cache gives up 10, not 10 + the margin."""
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler

    class Cache:
        nbytes = 20 * GIB

        def trim_to(self, n):
            mem["active"] -= self.nbytes - n
            self.nbytes = n

    mem = {"active": 105 * GIB}
    s = Scheduler(Host(None, Tok({})), working_set_bytes=100 * GIB)
    s._spike = 4 * GIB          # margin 5, limit 95
    s.cache = Cache()
    s._active = lambda: mem["active"]
    s._release = lambda: None
    s._guard_memory()
    assert s.cache.nbytes == 10 * GIB and mem["active"] == 95 * GIB


def test_out_of_memory_is_a_503_to_retry():
    from knurlogic.engine.runtime.scheduler import OutOfMemory
    from knurlogic.interfaces.http.openai import _status_of
    e = _status_of(OutOfMemory("stopped"))
    assert e.status == 503 and e.code == "insufficient_memory"


def test_a_prompt_that_would_not_fit_waits_or_is_refused_not_admitted():
    """A row prefills whole inside one step, so a long prompt must fit
    before it is admitted: the prompt cache gives way; if only one copy of
    its cache fits it goes in lean (no checkpoints); else it waits for
    running rows, or with none running it is refused."""
    import pytest
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB

    class Cache:
        nbytes = 4 * GIB

        def trim_to(self, n):
            mem["active"] -= self.nbytes - n
            self.nbytes = n

    mem = {"active": 95 * GIB}
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)
    s._spike = 4 * GIB          # measured: margin 5, limit 100
    s.cache = Cache()
    s._active = lambda: mem["active"]
    s._release = lambda: None

    def kv(n):   # 64 MiB of fixed state, 1 MiB per token
        return [type("KV", (), {"nbytes": 64 * 2**20 + n * 2**20})()]
    s._learn(list(range(512)), kv(512))
    s._learn(list(range(4096)), kv(4096))
    assert s._kv == (64 * 2**20, 2**20)          # fixed + slope, not a ratio
    # 5 GiB free under the limit; 1024 tokens x2 = ~2.1 GiB: full
    assert s._make_room(1024) == "full" and s.cache.nbytes == 4 * GIB
    # 3072 x2 = ~6.1 GiB: the cache gives up what it takes, still full
    assert s._make_room(3072) == "full" and s.cache.nbytes < 4 * GIB
    # 7000 tokens: twice (13.8) never fits, so nothing is evicted for it;
    # once (6.9) does with the cache giving up only what that takes: lean
    held = s.cache.nbytes
    assert s._make_room(7000) == "lean" and 0 < s.cache.nbytes < held
    # 16000 tokens do not fit at all: with a row running it waits ...
    s._rows = {1: S._Row(S.Job(P.ChatRequest(), P.PromptArgs()), None, [])}
    with pytest.raises(S._Wait):
        s._make_room(16000)
    # ... with none it is refused, and says why
    s._rows = {}
    with pytest.raises(S.OutOfMemory, match="16000 tokens"):
        s._make_room(16000)


def test_one_length_prices_a_hybrids_fixed_state_once():
    """Short runs alone: the recurrent layers (cannot trim) are fixed state,
    not bytes per token -- the ratio at one length priced a 24k prompt at
    4x its measured cache and refused it (GLM-5.3 Flash, drafting, M4)."""
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)

    class Rec:                                     # deltanet: 512 MiB
        nbytes = 512 * 2**20
        def is_trimmable(self): return False

    class KV:
        def __init__(self, n): self.nbytes = n * 2**10
        def is_trimmable(self): return True

    s._learn(list(range(1000)), [Rec(), KV(1000)])
    assert s._kv == (512 * 2**20, 2**10)
    assert s._cost(24000, 1) == 512 * 2**20 + 24000 * 2**10


def test_a_cache_that_could_never_make_room_is_not_evicted():
    """Two copies that would not fit with the prompt cache empty evict
    nothing: the lean admission that follows keeps the entries it (or the
    next request) could hit."""
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB

    class Cache:
        nbytes = 1 * GIB

        def trim_to(self, n):
            mem["active"] -= self.nbytes - n
            self.nbytes = n

    mem = {"active": 97 * GIB}
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)
    s._spike = 4 * GIB                             # limit 100: 3 GiB free
    s.cache = Cache()
    s._active = lambda: mem["active"]
    s._release = lambda: None
    s._kv = (0.0, float(2**20))                    # 1 MiB per token
    assert s._make_room(2500) == "lean"            # 2x5 > 3+1; 2.5 <= 3
    assert s.cache.nbytes == 1 * GIB


def test_the_step_margin_is_measured_not_published():
    """The largest spike measured, with a quarter again -- never below 5%
    of the working set: a spike learned on short contexts under-reads a
    longer one (GLM-5.3 aborted Metal at a 3.2 GiB learned margin)."""
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=120 * GIB)
    assert s._margin() == 6 * GIB                 # the floor: 5%
    peak = {"v": 0}
    s._active = lambda: 100 * GIB
    import mlx.core as mx
    real = mx.get_peak_memory
    try:
        mx.get_peak_memory = lambda: peak["v"]
        peak["v"] = 102 * GIB
        s._measure(100 * GIB)
        assert s._margin() == 6 * GIB              # 2.5 GiB is under it
        peak["v"] = 108 * GIB
        s._measure(100 * GIB)
        assert s._margin() == 10 * GIB             # 8 GiB spike x 1.25
        peak["v"] = 101 * GIB                      # a smaller one: kept max
        s._measure(100 * GIB)
        assert s._limit() == 110 * GIB
    finally:
        mx.get_peak_memory = real


def _rows_of(s, *contexts):
    """Running rows spanning these contexts (the margin reads only them)."""
    from knurlogic.engine.runtime import scheduler as S
    job = lambda n: type("J", (), {"prompt_tokens": n})()
    s._rows = {i: S._Row(job(n), None, []) for i, n in enumerate(contexts)}


def test_the_margin_follows_the_context_the_step_will_span():
    """The 27B on an M3 Ultra: the transient grew 1.58 -> 3.39 GiB as
    four agents' prompts grew, and the first step at a longer context than
    any measured ran past a margin the shorter ones had set."""
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=80 * GIB)
    s._tx = {"lo": (10_000, 1 * GIB), "hi": (50_000, 3 * GIB)}
    s._spike = 3 * GIB
    _rows_of(s, 50_000)
    assert s._margin() == 4 * GIB                 # 3.75 under the floor
    _rows_of(s, 60_000, 30_000)                  # 90k: 3 + 40k x 1/20k
    assert s._transient(90_000) == 5 * GIB
    assert s._margin() == int(6.25 * GIB)
    # admitting a prompt adds its tokens to the context the step spans
    assert s._limit(10_000) == 80 * GIB - int(5.5 * 1.25 * GIB)
    _rows_of(s, 20_000)                          # inside what was measured
    assert s._margin() == 4 * GIB


def test_one_context_measured_scales_the_transient_in_proportion():
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=200 * GIB)
    s._spike = 2 * GIB
    s._tx = {"lo": (40_000, 2 * GIB), "hi": (40_000, 2 * GIB)}
    assert s._transient(100_000) == 5 * GIB       # the safe side


def test_measure_records_the_transient_against_its_context():
    import mlx.core as mx
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=120 * GIB)
    s._active = lambda: 100 * GIB
    real = mx.get_peak_memory
    try:
        mx.get_peak_memory = lambda: 101 * GIB
        s._measure(100 * GIB, 20_000)
        mx.get_peak_memory = lambda: 103 * GIB
        s._measure(100 * GIB, 60_000)
        mx.get_peak_memory = lambda: int(100.5 * GIB)
        s._measure(100 * GIB, 70_000)            # a decode step: small
    finally:
        mx.get_peak_memory = real
    assert s._tx == {"lo": (20_000, GIB), "hi": (70_000, 3 * GIB)}
    assert s._spike == 3 * GIB


def test_other_processes_gpu_memory_comes_off_the_working_set():
    """iogpu.wired_limit_mb caps every process's GPU memory together."""
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    seen = {"v": 50 * GIB}
    s = Scheduler(Host(None, Tok({})), working_set_bytes=100 * GIB,
                  gpu_in_use=lambda: seen["v"])
    s._local_active = lambda: 45 * GIB
    s._cached = lambda: 2 * GIB
    assert s._limit() == 100 * GIB - 3 * GIB - 5 * GIB
    seen["v"] = 47 * GIB                         # they let go: the most
    s._others_at = 0.0                           # recent readings rule
    assert s._others_bytes() == 3 * GIB


def test_mlx_buffer_cache_is_cleared_before_a_step_it_would_crowd():
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=100 * GIB)
    freed = []
    s._active = s._local_active = lambda: 94 * GIB
    s._cached = lambda: 0 if freed else 3 * GIB
    s._release = lambda: freed.append(1)
    _rows_of(s, 1000)
    s._guard_memory()                            # 94 + 3 over a 95 limit
    assert freed == [1]


def test_an_admission_is_priced_as_the_copies_the_engine_makes():
    """Measured on the 27B (M3 Ultra; mlx active memory around each
    admission): 24.6k tokens segmented [8223, 16357, 1, 1] beside a running
    row grew memory 7.27 GiB; 41.0k tokens with a 24.6k hit, 7.96 alone
    and 10.65 beside a row. Priced as a row and one checkpoint they were
    3.4, 5.4 and 5.4."""
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=84 * GIB)
    s._kv = (153944064.0, 67190.3)                # as learned there
    row = S._Row(S.Job(P.ChatRequest(), P.PromptArgs()), None, [])
    cks = S._checkpoints([[0] * 8223, [0] * 16357, [0], [0]], 24582, 0)
    assert cks == [8223, 24580, 24581]
    s._rows = {1: row}
    assert abs(s._need(24582, cks) / GIB - 7.27) < 0.15
    cks = S._checkpoints([[0] * 24581, [0] * 16446, [0], [0]], 41029, 24581)
    assert cks == [41027, 41028]                  # none inside the hit
    assert abs(s._need(41029, cks) / GIB - 10.65) < 0.25
    s._rows = {}
    assert abs(s._need(41029, cks) / GIB - 7.96) < 0.25


def test_a_hit_inside_a_segment_keeps_that_segments_checkpoint():
    from knurlogic.engine.runtime.scheduler import _checkpoints
    segs = [[0] * 100, [0] * 100, [0] * 100]
    assert _checkpoints(segs, 300, 0) == [100, 200]
    assert _checkpoints(segs, 300, 150) == [200]
    assert _checkpoints(segs, 300, 200) == []


def test_397b_on_the_m4_admits_the_prompts_it_refused():
    """An M4 Max: 397B at 106.9 GiB active, a 5.2 GiB measured
    spike, and 31k/20k-token prompts needing ~1 GiB each were refused with
    nothing else running -- the margin was held back twice."""
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=120 * GIB)
    s._spike = int(5.2 * GIB)                    # margin 6.5, limit 113.5
    s._active = lambda: int(106.9 * GIB)
    s._release = lambda: None
    s.cache = type("C", (), {"nbytes": 0})()
    s._kv = (0.0, 1.1 * GIB / 31222)             # as measured: 1.1 GiB
    assert s._room_to_admit()
    assert s._make_room(31222) in ("full", "lean")
    assert s._make_room(20577) == "full"


def test_what_was_measured_goes_with_the_model():
    """A 35B's slope must not cost a 397B's prompt."""
    from knurlogic.engine.runtime.scheduler import Command, Scheduler

    class H:
        state, path, error = "ready", "/m/a", ""

        def load(self, p, **k):
            self.path = p
    s = Scheduler(H())
    s._kv, s._samples, s._spike = (1.0, 2.0), {"lo": (1, 1)}, 5 << 30
    c = Command("load", "/m/b")
    s._commands.put(c)
    s._do_commands()
    assert c.error == "" and s._samples == {} and s._spike == 0
    assert s._kv is None          # /m/b has no config to seed from


def test_the_spike_is_what_the_step_did_not_keep():
    """An admission's KV is growth; only the peak above where the step
    ENDED is transient."""
    import mlx.core as mx
    from knurlogic.engine.runtime.scheduler import GIB, Scheduler
    s = Scheduler(Host(None, Tok({})), working_set_bytes=120 * GIB)
    s._active = lambda: int(101.5 * GIB)
    real = mx.get_peak_memory
    try:
        mx.get_peak_memory = lambda: 102 * GIB
        s._measure(100 * GIB)
    finally:
        mx.get_peak_memory = real
    assert s._spike == GIB // 2


def test_a_waiting_prompt_neither_empties_the_cache_nor_blocks_the_line():
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB

    class Cache:
        nbytes = 2 * GIB

        def trim_to(self, n):
            raise AssertionError("a waiting prompt trimmed the cache")
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)
    s._spike = 4 * GIB                            # limit 100
    s._active = lambda: 95 * GIB
    s.cache = Cache()
    s._kv = (0.0, float(2**20))                   # 1 MiB per token
    big = S.Job(P.ChatRequest(), P.PromptArgs())
    big.prompt_tokens = 20000                     # ~19.5 GiB: cannot fit
    small = S.Job(P.ChatRequest(), P.PromptArgs())
    s._rows = {1: S._Row(S.Job(P.ChatRequest(), P.PromptArgs()), None, [])}
    admitted = []
    s._insert = lambda job: admitted.append(job)
    s._waiting = [big, small]
    s._admit_waiting()
    assert admitted == [small] and s._waiting == [big]


def test_a_held_prompt_waits_for_the_rows_it_found_not_for_newcomers():
    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime import scheduler as S
    GIB = S.GIB
    s = S.Scheduler(Host(None, Tok({})), working_set_bytes=105 * GIB)
    s._spike = 4 * GIB                            # limit 100
    s._active = lambda: 95 * GIB
    s.cache = type("C", (), {"nbytes": 0})()
    s._kv = (0.0, float(2**20))
    big = S.Job(P.ChatRequest(), P.PromptArgs())
    big.prompt_tokens = 20000
    s._insert = lambda job: None
    row = lambda: S._Row(S.Job(P.ChatRequest(), P.PromptArgs()), None, [])
    s._rows = {1: row()}
    s._waiting = [big]
    s._admit_waiting()
    assert s._waiting == [big] and big.waiting_on == {1}
    s._rows = {2: row()}                          # 1 finished, 2 arrived
    s._waiting = [big]
    s._admit_waiting()
    kind, err = big.outbox.get_nowait()
    assert kind == "error" and isinstance(err, S.OutOfMemory)
    assert s._waiting == []


def test_usage_says_what_the_request_took(sched):
    """TTFT, prefill and decode rates, measured where the steps run."""
    j = sched.submit(_job(sched.prompts[0], max_tokens=12))
    _text, usage = _collect(j)
    t = usage["knurlogic"]["timing"]
    assert t["ttft_s"] >= t["queue_s"] >= 0
    if usage["completion_tokens"] > 1:
        assert t["decode_tok_s"] > 0


def test_the_context_length_caps_a_request_live(sched, monkeypatch):
    """KNURLOGIC_CONTEXT_LENGTH is a cap, read at each admission: a prompt
    at or over it is refused, max_tokens is trimmed to fit under it."""
    p = sched.prompts[0]
    monkeypatch.setenv("KNURLOGIC_CONTEXT_LENGTH", str(len(p)))
    kind, err = sched.submit(_job(p, max_tokens=12)).outbox.get(timeout=60)
    assert kind == "error" and "context length" in str(err)
    monkeypatch.setenv("KNURLOGIC_CONTEXT_LENGTH", str(len(p) + 3))
    _text, usage = _collect(sched.submit(_job(p, max_tokens=12)))
    assert usage["completion_tokens"] <= 3
