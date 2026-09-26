"""The scheduler on the tiny model: requests in, deltas and usage out, on
its own thread. A stand-in tokenizer renders messages as token ids, so the
test is about scheduling, not templates."""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    s = Scheduler(Host(None, Tok({})), memory_limit_bytes=100 * GIB)
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


def test_out_of_memory_is_a_503_to_retry():
    from knurlogic.engine.runtime.scheduler import OutOfMemory
    from knurlogic.interfaces.http.openai import _status_of
    e = _status_of(OutOfMemory("stopped"))
    assert e.status == 503 and e.code == "insufficient_memory"
