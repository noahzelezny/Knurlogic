"""A running prefill's progress goes out as SSE keepalives, so a client
that hung up mid-prefill is noticed by the failed write (and its job
cancelled) instead of the prefill running on for nobody."""
from knurlogic.engine.runtime import prompt as P
from knurlogic.engine.runtime.scheduler import Job
from knurlogic.interfaces.http.openai import Reply


def test_prefill_progress_is_a_keepalive_per_chunk():
    job = Job(P.ChatRequest(messages=[]), P.PromptArgs())
    for d in (2048, 4096):
        job.outbox.put(("progress", (d, 137000)))
    job.outbox.put(("done", {}))
    r = Reply(job, {"chat": False, "model": "m", "include_usage": False,
                    "exclude": False, "stream": True})
    out = list(r.events(job.outbox.get()))
    assert out[:2] == [b": keepalive 2048/137000\n\n",
                       b": keepalive 4096/137000\n\n"]
    assert out[-1] == b"data: [DONE]\n\n"
