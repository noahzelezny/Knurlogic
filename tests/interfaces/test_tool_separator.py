"""The whitespace a model writes before its tool block (DeepSeek-V4's
"\n\n") is not shown as text: a reply that is only a tool call carries no
text part, streamed or not; real text before a call, or a reply with no
call, keeps its whitespace."""
import json
import queue
from types import SimpleNamespace

from knurlogic.engine.runtime.request import Delta
from knurlogic.interfaces.http.openai import Reply

CALL = {"id": "c1", "index": 0, "type": "function",
        "function": {"name": "write_file", "arguments": "{}"}}


def _reply(*deltas):
    q = queue.Queue()
    for d in deltas:
        q.put(("delta", d))
    q.put(("done", {}))
    ctx = {"chat": True, "exclude": False, "model": "m", "stream": True,
           "include_usage": False, "applied": []}
    return Reply(SimpleNamespace(outbox=q), ctx)


def _streamed(r):
    out = []
    for chunk in r.events(r.first()):
        line = chunk.decode()
        if line.startswith("data: {"):
            out.append(json.loads(line[6:])["choices"][0]["delta"])
    return out


def test_only_a_tool_call_carries_no_text():
    r = _reply(Delta(content="\n\n"), Delta(tool_calls=[CALL],
                                            finish="tool_calls"))
    msg = r.complete(r.first())["choices"][0]["message"]
    assert msg["content"] == "" and msg["tool_calls"]
    r = _reply(Delta(content="\n"), Delta(content="\n"),
               Delta(tool_calls=[CALL], finish="tool_calls"))
    assert [d for d in _streamed(r) if d.get("content")] == []


def test_real_text_and_text_without_a_call_keep_their_whitespace():
    r = _reply(Delta(content="Writing it.\n\n"),
               Delta(tool_calls=[CALL], finish="tool_calls"))
    assert "".join(d.get("content", "") for d in _streamed(r)) == \
        "Writing it.\n\n"
    r = _reply(Delta(content="a"), Delta(content="\n\n"), Delta(content="b"),
               Delta(finish="stop"))
    assert "".join(d.get("content", "") for d in _streamed(r)) == "a\n\nb"
    r = _reply(Delta(content="a"), Delta(content="\n"), Delta(finish="stop"))
    assert "".join(d.get("content", "") for d in _streamed(r)) == "a\n"
