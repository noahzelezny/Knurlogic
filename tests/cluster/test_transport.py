"""cluster/transport.answer: a reply, or None when the peer gave none --
unreachable, of another protocol, or its handler failed or refused."""
from knurlogic.cluster import protocol as P
from knurlogic.cluster import transport


def test_an_answer_or_none(monkeypatch):
    def post(page, kind, doc):
        return replies[page]
    replies = {"ok": {"ranks_here": [1]},
               # a Failure envelope, as decode_reply returns it
               "failed": {"error": "this machine's handler failed",
                          "failure_kind": "failure"}}
    assert transport.answer("ok", "JobState", {"job": "j"}, post) == \
        {"ranks_here": [1]}
    assert transport.answer("failed", "JobState", {}, post) is None

    def down(page, kind, doc):
        raise transport.PeerUnreachable("timed out")

    def other(page, kind, doc):
        raise P.VersionMismatch("another protocol")
    assert transport.answer("p", "JobState", {}, down) is None
    assert transport.answer("p", "JobState", {}, other) is None
