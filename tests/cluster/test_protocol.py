import pytest

from knurlogic.cluster import protocol as P

SAMPLES = {
    "Hello": P.Hello(id="n1", name="A", build="abc", boot_id="b"),
    "Heartbeat": P.Heartbeat(boot_id="b", jobs=[{"job": "ab", "phase": "ready"}]),
    "Survey": P.Survey(),
    "Residency": P.Residency(resident=[{"port": 1}]),
    "Prepare": P.Prepare(job="ab", rank=2, world=3, split="pipeline",
                         link="ring", hosts=["a:1"], nodes=[{"id": "x"}]),
    "PrepareReply": P.PrepareReply(ok=False, refused="no", free_port=9),
    "Start": P.Start(job="ab"),
    "Started": P.Started(job="ab", rank=1, pid=5, log="/l"),
    "RankStatus": P.RankStatus(job="ab", rank=0, phase="ready", step=3),
    "JobState": P.JobState(job="ab", ranks_here=[0], ended="x"),
    "Stop": P.Stop(job="ab", reason="r", failure_kind="machine"),
    "Stopped": P.Stopped(job="ab", killed=[1], exiting=[2], reason="r"),
    "Load": P.Load(identity="i", port=8, sets={"a": 1}),
    "Unload": P.Unload(port=8),
    "Failure": P.Failure(kind="memory", reason="oom", node="A"),
    "Shape": P.Shape(identity="i", world=2, split="tensor"),
    "MachineSet": P.MachineSet(allowance_gib=4.0, settings={"k": 1}),
    "Read": P.Read(path="/settings.json", query={"tune": "x"}, port=8),
    "Settings": P.Settings(port=8, values={"k": 1}),
}


def test_every_kind_has_a_sample():
    assert set(SAMPLES) == set(P.kinds())


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_round_trip(name):
    body = SAMPLES[name]
    assert type(body).from_wire(body.to_wire()) == body
    m = P.message(body, sender="n1", job="ab")
    assert P.Message.from_wire(m.to_wire()) == m
    assert m.to_wire()["kind"] == name


def test_unknown_field_ignored_and_missing_required_plain():
    assert P.Start.from_wire({"job": "ab", "zzz": 1}) == P.Start(job="ab")
    with pytest.raises(P.ProtocolError, match="job is missing"):
        P.Start.from_wire({})
    with pytest.raises(P.ProtocolError, match="unknown message kind"):
        P.Message.from_wire({"v": [1, 0], "kind": "Nope", "body": {}})


def test_failure_kind_validated():
    with pytest.raises(P.ProtocolError):
        P.Failure(kind="weird", reason="x")


def test_reply_wire_keys_are_todays():
    assert P.Started(job="ab", rank=1, pid=5).to_wire() == {
        "started": "ab", "rank": 1, "pid": 5}
    assert P.Stopped(job="ab").to_wire()["stopped"] == "ab"
    assert P.typed(P.Start(job="ab")) == {"job": "ab", "v": [1, 0]}


def test_any_different_major_refused_both_ways_with_the_text():
    m = P.message(P.Start(job="ab"), sender="Studio").to_wire()
    for major in (2, 0):
        with pytest.raises(P.VersionMismatch) as e:
            P.Message.from_wire({**m, "v": [major, 0]})
        assert str(e.value) == (f"Studio speaks protocol {major}, this "
                                f"machine 1: update knurlogic on the older one")


def test_higher_minor_accepted_and_bad_v_rejected():
    m = P.message(P.Start(job="ab")).to_wire()
    assert P.Message.from_wire({**m, "v": [1, 7]}).v == (1, 7)
    for bad in (None, "1.0", [1], [True, 0], [-1, 0]):
        with pytest.raises(P.ProtocolError):
            P.Message.from_wire({**m, "v": bad})
