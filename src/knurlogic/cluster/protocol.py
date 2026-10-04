"""The control-plane contract between knurlogic pages (docs/design/orchestration.md).

Stdlib only. Every message is an envelope `{v, kind, from, job?, seq, ts,
body}`; every kind is a frozen dataclass with `to_wire()` / `from_wire()`.
Unknown fields are ignored, a missing required field raises `ProtocolError`
with a plain message, any different major version is refused.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

VERSION = (1, 0)
FAILURE_KINDS = ("refusal", "memory", "machine", "failure")


class ProtocolError(ValueError):
    """A message that is not the contract; the text is for a person."""


class VersionMismatch(ProtocolError):
    pass


def version_refusal(who: str, theirs, ours=VERSION) -> str:
    return (f"{who} speaks protocol {theirs[0]}, this machine {ours[0]}: "
            f"update knurlogic on the older one")


def mismatch_text(name: str, theirs, ours=VERSION) -> str:
    """What the peer list says of a machine whose protocol major differs,
    or that speaks none (an older knurlogic): which machine to update."""
    if not isinstance(theirs, (list, tuple)) or not theirs:
        return (f"{name} speaks no protocol this machine knows, "
                f"this machine {ours[0]}: update knurlogic on {name}")
    older = theirs[0] < ours[0]
    return (f"{name} speaks protocol {theirs[0]}, this machine {ours[0]}: "
            f"update knurlogic on {name if older else 'this machine'}")


def check_version(v: Any, who: str = "a peer") -> tuple:
    """(major, minor) of a wire `v`; raises VersionMismatch when the major
    differs from ours (either way), ProtocolError when it is no version."""
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool)
                       and x >= 0 for x in v)):
        raise ProtocolError("v must be [major, minor]")
    if v[0] != VERSION[0]:
        raise VersionMismatch(version_refusal(who, v))
    return (v[0], v[1])


@dataclass(frozen=True)
class Body:
    """Base of every kind. `WIRE` renames a field on the wire (the keys
    today's replies already use); an optional field that is None is left
    off the wire."""
    KIND: ClassVar[str] = ""
    WIRE: ClassVar[dict] = {}
    KEEP: ClassVar[tuple] = ()      # None-valued fields that stay on the wire

    def to_wire(self) -> dict:
        out = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None and f.default is None and f.name not in self.KEEP:
                continue
            out[self.WIRE.get(f.name, f.name)] = val
        return out

    @classmethod
    def from_wire(cls, doc: Any):
        if not isinstance(doc, dict):
            raise ProtocolError(f"{cls.__name__} must be an object")
        kw = {}
        for f in fields(cls):
            key = cls.WIRE.get(f.name, f.name)
            if key in doc:
                kw[f.name] = doc[key]
            elif (f.default is dataclasses.MISSING
                  and f.default_factory is dataclasses.MISSING):
                raise ProtocolError(f"{cls.__name__}: {key} is missing")
        return cls(**kw)


_KINDS: dict = {}


def _register(cls):
    cls.KIND = cls.__name__
    _KINDS[cls.__name__] = cls
    return cls


@_register
@dataclass(frozen=True)
class Hello(Body):
    id: str
    name: str = ""
    build: str | None = None
    knurlogic: str | None = None
    mlx: str | None = None
    protocol: list = field(default_factory=lambda: list(VERSION))
    machine: dict = field(default_factory=dict)
    addresses: list = field(default_factory=list)
    boot_id: str | None = None


@_register
@dataclass(frozen=True)
class Heartbeat(Body):
    boot_id: str
    resident: str | None = None
    jobs: list = field(default_factory=list)


@_register
@dataclass(frozen=True)
class Survey(Body):
    pass


@_register
@dataclass(frozen=True)
class Residency(Body):
    resident: list = field(default_factory=list)
    recovery: list = field(default_factory=list)


@_register
@dataclass(frozen=True)
class Prepare(Body):
    job: str
    rank: int
    world: int
    split: str
    link: str
    identity: str | None = None
    hosts: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    port: int | None = None
    auto_port: bool = False
    extra: dict = field(default_factory=dict)


@_register
@dataclass(frozen=True)
class PrepareReply(Body):
    ok: bool
    machine: str | None = None
    rank: int | None = None
    refused: str | None = None
    free_port: int | None = None
    alert: str | None = None
    note: str | None = None
    #: the prompt chunk this rank's own room allows (None: not worked out)
    prefill_chunk: int | None = None


@_register
@dataclass(frozen=True)
class Start(Body):
    job: str
    #: the ring's prompt chunk once every rank has answered Prepare, and why
    prefill_chunk: int | None = None
    prefill_why: str | None = None


@_register
@dataclass(frozen=True)
class Started(Body):
    WIRE: ClassVar[dict] = {"job": "started"}
    job: str
    rank: int
    pid: int
    log: str | None = None


@_register
@dataclass(frozen=True)
class RankStatus(Body):
    job: str
    rank: int
    phase: str
    step: int | None = None
    busy: bool = False
    pid: int | None = None


@_register
@dataclass(frozen=True)
class JobState(Body):
    KEEP: ClassVar[tuple] = ("phase", "ended")
    job: str
    ranks_here: list = field(default_factory=list)
    prepared: bool = False
    stopping: bool = False
    phase: str | None = None
    processes: list = field(default_factory=list)
    ended: str | None = None
    ended_kind: str | None = None


@_register
@dataclass(frozen=True)
class Stop(Body):
    WIRE: ClassVar[dict] = {"failure_kind": "kind"}
    job: str
    reason: str = "stopped by another machine"
    failure_kind: str | None = None


@_register
@dataclass(frozen=True)
class Stopped(Body):
    """`killed` are processes this stop ended; `exiting` are ones still
    going when it answered (a peer may answer before its processes are
    gone)."""
    WIRE: ClassVar[dict] = {"job": "stopped"}
    job: str
    ranks_here: list = field(default_factory=list)
    killed: list = field(default_factory=list)
    exiting: list = field(default_factory=list)
    told: list = field(default_factory=list)
    reason: str = ""
    #: recovery records the stop cleared, here and on the pages it told
    cleared: list = field(default_factory=list)


@_register
@dataclass(frozen=True)
class Load(Body):
    identity: str | None = None
    name: str | None = None
    port: int | None = None
    tune: str | None = None
    sets: dict = field(default_factory=dict)
    force: bool = False
    draft: bool = True


@_register
@dataclass(frozen=True)
class Unload(Body):
    port: int


@_register
@dataclass(frozen=True)
class Failure(Body):
    """`kind` is refusal | memory | machine | failure; `requested` marks a
    stop somebody asked for."""
    kind: str
    reason: str
    node: str | None = None
    requested: bool = False

    def __post_init__(self):
        if self.kind not in FAILURE_KINDS:
            raise ProtocolError(f"Failure.kind is one of "
                                f"{', '.join(FAILURE_KINDS)}, not "
                                f"{self.kind!r}")


@_register
@dataclass(frozen=True)
class Shape(Body):
    identity: str | None = None
    name: str | None = None
    world: int | None = None
    split: str | None = None


@_register
@dataclass(frozen=True)
class MachineSet(Body):
    """A machine's own settings: the allowance, the strategy, and the
    knurlogic-wide settings (`settings`)."""
    allowance_gib: float | None = None
    strategy: str | None = None
    settings: dict | None = None


@_register
@dataclass(frozen=True)
class Read(Body):
    """A page's read of a peer's document: a fixed list of paths (the
    receiver's own allow-list), or -- with `port` -- a model server's
    /settings.json on that peer. Reads only."""
    path: str
    query: dict = field(default_factory=dict)
    port: int | None = None


@_register
@dataclass(frozen=True)
class Settings(Body):
    port: int | None = None
    values: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    body: Body
    sender: str = ""
    job: str | None = None
    seq: int = 0
    ts: float = 0.0
    re: int | None = None
    v: tuple = VERSION

    @property
    def kind(self) -> str:
        return self.body.KIND

    def to_wire(self) -> dict:
        out: dict = {"v": list(self.v), "kind": self.kind,
                     "from": self.sender, "seq": self.seq, "ts": self.ts,
                     "body": self.body.to_wire()}
        if self.job is not None:
            out["job"] = self.job
        if self.re is not None:
            out["re"] = self.re
        return out

    @classmethod
    def from_wire(cls, doc: Any) -> Message:
        if not isinstance(doc, dict):
            raise ProtocolError("a message is a JSON object")
        who = str(doc.get("from") or "a peer")[:60]
        v = check_version(doc.get("v"), who)
        k = doc.get("kind")
        if k not in _KINDS:
            raise ProtocolError(f"unknown message kind {str(k)[:40]!r}")
        body = _KINDS[k].from_wire(doc.get("body", {}))
        job, seq, ts, re_ = (doc.get("job"), doc.get("seq", 0),
                             doc.get("ts", 0.0), doc.get("re"))
        return cls(body=body, sender=str(doc.get("from") or ""),
                   job=None if job is None else str(job),
                   seq=seq if isinstance(seq, int) else 0,
                   ts=float(ts) if isinstance(ts, (int, float)) else 0.0,
                   re=re_ if isinstance(re_, int) else None, v=v)


_SEQ = [0]


def next_seq() -> int:
    _SEQ[0] += 1
    return _SEQ[0]


def message(body: Body, sender: str = "", job: str | None = None,
            re: int | None = None) -> Message:
    return Message(body=body, sender=sender, job=job, seq=next_seq(),
                   ts=time.time(), re=re)


def typed(body: Body) -> dict:
    """A reply body in today's wire keys plus `v`."""
    return {**body.to_wire(), "v": list(VERSION)}


def kinds() -> dict:
    return dict(_KINDS)
