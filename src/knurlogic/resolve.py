"""Turn an artifact + a memory budget into runtime settings.

The resolver owns the FINAL value of every knob. It does not write env files
and it does not hope one wins: vqlab F33 recorded an experiment that set
RTILE in `exo-env.sh`, which is sourced BEFORE a `ring-env.sh` that assigns
RTILE=32 unconditionally -- so the run benchmarked 32 twice and was reported
as "no difference." Anything that resolves settings must hand back one dict
and be the last word on it.

Headroom is an INPUT, not something this package detects. Machine inventory
belongs to whatever manages the machines.

ONE BOX OR SEVERAL. A cluster resolves the same knobs against a DIFFERENT
budget per node, because the boxes differ and because each node holds only
its shard. So `resolve()` takes either a byte count (one box, one
`Resolution`) or nodes (a `ClusterResolution`, one `Resolution` each). The
single-box call is the same call it always was; the cluster case was made
cheap now because threading a second budget through later is invasive.

What a node HOLDS is a placement question and placement belongs to whatever
does the sharding -- exo, here. When nobody says, this assumes the shard is
proportional to the node's working set, which is an ASSUMPTION and is
recorded as a note on every resolution that rides on it, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import settings as S
from .artifact import Artifact

GIB = 1 << 30


@dataclass
class Resolution:
    env: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_exports(self) -> str:
        return "\n".join(f"export {k}={v}" for k, v in sorted(self.env.items()))


@dataclass
class Node:
    """A box the artifact could run on, and what it would hold there.

    `working_set_bytes` is that box's usable GPU working set -- the same
    input the single-box call takes. `holds_bytes` is how much of the
    artifact lands on this node; None means "let the resolver assume
    proportional", which it will say it did.
    """
    name: str
    working_set_bytes: int
    holds_bytes: int | None = None


@dataclass
class ClusterResolution:
    """One `Resolution` per node, plus what is only true of the whole.

    It is deliberately NOT a Resolution: there is no single env dict for a
    cluster, and inventing one would put the wrong knobs on the wrong box.
    """
    nodes: dict = field(default_factory=dict)          # name -> Resolution
    inventory: list = field(default_factory=list)      # the Nodes, in order
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def working_set_bytes(self) -> int:
        return sum(n.working_set_bytes for n in self.inventory)

    def __iter__(self):
        return iter(self.nodes.items())

    def as_exports(self) -> str:
        """Per node, because nothing here is settable cluster-wide."""
        out = []
        for name, r in self.nodes.items():
            out.append(f"# {name}")
            out.append(r.as_exports())
        return "\n".join(out)


def _as_nodes(budget) -> list:
    """Accept a byte count, a Node, a sequence of Nodes, or {name: bytes}."""
    if isinstance(budget, Node):
        return [budget]
    if isinstance(budget, dict):
        return [n if isinstance(n, Node) else Node(str(k), int(n))
                for k, n in budget.items()]
    if isinstance(budget, (list, tuple)):
        return [n if isinstance(n, Node) else Node(f"node{i}", int(n))
                for i, n in enumerate(budget)]
    raise TypeError(f"cannot read {budget!r} as a memory budget or nodes")


def _shares(artifact: Artifact, nodes: list) -> dict:
    """How many bytes of the artifact each node holds.

    Declared shares win. Anything left is split proportionally to working
    set, which is what a placer with no other information does. A node with
    no working set gets nothing rather than a division by zero.
    """
    out = {n.name: n.holds_bytes for n in nodes if n.holds_bytes is not None}
    rest = [n for n in nodes if n.holds_bytes is None]
    remaining = max(artifact.bytes_on_disk - sum(out.values()), 0)
    pool = sum(max(n.working_set_bytes, 0) for n in rest)
    for n in rest:
        out[n.name] = (int(remaining * max(n.working_set_bytes, 0) / pool)
                       if pool > 0 else 0)
    return out


def decode_chunk_for(headroom_bytes: int, known: bool = True) -> int:
    """Chunk width that keeps the largest dense-expert transient bounded.

    transient = chunk * out * in * 2 bytes, and on a box where the model
    nearly fills RAM this is what caps context length -- it grew 3.35 MB/token
    on the 397B where KV-cache theory predicted 0.059. Smaller is also faster
    (128 -> 32 is 1.37x), so there is no speed/memory tradeoff to negotiate
    below the default; it is capped at the default rather than raised.
    """
    if not known:
        return S.DECODE_CHUNK_DEFAULT      # no budget given: leave the default
    if headroom_bytes <= 0:
        return S.DECODE_CHUNK_MIN          # does not fit: tightest, not default
    per = S.DECODE_CHUNK_BYTES_PER_UNIT * S.DECODE_CHUNK_HEADROOM_DIVISOR
    return max(S.DECODE_CHUNK_MIN,
               min(S.DECODE_CHUNK_DEFAULT, int(headroom_bytes / per)))


def resolve(artifact: Artifact, budget, profile: str = "v1.5"):
    """Resolve every knob for this artifact against a budget.

    `budget` is either a byte count -- one box, and the return is a
    `Resolution`, exactly as it always was -- or nodes (a `Node`, a sequence
    of them, or {name: working_set_bytes}), in which case the return is a
    `ClusterResolution` carrying one `Resolution` per node.

    `profile` is v1.5 (bit-exact vs the published runtime) or v2 (the two
    numerics-active flags on). An artifact shipping UNCHANGED weights gets
    v1.5: there is no quality gain to offset a numerics regression, however
    small. Only a repo shipping improved weights may take v2.
    """
    if profile not in S.RUNTIME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(S.RUNTIME_PROFILES)}")

    if isinstance(budget, (int, float)):
        return _resolve_one(artifact, int(budget), artifact.bytes_on_disk,
                            profile)
    return resolve_cluster(artifact, budget, profile)


def resolve_cluster(artifact: Artifact, budget,
                    profile: str = "v1.5") -> ClusterResolution:
    """Resolve per node, and say what is only true of the whole cluster."""
    nodes = _as_nodes(budget)
    if not nodes:
        raise ValueError("no nodes to resolve against")
    shares = _shares(artifact, nodes)
    assumed = [n.name for n in nodes if n.holds_bytes is None]

    c = ClusterResolution(inventory=nodes)
    for n in nodes:
        r = _resolve_one(artifact, n.working_set_bytes, shares[n.name],
                         profile)
        if n.holds_bytes is None:
            r.notes.append(
                f"shard size ASSUMED {shares[n.name]/GIB:.1f} GiB "
                f"(proportional to working set) -- placement was not "
                f"declared, so every headroom number here inherits that")
        c.nodes[n.name] = r

    total = sum(n.working_set_bytes for n in nodes)
    known = all(n.working_set_bytes > 0 for n in nodes)
    c.notes.append(
        f"{len(nodes)} nodes, {total/GIB:.1f} GiB of working set against a "
        f"{artifact.gib:.1f} GiB artifact")
    if assumed:
        c.notes.append(
            f"placement not declared for {', '.join(assumed)}; shards split "
            f"proportionally to working set")
    if known and total <= artifact.bytes_on_disk:
        c.warnings.append(
            f"the artifact is {artifact.gib:.1f} GiB and the whole cluster "
            f"has {total/GIB:.1f} GiB of working set -- it does not fit even "
            f"sharded, before any runtime overhead. More nodes, or a smaller "
            f"rung.")
    for name, r in c.nodes.items():
        c.warnings += [f"{name}: {w}" for w in r.warnings]
    return c


def _resolve_one(artifact: Artifact, working_set_bytes: int,
                 holds_bytes: int, profile: str) -> Resolution:
    """One box. `holds_bytes` is what this box holds of the artifact, which
    is the whole thing unless something sharded it."""
    r = Resolution()

    if not artifact.is_vq:
        r.notes.append("not a VQ artifact -- kernel knobs do not apply")
    elif not artifact.model_file:
        r.warnings.append(
            "VQ modules declared but config.json names no `model_file`: the "
            "bundled kernels will not be found and the load will fail")

    headroom = working_set_bytes - holds_bytes

    # --- the two knobs that decide runnable-vs-not --------------------------
    # VQ_DECODE_CHUNK bounds the dense-EXPERT decode transient, which only
    # exists on the VQ path. A stock affine artifact has no such buffer, so
    # emitting it would be cargo cult.
    chunk = decode_chunk_for(headroom, known=working_set_bytes > 0)
    if artifact.is_vq:
        r.env["VQ_DECODE_CHUNK"] = str(chunk)
    fits = working_set_bytes <= 0 or headroom > 0
    if artifact.is_vq and chunk < S.DECODE_CHUNK_DEFAULT and fits:
        r.notes.append(
            f"VQ_DECODE_CHUNK lowered to {chunk} ({headroom/GIB:.1f} GiB "
            f"headroom): bounds the dense-expert transient, which is what "
            f"caps context length on a full box")

    tight = working_set_bytes > 0 and headroom < S.TIGHT_HEADROOM_GIB * GIB
    r.env["VQLAB_PREFILL_CHUNK"] = str(
        S.PREFILL_CHUNK_TIGHT if tight else S.PREFILL_CHUNK_DEFAULT)
    if tight:
        r.notes.append(
            "prompt chunk narrowed: token-identical at every width, so this "
            "costs nothing but peak memory")

    r.env["VQLAB_CACHE_LIMIT_GB"] = str(S.CACHE_LIMIT_GB_DEFAULT)

    if working_set_bytes > 0 and headroom <= 0:
        r.warnings.append(
            f"{holds_bytes / GIB:.1f} GiB to hold against a "
            f"{working_set_bytes/GIB:.1f} GiB working set -- it does not fit "
            f"this box. No setting fixes that; it needs a bigger box or more "
            f"than one.")

    # --- performance knobs, each with a finding behind it -------------------
    if artifact.is_vq:
        for k, (v, why) in S.PERFORMANCE_DEFAULTS.items():
            r.env[k] = v
        r.env.update(S.RUNTIME_PROFILES[profile])
        r.notes.append(f"runtime profile {profile}")

    return r
