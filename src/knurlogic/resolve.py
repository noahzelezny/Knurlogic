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


#: Knobs `engine.serve` turns into argv or an mlx call, so they are real
#: whether or not an artifact's bundled runtime reads them.
ENGINE_CONSUMED = ("prefill_chunk", "cache_limit_gb", "prompt_concurrency")


def emit(r: Resolution, artifact: Artifact, logical: str, value) -> str | None:
    """Set a knob under the name THIS artifact reads, or not at all.

    Returns the name used, or None when the artifact ships a runtime and that
    runtime reads none of the aliases -- in which case emitting anything would
    be theatre. That is not hypothetical: VQLAB_PREFILL_CHUNK was emitted for
    every artifact and is read by none of the 37 bundled runtimes here.
    """
    names = S.KNOB_ALIASES.get(logical, (logical,))
    src = artifact.runtime_source()
    if not src:
        name = S.default_alias(logical) if logical in S.KNOB_ALIASES else names[0]
        r.env[name] = str(value)
        return name
    for name in names:
        if name in src:
            r.env[name] = str(value)
            return name
    if logical in ENGINE_CONSUMED:
        # The runtime does not read it, but the engine does: set it under the
        # name `settings.engine_settings` looks for, and it reaches argv.
        name = S.default_alias(logical)
        r.env[name] = str(value)
        return name
    r.notes.append(
        f"{logical} not emitted: this artifact's bundled runtime reads none "
        f"of {list(names)}, so any value would be a setting that does nothing")
    return None


def expert_transient_bytes_per_unit(artifact: Artifact):
    """(bytes per unit of decode chunk, why) -- from the ARTIFACT'S shape.

    The dense-expert transient is `chunk * out * in * 2`, and out/in are the
    model's: gate_up is [2 * moe_intermediate_size, hidden_size]. Both fields
    are already read off config.json; until now the resolver ignored them and
    used a constant frozen for one rung.

    That mattered more than it looks. The prefill spike is a FAMILY effect,
    not a box effect -- the same spike on M4 and M3, DeepSeek V4 far more
    dramatic than Qwen3.5 -- which is what this formula says it should be: H
    and M come from the config and the machine does not enter. A resolver
    keyed on the box alone cannot express that and will size the knob for the
    wrong model.
    """
    H, M = artifact.hidden_size, artifact.moe_intermediate_size
    if H and M:
        return 2 * M * H * 2, f"gate_up [2x{M}, {H}] from this artifact"
    w, x = S.DECODE_CHUNK_ASSUMED_SHAPE
    return S.DECODE_CHUNK_BYTES_PER_UNIT, (
        f"config declares no MoE expert shape, so the transient is sized "
        f"from the ASSUMED [{w}, {x}] -- the one rung the auto-sizer ran on")


def decode_chunk_for(headroom_bytes: int, known: bool = True,
                     bytes_per_unit: int | None = None) -> int:
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
    per = ((bytes_per_unit or S.DECODE_CHUNK_BYTES_PER_UNIT)
           * S.DECODE_CHUNK_HEADROOM_DIVISOR)
    return max(S.DECODE_CHUNK_MIN,
               min(S.DECODE_CHUNK_DEFAULT, int(headroom_bytes / per)))


def resolve(artifact: Artifact, budget, profile: str = "v1.5",
            tune: str = "balanced"):
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

    if tune not in S.TUNE_PROFILES:
        raise ValueError(f"tune must be one of {sorted(S.TUNE_PROFILES)}")
    if isinstance(budget, (int, float)):
        return _resolve_one(artifact, int(budget), artifact.bytes_on_disk,
                            profile, tune)
    return resolve_cluster(artifact, budget, profile, tune)


def resolve_cluster(artifact: Artifact, budget, profile: str = "v1.5",
                    tune: str = "balanced") -> ClusterResolution:
    """Resolve per node, and say what is only true of the whole cluster."""
    nodes = _as_nodes(budget)
    if not nodes:
        raise ValueError("no nodes to resolve against")
    shares = _shares(artifact, nodes)
    assumed = [n.name for n in nodes if n.holds_bytes is None]

    c = ClusterResolution(inventory=nodes)
    for n in nodes:
        r = _resolve_one(artifact, n.working_set_bytes, shares[n.name],
                         profile, tune)
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
                 holds_bytes: int, profile: str,
                 tune: str = "balanced") -> Resolution:
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
    per_unit, shape_why = expert_transient_bytes_per_unit(artifact)
    known = working_set_bytes > 0
    chunk = decode_chunk_for(headroom, known=known, bytes_per_unit=per_unit)
    frozen = decode_chunk_for(headroom, known=known)
    loosened = chunk > frozen
    if loosened and not S.DECODE_CHUNK_SHAPE_MAY_LOOSEN:
        # Tighten on the model's shape, never loosen on it -- see
        # DECODE_CHUNK_SHAPE_MAY_LOOSEN. Being wrong the other way is an OOM.
        chunk = frozen
    # Remember what HEADROOM alone decided, so the note below credits the
    # right cause. A message that blames headroom for a lowering the tune
    # profile made would send someone looking for memory they already have.
    chunk_from_headroom = chunk

    t = S.TUNE_PROFILES[tune]
    scale = t.get("decode_chunk_scale", 1.0)
    if scale != 1.0 and chunk > S.DECODE_CHUNK_MIN:
        chunk = max(S.DECODE_CHUNK_MIN, int(chunk * scale))
        r.notes.append(f"tune={tune}: transient bounded tighter than headroom "
                       f"requires (chunk {chunk})")

    if artifact.is_vq:
        emit(r, artifact, "decode_chunk", chunk)
        if known:
            r.notes.append(f"expert transient sized from {shape_why}")
        if loosened and not S.DECODE_CHUNK_SHAPE_MAY_LOOSEN:
            r.notes.append(
                f"this artifact's experts are small enough to justify a "
                f"larger chunk, and it was NOT taken: sizing from the model "
                f"may only tighten until a run measures the loosening "
                f"direction, because being wrong there is an OOM")
    fits = working_set_bytes <= 0 or headroom > 0
    if (artifact.is_vq and chunk_from_headroom < S.DECODE_CHUNK_DEFAULT
            and fits):
        r.notes.append(
            f"VQ_DECODE_CHUNK lowered to {chunk} ({headroom/GIB:.1f} GiB "
            f"headroom): bounds the dense-expert transient, which is what "
            f"caps context length on a full box")

    tight = working_set_bytes > 0 and headroom < S.TIGHT_HEADROOM_GIB * GIB
    family, family_why = S.prefill_chunk_for(artifact.model_type)
    prefill = min(family, S.PREFILL_CHUNK_TIGHT) if tight else family
    asked = t.get("VQLAB_PREFILL_CHUNK")
    if asked is not None and tune == "fast":
        # `fast` means spend headroom, never "narrower than was measured".
        asked = max(asked, family)
    if family != S.PREFILL_CHUNK_DEFAULT:
        r.notes.append(f"prompt chunk {family} {family_why}")
    if asked is not None and asked != prefill:
        # A tight box wins over the axis. `fast` cannot spend headroom that
        # is not there, and saying so is the difference between a knob and a
        # wish.
        if asked > prefill and tight:
            r.notes.append(
                f"tune={tune} asked for a {asked}-wide prompt chunk and did "
                f"not get it: {headroom / GIB:.1f} GiB of headroom is what "
                f"caps it, not the profile")
        else:
            prefill = asked
    emit(r, artifact, "prefill_chunk", prefill)
    if tight or tune == "safe":
        emit(r, artifact, "prompt_concurrency", S.PROMPT_CONCURRENCY_TIGHT)
        r.notes.append(
            "one prompt prefilled at a time: the transient is per prompt, so "
            "the engine's default of 8 together is 8x the spike")
    if prefill < S.PREFILL_CHUNK_DEFAULT:
        r.notes.append(
            "prompt chunk narrowed: token-identical at every width, so this "
            "costs nothing but peak memory")

    cache = float(t.get("VQLAB_CACHE_LIMIT_GB", S.CACHE_LIMIT_GB_DEFAULT))
    if cache > S.CACHE_LIMIT_GB_MAX:
        r.notes.append(f"tune={tune} capped: cache limit {cache} -> "
                       f"{S.CACHE_LIMIT_GB_MAX} GiB, above which nothing has "
                       f"been measured to improve")
        cache = S.CACHE_LIMIT_GB_MAX
    # Reclaimable is not free: it is still resident. On a box with little
    # headroom a large cache is the thing that turns a long prompt into an OOM.
    if known and cache * GIB > max(headroom, 0) / 2:
        room = max(round(max(headroom, 0) / 2 / GIB, 1), 1.0)
        if room < cache:
            r.notes.append(
                f"tune={tune} asked for a {cache} GiB reclaimable cache and "
                f"got {room}: it is reclaimable, not free, and there is only "
                f"{headroom / GIB:.1f} GiB of headroom to hold it in")
            cache = room
    emit(r, artifact, "cache_limit_gb", cache)
    if tune != "balanced":
        r.notes.append(f"tune={tune}: {t['why']}")

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
