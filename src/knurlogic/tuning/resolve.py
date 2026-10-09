"""Turn an artifact + a memory budget into runtime settings.

The resolver owns the final value of every knob and hands back one dict:
it never writes env files and hopes one wins. Headroom is an input, not
something this package detects. `resolve()` takes a byte count (one
machine, one `Resolution`) or nodes (a `ClusterResolution`, one budget per
node). When no placement is given, a node's shard is assumed proportional
to its working set, and that assumption is recorded as a note.

Design: docs/design/settings.md (resolve).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import context_window, fit, knobs, measured, numerics, presets


@dataclass
class Resolution:
    env: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    #: What a vision rung holds besides its text weights (`vision_budget`),
    #: or None for a text-only artifact.
    vision: dict | None = None
    #: knob -> the values THIS artifact may take, where they are narrower
    #: than knobs.KNOB_RANGE (KV bits on a family that refuses them)
    ranges: dict = field(default_factory=dict)
    #: the launch preset (= the tune): its name, the values it set, and --
    #: once `apply_preset_overrides` has run -- which of them an explicit
    #: per-model setting beat
    preset: dict = field(default_factory=dict)

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
    #: what the machine is, for rank order (rank_order) and, later,
    #: pipeline layer shares: "Apple M4 Max", its P-core clock, and its
    #: memory bandwidth when known
    chip: str | None = None
    p_core_ghz: float | None = None
    memory_bandwidth_gbs: float | None = None


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


#: Knobs `engine.model` turns into argv or an mlx call, so they are real
#: whether or not an artifact's bundled runtime reads them.
ENGINE_CONSUMED = ("prefill_chunk", "cache_limit_gb", "context_length", "mtp",
                   "mtp_dynamic", "kv_bits", "kv_kernel", "cross_chip",
                   "long_context", "preset", "vision", "thinking_default")


def emit(r: Resolution, artifact: Artifact, logical: str, value) -> str | None:
    """Set a knob under the name THIS artifact reads, or not at all.

    Returns the name used, or None when the artifact ships a runtime and that
    runtime reads none of the aliases -- in which case emitting anything would
    be theatre. That is not hypothetical: VQLAB_PREFILL_CHUNK was emitted for
    every artifact and is read by none of the 37 bundled runtimes here.
    """
    names = knobs.KNOB_ALIASES.get(logical, (logical,))
    src = artifact.runtime_source()
    if not src:
        name = (knobs.default_alias(logical)
                if logical in knobs.KNOB_ALIASES else names[0])
        r.env[name] = str(value)
        return name
    for name in names:
        if name in src:
            r.env[name] = str(value)
            return name
    if logical in ENGINE_CONSUMED:
        # The runtime does not read it, but the engine does: set it under the
        # name `knobs.engine_settings` looks for, and it reaches argv.
        name = knobs.default_alias(logical)
        r.env[name] = str(value)
        return name
    r.notes.append(
        f"{logical} not emitted: this artifact's bundled runtime reads none "
        f"of {list(names)}, so any value would be a setting that does nothing")
    return None


def emit_cache_limit(r: Resolution, artifact: Artifact, gib) -> None:
    """The cache limit, one value: knurlogic's own (the engine sets mlx's
    free-buffer cache ceiling from it), and the same value under the name
    a VQ bundle's runtime reads, which sets that ceiling again at load."""
    r.env["KNURLOGIC_CACHE_LIMIT_GB"] = str(gib)
    src = artifact.runtime_source()
    old = knobs.KNOB_ALIASES["cache_limit_gb"][1:]
    name = next((n for n in old if n in src), None) if src else \
        (old[0] if artifact.is_vq else None)
    if name:
        r.env[name] = str(gib)


def resolve(artifact: Artifact, budget, profile: str | None = None,
            tune: str = "default", store_bytes: int | None = None,
            holds_bytes: int | None = None, kv_bits=None,
            long_context=None, vision: bool = True):
    """Resolve every knob for this artifact against a budget.

    `budget` is either a byte count -- one box, and the return is a
    `Resolution`, exactly as it always was -- or nodes (a `Node`, a sequence
    of them, or {name: working_set_bytes}), in which case the return is a
    `ClusterResolution` carrying one `Resolution` per node.

    `profile` is None (the default): each model keeps the numerics it
    SHIPPED (`numerics_for`). "v1.5" or "v2" forces that profile's two
    numerics-active flags, and only because someone asked -- it used to
    default to v1.5 and so silently turned off bf16 I/O on the rungs that
    shipped it on.

    `store_bytes` is a live image store's `budget_bytes()`, when one exists;
    otherwise a vision rung is budgeted at the store's default bound
    (`vision_budget`).

    `holds_bytes`: what this box holds of the artifact when it is one rank
    of a split (its share); default, the whole artifact.

    `kv_bits`: the KV precision the model will launch with (a launch
    setting), for what its KV is counted at; None = bf16.

    `long_context`: KNURLOGIC_LONG_CONTEXT ('off' | 'yarn'), which moves
    the context cap to the YaRN window and is checked against the room the
    KV of that context needs (`long_context_room`).

    `vision`: KNURLOGIC_VISION. Off, a vision rung holds no tower, image
    store or image KV: none is counted, and a whole artifact's tower bytes
    are taken out of what the box holds (`vision_freed_bytes`).
    """
    tune = presets.preset_of(tune)
    if profile is not None and profile not in numerics.RUNTIME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(numerics.RUNTIME_PROFILES)}")

    if isinstance(budget, (int, float)):
        holds = artifact.bytes_on_disk if holds_bytes is None \
            else int(holds_bytes)
        r = _resolve_one(artifact, int(budget),
                         holds - (fit.vision_off_tower(artifact)
                                  if not vision and holds_bytes is None
                                  else 0),
                         profile, tune, store_bytes=store_bytes,
                         vision=vision, kv_bits=kv_bits,
                         long_context=long_context)
        if fit.vision_budget(artifact) is not None:
            # a launch setting of a vision rung only (Settings -> Models)
            emit(r, artifact, "vision", "on" if vision else "off")
        if not vision and fit.vision_budget(artifact) is not None:
            r.notes.append(
                f"vision off (KNURLOGIC_VISION=off): "
                f"{fit.vision_freed_bytes(artifact, kv_bits) / fit.GIB:.2f} GiB of "
                f"tower, image store and image KV not held")
        return r
    return resolve_cluster(artifact, budget, profile, tune,
                           store_bytes=store_bytes, vision=vision)


def resolve_cluster(artifact: Artifact, budget, profile: str | None = None,
                    tune: str = "default",
                    store_bytes: int | None = None,
                    vision: bool = True) -> ClusterResolution:
    """Resolve per node, and say what is only true of the whole cluster."""
    nodes = _as_nodes(budget)
    if not nodes:
        raise ValueError("no nodes to resolve against")
    shares = _shares(artifact, nodes)
    assumed = [n.name for n in nodes if n.holds_bytes is None]

    c = ClusterResolution(inventory=nodes)
    for i, n in enumerate(nodes):
        # The tower, the image store and the requests' image KV live where
        # the request enters -- the first node -- not spread with the shard.
        r = _resolve_one(artifact, n.working_set_bytes, shares[n.name],
                         profile, tune, store_bytes=store_bytes,
                         vision=(i == 0 and vision))
        if n.holds_bytes is None:
            r.notes.append(
                f"shard size ASSUMED {shares[n.name]/fit.GIB:.1f} GiB "
                f"(proportional to working set) -- placement was not "
                f"declared, so every headroom number here inherits that")
        c.nodes[n.name] = r

    total = sum(n.working_set_bytes for n in nodes)
    known = all(n.working_set_bytes > 0 for n in nodes)
    c.notes.append(
        f"{len(nodes)} nodes, {total/fit.GIB:.1f} GiB of working set against a "
        f"{artifact.gib:.1f} GiB artifact")
    if assumed:
        c.notes.append(
            f"placement not declared for {', '.join(assumed)}; shards split "
            f"proportionally to working set")
    if known and total <= artifact.bytes_on_disk:
        c.warnings.append(
            f"the artifact is {artifact.gib:.1f} GiB and the whole cluster "
            f"has {total/fit.GIB:.1f} GiB of working set -- it does not fit even "
            f"sharded, before any runtime overhead. More nodes, or a smaller "
            f"rung.")
    _ring_consistent(c)
    for name, r in c.nodes.items():
        c.warnings += [f"{name}: {w}" for w in r.warnings]
    return c


def _ring_consistent(c: ClusterResolution) -> None:
    """One prompt chunk on every rank: the smallest any node needs.

    Per-node resolution is right for per-node memory and wrong for this. A
    pipeline's ranks process the same chunks, so a low-headroom node's narrow chunk
    has to be everyone's -- and a rank that disagrees is a desync, not a
    tuning difference.
    """
    want = {n: knobs.engine_settings(r.env).get("prefill_step_size")
            for n, r in c.nodes.items()}
    got = [v for v in want.values() if v]
    if not got:
        return
    ring = min(got)
    for name, r in c.nodes.items():
        for alias in knobs.KNOB_ALIASES["prefill_chunk"]:
            if alias in r.env:
                r.env[alias] = str(ring)
        if want[name] and want[name] != ring:
            r.notes.append(
                f"prompt chunk {ring}, not the {want[name]} this node alone "
                f"would take: it is ring-wide, and every rank must match")
    if len(set(got)) > 1:
        c.notes.append(f"prompt chunk {ring} on every rank (the tightest "
                       f"node's); ranks that disagree desync")


def _resolve_one(artifact: Artifact, working_set_bytes: int,
                 holds_bytes: int, profile: str | None,
                 tune: str = "default", store_bytes: int | None = None,
                 vision: bool = True, kv_bits=None,
                 long_context=None) -> Resolution:
    """One box. `holds_bytes` is what this box holds of the artifact, which
    is the whole thing unless something sharded it. A vision rung also
    holds its tower, its image store and its image KV (`vision_budget`),
    counted here, before the headroom every knob below is sized from."""
    r = Resolution()
    vb = fit.vision_budget(artifact, store_bytes, kv_bits) if vision else None
    if vb is not None:
        r.vision = vb
        holds_bytes = holds_bytes + vb["extra_bytes"]
        r.notes.extend(vb["notes"])

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
    per_unit, shape_why = fit.expert_transient_bytes_per_unit(artifact)
    known = working_set_bytes > 0
    chunk = fit.decode_chunk_for(headroom, known=known, bytes_per_unit=per_unit)
    frozen = fit.decode_chunk_for(headroom, known=known)
    loosened = chunk > frozen
    if loosened and not measured.DECODE_CHUNK_SHAPE_MAY_LOOSEN:
        # Tighten on the model's shape, never loosen on it -- see
        # DECODE_CHUNK_SHAPE_MAY_LOOSEN. Being wrong the other way is an OOM.
        chunk = frozen
    # Remember what HEADROOM alone decided, so the note below credits the
    # right cause. A message that blames headroom for a lowering the tune
    # profile made would send someone looking for memory they already have.
    chunk_from_headroom = chunk

    t = presets.TUNE_PROFILES[tune]
    scale = t.get("decode_chunk_scale", 1.0)
    if scale != 1.0 and chunk > measured.DECODE_CHUNK_MIN:
        chunk = max(measured.DECODE_CHUNK_MIN, int(chunk * scale))
        r.notes.append(f"tune={tune}: transient bounded tighter than headroom "
                       f"requires (chunk {chunk})")

    if artifact.is_vq:
        emit(r, artifact, "decode_chunk", chunk)
        if known:
            r.notes.append(f"expert transient sized from {shape_why}")
        if loosened and not measured.DECODE_CHUNK_SHAPE_MAY_LOOSEN:
            r.notes.append(
                "this artifact's experts are small enough to justify a "
                "larger chunk, and it was NOT taken: sizing from the model "
                "may only tighten until a run measures the loosening "
                "direction, because being wrong there is an OOM")
    fits = working_set_bytes <= 0 or headroom > 0
    if (artifact.is_vq and chunk_from_headroom < measured.DECODE_CHUNK_DEFAULT
            and fits):
        r.notes.append(
            f"VQ_DECODE_CHUNK lowered to {chunk} ({headroom/fit.GIB:.1f} GiB "
            f"headroom): bounds the dense-expert transient, which is what "
            f"caps context length on a full box")

    cache = float(t.get("KNURLOGIC_CACHE_LIMIT_GB", measured.CACHE_LIMIT_GB_DEFAULT))
    if cache > measured.CACHE_LIMIT_GB_MAX:
        r.notes.append(f"tune={tune} capped: cache limit {cache} -> "
                       f"{measured.CACHE_LIMIT_GB_MAX} GiB, above which nothing has "
                       f"been measured to improve")
        cache = measured.CACHE_LIMIT_GB_MAX
    # Reclaimable is not free: it is still resident. On a box with little
    # headroom a large cache is the thing that turns a long prompt into an OOM.
    if known and cache * fit.GIB > max(headroom, 0) / 2:
        room = max(round(max(headroom, 0) / 2 / fit.GIB, 1), 1.0)
        if room < cache:
            r.notes.append(
                f"tune={tune} asked for a {cache} GiB reclaimable cache and "
                f"got {room}: it is reclaimable, not free, and there is only "
                f"{headroom / fit.GIB:.1f} GiB of headroom to hold it in")
            cache = room
    family, family_why = measured.prefill_chunk_for(artifact.model_type)
    asked = t.get("KNURLOGIC_PREFILL_CHUNK")
    if asked is not None:
        # lean: narrow whatever the room
        prefill = asked
        if family > asked:
            r.notes.append(
                f"prompt chunk {asked}: tune={tune} keeps it narrow; "
                f"{family} was {family_why}")
    else:
        prefill, why = fit.prefill_chunk_by_room(
            artifact, headroom if known else None, working_set_bytes,
            int(cache * fit.GIB), kv_bits, family, family_why)
        r.notes.append(why)
    emit(r, artifact, "prefill_chunk", prefill)
    window, _ = context_window.model_window(
        _long_context_cfg(r, artifact, long_context, working_set_bytes,
                          holds_bytes, kv_bits))
    if window:
        # the model's own window: the cap a person lowers, never raises past
        # (checks.check_knob refuses more), so the control stops there
        emit(r, artifact, "context_length", window)
        # a family with documented YaRN is offered up to its YaRN window:
        # a context past the native one turns long context on at launch
        # (context_window.settle_context)
        top = context_window.context_ceiling(artifact.model_type,
                                             artifact.raw_config) or window
        top = max(top, window)
        steps = [v for v in knobs.KNOB_RANGE["KNURLOGIC_CONTEXT_LENGTH"][0]
                 if v < top and v != window]
        r.ranges["KNURLOGIC_CONTEXT_LENGTH"] = sorted(
            set(steps + [window, top]))
    launch, launch_notes = presets.preset_launch(tune, artifact.model_type)
    if prefill < measured.PREFILL_CHUNK_DEFAULT:
        r.notes.append(
            "prompt chunk narrowed: token-identical at every width, so this "
            "costs nothing but peak memory")

    emit_cache_limit(r, artifact, cache)
    if tune != "default":
        r.notes.append(f"tune={tune}: {t['why']}")
    model_launch(r, artifact, kv_bits, tune)
    r.notes.extend(launch_notes)
    _preset_record(r, tune, launch)

    if working_set_bytes > 0 and headroom <= 0:
        r.warnings.append(
            f"{holds_bytes / fit.GIB:.1f} GiB to hold against a "
            f"{working_set_bytes/fit.GIB:.1f} GiB working set -- it does not fit "
            f"this box. No setting fixes that; it needs a bigger box or more "
            f"than one.")

    # --- numerics: the model's own, from what it ships ----------------------
    if artifact.is_vq:
        env, note = numerics.numerics_for(artifact, profile)
        r.env.update(env)
        r.notes.append(note)

    return r


def _preset_record(r: Resolution, tune: str, launch: dict) -> None:
    """Which env values the preset put there: its launch settings, plus the
    prompt chunk / cache limit its profile names."""
    t = presets.TUNE_PROFILES[tune]
    logicals = set(launch)
    if "KNURLOGIC_PREFILL_CHUNK" in t:
        logicals.add("prefill_chunk")
    if "KNURLOGIC_CACHE_LIMIT_GB" in t:
        logicals.add("cache_limit_gb")
    names = {n for lg in logicals for n in knobs.KNOB_ALIASES.get(lg, (lg,))}
    r.preset = {"name": tune, "why": t.get("why", ""),
                "from_preset": {k: v for k, v in sorted(r.env.items())
                                if k in names},
                "overridden": {}}


def apply_preset_overrides(r: Resolution, overrides: dict) -> dict:
    """Record which of the preset's values an explicit setting beat:
    {name: {"preset": value, "set": value}}. The explicit value wins; this
    only says so. Returns r.preset."""
    fp = r.preset.setdefault("from_preset", {})
    ov = r.preset.setdefault("overridden", {})
    for k, v in (overrides or {}).items():
        if k in fp and str(fp[k]) != str(v):
            ov[k] = {"preset": fp.pop(k), "set": str(v)}
    return r.preset


def preset_env(artifact: Artifact, tune: str) -> dict:
    """The preset's model launch settings as env names (MODEL_KNOBS), for a
    caller that must read them before resolving (serve: KV bits change
    what the context costs)."""
    launch, _ = presets.preset_launch(tune, artifact.model_type)
    return {knobs.KNOB_ALIASES[k][0]: str(v) for k, v in launch.items()
            if knobs.KNOB_ALIASES[k][0] in knobs.MODEL_KNOBS}


def model_launch(r: Resolution, artifact: Artifact, kv_bits=None,
                 tune: str = presets.PRESET_DEFAULT) -> None:
    """The model's own launch settings: MTP drafting and its controller
    where a head ships beside the weights, and the KV precision this
    family allows (every value but bf16 refused, with the reason, where
    its caches cannot be quantized)."""
    from knurlogic.engine.mtp import find_head
    try:
        head = find_head(artifact.path)
    except (OSError, ValueError):
        head = None
    launch, _ = presets.preset_launch(tune, artifact.model_type)
    if head is not None:
        emit(r, artifact, "mtp", launch.get("mtp", "on"))
        emit(r, artifact, "mtp_dynamic", launch.get("mtp_dynamic", "on"))
    bits, why = measured.kv_quant_for(artifact.model_type)
    emit(r, artifact, "kv_bits", launch.get("kv_bits", "bf16"))
    # the decode kernel reads 8-bit K/V only: shown where it can matter
    if str(kv_bits if kv_bits is not None
           else launch.get("kv_bits", "bf16")) == "8":
        emit(r, artifact, "kv_kernel", launch.get("kv_kernel", "on"))
    emit(r, artifact, "cross_chip", launch.get("cross_chip", "off"))
    # the level a silent request gets: shown where the template has levels
    try:
        from knurlogic.engine.model import thinking
        native = thinking.levels(
            thinking.template_of(artifact.path)).get("native") or []
    except (OSError, ValueError, ImportError):
        native = []
    if native:
        emit(r, artifact, "thinking_default",
             launch.get("thinking_default", "model"))
        # the template's own names (GLM: off, low, high, max), not the
        # ladder's, whose words mean other levels there
        r.ranges["KNURLOGIC_THINKING_DEFAULT"] = [
            str(n["name"]) for n in native]
    emit(r, artifact, "preset", tune)
    r.ranges["KNURLOGIC_KV_BITS"] = ["bf16"] + [
        str(b) for b in bits if str(b) in knobs.KV_BITS_OFFERED]
    if not bits:
        r.notes.append(f"KV cache stays bf16: {why}")
    elif kv_bits is not None:
        r.notes.append(f"KV cache counted at {kv_bits} bits "
                       f"({measured.kv_bytes_per_element(kv_bits):.3g} bytes per "
                       f"element against bf16's 2): {why}")


def _long_context_cfg(r: Resolution, artifact: Artifact, long_context,
                      working_set_bytes: int, holds_bytes: int,
                      kv_bits) -> dict:
    """The config the window is read from under KNURLOGIC_LONG_CONTEXT,
    with the knob emitted (off / yarn where the family's model card
    documents YaRN) and the KV room for the YaRN window warned about."""
    cfg = artifact.raw_config or {}
    mt = artifact.model_type
    if context_window.long_context_family(mt) is None:
        return cfg
    mode = context_window.long_context_of(long_context)
    emit(r, artifact, "long_context", mode)
    r.ranges["KNURLOGIC_LONG_CONTEXT"] = list(context_window.LONG_CONTEXT_VALUES)
    if mode == "off":
        return cfg
    cfg = context_window.with_long_context(cfg, mode)
    window, why = context_window.model_window(cfg)
    r.notes.append(f"long context (YaRN): the cap is {window:,} tokens "
                   f"({why}); static YaRN may cost a little quality on "
                   f"short prompts (Qwen's model card)")
    if working_set_bytes > 0:
        short = context_window.long_context_room(
            artifact, working_set_bytes, holds_bytes, window, kv_bits)
        if short:
            r.warnings.append(short)
    return cfg
def kv_refusal(artifact: Artifact, kv_bits) -> str | None:
    """Why this artifact cannot launch with `kv_bits`, or None."""
    if kv_bits is None:
        return None
    bits, why = measured.kv_quant_for(artifact.model_type)
    if int(kv_bits) not in bits:
        return (f"KV cache at {kv_bits} bits is refused for "
                f"{artifact.model_type}: {why}")
    return None
