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
does the sharding. When nobody says, this assumes the shard is
proportional to the node's working set, which is an ASSUMPTION and is
recorded as a note on every resolution that rides on it, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knurlogic.tuning import settings as S
from knurlogic.machine.artifact import Artifact

GIB = 1 << 30


@dataclass
class Resolution:
    env: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    #: What a vision rung holds besides its text weights (`vision_budget`),
    #: or None for a text-only artifact.
    vision: dict | None = None
    #: knob -> the values THIS artifact may take, where they are narrower
    #: than settings.KNOB_RANGE (KV bits on a family that refuses them)
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


#: Knobs `engine.serve` turns into argv or an mlx call, so they are real
#: whether or not an artifact's bundled runtime reads them.
ENGINE_CONSUMED = ("prefill_chunk", "cache_limit_gb", "context_length", "mtp", "mtp_dynamic", "kv_bits",
                   "cross_chip", "preset")


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


def resolve(artifact: Artifact, budget, profile: str | None = None,
            tune: str = "balanced", store_bytes: int | None = None,
            holds_bytes: int | None = None, kv_bits=None):
    """Resolve every knob for this artifact against a budget.

    `budget` is either a byte count -- one box, and the return is a
    `Resolution`, exactly as it always was -- or nodes (a `Node`, a sequence
    of them, or {name: working_set_bytes}), in which case the return is a
    `ClusterResolution` carrying one `Resolution` per node.

    `profile` is None (the default): each rung keeps the numerics it
    SHIPPED (`numerics_for`). "v1.5" or "v2" forces that profile's two
    numerics-active flags, and only because someone asked -- it used to
    default to v1.5 and so silently turned off bf16 I/O on the rungs that
    shipped it on (design D1).

    `store_bytes` is a live image store's `budget_bytes()`, when one exists;
    otherwise a vision rung is budgeted at the store's default bound
    (`vision_budget`).

    `holds_bytes`: what this box holds of the artifact when it is one rank
    of a split (its share); default, the whole artifact.

    `kv_bits`: the KV precision the model will launch with (a launch
    setting), for what its KV is counted at; None = bf16.
    """
    if profile is not None and profile not in S.RUNTIME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(S.RUNTIME_PROFILES)}")

    if tune not in S.TUNE_PROFILES:
        raise ValueError(f"tune must be one of {sorted(S.TUNE_PROFILES)}")
    if isinstance(budget, (int, float)):
        holds = artifact.bytes_on_disk if holds_bytes is None \
            else int(holds_bytes)
        return _resolve_one(artifact, int(budget), holds,
                            profile, tune, store_bytes=store_bytes,
                            kv_bits=kv_bits)
    return resolve_cluster(artifact, budget, profile, tune,
                           store_bytes=store_bytes)


def resolve_cluster(artifact: Artifact, budget, profile: str | None = None,
                    tune: str = "balanced",
                    store_bytes: int | None = None) -> ClusterResolution:
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
                         vision=(i == 0))
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
    _ring_consistent(c)
    for name, r in c.nodes.items():
        c.warnings += [f"{name}: {w}" for w in r.warnings]
    return c


def _ring_consistent(c: ClusterResolution) -> None:
    """One prompt chunk on every rank: the smallest any node needs.

    Per-node resolution is right for per-node memory and wrong for this. A
    pipeline's ranks process the same chunks, so a tight node's narrow chunk
    has to be everyone's -- and a rank that disagrees is a desync, not a
    tuning difference.
    """
    want = {n: S.engine_settings(r.env).get("prefill_step_size")
            for n, r in c.nodes.items()}
    got = [v for v in want.values() if v]
    if not got:
        return
    ring = min(got)
    for name, r in c.nodes.items():
        for alias in S.KNOB_ALIASES["prefill_chunk"]:
            if alias in r.env:
                r.env[alias] = str(ring)
        if want[name] and want[name] != ring:
            r.notes.append(
                f"prompt chunk {ring}, not the {want[name]} this node alone "
                f"would take: it is ring-wide, and every rank must match")
    if len(set(got)) > 1:
        c.notes.append(f"prompt chunk {ring} on every rank (the tightest "
                       f"node's); ranks that disagree desync")


def _tower_bytes(artifact: Artifact) -> tuple:
    """(tower bytes, bytes of them OUTSIDE what bytes_on_disk counted,
    tensors) from the safetensors headers -- read, never guessed.
    bytes_on_disk sums the artifact directory's top-level *.safetensors;
    anything deeper would be extra."""
    import json
    import struct

    root = artifact.path
    total = outside = count = 0
    try:
        files = sorted(root.rglob("*.safetensors"))
    except OSError:
        return 0, 0, 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                if n <= 0 or n > (1 << 28):
                    continue
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            continue
        if not isinstance(header, dict):
            continue
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if not k.startswith(S.VISION_TOWER_PREFIXES):
                continue
            try:
                a, b = v.get("data_offsets", (0, 0))
                a, b = int(a), int(b)
            except (TypeError, ValueError):
                continue          # a malformed entry, not a crash
            total += b - a
            count += 1
            if f.parent != root:
                outside += b - a
    return total, outside, count


def kv_bytes_per_token(tc: dict, kv_bits=None) -> tuple:
    """(bytes, why): K and V for one token over the layers whose cache
    grows with context. Hybrid models (Qwen3.5's linear layers, gemma's
    sliding windows) are counted by their full-attention layers only.
    `kv_bits` (8/6/4, None = bf16): the attention K/V's stored precision
    (KNURLOGIC_KV_BITS); an MLA latent is never quantized."""
    layers = int(tc.get("num_hidden_layers") or 0)
    types = tc.get("layer_types")
    interval = tc.get("full_attention_interval")
    if isinstance(types, list) and tc.get("kv_lora_rank"):
        # MLA (GLM-5.3's deepseek_sparse_attention): a layer caches its
        # compressed latent and the DSA indexer's key, not K and V per
        # head -- counted as full attention it read 0 layers (GLM's are
        # not named full_attention), and as K,V per head it would be ~10x
        mla = sum(1 for t in types if t != "linear_attention")
        width = (int(tc.get("kv_lora_rank") or 0)
                 + int(tc.get("qk_rope_head_dim") or 0)
                 + int(tc.get("index_head_dim") or 0))
        per = mla * width * S.VISION_KV_DTYPE_BYTES
        return per, (f"{mla} MLA layers of {len(types)} x {width} "
                     f"(latent + rope + indexer key) x bf16")
    if isinstance(types, list) and types:
        full = sum(1 for t in types if t == "full_attention")
        how = f"{full} full-attention of {len(types)} layers"
    elif interval:
        full = layers // int(interval)
        how = f"{full} full-attention layers (every {interval}th of {layers})"
    else:
        full = layers
        how = f"{layers} layers"
    heads = int(tc.get("num_attention_heads") or 0)
    kv = int(tc.get("num_key_value_heads") or heads or 0)
    hd = int(tc.get("head_dim") or (
        int(tc.get("hidden_size") or 0) // heads if heads else 0))
    el = S.kv_bytes_per_element(kv_bits)
    per = int(2 * full * kv * hd * el)
    dt = "bf16" if kv_bits is None else f"{kv_bits}-bit (+ group scales)"
    return per, f"{how} x {kv} KV heads x {hd} dims x K,V x {dt}"


def step_margin(working_set_bytes: int) -> int:
    """The floor of the scheduler's step margin (engine/runtime/scheduler.py
    `_margin`): 5% of the working set, at least 4 GiB. Repeated here, not
    imported, because the scheduler lives on the mlx side of the line and
    this has to answer before anything loads."""
    return max(4 * GIB, int(working_set_bytes) // 20)


def context_room(working_set_bytes: int, weights_bytes: int,
                 cfg: dict, kv_bits=None) -> dict:
    """What a model that fits leaves for its conversations: the working set
    (or allowance) less the weights less the step margin, and about how
    many tokens of context that is at the model's KV bytes per token --
    shared by every conversation at once, not each one's.

    Why it is said at all: GLM-5.3 2.7bpw "fit" on the 128 GB M4 and left
    about 6 GiB, and four long agent conversations could never run
    (2026-09-26). A fit that leaves no room to talk is not much of a fit.
    `small` is under a fifth of the model's own window, or under 2 GiB."""
    cfg = cfg or {}
    tc = cfg.get("text_config") or cfg
    per, why = kv_bytes_per_token(tc, kv_bits)
    window = int(tc.get("max_position_embeddings")
                 or cfg.get("max_position_embeddings") or 0)
    ws = int(working_set_bytes or 0)
    margin = step_margin(ws)
    left = max(ws - int(weights_bytes) - margin, 0)
    tokens = left // per if per else 0
    small = left < 2 * GIB or bool(per and window and tokens < window / 5)
    return {"fits": bool(ws) and int(weights_bytes) <= ws,
            "working_set_bytes": ws, "weights_bytes": int(weights_bytes),
            "margin_bytes": margin, "left_bytes": left,
            "kv_bytes_per_token": per, "kv_why": why,
            "tokens": tokens, "window": window, "small": small,
            "text": room_text(left, tokens, per)}


def room_for(weights_bytes: int, cfg: dict,
             working_set_bytes: int | None = None, kv_bits=None) -> dict:
    """`context_room` against THIS machine: its GPU working set under the
    knurlogic allowance. Not memory available now -- what other programs
    hold today comes and goes; the working set is what the scheduler's
    guard will count against for the life of the load."""
    if working_set_bytes is None:
        from knurlogic.machine import allowance, wired
        working_set_bytes = allowance.cap(wired.detected_working_set_bytes())
    return context_room(working_set_bytes, weights_bytes, cfg, kv_bits)


def room_text(left: int, tokens: int, per: int) -> str:
    """'leaves 6 GiB, about 400k tokens of context across all
    conversations' -- one sentence, shared by the page and doctor."""
    t = (f"{tokens / 1e6:.1f}M" if tokens >= 1e6 else
         f"{tokens / 1e3:.0f}k" if tokens >= 1e3 else str(tokens))
    return (f"leaves {left / GIB:.0f} GiB"
            + (f", about {t} tokens of context across all conversations"
               if per else " (its KV size per token is not known)"))


def vision_budget(artifact: Artifact, store_bytes: int | None = None,
                  kv_bits=None) -> dict | None:
    """What a vision rung holds besides its text weights, term by term,
    or None for an artifact with no `vision_config`.

    Stdlib only (no mlx, no family import): tuning/ must not import mlx,
    and this has to answer BEFORE a load. `extra_bytes` is what the
    resolver adds to what the box holds; each term has its note."""
    cfg = artifact.raw_config or {}
    if not isinstance(cfg.get("vision_config"), dict):
        return None
    from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES

    tower, outside, n = _tower_bytes(artifact)
    live = store_bytes is not None
    store = int(store_bytes) if live else DEFAULT_MAX_BYTES
    tc = cfg.get("text_config") or cfg
    per_tok, kv_why = kv_bytes_per_token(tc, kv_bits)
    toks = S.VISION_KV_IMAGES * S.VISION_KV_TOKENS_PER_IMAGE
    kv = per_tok * toks
    notes = [
        (f"vision tower: {tower / GIB:.2f} GiB in {n} tensors, read from "
         f"the safetensors headers; "
         + (f"{outside / GIB:.2f} GiB of it sits outside the files the "
            f"artifact size counts, so that much is added"
            if outside else "already inside the artifact's size, so "
                            "nothing is added for it")),
        (f"image store: {store / GIB:.2f} GiB reserved -- "
         + ("the live store's budget (its bound, or more while prompt-cache "
            "entries pin images in use)" if live else
            "the store's default bound (engine/vision/store.py "
            "DEFAULT_MAX_BYTES)")),
        (f"image KV allowance: {kv / GIB:.2f} GiB for "
         f"{S.VISION_KV_IMAGES} images x {S.VISION_KV_TOKENS_PER_IMAGE} "
         f"tokens ({kv_why}) -- an ALLOWANCE for the context images add, "
         f"not a measurement"),
    ]
    return {"tower_bytes": tower, "tower_outside_bytes": outside,
            "tower_tensors": n, "store_bytes": store,
            "store_is_live": live, "kv_allowance_bytes": kv,
            "kv_bytes_per_token": per_tok, "kv_tokens": toks,
            "extra_bytes": outside + store + kv, "notes": notes}


def _resolve_one(artifact: Artifact, working_set_bytes: int,
                 holds_bytes: int, profile: str | None,
                 tune: str = "balanced", store_bytes: int | None = None,
                 vision: bool = True, kv_bits=None) -> Resolution:
    """One box. `holds_bytes` is what this box holds of the artifact, which
    is the whole thing unless something sharded it. A vision rung also
    holds its tower, its image store and its image KV (`vision_budget`),
    counted here, before the headroom every knob below is sized from."""
    r = Resolution()
    vb = vision_budget(artifact, store_bytes, kv_bits) if vision else None
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

    tight = working_set_bytes > 0 and \
        headroom < S.tight_headroom_bytes(working_set_bytes)
    family, family_why = S.prefill_chunk_for(artifact.model_type)
    # the small default everywhere (settings.PREFILL_CHUNK_DEFAULT says why);
    # a family's measured width is spent only when asked for: tune=fast on
    # a box with room for it
    prefill = S.PREFILL_CHUNK_DEFAULT
    if tune == "fast" and not tight:
        prefill = max(prefill, family)
    asked = t.get("VQLAB_PREFILL_CHUNK")
    if asked is not None and tune == "fast":
        asked = max(asked, family)
    if family > S.PREFILL_CHUNK_DEFAULT and prefill < family:
        r.notes.append(
            f"prompt chunk {prefill}: {family} was {family_why}, and buys "
            f"some prefill for a step transient several times "
            f"larger -- take it with tune=fast or per base model")
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
    window, _ = S.model_window(artifact.raw_config or {})
    if window:
        # the model's own window: the cap a person lowers, never raises past
        # (settings.check_knob refuses more), so the control stops there
        emit(r, artifact, "context_length", window)
        steps = [v for v in S.KNOB_RANGE["KNURLOGIC_CONTEXT_LENGTH"][0]
                 if v < window]
        r.ranges["KNURLOGIC_CONTEXT_LENGTH"] = steps + [window]
    launch, launch_notes = S.preset_launch(tune, artifact.model_type)
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
    model_launch(r, artifact, kv_bits, tune)
    r.notes.extend(launch_notes)
    _preset_record(r, tune, launch)

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
        env, note = numerics_for(artifact, profile)
        r.env.update(env)
        r.notes.append(note)

    return r


def _preset_record(r: Resolution, tune: str, launch: dict) -> None:
    """Which env values the preset put there: its launch settings, plus the
    prompt chunk / cache limit its profile names."""
    t = S.TUNE_PROFILES[tune]
    logicals = set(launch)
    if "VQLAB_PREFILL_CHUNK" in t:
        logicals.add("prefill_chunk")
    if "VQLAB_CACHE_LIMIT_GB" in t:
        logicals.add("cache_limit_gb")
    names = {n for lg in logicals for n in S.KNOB_ALIASES.get(lg, (lg,))}
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
    launch, _ = S.preset_launch(tune, artifact.model_type)
    return {S.KNOB_ALIASES[k][0]: str(v) for k, v in launch.items()
            if S.KNOB_ALIASES[k][0] in S.MODEL_KNOBS}


def model_launch(r: Resolution, artifact: Artifact, kv_bits=None,
                 tune: str = S.PRESET_DEFAULT) -> None:
    """The model's own launch settings: MTP drafting and its controller
    where a head ships beside the weights, and the KV precision this
    family allows (every value but bf16 refused, with the reason, where
    its caches cannot be quantized)."""
    from knurlogic.engine.mtp import find_head
    try:
        head = find_head(artifact.path)
    except Exception:
        head = None
    launch, _ = S.preset_launch(tune, artifact.model_type)
    if head is not None:
        emit(r, artifact, "mtp", launch.get("mtp", "on"))
        emit(r, artifact, "mtp_dynamic", launch.get("mtp_dynamic", "on"))
    bits, why = S.kv_quant_for(artifact.model_type)
    emit(r, artifact, "kv_bits", launch.get("kv_bits", "bf16"))
    emit(r, artifact, "cross_chip", launch.get("cross_chip", "off"))
    emit(r, artifact, "preset", tune)
    r.ranges["KNURLOGIC_KV_BITS"] = ["bf16"] + [str(b) for b in bits]
    if not bits:
        r.notes.append(f"KV cache stays bf16: {why}")
    elif kv_bits is not None:
        r.notes.append(f"KV cache counted at {kv_bits} bits "
                       f"({S.kv_bytes_per_element(kv_bits):.3g} bytes per "
                       f"element against bf16's 2): {why}")


def kv_refusal(artifact: Artifact, kv_bits) -> str | None:
    """Why this artifact cannot launch with `kv_bits`, or None."""
    if kv_bits is None:
        return None
    bits, why = S.kv_quant_for(artifact.model_type)
    if int(kv_bits) not in bits:
        return (f"KV cache at {kv_bits} bits is refused for "
                f"{artifact.model_type}: {why}")
    return None


def _numerics_source(artifact: Artifact, flag: str, source: str):
    if source == "declared":
        d = artifact.declared_knobs().get(flag)
        if isinstance(d, dict) and d.get("default") is not None:
            return str(d["default"])
        return None
    if source == "published":
        from knurlogic.engine.vq import rungs
        row = rungs.rung(artifact.path)
        if not row:
            return None
        # an arc6-era bundle has no such flag; its arithmetic is the flag
        # off, which rungs.json records as an inferred knob
        return (row.get("published_defaults", {}).get(flag)
                or row.get("knobs", {}).get(flag)
                or ("0" if flag in row.get("inferred_knobs", ()) else None))
    if source == "bundled":
        from knurlogic.engine.vq import rungs
        return rungs.flag_defaults(artifact.runtime_source()).get(flag)
    raise ValueError(source)


def numerics_for(artifact: Artifact, profile: str | None = None):
    """(env, note): the numerics-active flags for this artifact.

    A profile someone ASKED for wins, and the note names what it overrode.
    Otherwise every flag comes from the rung itself, first source in
    S.NUMERICS_SOURCES that answers. The bug this replaces: a v1.5 default
    applied to every VQ artifact, forcing Flash-Next 2.1 and Qwen3.6-35B-A3B
    3.8/4.6/5.4 -- published with both flags ON -- to run off (F103/F105:
    numerics-active, up to +0.97% ppl)."""
    own, where = {}, {}
    for flag in S.NUMERICS_FLAGS:
        for src in S.NUMERICS_SOURCES:
            v = _numerics_source(artifact, flag, src)
            if v is not None:
                own[flag], where[flag] = v, src
                break
    if profile is not None:
        forced = dict(S.RUNTIME_PROFILES[profile])
        changed = {f: (own[f], v) for f, v in forced.items()
                   if f in own and own[f] != v}
        note = f"runtime profile {profile} (asked for)"
        if changed:
            note += " -- overrides what this rung shipped: " + ", ".join(
                f"{f} {a}->{b}" for f, (a, b) in changed.items())
        return forced, note
    if not own:
        return {}, ("numerics: nothing declares them -- the runtime's own "
                    "defaults stand")
    srcs = sorted(set(where.values()))
    return own, ("numerics as shipped (" + ", ".join(
        f"{f}={v}" for f, v in own.items()) + f"; from {'/'.join(srcs)})")


# ------------------------------------------------------------ tensor split
#
# One model served by N ranks, every layer's weights split N ways (the
# qwen3_5 families: engine/runtime/tensor.py does the split). Pure
# arithmetic over the config and the safetensors headers, so a refusal is
# said -- with its numbers -- before anything loads.

#: the model types engine/runtime/tensor.py knows how to split
TENSOR_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")

#: layer weights split N ways under tensor; everything else is replicated
_TENSOR_SHARDED = (
    ".linear_attn.conv1d.", ".linear_attn.in_proj_", ".linear_attn.dt_bias",
    ".linear_attn.A_log", ".linear_attn.out_proj.",
    ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
    ".self_attn.o_proj.",
    ".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.",
    ".mlp.shared_expert.", ".mlp.switch_mlp.",
    ".mlp.experts.",
)


def tensor_sharded(name: str) -> bool:
    """Is this weight split across ranks under tensor? A VQ codebook never
    is: it is a lookup table the codes index, replicated whole."""
    if name.endswith("codebook"):
        return False
    return ".layers." in name and any(s in name for s in _TENSOR_SHARDED)


def tensor_refusals(cfg: dict, n: int) -> list:
    """Why this config cannot be split `n` ways, one line per reason with
    its arithmetic; [] when it can."""
    out = []
    if n < 2:
        return out
    tc = cfg.get("text_config", cfg)
    types = {cfg.get("model_type"), tc.get("model_type")}
    if not types & set(TENSOR_TYPES):
        out.append(f"tensor split knows {', '.join(TENSOR_TYPES[:2])}; this "
                   f"is {cfg.get('model_type')!r}")
        return out

    def div(what, v, why=""):
        if v is None:
            return
        if int(v) % n:
            out.append(f"{what} = {v} is not divisible by {n} ranks "
                       f"({v} / {n} = {int(v) / n:g}){why}")

    div("num_attention_heads", tc.get("num_attention_heads"))
    kv = tc.get("num_key_value_heads")
    if kv:
        if kv >= n:
            div("num_key_value_heads", kv)
        elif n % kv:
            out.append(f"num_key_value_heads = {kv} is fewer than {n} ranks "
                       f"and does not divide them ({n} % {kv} = {n % kv}): "
                       f"the heads cannot be repeated evenly")
    div("linear_num_key_heads", tc.get("linear_num_key_heads"))
    div("linear_num_value_heads", tc.get("linear_num_value_heads"))
    if tc.get("num_experts"):
        div("moe_intermediate_size", tc.get("moe_intermediate_size"))
        div("shared_expert_intermediate_size",
            tc.get("shared_expert_intermediate_size"))
    else:
        div("intermediate_size", tc.get("intermediate_size"))

    # sharded-to-all layers split their INPUT axis: each rank's slice must
    # start on a quantization group
    q = cfg.get("quantization") or {}
    g = q.get("group_size") if isinstance(q, dict) else None
    if g:
        hd = tc.get("head_dim") or (tc.get("hidden_size", 0)
                                    // max(tc.get("num_attention_heads") or 1, 1))
        ins = {"self_attn.o_proj": (tc.get("num_attention_heads") or 0) * hd,
               "linear_attn.out_proj": (tc.get("linear_num_value_heads") or 0)
               * (tc.get("linear_value_head_dim") or 0),
               "shared_expert.down_proj":
                   tc.get("shared_expert_intermediate_size") or 0}
        for what, IN in ins.items():
            if IN and (IN // n) % g:
                out.append(f"{what}: input {IN} / {n} = {IN // n} is not a "
                           f"multiple of the quantization group {g}")

    if cfg.get("vq_linear"):
        out.append(f"{len(cfg['vq_linear'])} VQ dense linear(s) (vq_linear): "
                   f"not split by tensor in this build")
    if cfg.get("vq_embed"):
        out.append(f"{len(cfg['vq_embed'])} VQ embedding(s) (vq_embed): not "
                   f"split by tensor in this build")
    for path, m in sorted((cfg.get("vq_modules") or {}).items()):
        IN, OUT = int(m.get("in", 0)), int(m.get("out", 0))
        G, D = int(m.get("group", 64)), int(m.get("dim", 1))
        packed = bool(m.get("pack_bits"))
        if path.endswith("down_proj"):
            # sharded-to-all: codes split on their input axis. Packed codes
            # are uint32 words holding 32 codes per BITS words, so a slice
            # must hold whole 32-code blocks -- 32*dim inputs -- and whole
            # scale groups.
            unit = max(G, 32 * D) if packed else max(G, D)
            if IN % n or (IN // n) % unit:
                out.append(
                    f"{path}: input {IN} / {n} = {IN / n:g}, not a multiple "
                    f"of {unit} (max(group {G}, "
                    + (f"32 x dim {D}" if packed else f"dim {D}")
                    + ")): a rank's slice would cut a "
                    + ("packed code word" if packed else "scale group"))
        elif OUT % n:
            out.append(f"{path}: output {OUT} / {n} = {OUT / n:g}")
        if len(out) > 12:
            out.append("... (and more)")
            break
    return out


def tensor_placement_of(tensors: dict, n: int) -> dict:
    """{name: bytes} -> what each of `n` ranks holds under tensor: the
    sharded weights' Nth plus every replicated one."""
    sharded = sum(b for k, b in tensors.items() if tensor_sharded(k))
    replicated = sum(b for k, b in tensors.items() if not tensor_sharded(k))
    return {"ranks": n, "sharded_bytes": sharded,
            "replicated_bytes": replicated,
            "per_rank_bytes": -(-sharded // max(n, 1)) + replicated}


def tensor_placement(artifact: Artifact, n: int) -> dict:
    """tensor_placement_of over the artifact's top-level safetensors
    headers (the tower and a packed MTP head are not the trunk: neither
    loads under tensor)."""
    import json
    import struct

    sizes = {}
    for f in sorted(artifact.path.glob("*.safetensors")):
        if f.name.startswith(("mtp", "model-vision")):
            continue
        try:
            with open(f, "rb") as fh:
                (hn,) = struct.unpack("<Q", fh.read(8))
                if hn <= 0 or hn > (1 << 28):
                    continue
                header = json.loads(fh.read(hn))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.startswith(S.VISION_TOWER_PREFIXES):
                continue
            a, b = v.get("data_offsets", (0, 0))
            sizes[k] = int(b) - int(a)
    return tensor_placement_of(sizes, n)


# ------------------------------------------------------------- rank order
#
# Nobody types --rank. The cluster page asks this for the order, passes
# --rank/--world to each machine's `serve`, and lets the user drag a
# machine to the front (an explicit order, which wins).
#
# Rank 0 does the CPU and Python side -- HTTP, the scheduler, tokenizing,
# sampling, encoding the plan -- while under tensor every rank's GPU work
# is equal. So the leader is the fastest single core: the newest chip
# generation, then the higher P-core clock; then the most free memory (it
# may host extras, like the vision tower); then the order given.

#: link kinds, fastest first
LINK_SPEED = ("rdma", "tb5", "tb4", "ethernet", "wifi")


def _link_rank(kind) -> int:
    k = str(kind or "").lower()
    return LINK_SPEED.index(k) if k in LINK_SPEED else len(LINK_SPEED)


def chip_generation(chip) -> int:
    """"Apple M4 Max" -> 4; 0 when it does not say."""
    import re
    m = re.search(r"\bM(\d+)\b", str(chip or ""))
    return int(m.group(1)) if m else 0


def leader_key(m: dict, position: int = 0) -> tuple:
    """Sort key for the leader, smallest first: newest chip generation,
    higher P-core clock, more free memory, earlier in the given order."""
    free = m.get("free_bytes")
    if free is None:
        free = m.get("working_set_bytes") or 0
    return (-chip_generation(m.get("chip")),
            -float(m.get("p_core_ghz") or 0.0), -int(free), position)


def rank_order(machines: list, explicit: list | None = None) -> list:
    """Machine names in rank order.

    `machines`: [{"name", "chip" ("Apple M4 Max"), "p_core_ghz",
    "free_bytes" (else "working_set_bytes"), "links": {other name: link
    kind}, ...}]; anything else (memory bandwidth, ...) rides along for
    later placement and is ignored here. Rank 0 is `leader_key`'s first.
    Each next rank is the unplaced machine the previous one reaches over
    the fastest link (LINK_SPEED; the ring follows the links), ties by
    `leader_key`.

    `explicit`: names in the order wanted (the page's drag). Every machine
    must appear once; it is returned as is."""
    names = [m["name"] for m in machines]
    if len(set(names)) != len(names):
        raise ValueError(f"machine names repeat: {names}")
    if explicit is not None:
        if sorted(explicit) != sorted(names):
            raise ValueError(f"explicit order {list(explicit)} is not a "
                             f"permutation of the machines {names}")
        return list(explicit)
    if not machines:
        return []
    pos = {n: i for i, n in enumerate(names)}
    by = {m["name"]: m for m in machines}
    order = [min(machines, key=lambda m: leader_key(m, pos[m["name"]]))["name"]]
    left = [n for n in names if n != order[0]]
    while left:
        here = by[order[-1]].get("links") or {}
        nxt = min(left, key=lambda n: (_link_rank(here.get(n)),)
                  + leader_key(by[n], pos[n]))
        order.append(nxt)
        left.remove(nxt)
    return order


# ---------------------------------------------------------- pipeline split
#
# One model served by N ranks, each holding a contiguous run of layers
# (engine/runtime/pipeline.py). Rank 0 -- the leader, which samples -- holds
# the LAST layers, so the logits are born where they are used and nothing
# is gathered; rank N-1 holds the first layers and embeds. Pure arithmetic,
# so the split is said, with its reason, before anything loads.

#: model types engine/runtime/pipeline.py knows how to slice (each has its
#: own index fixups there); anything else is refused with a reason
PIPELINE_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text",
                  "qwen3_5_moe_text", "glm5_next", "qwen4_exp",
                  "qwen4_exp_text")

_PIPELINE_WHY_NOT = {
    "gemma4": "gemma4 shares KV across layers (a layer reads a cache another "
              "layer wrote), so a cut between them would need that cache on "
              "two ranks; not built",
}

#: memory bandwidth (GB/s) of chips that come in ONE bandwidth only. A Max
#: that is sold binned (M3 Max 300/400, M4 Max 410/546) is not here:
#: unknown is said, never guessed.
CHIP_BANDWIDTH_GBS = {
    "Apple M1 Max": 400.0, "Apple M1 Ultra": 800.0,
    "Apple M2 Max": 400.0, "Apple M2 Ultra": 800.0,
    "Apple M3 Ultra": 819.0, "Apple M4 Pro": 273.0,
}


def chip_bandwidth_gbs(chip) -> float | None:
    return CHIP_BANDWIDTH_GBS.get(str(chip or "").strip())


def pipeline_refusals(cfg: dict, n: int) -> list:
    """Why this config cannot be pipelined `n` ways; [] when it can."""
    if n < 2:
        return []
    tc = cfg.get("text_config", cfg) or {}
    types = {cfg.get("model_type"), tc.get("model_type")}
    if not types & set(PIPELINE_TYPES):
        mt = cfg.get("model_type")
        why = next((w for k, w in _PIPELINE_WHY_NOT.items()
                    if any(str(t or "").startswith(k) for t in types)), None)
        return [f"pipeline split knows {', '.join(PIPELINE_TYPES[:2])}, "
                f"glm5_next and qwen4_exp; this is {mt!r}"
                + (f": {why}" if why else "")]
    L = tc.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if L is not None and int(L) < n:
        return [f"num_hidden_layers = {L} is fewer than {n} ranks: every "
                f"rank holds at least one layer"]
    return []


def layer_bytes_of(tensors: dict, n_layers: int) -> tuple:
    """{name: bytes} -> ([bytes of layer i for i < n_layers], bytes outside
    the layers). A tensor of layer i is one whose name holds `.layers.i.`;
    an index >= n_layers, or a name under `mtp.` (a grafted head), counts
    as outside."""
    import re
    per = [0] * n_layers
    other = 0
    rx = re.compile(r"\.layers\.(\d+)\.")
    for k, b in tensors.items():
        m = None if k.split(".")[0] == "mtp" else rx.search(k)
        i = int(m.group(1)) if m else -1
        if 0 <= i < n_layers:
            per[i] += int(b)
        else:
            other += int(b)
    return per, other


def _largest_remainder(n: int, weights: list, floor: list, cap: list) -> list:
    """n items shared by weight, each share within [floor, cap]; ties go
    to the lower index. Deterministic: plain arithmetic, no sort instability
    (keys carry the index)."""
    k = len(weights)
    out = list(floor)
    left = n - sum(out)
    free = [i for i in range(k) if out[i] < cap[i]]
    while left > 0 and free:
        tot = sum(weights[i] for i in free)
        if tot <= 0:
            want = {i: left / len(free) for i in free}
        else:
            want = {i: left * weights[i] / tot for i in free}
        give = {i: min(int(want[i]), cap[i] - out[i]) for i in free}
        if sum(give.values()) == 0:
            # remainders: the largest fractional part first, lower rank on a tie
            order = sorted(free, key=lambda i: (-(want[i] - int(want[i])), i))
            give = {i: 0 for i in free}
            for i in order[:left]:
                give[i] = 1
        for i, g in give.items():
            out[i] += g
            left -= g
        free = [i for i in range(k) if out[i] < cap[i]]
    return out


def _byte_bounds(layer_bytes: list, weights: list, cap: list):
    """Contiguous runs by the real bytes: rank n-1 takes the first layers,
    each rank's run as near its weight's share of the bytes still unplaced
    as whole layers allow, within what it can hold, leaving every later rank
    one layer; rank 0 takes the rest. [(start, end) per rank], or None when
    a rank would hold more than it can."""
    n, L = len(weights), len(layer_bytes)
    bounds = [None] * n
    at, left = 0, float(sum(layer_bytes))
    for r in range(n - 1, 0, -1):
        wsum = sum(weights[:r + 1])
        target = left * (weights[r] / wsum if wsum > 0 else 1.0 / (r + 1))
        end, acc = at, 0
        while end < L - r:            # leave ranks r-1 .. 0 a layer each
            nxt = acc + layer_bytes[end]
            if nxt > cap[r] or (end > at and
                                abs(nxt - target) > abs(acc - target)):
                break
            acc, end = nxt, end + 1
        if end == at:
            return None
        bounds[r] = (at, end)
        left -= acc
        at = end
    bounds[0] = (at, L)
    if any(sum(layer_bytes[a:b]) > cap[r] for r, (a, b) in enumerate(bounds)):
        return None
    return bounds


def pipeline_shares(layer_bytes: list, ranks: list, other_bytes: int = 0) -> dict:
    """Which layers each rank holds.

    `layer_bytes`: bytes of each layer, in order. `ranks`: in rank order,
    [{"name", "working_set_bytes", "memory_bandwidth_gbs" (None: unknown)}].
    `other_bytes`: what every rank holds besides its layers (embeddings,
    final norm, lm_head -- replicated).

    A rank's weight is what it can hold (working set less the replicated
    bytes), times its memory bandwidth when EVERY rank's is known (decode
    reads each layer's weights once per step, so a faster rank should read
    more of them); a mix of known and unknown bandwidths weighs by capacity
    only, and says so. What a rank can hold leaves its step margin
    (step_margin) free; the reason says what each rank leaves. Every rank holds at least one layer and no rank more
    than fits; rank 0 holds the LAST run of layers, rank N-1 the first.

    -> {"layers": [count per rank], "bounds": [(start, end) per rank],
        "bytes": [layer bytes per rank], "weights": [...], "reason": str}
    Raises ValueError, with the arithmetic, when it cannot be done."""
    n, L = len(ranks), len(layer_bytes)
    if n < 1:
        raise ValueError("no ranks")
    if L < n:
        raise ValueError(f"{L} layers cannot give each of {n} ranks one")
    names = [str(r.get("name", f"rank{i}")) for i, r in enumerate(ranks)]
    wss = [int(r.get("working_set_bytes") or 0) for r in ranks]
    # a rank's layers leave its step margin free, as a single machine's
    # weights do (the scheduler's floor: 5%, at least 4 GiB)
    cap = [w - int(other_bytes) - step_margin(w) for w in wss]
    for i, c in enumerate(cap):
        if c <= 0:
            raise ValueError(
                f"{names[i]}: working set {wss[i] / GIB:.1f} GiB "
                f"holds none of the layers after the {other_bytes / GIB:.1f} "
                f"GiB every rank keeps (embeddings, norm, lm_head) and its "
                f"{step_margin(wss[i]) / GIB:.1f} GiB step margin")
    bws = [r.get("memory_bandwidth_gbs") for r in ranks]
    known = all(b for b in bws)
    weights = [cap[i] * (float(bws[i]) if known else 1.0) for i in range(n)]
    avg = sum(layer_bytes) / L if L else 0
    # a rank's ceiling in layers, by the average layer; checked exactly below
    ceil = [max(1, min(L, int(cap[i] // avg))) if avg else L for i in range(n)]
    if sum(ceil) < L:
        raise ValueError(
            f"{L} layers x {avg / GIB:.2f} GiB average = "
            f"{sum(layer_bytes) / GIB:.1f} GiB; the ranks hold "
            + " + ".join(f"{names[i]} {cap[i] / GIB:.1f}" for i in range(n))
            + f" = {sum(cap) / GIB:.1f} GiB of layers")
    counts = _largest_remainder(L, weights, [1] * n, ceil)
    # stage order: rank n-1 first ... rank 0 last
    bounds = [None] * n
    at = 0
    for r in range(n - 1, -1, -1):
        bounds[r] = (at, at + counts[r])
        at += counts[r]
    got = [sum(layer_bytes[a:b]) for a, b in bounds]
    over = [i for i in range(n) if got[i] > cap[i]]
    if over:
        # counting by the average layer is wrong when layers are not alike
        # (Qwen3.8 Flash's layer 1 carries a 42 GiB n-gram embedding): cut
        # by the real bytes instead, and refuse only when that fails too
        alt = _byte_bounds(layer_bytes, weights, cap)
        if alt is not None:
            bounds = alt
            counts = [b - a for a, b in bounds]
            got = [sum(layer_bytes[a:b]) for a, b in bounds]
            over = []
    if over:
        i = over[0]
        raise ValueError(
            f"{names[i]}: layers {bounds[i][0]}..{bounds[i][1] - 1} are "
            f"{got[i] / GIB:.1f} GiB against {cap[i] / GIB:.1f} GiB it can "
            f"hold")
    how = ("capacity x memory bandwidth" if known else
           "capacity only (memory bandwidth unknown on "
           + ", ".join(names[i] for i in range(n) if not bws[i]) + ")")
    reason = (f"{L} layers by {how}: " + "; ".join(
        f"rank {i} {names[i]} holds {counts[i]} (layers {bounds[i][0]}.."
        f"{bounds[i][1] - 1}, {got[i] / GIB:.1f} of {cap[i] / GIB:.1f} GiB"
        f", leaves {(wss[i] - int(other_bytes) - got[i]) / GIB:.1f} GiB"
        + (f", {float(bws[i]):g} GB/s" if bws[i] else "") + ")"
        for i in range(n)) + "; rank 0 holds the last layers and samples")
    return {"layers": counts, "bounds": [tuple(b) for b in bounds],
            "bytes": got, "weights": weights, "reason": reason}


def pipeline_layer_bytes(artifact: Artifact) -> tuple:
    """layer_bytes_of over the artifact's top-level safetensors headers
    (the tower is not the trunk; an MTP sidecar is counted as replicated)."""
    import json
    import struct

    cfg = artifact.raw_config or {}
    tc = cfg.get("text_config", cfg) or {}
    L = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    sizes = {}
    for f in sorted(artifact.path.glob("*.safetensors")):
        if f.name.startswith("model-vision"):
            continue
        try:
            with open(f, "rb") as fh:
                (hn,) = struct.unpack("<Q", fh.read(8))
                if hn <= 0 or hn > (1 << 28):
                    continue
                header = json.loads(fh.read(hn))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.startswith(S.VISION_TOWER_PREFIXES):
                continue
            a, b = v.get("data_offsets", (0, 0))
            sizes["mtp." + k if f.name.startswith("mtp") else k] = \
                int(b) - int(a)
    return layer_bytes_of(sizes, L)
