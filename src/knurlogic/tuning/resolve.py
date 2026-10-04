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

import re
from dataclasses import dataclass, field

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import settings as S

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
ENGINE_CONSUMED = ("prefill_chunk", "cache_limit_gb", "context_length", "mtp",
                   "mtp_dynamic", "kv_bits", "kv_kernel", "cross_chip",
                   "long_context", "preset", "vision")


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


def emit_cache_limit(r: Resolution, artifact: Artifact, gib) -> None:
    """The cache limit, one value: knurlogic's own (the engine sets mlx's
    free-buffer cache ceiling from it), and the same value under the name
    a VQ bundle's runtime reads, which sets that ceiling again at load."""
    r.env["KNURLOGIC_CACHE_LIMIT_GB"] = str(gib)
    src = artifact.runtime_source()
    old = S.KNOB_ALIASES["cache_limit_gb"][1:]
    name = next((n for n in old if n in src), None) if src else \
        (old[0] if artifact.is_vq else None)
    if name:
        r.env[name] = str(gib)


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


def prefill_chunk_by_room(artifact: Artifact, headroom, working_set_bytes: int,
                          cache_bytes: int, kv_bits, family: int,
                          family_why: str) -> tuple:
    """(width, one-line why) for the prompt chunk, read from the room.

    The widest ladder width, capped at the family's measured best, whose
    predicted step transient fits in the memory reserved for transients
    (the step margin / first-request reserve, capped by `headroom`); else
    step down, floor 512. `headroom` is
    what the load budget -- the room actually free at launch -- leaves
    above what this box holds; None means the budget is unknown."""
    floor = S.PREFILL_CHUNK_DEFAULT
    if family <= floor:
        return floor, f"prompt chunk {floor}: {family_why}"
    if headroom is None:
        return floor, (f"prompt chunk {floor}: the room free at launch is "
                       f"not known, so none of the measured {family} is "
                       f"spent")
    cfg = artifact.raw_config or {}
    tc = cfg.get("text_config") or cfg
    per_tok, _ = kv_bytes_per_token(tc, kv_bits)
    kv = per_tok * S.PREFILL_KV_ALLOWANCE_TOKENS
    hidden = artifact.hidden_size or S.DECODE_CHUNK_ASSUMED_SHAPE[1]

    def transient(w):
        return w * hidden * S.PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN

    # The memory the launch RESERVES for transients: the step margin, or
    # the first request's transient and a quarter again when larger (the
    # same reserve the fit holds free, rank_margin). A chunk whose
    # predicted transient fits in it is already paid for; the room left
    # after weights, KV and cache is not what a transient is charged to.
    # It is never more than the headroom actually there.
    reserve = max(step_margin(working_set_bytes),
                  int(1.25 * max(S.FIT_TRANSIENT_FLOOR,
                                 transient(S.FIT_PREFILL_CHUNK))))
    allowed = min(reserve, max(int(headroom), 0))
    room = max(int(headroom) - step_margin(working_set_bytes) - kv
               - int(cache_bytes), 0)

    width = floor
    for w in S.PREFILL_CHUNK_LADDER:
        if floor < w <= family and transient(w) <= allowed:
            width = w
    nxt = min((w for w in S.PREFILL_CHUNK_LADDER if w > width), default=0)
    return width, (
        f"prompt chunk {width} (family best {family}): its step transient "
        f"~{transient(width) / GIB:.2f} GiB"
        + (f" (next up, {nxt}: ~{transient(nxt) / GIB:.2f})"
           if width < family else "")
        + f" vs {allowed / GIB:.2f} GiB reserved for transients (step "
        f"margin / first-request reserve; {room / GIB:.1f} GiB more room "
        f"after weights, {kv / GIB:.1f} GiB KV and {cache_bytes / GIB:.1f} "
        f"GiB cache) -- the family best when its transient fits the "
        f"reserve, else the widest that does, floor {floor}")


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
    tune = S.preset_of(tune)
    if profile is not None and profile not in S.RUNTIME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(S.RUNTIME_PROFILES)}")

    if isinstance(budget, (int, float)):
        holds = artifact.bytes_on_disk if holds_bytes is None \
            else int(holds_bytes)
        r = _resolve_one(artifact, int(budget),
                         holds - (_vision_off_tower(artifact)
                                  if not vision and holds_bytes is None
                                  else 0),
                         profile, tune, store_bytes=store_bytes,
                         vision=vision, kv_bits=kv_bits,
                         long_context=long_context)
        if vision_budget(artifact) is not None:
            # a launch setting of a vision rung only (Settings -> Models)
            emit(r, artifact, "vision", "on" if vision else "off")
        if not vision and vision_budget(artifact) is not None:
            r.notes.append(
                f"vision off (KNURLOGIC_VISION=off): "
                f"{vision_freed_bytes(artifact, kv_bits) / GIB:.2f} GiB of "
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
    pipeline's ranks process the same chunks, so a low-headroom node's narrow chunk
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
    (KNURLOGIC_KV_BITS); an MLA latent is stored at those bits too, its
    rope key and DSA indexer key stay bf16."""
    layers = int(tc.get("num_hidden_layers") or 0)
    types = tc.get("layer_types")
    interval = tc.get("full_attention_interval")
    ratios = tc.get("compress_ratios")
    if tc.get("model_type") == "deepseek_v4":
        # DeepSeek-V4: every layer keeps a fixed sliding window (bounded,
        # not per token); what grows is a compressed pool of one head_dim
        # row per `ratio` tokens on each layer with a ratio, plus, on the
        # ratio-4 layers, the indexer's pool of one index_head_dim row per
        # 4 tokens. All bf16 (DeepseekV4Cache; KV quantization refused).
        ratios = [int(r) for r in (ratios or [])][:layers]
        hd = int(tc.get("head_dim") or 0)
        ihd = int(tc.get("index_head_dim") or 0)
        el: float = S.BF16_BYTES
        per = sum(hd / r + (ihd / r if r == 4 else 0)
                  for r in ratios if r > 0) * el
        n = sum(1 for r in ratios if r > 0)
        n4 = sum(1 for r in ratios if r == 4)
        return int(round(per)), (
            f"{n} compressed layers of {layers} ({n4} at 1/4 with the "
            f"indexer's {ihd}-dim pool, {n - n4} at 1/{max(ratios)}) x "
            f"{hd}-dim rows x bf16; the {tc.get('sliding_window')}-token "
            f"window is bounded")
    if isinstance(types, list) and tc.get("kv_lora_rank"):
        # MLA (GLM-5.3's deepseek_sparse_attention): a layer caches its
        # compressed latent and the DSA indexer's key, not K and V per
        # head -- counted as full attention it read 0 layers (GLM's are
        # not named full_attention), and as K,V per head it would be ~10x
        # head. The latent is stored once (K = V); the indexer's row is
        # its key, its pool gate scores (both index_head_dim) and a valid
        # flag, bf16 at every kv_bits (glm5_next language.py).
        mla = sum(1 for t in types if t != "linear_attention")
        latent = int(tc.get("kv_lora_rank") or 0)
        ihd = int(tc.get("index_head_dim") or 0)
        exact = (int(tc.get("qk_rope_head_dim") or 0)
                 + (2 * ihd + 1 if ihd else 0))
        el = S.kv_bytes_per_element(kv_bits)
        per = int(round(mla * (latent * el + exact * S.BF16_BYTES)))
        dt = "bf16" if kv_bits is None else f"{kv_bits}-bit"
        return per, (f"{mla} MLA layers of {len(types)} x ({latent} latent "
                     f"once x {dt} + {exact} rope, indexer key, gate and valid "
                     f"x bf16)")
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


def fit_reserve(cfg: dict, kv_bits=None) -> dict:
    """What a rank must keep free beyond its weights, from the config
    alone: {"transient_bytes": the first request's transient at the
    smallest prompt chunk (max of the measured flat part and the hidden-
    size line, settings.FIT_*), "kv_bytes": the KV of
    FIT_MIN_CONTEXT_TOKENS tokens at `kv_bits`}. KV is charged whole to
    every rank, not by the layers it holds: a rank's share is not known
    until this is, and the overcharge is one minimum context."""
    cfg = cfg or {}
    tc = cfg.get("text_config") or cfg
    hidden = int(tc.get("hidden_size") or S.DECODE_CHUNK_ASSUMED_SHAPE[1])
    transient = max(S.FIT_TRANSIENT_FLOOR,
                    S.FIT_PREFILL_CHUNK * hidden
                    * S.PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN)
    per, _ = kv_bytes_per_token(tc, kv_bits)
    return {"transient_bytes": int(transient),
            "kv_bytes": int(per * S.FIT_MIN_CONTEXT_TOKENS)}


def rank_margin(working_set_bytes: int, reserve: dict | None = None) -> int:
    """What a rank of this working set keeps free beyond its weights: the
    scheduler's step margin (`step_margin`, or the first request's transient
    and a quarter again, as the scheduler's `_margin` takes the larger),
    plus the KV of a minimum context. `reserve` (fit_reserve) None = the
    step margin alone."""
    m = step_margin(working_set_bytes)
    if not reserve:
        return m
    return max(m, int(int(reserve.get("transient_bytes") or 0) * 1.25)) \
        + int(reserve.get("kv_bytes") or 0)


def _vision_off_tower(artifact: Artifact) -> int:
    """The tower bytes inside the artifact's size: what vision off takes
    out of the weights (the text load never reads them; `vision.bind`
    does). 0 for an artifact with no vision_config."""
    if vision_budget(artifact) is None:
        return 0
    tower, outside, _ = _tower_bytes(artifact)
    return max(int(tower) - int(outside), 0)


def vision_freed_bytes(artifact: Artifact, kv_bits=None) -> int:
    """What KNURLOGIC_VISION=off frees on one machine: the tower, the image
    store's bound and the image KV allowance. 0 without a vision_config."""
    vb = vision_budget(artifact, kv_bits=kv_bits)
    if vb is None:
        return 0
    return int(vb["tower_bytes"] + vb["store_bytes"]
               + vb["kv_allowance_bytes"])


def mtp_head_bytes(artifact: Artifact) -> int:
    """The packed MTP head's bytes (its mtp-head*.safetensors sidecars):
    what MTP off frees. 0 without one."""
    return sum(f.stat().st_size for f in artifact.path.glob(
        "mtp-head*.safetensors") if f.is_file())


def single_fit_check(artifact: Artifact, budget_bytes: int,
                     draft: bool = True, kv_bits=None,
                     vision: bool = True) -> dict:
    """How a single-machine load sits in `budget_bytes`:
    {"state": "fits" | "cannot", "why": str, "head_bytes"}.

    Counts what will really be bound: the artifact's weights, the MTP head
    only when drafting is on (its sidecar is inside the artifact's size,
    so it is taken OUT when off), and a vision rung's extra bytes. The fit
    line is the weights plus the minimum step margin (`step_margin`), not
    the fuller reserve a cluster rank plans its shares around (`rank_margin`
    of `fit_reserve`): for real models the two differ by ~0.1 GiB, too thin
    a band to be worth a third answer."""
    ws = int(budget_bytes or 0)
    out: dict = {"state": "fits", "why": "", "head_bytes": 0}
    if ws <= 0:
        return out
    head = mtp_head_bytes(artifact)
    vb = vision_budget(artifact, kv_bits=kv_bits)
    extra = int((vb or {}).get("extra_bytes") or 0)
    freed = vision_freed_bytes(artifact, kv_bits)
    base = int(artifact.bytes_on_disk) + extra
    need = base - (0 if draft else head) - (0 if vision else freed)
    floor = step_margin(ws)
    if need + floor <= ws:
        return out
    out["head_bytes"] = head if draft else 0
    out["vision_bytes"] = freed if vision else 0
    out["state"] = "cannot"
    out["why"] = (
        f"{artifact.path.name} needs {need / GIB:.1f} GiB (weights"
        + (f" incl. the {head / GIB:.1f} GiB MTP head" if draft and head
           else "")
        + (f", {freed / GIB:.1f} GiB vision" if vision and freed else "")
        + f") plus {floor / GIB:.1f} GiB step margin; the budget is "
        f"{ws / GIB:.1f} GiB")
    # what turning parts off would free, smallest change first
    mtp_off = draft and head and need - head + floor <= ws
    vis_off = vision and freed and need - freed + floor <= ws
    both = (draft and head and vision and freed
            and need - head - freed + floor <= ws)
    if mtp_off and vis_off:
        out["why"] += ("; turn MTP off (Settings \u2192 Presets) or vision "
                       "off (VISION in Load model) to fit")
    elif mtp_off:
        out["why"] += "; turn MTP off (Settings \u2192 Presets) to fit"
    elif vis_off:
        out["why"] += ("; turn vision off (VISION in Load model, "
                       "KNURLOGIC_VISION=off) to fit")
    elif both:
        out["why"] += ("; turn MTP off (Settings \u2192 Presets) and vision "
                       "off (VISION in Load model) to fit")
    return out


def single_fit(artifact: Artifact, budget_bytes: int, draft: bool = True,
               kv_bits=None, vision: bool = True) -> str:
    """"" unless a single-machine load cannot fit at all (the weights and the
    minimum step margin), else why not: see `single_fit_check`."""
    c = single_fit_check(artifact, budget_bytes, draft, kv_bits, vision)
    return c["why"] if c["state"] == "cannot" else ""


def context_room(working_set_bytes: int, weights_bytes: int,
                 cfg: dict, kv_bits=None) -> dict:
    """What a model that fits leaves for its conversations: the working set
    (or allowance) less the weights less the step margin, and about how
    many tokens of context that is at the model's KV bytes per token --
    shared by every conversation at once, not each one's.

    Why it is said at all: GLM-5.3 2.7bpw "fits" a 128 GB M4 Max and
    leaves about 6 GiB, where four long agent conversations cannot run.
    A fit that leaves no room to talk is not much of a fit.
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
    # the step margin is kept free like a rank's (rank_margin): weights that
    # fill the budget "fit" only until the first request swaps
    return {"fits": bool(ws) and int(weights_bytes) + margin <= ws,
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
    or None for an artifact with no vision tower (a `vision_config`, or
    DeepSeek-V4's flat `vision_n_layers` with its `vision.*` tensors).

    Stdlib only (no mlx, no family import): tuning/ must not import mlx,
    and this has to answer BEFORE a load. `extra_bytes` is what the
    resolver adds to what the box holds; each term has its note."""
    from knurlogic.engine.vision.registry import has_vision_config
    cfg = artifact.raw_config or {}
    if not has_vision_config(cfg, artifact.path):
        return None
    from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES

    tower, outside, n = _tower_bytes(artifact)
    live = store_bytes is not None
    store = int(store_bytes) if store_bytes is not None else DEFAULT_MAX_BYTES
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
                 tune: str = "default", store_bytes: int | None = None,
                 vision: bool = True, kv_bits=None,
                 long_context=None) -> Resolution:
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
                "this artifact's experts are small enough to justify a "
                "larger chunk, and it was NOT taken: sizing from the model "
                "may only tighten until a run measures the loosening "
                "direction, because being wrong there is an OOM")
    fits = working_set_bytes <= 0 or headroom > 0
    if (artifact.is_vq and chunk_from_headroom < S.DECODE_CHUNK_DEFAULT
            and fits):
        r.notes.append(
            f"VQ_DECODE_CHUNK lowered to {chunk} ({headroom/GIB:.1f} GiB "
            f"headroom): bounds the dense-expert transient, which is what "
            f"caps context length on a full box")

    cache = float(t.get("KNURLOGIC_CACHE_LIMIT_GB", S.CACHE_LIMIT_GB_DEFAULT))
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
    family, family_why = S.prefill_chunk_for(artifact.model_type)
    asked = t.get("KNURLOGIC_PREFILL_CHUNK")
    if asked is not None:
        # lean: narrow whatever the room
        prefill = asked
        if family > asked:
            r.notes.append(
                f"prompt chunk {asked}: tune={tune} keeps it narrow; "
                f"{family} was {family_why}")
    else:
        prefill, why = prefill_chunk_by_room(
            artifact, headroom if known else None, working_set_bytes,
            int(cache * GIB), kv_bits, family, family_why)
        r.notes.append(why)
    emit(r, artifact, "prefill_chunk", prefill)
    window, _ = S.model_window(
        _long_context_cfg(r, artifact, long_context, working_set_bytes,
                          holds_bytes, kv_bits))
    if window:
        # the model's own window: the cap a person lowers, never raises past
        # (settings.check_knob refuses more), so the control stops there
        emit(r, artifact, "context_length", window)
        # a family with documented YaRN is offered up to its YaRN window:
        # a context past the native one turns long context on at launch
        # (settings.settle_context)
        top = S.context_ceiling(artifact.model_type,
                                artifact.raw_config) or window
        top = max(top, window)
        steps = [v for v in S.KNOB_RANGE["KNURLOGIC_CONTEXT_LENGTH"][0]
                 if v < top and v != window]
        r.ranges["KNURLOGIC_CONTEXT_LENGTH"] = sorted(
            set(steps + [window, top]))
    launch, launch_notes = S.preset_launch(tune, artifact.model_type)
    if prefill < S.PREFILL_CHUNK_DEFAULT:
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
            f"{holds_bytes / GIB:.1f} GiB to hold against a "
            f"{working_set_bytes/GIB:.1f} GiB working set -- it does not fit "
            f"this box. No setting fixes that; it needs a bigger box or more "
            f"than one.")

    # --- numerics: the model's own, from what it ships ----------------------
    if artifact.is_vq:
        env, note = numerics_for(artifact, profile)
        r.env.update(env)
        r.notes.append(note)

    return r


def _preset_record(r: Resolution, tune: str, launch: dict) -> None:
    """Which env values the preset put there: its launch settings, plus the
    prompt chunk / cache limit its profile names."""
    t = S.TUNE_PROFILES[tune]
    logicals = set(launch)
    if "KNURLOGIC_PREFILL_CHUNK" in t:
        logicals.add("prefill_chunk")
    if "KNURLOGIC_CACHE_LIMIT_GB" in t:
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
    except (OSError, ValueError):
        head = None
    launch, _ = S.preset_launch(tune, artifact.model_type)
    if head is not None:
        emit(r, artifact, "mtp", launch.get("mtp", "on"))
        emit(r, artifact, "mtp_dynamic", launch.get("mtp_dynamic", "on"))
    bits, why = S.kv_quant_for(artifact.model_type)
    emit(r, artifact, "kv_bits", launch.get("kv_bits", "bf16"))
    # the decode kernel reads 8-bit K/V only: shown where it can matter
    if str(kv_bits if kv_bits is not None
           else launch.get("kv_bits", "bf16")) == "8":
        emit(r, artifact, "kv_kernel", launch.get("kv_kernel", "on"))
    emit(r, artifact, "cross_chip", launch.get("cross_chip", "off"))
    emit(r, artifact, "preset", tune)
    r.ranges["KNURLOGIC_KV_BITS"] = ["bf16"] + [
        str(b) for b in bits if str(b) in S.KV_BITS_OFFERED]
    if not bits:
        r.notes.append(f"KV cache stays bf16: {why}")
    elif kv_bits is not None:
        r.notes.append(f"KV cache counted at {kv_bits} bits "
                       f"({S.kv_bytes_per_element(kv_bits):.3g} bytes per "
                       f"element against bf16's 2): {why}")


def _long_context_cfg(r: Resolution, artifact: Artifact, long_context,
                      working_set_bytes: int, holds_bytes: int,
                      kv_bits) -> dict:
    """The config the window is read from under KNURLOGIC_LONG_CONTEXT,
    with the knob emitted (off / yarn where the family's model card
    documents YaRN) and the KV room for the YaRN window warned about."""
    cfg = artifact.raw_config or {}
    mt = artifact.model_type
    if S.long_context_family(mt) is None:
        return cfg
    mode = S.long_context_of(long_context)
    emit(r, artifact, "long_context", mode)
    r.ranges["KNURLOGIC_LONG_CONTEXT"] = list(S.LONG_CONTEXT_VALUES)
    if mode == "off":
        return cfg
    cfg = S.with_long_context(cfg, mode)
    window, why = S.model_window(cfg)
    r.notes.append(f"long context (YaRN): the cap is {window:,} tokens "
                   f"({why}); static YaRN may cost a little quality on "
                   f"short prompts (Qwen's model card)")
    if working_set_bytes > 0:
        short = long_context_room(artifact, working_set_bytes, holds_bytes,
                                  window, kv_bits)
        if short:
            r.warnings.append(short)
    return cfg


def long_context_room(artifact: Artifact, working_set_bytes: int,
                      holds_bytes: int, context: int, kv_bits=None):
    """None when this box can hold the KV of `context` tokens beside what
    it holds of the model and the step margin, else the sentence saying
    how short it is and what would fit."""
    tc = (artifact.raw_config or {}).get("text_config") \
        or artifact.raw_config or {}
    per, why = kv_bytes_per_token(tc, kv_bits)
    if not per or not context:
        return None
    need = per * int(context)
    left = max(int(working_set_bytes) - int(holds_bytes)
               - step_margin(working_set_bytes), 0)
    if need <= left:
        return None
    fits = left // per
    half = ("" if kv_bits is not None else
            f"; 8-bit KV (KNURLOGIC_KV_BITS=8) needs "
            f"{kv_bytes_per_token(tc, 8)[0] * int(context) / GIB:.1f} GiB")
    return (f"{int(context):,} tokens of context need {need / GIB:.1f} GiB "
            f"of KV ({per:,} bytes per token: {why}) and this box leaves "
            f"{left / GIB:.1f} GiB after the model and the step margin -- "
            f"about {fits:,} tokens. Lower KNURLOGIC_CONTEXT_LENGTH{half}")


def kv_refusal(artifact: Artifact, kv_bits) -> str | None:
    """Why this artifact cannot launch with `kv_bits`, or None."""
    if kv_bits is None:
        return None
    bits, why = S.kv_quant_for(artifact.model_type)
    if int(kv_bits) not in bits:
        return (f"KV cache at {kv_bits} bits is refused for "
                f"{artifact.model_type}: {why}")
    return None


#: How a runtime spells an env flag and its default in source.
_FLAG_DEFAULT = re.compile(
    r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]+)"\s*,\s*\n?\s*("?[^")\s]*"?)\s*\)')


def _flag_defaults(src: str) -> dict:
    """{flag: default string} for every `os.environ.get("X", d)` in a
    runtime's source; the first occurrence wins."""
    out: dict = {}
    for m in _FLAG_DEFAULT.finditer(src):
        out.setdefault(m.group(1), m.group(2).strip('"'))
    return out


def _numerics_source(artifact: Artifact, flag: str, source: str):
    if source == "declared":
        d = artifact.declared_knobs().get(flag)
        if isinstance(d, dict) and d.get("default") is not None:
            return str(d["default"])
        return None
    if source == "bundled":
        return _flag_defaults(artifact.runtime_source()).get(flag)
    raise ValueError(source)


def numerics_for(artifact: Artifact, profile: str | None = None):
    """(env, note): the numerics-active flags for this artifact.

    A profile someone ASKED for wins, and the note names what it overrode.
    Otherwise every flag comes from the artifact itself, first source in
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
# qwen3_5 families, qwen4_exp and deepseek_v4: engine/runtime/tensor.py does
# the split).
# Pure arithmetic over the config and the safetensors headers, so a refusal is
# said -- with its numbers -- before anything loads.

def _tensor_maps() -> tuple:
    from knurlogic.engine import families
    m = families.build_maps()
    return m["tensor"], m["tensor_archs"]


#: model type -> its family's `tensor` entry (engine/families/<family>):
#: the model types engine/runtime/tensor.py knows how to split, and the
#: config keys each needs divisible by the ranks; and those architectures
_TENSOR, _TENSOR_ARCHS = _tensor_maps()
TENSOR_TYPES = tuple(_TENSOR)


def tensor_sharded(name: str) -> bool:
    """Is this weight split across ranks under tensor? (tensor_rules: a VQ
    codebook never is.)"""
    from knurlogic.engine.runtime.tensor_rules import sharded
    return sharded(name)


def _block_width(tc: dict, inp: dict):
    """An input's width: the product of its config keys; None when one is
    absent and has no default."""
    w = 1
    for k in inp["keys"]:
        v = tc.get(k)
        if v is None:
            v = (inp.get("defaults") or {}).get(k)
        elif not v and k in (inp.get("defaults") or {}):
            v = inp["defaults"][k]
        if v is None:
            return None
        w *= int(v)
    return w


def _block_refusals(types: set, tc: dict, n: int) -> list:
    """A family whose activations are rounded in blocks along a linear's
    input (its manifest's `tensor_split`: DeepSeek-V4's act_quant, block
    128, architecture edit 21): a split that cuts such an input must leave
    each rank whole blocks, or the split model rounds other blocks than
    the whole one."""
    from knurlogic.engine import families
    rules = families.build_maps()["tensor_split"]
    out: list = []
    seen: set = set()
    for t in sorted(t for t in types if t in rules):
        rule = rules[t]
        b = int(rule["act_quant_block"])
        for inp in rule["inputs"]:
            if inp["what"] in seen:
                continue
            seen.add(inp["what"])
            v = _block_width(tc, inp)
            if v is None or v % n:
                continue
            if (v // n) % b:
                out.append(
                    f"{inp['what']} = {v}: a rank's {v // n} is not whole "
                    f"{b}-blocks ({v // n} % {b} = {v // n % b})"
                    f": its activation rounding (act_quant, block {b}) "
                    f"would differ from the whole model's")
    return out


def tensor_refusals(cfg: dict, n: int) -> list:
    """Why this config cannot be split `n` ways, one line per reason with
    its arithmetic; [] when it can."""
    out: list = []
    if n < 2:
        return out
    tc = cfg.get("text_config", cfg)
    types = {cfg.get("model_type"), tc.get("model_type")}
    if not types & set(TENSOR_TYPES):
        out.append(f"tensor split knows {', '.join(_TENSOR_ARCHS)}; this "
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
    # a family's own axes a rank must hold whole (its manifest's
    # `tensor.divisible`, with why)
    for k in dict.fromkeys(k for t in sorted(types & set(_TENSOR))
                           for k in _TENSOR[t].get("divisible", ())):
        div(k, tc.get(k))
    out += _block_refusals(types, tc, n)
    # the arrays' own axes (intermediate sizes, quantization groups, VQ
    # code rows) are tensor_header_refusals': the headers answer them
    if cfg.get("vq_linear"):
        out.append(f"{len(cfg['vq_linear'])} VQ dense linear(s) (vq_linear): "
                   f"not split by tensor in this build")
    if cfg.get("vq_embed"):
        out.append(f"{len(cfg['vq_embed'])} VQ embedding(s) (vq_embed): not "
                   f"split by tensor in this build")
    for path, m in sorted((cfg.get("vq_modules") or {}).items()):
        IN = int(m.get("in", 0))
        G, D = int(m.get("group", 64)), int(m.get("dim", 1))
        if path.endswith("down_proj") and m.get("pack_bits"):
            # sharded-to-all: codes split on their input axis. Packed codes
            # are uint32 words holding 32 codes per BITS words, so a slice
            # must hold whole 32-code blocks -- 32*dim inputs (the headers
            # see words, not codes: this one the config answers)
            unit = max(G, 32 * D)
            if IN % n or (IN // n) % unit:
                out.append(
                    f"{path}: input {IN} / {n} = {IN / n:g}, not a multiple "
                    f"of {unit} (max(group {G}, 32 x dim {D})): a rank's "
                    f"slice would cut a packed code word")
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


#: path -> (stat stamp of its shards, trunk headers): the picker asks
#: every model's headers on each listing, ~80 of them, many over SMB
_HEADERS: dict = {}


def trunk_headers(path) -> dict:
    """{name: (shape, bytes)} from the artifact's top-level safetensors
    headers -- 8 bytes and a JSON each, no weights -- minus the tower and a
    packed MTP head (neither is split: rank 0 alone holds them,
    `leader_bytes`). Cached on each shard's
    (size, mtime_ns, ctime_ns)."""
    import json
    import struct
    from pathlib import Path

    files = sorted(f for f in Path(path).glob("*.safetensors")
                   if not f.name.startswith(("mtp", "model-vision")))
    try:
        stamp = tuple((f.name, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                      for f in files for st in (f.stat(),))
    except OSError:
        stamp = None
    hit = _HEADERS.get(str(path))
    if stamp is not None and hit and hit[0] == stamp:
        return hit[1]
    out = {}
    for f in files:
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
            out[k] = (tuple(v.get("shape") or ()), int(b) - int(a))
    if stamp is not None:
        _HEADERS[str(path)] = (stamp, out)
    return out


def tensor_placement(artifact: Artifact, n: int) -> dict:
    """tensor_placement_of over the artifact's trunk headers."""
    return tensor_placement_of(
        {k: b for k, (_, b) in trunk_headers(artifact.path).items()}, n)


def tensor_header_refusals(path, cfg: dict, n: int) -> list:
    """Why the arrays on disk cannot be cut `n` ways by the split's own
    rules (engine/runtime/tensor_rules), each with its numbers."""
    from knurlogic.engine.runtime.tensor_rules import refusals, skipzero_split
    if n < 2:
        return []
    tc = cfg.get("text_config", cfg)
    kd = (tc.get("linear_num_key_heads") or 0) * \
        (tc.get("linear_key_head_dim") or 0)
    shapes = {k: s for k, (s, _) in trunk_headers(path).items()}
    sz = skipzero_split(path)
    out = []
    if sz:
        # the runtime cuts these by output row, per expert: the rows must
        # divide (it refuses at load; said here before any rank starts)
        for p, m in sorted(((cfg.get("vq_skipzero") or {}).get("modules")
                            or {}).items()):
            OUT = int(m.get("out") or ((cfg.get("vq_modules") or {})
                                         .get(p) or {}).get("out") or 0)
            if p.endswith(("gate_proj", "up_proj")) and OUT % n:
                out.append(f"{p}: {OUT} output rows do not divide by {n}")
    return out + refusals(shapes, n, kd, int(tc.get("num_key_value_heads")
                                            or 0), sz)


def tensor_unverified(path) -> dict:
    """{(rule path, unknown parameters): a layer holding them} -- the
    modules whose layout no split rule knows, for a launch to run
    (engine/runtime/viability)."""
    from knurlogic.engine.runtime.tensor_rules import skipzero_split, unverified
    return unverified({k: s for k, (s, _) in trunk_headers(path).items()},
                      skipzero_split(path))


def tensor_split_refusals(path, n: int, cfg: dict | None = None) -> list:
    """Everything the config and the headers say against splitting the
    artifact at `path` `n` ways; [] when nothing does."""
    import json
    from pathlib import Path
    if cfg is None:
        cfg = json.loads((Path(path) / "config.json").read_text())
    why = tensor_refusals(cfg, n)
    if why or n < 2:
        return why
    return tensor_header_refusals(path, cfg, n)


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
                  "qwen4_exp_text", "deepseek_v4")

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
                f"glm5_next, qwen4_exp and deepseek_v4; this is {mt!r}"
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
            # remainders: the largest fractional part first, lower rank on a
            # tie
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
    bounds: list = [None] * n
    at, left = 0, float(sum(layer_bytes))
    for r in range(n - 1, 0, -1):
        wsum = sum(weights[:r + 1])
        target = left * (weights[r] / wsum if wsum > 0 else 1.0 / (r + 1))
        end, acc = at, 0
        while end < L - r:            # leave ranks r-1 .. 0 a layer each
            nxt = acc + layer_bytes[end]
            if nxt > cap[r] or (end > at and
                                abs(nxt - target) >= abs(acc - target)):
                # a tie leaves the layer to the lower ranks (rank 0 last)
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


def pipeline_shares(layer_bytes: list, ranks: list, other_bytes: int = 0,
                    leader_bytes: int = 0, reserve: dict | None = None) -> dict:
    """Which layers each rank holds.

    `layer_bytes`: bytes of each layer, in order. `ranks`: in rank order,
    [{"name", "working_set_bytes", "memory_bandwidth_gbs" (None: unknown)}].
    `other_bytes`: what every rank holds besides its layers (embeddings,
    final norm, lm_head -- replicated). `leader_bytes`: what rank 0 alone
    holds besides (the MTP head and the vision tower:
    `leader_bytes`), so rank 0 takes fewer layers for them.

    A rank's weight is what it can hold (working set less the replicated
    bytes), times its memory bandwidth when EVERY rank's is known (decode
    reads each layer's weights once per step, so a faster rank should read
    more of them); a mix of known and unknown bandwidths weighs by capacity
    only, and says so. What a rank can hold leaves its step margin
    (step_margin) free; the reason says what each rank leaves. Every rank holds
    at least one layer and no rank more
    than fits; `reserve` (fit_reserve) is what each rank also keeps free:
    the first request's transient and a minimum context's KV, so an uneven
    split respects what the fit check does; rank 0 holds the LAST run of
    layers, rank N-1 the first.

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
    held = [int(other_bytes) + (int(leader_bytes) if i == 0 else 0)
            for i in range(n)]
    cap = [w - held[i] - rank_margin(w, reserve) for i, w in enumerate(wss)]
    for i, c in enumerate(cap):
        if c <= 0:
            raise ValueError(
                f"{names[i]}: working set {wss[i] / GIB:.1f} GiB "
                f"holds none of the layers after the {held[i] / GIB:.1f} "
                f"GiB it keeps besides them (embeddings, norm, lm_head"
                + (", and on rank 0 the MTP head and vision tower"
                   if i == 0 and leader_bytes else "")
                + f") and the {rank_margin(wss[i], reserve) / GIB:.1f} GiB "
                f"it keeps free (step margin or first-request transient, "
                f"plus a minimum context's KV)")
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
    bounds: list = [None] * n
    at = 0
    for r in range(n - 1, -1, -1):
        bounds[r] = (at, at + counts[r])
        at += counts[r]
    # Cut by the real bytes, not by counting layers: layers are not alike
    # (Qwen3.8 Flash's layer 1 carries a 42 GiB n-gram embedding). Counted,
    # an M3 Ultra rank took layers 0..18 -- 63.5 GiB of 110 -- and fit, but
    # with 13 GiB left for every prompt's KV while the M4 Max rank kept
    # 70 GiB free, and long prompts were refused. The count is the fallback,
    # not the rule.
    alt = _byte_bounds(layer_bytes, weights, cap)
    if alt is not None:
        bounds = alt
        counts = [b - a for a, b in bounds]
    got = [sum(layer_bytes[a:b]) for a, b in bounds]
    over = [i for i in range(n) if got[i] > cap[i]]
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
        f", leaves {(wss[i] - held[i] - got[i]) / GIB:.1f} GiB"
        + (f", {float(bws[i]):g} GB/s" if bws[i] else "") + ")"
        for i in range(n)) + "; rank 0 holds the last layers and samples")
    return {"layers": counts, "bounds": [tuple(b) for b in bounds],
            "bytes": got, "weights": weights, "reason": reason}


def pipeline_layer_bytes(artifact: Artifact) -> tuple:
    """layer_bytes_of over the artifact's top-level safetensors headers.
    Neither the tower nor an MTP head is the trunk, and neither is on every
    rank: rank 0 alone holds them (`leader_bytes`)."""
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
            if f.name.startswith("mtp") or k.split(".")[0] == "mtp":
                continue
            a, b = v.get("data_offsets", (0, 0))
            sizes[k] = int(b) - int(a)
    return layer_bytes_of(sizes, L)


def leader_bytes(artifact: Artifact, vision: bool = True,
                 mtp: bool = True) -> int:
    """What rank 0 of a split model (pipeline or tensor) holds and no other
    rank does: the MTP head when it drafts (`mtp`; the followers only run
    the verify rows) and the vision tower when it serves images (`vision`;
    it encodes at tokenize and ships the image rows, a follower binds the
    family without one). Read off the safetensors headers."""
    import json
    import struct

    total = 0
    for f in sorted(artifact.path.glob("*.safetensors")) if mtp else ():
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
            if f.name.startswith("mtp") or k.split(".")[0] == "mtp":
                a, b = v.get("data_offsets", (0, 0))
                total += int(b) - int(a)
    return total + (_tower_bytes(artifact)[0] if vision else 0)
