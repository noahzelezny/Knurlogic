"""Will it fit: the memory arithmetic behind a launch.

KV bytes per token, the step margin and fit reserve a rank keeps free, the
room left for context, what vision and an MTP head cost, the single-Mac fit
check, and the chunk widths the room allows (decode and prefill).
Headroom is an input: a byte count from machine/, never measured here.
"""

from __future__ import annotations

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import measured

GIB = 1 << 30


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
    w, x = measured.DECODE_CHUNK_ASSUMED_SHAPE
    return measured.DECODE_CHUNK_BYTES_PER_UNIT, (
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
        return measured.DECODE_CHUNK_DEFAULT      # no budget given: leave the default
    if headroom_bytes <= 0:
        return measured.DECODE_CHUNK_MIN          # does not fit: tightest, not default
    per = ((bytes_per_unit or measured.DECODE_CHUNK_BYTES_PER_UNIT)
           * measured.DECODE_CHUNK_HEADROOM_DIVISOR)
    return max(measured.DECODE_CHUNK_MIN,
               min(measured.DECODE_CHUNK_DEFAULT, int(headroom_bytes / per)))


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
    floor = measured.PREFILL_CHUNK_DEFAULT
    if family <= floor:
        return floor, f"prompt chunk {floor}: {family_why}"
    if headroom is None:
        return floor, (f"prompt chunk {floor}: the room free at launch is "
                       f"not known, so none of the measured {family} is "
                       f"spent")
    cfg = artifact.raw_config or {}
    tc = cfg.get("text_config") or cfg
    per_tok, _ = kv_bytes_per_token(tc, kv_bits)
    kv = per_tok * measured.PREFILL_KV_ALLOWANCE_TOKENS
    hidden = artifact.hidden_size or measured.DECODE_CHUNK_ASSUMED_SHAPE[1]

    def transient(w):
        return w * hidden * measured.PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN

    # The memory the launch RESERVES for transients: the step margin, or
    # the first request's transient and a quarter again when larger (the
    # same reserve the fit holds free, rank_margin). A chunk whose
    # predicted transient fits in it is already paid for; the room left
    # after weights, KV and cache is not what a transient is charged to.
    # It is never more than the headroom actually there.
    reserve = max(step_margin(working_set_bytes),
                  int(1.25 * max(measured.FIT_TRANSIENT_FLOOR,
                                 transient(measured.FIT_PREFILL_CHUNK))))
    allowed = min(reserve, max(int(headroom), 0))
    room = max(int(headroom) - step_margin(working_set_bytes) - kv
               - int(cache_bytes), 0)

    width = floor
    for w in measured.PREFILL_CHUNK_LADDER:
        if floor < w <= family and transient(w) <= allowed:
            width = w
    nxt = min((w for w in measured.PREFILL_CHUNK_LADDER if w > width), default=0)
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
            if not k.startswith(measured.VISION_TOWER_PREFIXES):
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
        el: float = measured.BF16_BYTES
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
        el = measured.kv_bytes_per_element(kv_bits)
        per = int(round(mla * (latent * el + exact * measured.BF16_BYTES)))
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
    el = measured.kv_bytes_per_element(kv_bits)
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
    hidden = int(tc.get("hidden_size") or measured.DECODE_CHUNK_ASSUMED_SHAPE[1])
    transient = max(measured.FIT_TRANSIENT_FLOOR,
                    measured.FIT_PREFILL_CHUNK * hidden
                    * measured.PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN)
    per, _ = kv_bytes_per_token(tc, kv_bits)
    return {"transient_bytes": int(transient),
            "kv_bytes": int(per * measured.FIT_MIN_CONTEXT_TOKENS)}


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
        from knurlogic.machine.memory import allowance, wired
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
    toks = measured.VISION_KV_IMAGES * measured.VISION_KV_TOKENS_PER_IMAGE
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
         f"{measured.VISION_KV_IMAGES} images x {measured.VISION_KV_TOKENS_PER_IMAGE} "
         f"tokens ({kv_why}) -- an ALLOWANCE for the context images add, "
         f"not a measurement"),
    ]
    return {"tower_bytes": tower, "tower_outside_bytes": outside,
            "tower_tensors": n, "store_bytes": store,
            "store_is_live": live, "kv_allowance_bytes": kv,
            "kv_bytes_per_token": per_tok, "kv_tokens": toks,
            "extra_bytes": outside + store + kv, "notes": notes}
