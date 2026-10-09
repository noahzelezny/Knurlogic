"""The knob registry: what each knob is (KNOB_DOC, KNOB_HELP,
KNOB_TITLES), the env names it goes by (KNOB_ALIASES), who it is for
(knob_tier), what it can be turned to (KNOB_RANGE, KNOB_BOUNDS), and the
readers that turn a set value into what the engine takes.

Whether a value is allowed is tuning/checks; whether a change applies to a
running server is tuning/live.
"""

from __future__ import annotations

from knurlogic.tuning import context_window, measured, numerics, presets

#: the page's name for each launch setting (Settings -> Models / Knurlogic;
#: views/settings/knobs.js KNOB_TITLE): what every refusal calls it
KNOB_TITLES: dict = {
    **{r["name"]: r["title"] for r in presets.PRESET_ROWS},
    "KNURLOGIC_PRESET": "Preset",
    "KNURLOGIC_CONTEXT_LENGTH": "Context length",
    "KNURLOGIC_CROSS_CHIP": "Per-chip rounding",
    "KNURLOGIC_MTP": "MTP",
    "KNURLOGIC_MTP_DYNAMIC": "MTP dynamic",
    "KNURLOGIC_VISION": "Vision",
    "KNURLOGIC_KV_KERNEL": "KV kernel",
    "KNURLOGIC_LONG_CONTEXT": "Long context",
    "KNURLOGIC_THINKING_DEFAULT": "Thinking default",
}
KNOB_TITLES["VQLAB_PREFILL_CHUNK"] = KNOB_TITLES["KNURLOGIC_PREFILL_CHUNK"]
KNOB_TITLES["VQ_CACHE_LIMIT_GB"] = KNOB_TITLES["KNURLOGIC_CACHE_LIMIT_GB"]
KNOB_TITLES["VQLAB_CACHE_LIMIT_GB"] = KNOB_TITLES["KNURLOGIC_CACHE_LIMIT_GB"]


def knob_title(name: str) -> str:
    return KNOB_TITLES.get(name, name)


# --- what each knob IS, in one line, for anything that shows it to a person -
# A settings panel that lists names and values is a config file with a
# stylesheet. The reason to show a knob at all is the sentence next to it:
# what it does, and what run says so. Anything added to the resolver should
# be added here too, or it will appear in the UI as a bare string.
KNOB_DOC = {
    # (what it is, its TRADE-OFF: what a change buys and what it costs).
    # Numbers are measurements with their run; where none exists the cost
    # is said in words, never invented.
    "VQ_DECODE_CHUNK": (
        "experts decoded to dense fp16 per prefill chunk",
        "THE memory knob: prefill grew 3.35 MB/token where KV-cache theory "
        "predicted 0.059. No trade at the top: smaller is also faster (128 "
        "-> 32 is 1.37x), so it is capped at 32. Lower still shrinks the "
        "prefill memory spike further; the speed below 32 is not measured."),
    "KNURLOGIC_PREFILL_CHUNK": (
        "how many prompt tokens are processed at once",
        "wider can prefill faster -- ~1.8x from 512 to 2048 on the "
        "35B-A3B VQ (645 -> 1163 tok/s at 28.7k tokens, M4, 4096 no "
        "better), up to ~30% on the 397B VQ, nothing on Qwen3.8 Flash -- "
        "but each step's memory spike grows 9-23x from 512 to 4096, which "
        "is what runs a long prompt or parallel agents out of memory. "
        "Unset, it is read from the room free at launch: the widest "
        "width up to the family's measured best whose predicted spike "
        "fits in 10% of the room left after weights, KV and cache, else "
        "512 (lean: always 512). Output is identical at "
        "every width."),
    "KNURLOGIC_CONTEXT_LENGTH": (
        "the longest conversation (prompt + answer, in tokens) a request may "
        "use",
        "longer lets a request run longer, and every token of it holds KV "
        "memory while it runs, so fewer long conversations or agents fit at "
        "once. Shorter caps that memory; a longer prompt is refused (400) "
        "and max_tokens trimmed to fit. Nothing is reserved up front. The "
        "default is the model's own window."),
    "KNURLOGIC_MTP": (
        "multi-token prediction: draft with the head packed beside the "
        "weights",
        "on: steps that draft can emit several tokens, usually faster "
        "replies, with the same output distribution -- at the cost of "
        "keeping the head in memory. Off frees that memory and every step "
        "is a plain, slower one."),
    "KNURLOGIC_VISION": (
        "load the vision tower, its image store and the image KV allowance",
        "on (the default): a model with a vision tower takes images. Off: "
        "the tower is not loaded and no image store or image KV is held, "
        "so that memory goes to headroom (a wider prompt chunk, or a fit "
        "that would not otherwise); a request with an image gets a 400 "
        "saying vision is off for this launch."),
    "KNURLOGIC_MTP_DYNAMIC": (
        "switch between drafting and plain steps by their measured cost",
        "on: each regime is timed per batch width and the cheaper one "
        "taken, so it is usually faster -- but timing varies as it "
        "switches. Off: every step drafts while a head is bound, steady "
        "timing, paying for drafts even where a plain step would be "
        "cheaper. Only with MTP on."),
    "KNURLOGIC_KV_BITS": (
        "precision of the attention KV cache: bf16, or 8, 6 or 4 bits",
        "fewer bits hold more context and more agents: 8-bit takes about "
        "53% of bf16's memory (6: 41%, 4: 28%). The cost is speed and a "
        "little precision: at 8 bits decode reads the cache through a "
        "fused kernel (KNURLOGIC_KV_KERNEL), measured ~5% slower than bf16 "
        "at 6k tokens of context and ~6% at 16k (M4, kernel on; ~7% / ~18% "
        "with it off), no measured prefill cost; 6/4 dequantize K/V every "
        "step and are not measured. 8-bit moves the tiny test models' "
        "logits by ~0.3% of their range; a planted-needle answer stayed "
        "exact on Qwen3.6-35B at 26k tokens. Attention K/V only (GLM: its "
        "MLA latent; GLM takes 8 only)."),
    "KNURLOGIC_KV_KERNEL": (
        "8-bit KV decode through the fused kernel (on) or dequantize + "
        "attention (off)",
        "only with an 8-bit KV cache: decode steps read the 8-bit K/V "
        "directly instead of dequantizing the whole cache each step "
        "(engine/kvattn.py). No trade measured: M4 Qwen3.6-35B decode is "
        "+4% faster at 6k, +15% at 16k context than off, and the needle "
        "answer is exact either way. Off is for A/B -- check /status.json "
        "kv_kernel hits vs misses to see "
        "which path is actually live (GLM's MLA latent and gemma4's "
        "KV-shared layers always take dequantize + attention). A row with "
        "every key masked returns 0 here where mlx sdpa returns NaN. "
        "Prefill is dequantize + attention either way."),
    "KNURLOGIC_THINKING_DEFAULT": (
        "the thinking level a request that names none is served at "
        "(default: the template's own)",
        "for a client that sends no reasoning_effort, or whose control is "
        "broken: the level goes through the same translation as a "
        "request's own (engine/model/thinking), to the nearest native "
        "level at or above it, and usage.knurlogic.thinking says it was "
        "the server's default. A request that names a level still wins. "
        "GLM-5.3's own default is max: on a 339k-token conversation it "
        "thought for hours without committing to an answer. The trade: a "
        "higher level answers slower and spends more of the context on "
        "thinking; a lower one is faster but loses reasoning on hard "
        "questions. Live: applies to the next request."),
    "KNURLOGIC_LONG_CONTEXT": (
        "reach past the model's trained window: off, or yarn (Qwen's "
        "documented YaRN rope scaling, factor 4 over 262,144 -> ~1M tokens)",
        "yarn raises this model's context cap to 1,048,576 tokens (Qwen "
        "documents 1,010,000 for Qwen3.5/3.6, 1,000,000 for Qwen3.8), set "
        "on the loaded config at load time, the artifact untouched. The "
        "cost, per Qwen: static YaRN applies the same scaling to every "
        "prompt, so short prompts may get slightly worse -- leave it off "
        "unless you need past 262k. And every token of context holds KV "
        "memory: at 1,048,576 tokens bf16 KV is 20 GiB on Qwen3.6-35B-A3B, "
        "24 GiB on Qwen3.8 Flash, 30 GiB on the 397B, 64 GiB on "
        "Qwen3.8-27B (all conversations together); 8-bit KV "
        "(KNURLOGIC_KV_BITS=8) takes 53% of that. The load is "
        "refused where the box cannot hold the chosen context's KV. Only "
        "for families whose model card documents it (qwen3_5, "
        "qwen3_5_moe, qwen4_exp)."),
    "KNURLOGIC_CROSS_CHIP": (
        "identical results across chips: a split over an M3 and an M4 "
        "gives the same tokens as a split over two of one",
        "on costs speed: mlx picks a different quantized-matmul kernel "
        "for 9-31 rows on each GPU generation, so on pads those calls to 32 "
        "rows, +2-6% time on them. Off is faster, and a mixed-chip split "
        "may then produce different (equally valid) tokens than a same-chip "
        "one -- it cannot desync, rank 0 samples every token. auto: on only "
        "when a cluster job's machines have different GPU architectures."),
    "KNURLOGIC_PRESET": (
        "launch preset for this model: default or lean -- the knurlogic "
        "strategy unless set here",
        "one named bundle of the settings below. default is the measured "
        "settings; lean buys context and agents with some speed and "
        "precision (8-bit KV, 512-token prompt chunks, MTP off). Any "
        "setting changed beside it beats the preset's value."),
    "VQ_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "KNURLOGIC_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "VQLAB_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "VQ_GEMMSEG_BF16IO": ("bf16 IO in the segmented GEMM",
                          "numerics-active: changing it changes "
                          "the output, up to +0.97% ppl, on weights fitted "
                          "the other way. Each model keeps what it shipped."),
    "VQ_DECODE_BF16IO": ("bf16 IO on the decode path",
                         "numerics-active: changing it changes "
                         "the output (up to +0.97% ppl). Each model keeps "
                         "what it shipped."),
}


#: The (i) beside a knob in Settings -> Models: what it does for the person
#: choosing, short and plain. KNOB_DOC keeps the measurements behind it.
KNOB_HELP = {
    "VQ_DECODE_CHUNK": "How much of the model is unpacked at once; lower "
                       "uses less memory and is also faster.",
    "KNURLOGIC_PREFILL_CHUNK": "How many prompt tokens are read at once; "
                               "wider is faster on long prompts but needs "
                               "more memory.",
    "KNURLOGIC_CONTEXT_LENGTH": "How many tokens the model can hold in "
                                "memory; more uses more memory.",
    "KNURLOGIC_MTP": "Guesses several tokens per step: usually faster, for "
                     "a little more memory.",
    "KNURLOGIC_VISION": "Reads images. Off frees the vision tower's "
                        "memory for a model you only send text.",
    "KNURLOGIC_MTP_DYNAMIC": "Guesses ahead only where that is faster; off "
                             "gives steadier timing.",
    "KNURLOGIC_KV_BITS": "8-bit holds about twice the conversation in the "
                         "same memory, slightly slower.",
    "KNURLOGIC_KV_KERNEL": "A faster way to read an 8-bit cache; leave on.",
    "KNURLOGIC_THINKING_DEFAULT": "How hard the model thinks when a client "
                                  "doesn't say. default = the model's own.",
    "KNURLOGIC_CACHE_LIMIT_GB": "Freed memory held back for reuse instead of "
                                "returned to the system. No measured speed "
                                "difference; less leaves more memory free.",
    "VQ_GEMMSEG_BF16IO": "Changes the output slightly; keep what the model "
                         "shipped with.",
    "VQ_DECODE_BF16IO": "Changes the output slightly; keep what the model "
                        "shipped with.",
    "KNURLOGIC_PRESET": "Default follows the Knurlogic presets.",
    "KNURLOGIC_CROSS_CHIP": "On: faster, but a split across different "
                            "chips can give different (still valid) tokens. "
                            "Turn off if you need identical output; "
                            "slightly slower.",
    "KNURLOGIC_COMPACT_AUTO": "Compacts the conversation automatically "
                              "when it reaches the Auto compact point, even if "
                              "the client didn't ask.",
    "KNURLOGIC_COMPACT_TRIGGER": "How full the context window gets before "
                                 "automatic compaction starts.",
    "KNURLOGIC_COMPACT_KEEP_TURNS": "How many of the most recent messages "
                                    "stay word-for-word when the rest is "
                                    "summarized.",
    "KNURLOGIC_COMPACT_TOOL_RESULTS": "Distill keeps a summary of tool "
                                      "results, making compaction slower. "
                                      "Clear is faster, but the model is "
                                      "more likely to repeat calls.",
}


# --- the names are the ARTIFACT'S, not ours --------------------------------
# A knob has a LOGICAL name here and a list of env names, preferred first.
# The resolver emits whichever one the target artifact's bundled runtime
# actually reads. The old names are still accepted from a saved setting or
# `--set`. The cache limit is one value, knurlogic's (mlx's free-buffer
# cache ceiling, which the engine sets): a VQ bundle's runtime sets the same
# ceiling from VQ_CACHE_LIMIT_GB (published ones: VQLAB_CACHE_LIMIT_GB), so
# the resolver passes knurlogic's value through under that name as well
# (resolve.emit_cache_limit).
KNOB_ALIASES = {
    "cache_limit_gb": ("KNURLOGIC_CACHE_LIMIT_GB", "VQ_CACHE_LIMIT_GB",
                       "VQLAB_CACHE_LIMIT_GB"),
    "prefill_chunk": ("KNURLOGIC_PREFILL_CHUNK", "VQLAB_PREFILL_CHUNK"),
    "decode_chunk": ("VQ_DECODE_CHUNK",),
    "prompt_concurrency": ("KNURLOGIC_PROMPT_CONCURRENCY",),
    "context_length": ("KNURLOGIC_CONTEXT_LENGTH",),
    "mtp": ("KNURLOGIC_MTP",),
    "mtp_dynamic": ("KNURLOGIC_MTP_DYNAMIC",),
    "vision": ("KNURLOGIC_VISION",),
    "kv_bits": ("KNURLOGIC_KV_BITS",),
    # the 8-bit KV decode kernel (engine/kvattn): on unless "off"; A/B knob
    "kv_kernel": ("KNURLOGIC_KV_KERNEL",),
    # the level a request that names none is served at (engine/model/thinking)
    "thinking_default": ("KNURLOGIC_THINKING_DEFAULT",),
    "cross_chip": ("KNURLOGIC_CROSS_CHIP",),
    "long_context": ("KNURLOGIC_LONG_CONTEXT",),
    "preset": ("KNURLOGIC_PRESET",),
}

#: launch settings of the model itself, read when it loads: the same on
#: every rank of a split (cluster/launch passes them ring-wide)
MODEL_KNOBS = ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC", "KNURLOGIC_VISION",
               "KNURLOGIC_KV_BITS",
               "KNURLOGIC_KV_KERNEL", "KNURLOGIC_CROSS_CHIP",
               "KNURLOGIC_LONG_CONTEXT", "KNURLOGIC_PRESET")


# --- which knobs the ENGINE consumes ----------------------------------------
# Most knobs are read by an artifact's bundled runtime. These are not: the
# scheduler takes the prompt chunk directly (interfaces/http.scheduler_
# options), and the buffer cache is a process-global mlx setting. Emitting
# them as environment variables and stopping there is how they were, for a
# while, settings that did nothing -- the resolver explained a prompt chunk
# the server never saw.
ENGINE_KNOB_NAMES = tuple(n for k in ("prefill_chunk", "cache_limit_gb",
                                      "context_length", "mtp",
                                      "mtp_dynamic", "vision", "kv_bits",
                                      "kv_kernel", "thinking_default",
                                      "cross_chip", "long_context",
                                      "preset")
                          for n in KNOB_ALIASES[k])


def vision_of(sets: dict | None) -> bool:
    """KNURLOGIC_VISION from launch settings: True (the default) unless
    set off; a bad value is the default here (serve refuses it)."""
    try:
        return on_off((sets or {}).get("KNURLOGIC_VISION"), True)
    except ValueError:
        return True


def mtp_of(sets: dict | None) -> bool:
    """KNURLOGIC_MTP from launch settings: True (the default) unless set
    off; a bad value is the default here (serve refuses it)."""
    try:
        return on_off((sets or {}).get("KNURLOGIC_MTP"), True)
    except ValueError:
        return True


def on_off(v, default: bool = True) -> bool:
    """'on'/'off' (and 1/0, true/false); None or '' is `default`."""
    s = str(v if v is not None else "").strip().lower()
    if not s:
        return default
    if s in ("on", "1", "true", "yes"):
        return True
    if s in ("off", "0", "false", "no"):
        return False
    raise ValueError(f"{v!r}: on or off")


#: what KNURLOGIC_KV_BITS may be; "bf16" is the unquantized cache
KV_BITS_VALUES = ["bf16", "8", "6", "4"]
#: what Settings offers: 6 and 4 have no fused decode kernel and were never
#: measured on a real model, so they are taken from env / --set only
KV_BITS_OFFERED = ("bf16", "8")


#: the Thinking default's values: a model's own level names, which the
#: resolver narrows the range to per model (GLM: off, low, high, max); unset
#: (the page's "default") is the template's own default
THINKING_DEFAULT_VALUES = ["off", "low", "medium", "high", "max"]


def thinking_default_of(v) -> str:
    """'model' (unset) or a level name: one word. Which names a model has
    is its template's (resolve narrows the range; the server maps it)."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "default", "model"):
        return "model"
    if not s.isalpha():
        raise ValueError(f"{s!r} isn't a thinking level")
    return s


def kv_bits_of(v):
    """'bf16'/''/None -> None; '8'/'6'/'4' -> int; anything else raises.
    The same reading as engine/kvquant.parse_bits, without mlx."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "bf16", "16", "none", "off"):
        return None
    if s not in KV_BITS_VALUES:
        n = f"{s}-bit" if s.isdigit() else repr(s)
        raise ValueError(f"KV cache: {n} isn't supported; use bf16 or 8-bit")
    return int(s)


def cross_chip_of(v) -> str:
    """'auto' | 'on' | 'off' (engine/crosschip.parse, without mlx)."""
    from knurlogic.engine.crosschip import parse
    return parse(v)


def engine_settings(env: dict) -> dict:
    """{prefill_step_size, cache_limit_gb, mtp, ...} from a
    resolved environment, whichever alias it was emitted under. Absent means
    the engine's own default stands."""
    out = {}
    for logical, key, cast in (("prefill_chunk", "prefill_step_size", int),
                               ("cache_limit_gb", "cache_limit_gb", float),
                               ("mtp", "mtp", on_off),
                               ("mtp_dynamic", "mtp_dynamic", on_off),
                               ("vision", "vision", on_off),
                               ("kv_bits", "kv_bits", kv_bits_of),
                               ("cross_chip", "cross_chip", cross_chip_of),
                               ("long_context", "long_context",
                                context_window.long_context_of)):
        for name in KNOB_ALIASES[logical]:
            if name in env:
                out[key] = (cast(float(env[name])) if cast in (int, float)
                            else cast(env[name]))
                break
    return out

def canonical_sets(sets: dict | None) -> dict:
    """Explicit settings with an accepted old name moved to the name the
    resolver emits first (VQLAB_PREFILL_CHUNK -> KNURLOGIC_PREFILL_CHUNK,
    VQ_CACHE_LIMIT_GB -> KNURLOGIC_CACHE_LIMIT_GB), so an explicit value cannot
    lose to the resolver's under another alias. Where both are given, the
    current name wins."""
    out = dict(sets or {})
    for names in KNOB_ALIASES.values():
        for old in names[1:]:
            if old in out:
                v = out.pop(old)
                out.setdefault(names[0], v)
    return out


def legacy_mirror(env: dict, forced: dict) -> dict:
    """An explicit value set under a knob's current name, copied to the old
    name the resolver emitted for this artifact's bundled runtime (which
    reads only that one). Returns the extra {name: value} to set."""
    out = {}
    for names in KNOB_ALIASES.values():
        if names[0] in forced:
            for old in names[1:]:
                if old in env and old not in forced:
                    out[old] = forced[names[0]]
    return out


#: The env name for a logical knob when no bundled runtime can be asked.
def default_alias(logical: str) -> str:
    return KNOB_ALIASES[logical][0]


# --- who is a knob FOR ------------------------------------------------------
# Those bundled runtimes read ~38 environment variables. Most are kernel
# internals -- tile widths, register buffers, SIMD group sizes -- and nobody
# outside the person writing the kernel has a reason to touch them. Showing
# all 38 would be the busy-panel mistake: every knob visible, none of them
# weighted, the eye with nowhere to go.
#
# So three tiers, by who would reach for it:
#
#   reach     will it run, will it OOM, how fast. The memory knobs and the
#             tune axis. These are the page.
#   deeper    measured performance and numerics flags. Real effects, real
#             findings behind them, but you go looking on purpose.
#   kernel    read by the runtime, no measured answer here. NOT defaulted and
#             NOT hidden: listed if someone digs, labelled as the runtime's
#             own business. Inventing defaults for unmeasured knobs is how
#             the frozen 2048*4096*2 constant happened.
KNOB_TIER_REACH = ("VQ_DECODE_CHUNK", "VQ_CACHE_LIMIT_GB",
                   "KNURLOGIC_CACHE_LIMIT_GB",
                   "VQLAB_CACHE_LIMIT_GB", "KNURLOGIC_PREFILL_CHUNK",
                   "VQLAB_PREFILL_CHUNK",
                   "KNURLOGIC_CONTEXT_LENGTH",
                   "KNURLOGIC_THINKING_DEFAULT") + MODEL_KNOBS


def knob_tier(name: str) -> str:
    if name in KNOB_TIER_REACH:
        return "reach"
    if name in numerics.NUMERICS_FLAGS:
        return "deeper"
    return "kernel"


# --- what a knob can be turned TO -------------------------------------------
# Only the knobs a person reaches for get a range. The rest are named, not
# turned. A range is (values, unit): discrete, because these are discrete --
# a continuous slider over a chunk width would invent positions that no run
# ever measured.
#
# The maximum is a MEASUREMENT, not a taste. VQ_DECODE_CHUNK stops at 32
# because 128 -> 32 is 1.37x on every rung measured and nothing above it was
# ever better; the control should stop where the evidence stops.
KNOB_RANGE: dict = {
    # 4096 is the widest ever measured (qwen3_5); the table is where a wider
    # one would have to be earned first.
    "VQLAB_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    "KNURLOGIC_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    # powers of two up to the longest window a released model has; the
    # control stops at the model's own (the resolver's value)
    "KNURLOGIC_CONTEXT_LENGTH": ([8192, 16384, 32768, 65536, 131072,
                                  262144, 524288, 1048576], "tokens"),
    "VQ_DECODE_CHUNK": ([4, 8, 16, 32], ""),
    "KNURLOGIC_MTP": (["on", "off"], ""),
    "KNURLOGIC_MTP_DYNAMIC": (["on", "off"], ""),
    "KNURLOGIC_VISION": (["on", "off"], ""),
    # narrowed per family by the resolver (Resolution.ranges): a family
    # that refuses quantized KV offers bf16 alone
    "KNURLOGIC_KV_BITS": (KV_BITS_VALUES, "bits"),
    "KNURLOGIC_KV_KERNEL": (["on", "off"], ""),
    "KNURLOGIC_THINKING_DEFAULT": (THINKING_DEFAULT_VALUES, ""),
    "KNURLOGIC_CROSS_CHIP": (["off", "on", "auto"], ""),
    # narrowed per family by the resolver: ["off"] where no model card
    # documents YaRN
    "KNURLOGIC_LONG_CONTEXT": (["off", "yarn"], ""),
    "KNURLOGIC_PRESET": (list(presets.PRESETS), ""),
    "VQ_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                          "GiB"),
    "VQLAB_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                             "GiB"),
    "KNURLOGIC_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                                 "GiB"),
}


#: numeric knobs: (type, lowest, highest or None, what it counts)
KNOB_BOUNDS = {
    "KNURLOGIC_PREFILL_CHUNK": (int, 16, 4096, "tokens"),
    "VQLAB_PREFILL_CHUNK": (int, 16, 4096, "tokens"),
    "KNURLOGIC_CONTEXT_LENGTH": (int, 256, None, "tokens"),
    "VQ_DECODE_CHUNK": (int, measured.DECODE_CHUNK_MIN,
                        measured.DECODE_CHUNK_DEFAULT, ""),
    "VQ_CACHE_LIMIT_GB": (float, 0.0, measured.CACHE_LIMIT_GB_MAX, "GiB"),
    "VQLAB_CACHE_LIMIT_GB": (float, 0.0, measured.CACHE_LIMIT_GB_MAX, "GiB"),
    "KNURLOGIC_CACHE_LIMIT_GB": (float, 0.0, measured.CACHE_LIMIT_GB_MAX, "GiB"),
}
