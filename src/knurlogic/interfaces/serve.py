"""`knurlogic serve` -- an OpenAI-compatible endpoint that loads these models.

    knurlogic serve <artifact> [--host H] [--port P] [--working-set-gib N]

knurlogic resolves the environment, registers the architecture, and serves
with its own server (interfaces/http over engine/runtime); mlx-lm is the
library underneath, not the server. The environment is set before the
server loads anything, because a VQ artifact's bundled runtime reads its
knobs at import, inside the model load. Serving across machines is
knurlogic's own (cluster/ and `--host cluster`).

Design: docs/design/server.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from knurlogic.engine import arch, mtp
from knurlogic.engine import serve as engine
from knurlogic.interfaces.page import documents
from knurlogic.machine import status, wired
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


#: prompt-cache entries one conversation keeps: its system prompt, turns
#: and segment checkpoints (on a hybrid model each checkpoint is a whole
#: entry). A server-wide cap of 10 evicts four agents' entries on a
#: hybrid model: each agent needs its own
PROMPT_CACHE_PER_AGENT = 10
#: a ring counts at most this many concurrent agents (its decode
#: concurrency, which every rank has from the same argv)
PROMPT_CACHE_AGENTS_MAX = 8
#: one machine: the entry cap under the byte cap -- bytes decide
PROMPT_CACHE_ENTRIES_BYTE_SIZED = 256
#: one machine: the share of the headroom (working set less the model)
#: the prompt cache may hold; the memory guard takes it back before a
#: request's own KV would not fit
PROMPT_CACHE_HEADROOM_SHARE = 0.5


def prompt_cache_policy(serving: dict, world: int, working_set: int,
                        holds: int) -> tuple:
    """(size, bytes or None, why) of the prompt cache.

    One machine: sized by bytes -- PROMPT_CACHE_HEADROOM_SHARE of what the
    working set leaves beside the model -- with a generous entry cap, so
    it is memory, not a count, that evicts. A ring: count-based by design
    (every rank must evict the same entries; bytes are each rank's own),
    so the count is PROMPT_CACHE_PER_AGENT per concurrent agent. An
    explicit --prompt-cache-size / --prompt-cache-gib wins."""
    size = int(serving.get("prompt_cache_size") or 0)
    cap = serving.get("prompt_cache_bytes")
    dc = int(serving.get("decode_concurrency") or 32)
    agents = max(1, min(dc, PROMPT_CACHE_AGENTS_MAX))
    per = (f"{PROMPT_CACHE_PER_AGENT} per agent x {agents} concurrent "
           f"agents (decode concurrency {dc}, at most "
           f"{PROMPT_CACHE_AGENTS_MAX} counted)")
    if world > 1:
        if size:
            return size, None, (f"{size} entries (--prompt-cache-size); a "
                                f"ring's prompt cache is count-based")
        n = PROMPT_CACHE_PER_AGENT * agents
        return n, None, (f"{n} entries: {per}; a ring's prompt cache is "
                         f"count-based -- every rank evicts the same "
                         f"entries -- and the memory guard still takes "
                         f"entries back first")
    if cap:
        n = size or PROMPT_CACHE_ENTRIES_BYTE_SIZED
        return n, int(cap), (f"{int(cap) / GIB:.1f} GiB "
                             f"(--prompt-cache-gib), up to {n} entries")
    room = int(working_set or 0) - int(holds or 0)
    if room > 0:
        b = int(room * PROMPT_CACHE_HEADROOM_SHARE)
        n = size or PROMPT_CACHE_ENTRIES_BYTE_SIZED
        return n, b, (f"{b / GIB:.1f} GiB ({PROMPT_CACHE_HEADROOM_SHARE:.0%}"
                      f" of the {room / GIB:.1f} GiB the working set leaves "
                      f"beside the model), up to {n} entries: sized by "
                      f"memory, not a count")
    n = size or PROMPT_CACHE_PER_AGENT * agents
    return n, None, (f"{n} entries" + ("" if size else f": {per}")
                     + "; the working set is unknown, so not sized by bytes")


def pipeline_share_bytes(per: list, other: int, rank: int, world: int,
                         counts=None, leader: int = 0) -> int:
    """What pipeline rank `rank` holds: its layers (rank 0 the LAST
    `counts[0]`, as launch.prepare places them) plus what every rank
    holds, plus `leader` (the head and tower) on rank 0. Counts not given
    yet (the resolver's split is made once the ring is up): an even share
    of the layers."""
    own = int(other) + (int(leader) if rank == 0 else 0)
    if counts:
        start = sum(counts[rank + 1:])
        return sum(per[start:start + counts[rank]]) + own
    return -(-sum(per) // max(world, 1)) + own



#: the serve flag that sets a knob, where one exists (else --set K=V)
_FLAG_FOR = {"KNURLOGIC_KV_BITS": "--kv-bits",
             "KNURLOGIC_CONTEXT_LENGTH": "--context-length",
             "KNURLOGIC_MTP_DYNAMIC": "--mtp-dynamic"}


def stock_runtime_line() -> str:
    """The positive statement for a model that ships no model.py."""
    try:
        from importlib.metadata import version
        v = version("mlx-lm")
    except Exception:    # an mlx-lm without metadata still runs the model
        v = "unknown"
    return f"runtime   stock mlx-lm {v} (no bundled model.py)"


def ignored_env(env: dict, forced: dict, environ) -> list:
    """One line per knob set in the environment that the resolver's value
    replaces. Serve takes knobs from its flags and --set, not from the
    environment (the resolver writes every knob it emits); a value set
    there was dropped without a word (KNURLOGIC_KV_BITS=8 ran bf16)."""
    out = []
    for k, v in sorted(env.items()):
        have = environ.get(k)
        if k in forced or have is None or have == v:
            continue
        flag = _FLAG_FOR.get(k, f"--set {k}=...")
        out.append(f"  note: {k}={have} in the environment is ignored "
                   f"(runs {v}); use {flag}")
    return out

#: `knurlogic serve`'s exit status when it refuses to start (bad settings,
#: a model it cannot load as asked): deterministic, so the page fails the
#: load with the REFUSING line and never relaunches it (cluster/recovery).
#: EX_CONFIG, distinct from a crash (1) and from argparse's usage error (2).
REFUSED_EXIT = 78
#: the prefix of every refusal line serve prints; the page reads it back
REFUSING = "REFUSING: "


def settings_refusal(a, overrides) -> str | None:
    """The first value a launch would use that its own setting refuses,
    named in the page's words and where to fix it: this model's settings
    (Settings -> Models) or the saved knurlogic-wide ones (Settings ->
    Knurlogic). Nothing is dropped silently. Launch knobs in the
    environment are not read at all (ignored_env says so), so they are not
    checked here."""
    from knurlogic.machine import preferences
    from knurlogic.tuning import settings as S
    # past the native window is long context where the family has YaRN
    # (settings.settle_context turns it on), so the ceiling is the YaRN one
    w = S.context_ceiling(getattr(a, "model_type", ""),
                          getattr(a, "raw_config", None) or {})
    why = next((m for m in (S.check_knob(k, v, w) for k, v in
                            S.canonical_sets(dict(overrides or {})).items())
                if m), None)
    if why:
        return f"{why} (Settings \u2192 Models)"
    why = next((m for _, m in preferences.invalid()), None)
    return f"{why} (Settings \u2192 Knurlogic)" if why else None


def launch_refusal(a, overrides, tune: str = "default") -> str | None:
    """None when `overrides` (a launch's --set values) and the preset `tune`
    can start `a`, else why not -- the deterministic refusals `run` makes
    before it loads a thing, in the same words, so a page or the MCP can
    refuse the launch BEFORE a process (or a ring of them) is started."""
    from knurlogic.machine import preferences
    from knurlogic.tuning import settings as S
    from knurlogic.tuning.resolve import kv_refusal, preset_env
    # a context past the native window is settled (turned on / lowered)
    # before the values are checked, as run does
    sets, _ = S.settle_context(a.model_type, a.raw_config,
                               S.canonical_sets(dict(overrides or {})))
    why = settings_refusal(a, sets)
    if why:
        return why
    sets = preferences.launch_sets(sets)
    why = documents.refuse_sets(a, sets)
    if why:
        return why
    try:
        tune = S.preset_of(sets.get("KNURLOGIC_PRESET"), tune)
        launch = S.engine_settings({**preset_env(a, tune),
                                    **{k: v for k, v in sets.items()
                                       if k in S.MODEL_KNOBS}})
    except ValueError as e:
        return str(e)
    return (kv_refusal(a, launch.get("kv_bits"))
            or S.long_context_refusal(a.model_type,
                                      launch.get("long_context", "off"))
            or None)


def launch_fit(a, overrides, tune: str = "default", draft: bool = True,
               budget_bytes: int | None = None) -> dict:
    """`tuning.resolve.single_fit_check` for a launch's settings against
    `budget_bytes` (default: the load budget): the same check `run` makes,
    so the MCP and the page can refuse before a process starts."""
    from knurlogic.machine import preferences, wired
    from knurlogic.tuning import settings as S
    from knurlogic.tuning.resolve import preset_env, single_fit_check
    if budget_bytes is None:
        budget_bytes = wired.load_budget()["bytes"]
    try:
        sets, _ = S.settle_context(a.model_type, a.raw_config,
                                   S.canonical_sets(dict(overrides or {})))
        sets = preferences.launch_sets(sets)
        tune = S.preset_of(sets.get("KNURLOGIC_PRESET"), tune)
        launch = S.engine_settings({**preset_env(a, tune),
                                    **{k: v for k, v in sets.items()
                                       if k in S.MODEL_KNOBS}})
    except ValueError:
        launch = {}
    if launch.get("mtp") is False:
        draft = False
    return single_fit_check(a, budget_bytes, draft, launch.get("kv_bits"),
                            launch.get("vision", True))


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str | None, tune: str = "default",
        overrides: dict | None = None, draft: bool = True,
        serving: dict | None = None, ring: dict | None = None) -> int:
    a = Artifact.load(path)
    print(f"artifact  {a.path.name}  ({a.model_type}, {a.gib:.1f} GiB)")
    print(f"engine    {engine.describe()}")

    # the same registration a switch gets (interfaces/loading.py)
    from knurlogic.interfaces import loading
    needed = arch.modules_for_artifact(a)
    print(f"registered {needed} from knurlogic's vendored set" if needed
          else f"no mapping for {a.model_type!r}; relying on what the "
               f"engine ships")
    missing = loading.register(a)
    if missing:
        print(f"REFUSING: no implementation for {missing}. This artifact "
              f"cannot load, and starting a server that 500s on every request "
              f"helps nobody.", file=sys.stderr)
        return REFUSED_EXIT

    ring = ring or {}
    world = int(ring.get("world") or 1)
    share = None
    if world > 1:
        ring_why = _ring_refusals(a, ring, working_set_gib, overrides or {})
        if ring_why:
            print(f"REFUSING a {world}-rank {ring.get('split')} split of "
                  f"{a.path.name}:", file=sys.stderr)
            for w in ring_why:
                print(f"  - {w}", file=sys.stderr)
            return REFUSED_EXIT
        if ring.get("split") == "pipeline":
            from knurlogic.tuning import resolve as R
            per, other = R.pipeline_layer_bytes(a)
            # rank 0 holds the tower only with vision on and the head only
            # with MTP on -- the ring-wide sets, never this rank's --no-draft,
            # so every rank computes the same split
            from knurlogic.tuning.settings import (canonical_sets, mtp_of,
                                                   vision_of)
            rs = canonical_sets(dict(overrides or {}))
            lead = R.leader_bytes(a, vision=vision_of(rs), mtp=mtp_of(rs))
            bw = ring.get("bandwidth_gbs") or R.chip_bandwidth_gbs(_chip())
            # every rank's working set and bandwidth are gathered once the
            # ring is up; the split is computed the same way on every rank
            ring["pipeline"] = {
                "layer_bytes": per, "other_bytes": other,
                "leader_bytes": lead,
                "reserve": R.fit_reserve(a.raw_config),
                "working_set": int(working_set_gib * GIB),
                "bandwidth_gbs": bw, "counts": ring.get("layers") or None}
            share = pipeline_share_bytes(per, other, int(ring["rank"]),
                                         world, ring.get("layers") or None,
                                         leader=lead)
            print(f"pipeline  rank {ring['rank']} of {world} over "
                  f"{ring['link']}: {len(per)} layers, "
                  f"{sum(per) / GIB:.1f} GiB split by layer + "
                  f"{other / GIB:.1f} GiB on every rank; memory bandwidth "
                  + (f"{bw:g} GB/s" if bw else "unknown here"))
        else:
            from knurlogic.tuning.resolve import leader_bytes, tensor_placement
            from knurlogic.tuning.settings import (canonical_sets, mtp_of,
                                                   vision_of)
            pl = tensor_placement(a, world)
            # rank 0 alone holds the head (MTP on) and the tower (vision on)
            rs = canonical_sets(dict(overrides or {}))
            share = int(pl["per_rank_bytes"]) + (leader_bytes(
                a, vision=vision_of(rs), mtp=mtp_of(rs))
                if int(ring["rank"]) == 0 else 0)
            print(f"tensor    rank {ring['rank']} of {world} over "
                  f"{ring['link']}: holds ~{pl['per_rank_bytes'] / GIB:.1f} "
                  f"GiB ({pl['sharded_bytes'] / GIB:.1f} split {world} ways "
                  f"+ {pl['replicated_bytes'] / GIB:.1f} replicated)")
        _ring_env(ring)
        _ring_marker(ring)
        # the ring-wide knobs beat the resolver like any --set
        overrides = dict(overrides or {})
        overrides.pop("VQLAB_PREFILL_CHUNK", None)
        overrides["KNURLOGIC_PREFILL_CHUNK"] = str(int(ring["prefill_chunk"]))
        if ring.get("decode_chunk"):
            overrides["VQ_DECODE_CHUNK"] = str(int(ring["decode_chunk"]))

    # The model's own launch settings (MTP, its controller, KV precision):
    # read before the resolver, since the KV bits change what the context
    # costs, and refused here with the reason rather than at load.
    from knurlogic.tuning import settings as S
    from knurlogic.tuning.resolve import apply_preset_overrides, kv_refusal, preset_env
    # A launch preset IS the tune: a per-model KNURLOGIC_PRESET (Settings
    # -> Models, carried ring-wide like every launch set) picks it over
    # --tune; its launch values are defaults every explicit set beats.
    overrides = S.canonical_sets(overrides)
    # identical results across chips is knurlogic-wide (Settings ->
    # Knurlogic, machine/preferences), not a model's: saved, it beats the
    # preset's value; an explicit --set still beats it
    from knurlogic.machine import preferences
    # a context past the native window turns long context on (or is lowered
    # to what the model reaches): a value never refuses a launch for that
    overrides, settled = S.settle_context(
        a.model_type, a.raw_config, S.canonical_sets(dict(overrides or {})))
    for n in settled:
        print(f"  note: {n}")
    why = settings_refusal(a, overrides)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return REFUSED_EXIT
    overrides = preferences.launch_sets(overrides)
    why = documents.refuse_sets(a, overrides)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return REFUSED_EXIT
    try:
        tune = S.preset_of(overrides.pop("KNURLOGIC_PRESET", None), tune)
    except ValueError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return REFUSED_EXIT
    try:
        launch = S.engine_settings({**preset_env(a, tune),
                                    **{k: v for k, v in overrides.items()
                                       if k in S.MODEL_KNOBS}})
    except ValueError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return REFUSED_EXIT
    kv_bits = launch.get("kv_bits")
    why = kv_refusal(a, kv_bits)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return REFUSED_EXIT
    long_context = launch.get("long_context", "off")
    why = S.long_context_refusal(a.model_type, long_context)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return REFUSED_EXIT
    if launch.get("mtp") is False:
        draft = False
    vision = launch.get("vision", True)
    # identical rounding across GPU architectures (engine/crosschip.py):
    # off by default; auto is on when this job's machines differ
    from knurlogic.engine import crosschip
    cross = crosschip.resolve(launch.get("cross_chip", "off"),
                              ring.get("chips") if world > 1 else None)

    ws = int(working_set_gib * GIB)
    if ws == 0:
        b = wired.load_budget()
        ws = b["bytes"]
        if ws:
            print(f"budget    {ws / GIB:.1f} GiB, limited by {b['limited_by']} "
                  f"(working set {b['working_set_bytes'] / GIB:.1f}, "
                  f"available now {b['available_bytes'] / GIB:.1f}; "
                  f"--working-set-gib overrides)")
    if world == 1 and ws:
        # The weights, the MTP head and vision bytes that will be bound,
        # and the step margin: a load that fills the budget swaps
        # on its first request instead of failing here.
        from knurlogic.tuning.resolve import single_fit_check
        chk = single_fit_check(a, ws, draft, kv_bits, vision)
        if chk["state"] == "cannot":
            print(f"REFUSING: {chk['why']}", file=sys.stderr)
            return REFUSED_EXIT
    adv = wired.advise(a.bytes_on_disk)
    if adv.get("action") == "raise":
        print("\n" + wired.render(adv) + "\n")
    # a rank holds its share, not the artifact: judged against the whole
    # 115 GiB, every rank of a 397B split would warn "does not fit this
    # box"
    r = resolve(a, ws, profile=profile, tune=tune, holds_bytes=share,
                kv_bits=kv_bits, long_context=long_context, vision=vision)
    apply_preset_overrides(r, overrides)
    if long_context != "off" and ws:
        # YaRN: the KV of the chosen context must fit, or the load is
        # refused here rather than an OOM a million tokens in
        from knurlogic.tuning.resolve import long_context_room
        ctx = int(overrides.get("KNURLOGIC_CONTEXT_LENGTH")
                  or r.env.get("KNURLOGIC_CONTEXT_LENGTH") or 0)
        why = long_context_room(a, ws, a.bytes_on_disk if share is None
                                else share, ctx, kv_bits)
        if why:
            print(f"REFUSING: KNURLOGIC_LONG_CONTEXT=yarn: {why}",
                  file=sys.stderr)
            return REFUSED_EXIT
    print(f"preset    {tune}" + (
        f" (overridden: {', '.join(sorted(r.preset['overridden']))})"
        if r.preset.get("overridden") else ""))
    # An explicit --set WINS over the resolver. Most of these knobs are read
    # at import and compiled into kernel source, so startup is the only
    # moment they can be chosen at all -- which makes "the resolver decides
    # and you may not" the wrong default for the one place it is possible.
    # Reported as overridden rather than applied quietly.
    forced = dict(overrides or {})
    # an explicit value under a knob's current name also reaches the old
    # name this artifact's bundled runtime reads
    forced.update(S.legacy_mirror(r.env, forced))
    for line in ignored_env(r.env, forced, os.environ):
        print(line)
    for k, v in sorted(r.env.items()):
        if k in forced:
            continue
        os.environ[k] = v
        print(f"  {k}={v}")
    for k, v in sorted(forced.items()):
        was = r.env.get(k)
        os.environ[k] = v
        print(f"  {k}={v}   (overridden"
              + (f", resolver said {was}" if was is not None else "") + ")")
    notes = r.notes
    if world > 1:
        # the ring's chunk is the one this rank runs: its own room's note
        # would state a chunk it will not use
        notes = [n for n in notes if not n.startswith("prompt chunk")]
        notes.insert(0, f"prompt chunk {int(ring['prefill_chunk'])}"
                     + (f" ({ring['prefill_why']})"
                        if ring.get("prefill_why") else " (ring-wide)"))
    for n in notes:
        print(f"  note: {n}")
    for w in r.warnings:
        print(f"  WARNING: {w}", file=sys.stderr)

    if a.model_file:
        print(f"\n{a.path.name} ships its own runtime ({a.model_file}) "
              f"and it WILL be executed -- that is where its kernels live.")
    else:
        print(stock_runtime_line())
        from knurlogic.engine.serve.load import vq_without_runtime
        why = vq_without_runtime(a.path)
        if why:
            print(f"\nREFUSING: {why}", file=sys.stderr)
            return REFUSED_EXIT

    rows = arch.check(a.model_type)

    def _resolve_for(ws_bytes, tune_name):
        return resolve(a, ws_bytes, profile=profile, tune=tune_name,
                       holds_bytes=share, kv_bits=kv_bits,
                       long_context=long_context, vision=vision)

    # `top` plus `ps` costs about a third of a second, and the page polls
    # status every two. Cached just long enough that a poll is free and a
    # model load still shows up on the next one.
    _mm: dict = {"at": 0.0, "doc": None}

    def _memory_map():
        import time

        from knurlogic.machine import loaded
        now = time.time()
        if _mm["doc"] is None or now - _mm["at"] > 4.0:
            try:
                _mm["doc"] = loaded.memory_map()
            except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                    AttributeError):
                _mm["doc"] = None
            _mm["at"] = now
        return _mm["doc"]

    def _status(requests):
        # Served through `aggregate` even though there is exactly one node:
        # /status.json is the contract, and a client that learns the cluster
        # shape now does not get rewritten when a second node shows up.
        snap = status.aggregate([status.snapshot(
            artifact=a, arch_rows=rows, env=r.env, requests=requests,
            node="local", memory_map=_memory_map())])
        # The wired limit belongs here because this is where somebody looks
        # when a model will not load. Advice only -- knurlogic never sets it.
        snap["wired"] = wired.advise(a.bytes_on_disk)
        snap["drafting"] = engine.drafting_status()
        from knurlogic.engine.serve import state as _st
        snap["cross_chip"] = dict(_st.SERVED.get("cross_chip") or cross)
        # 8-bit KV decode kernel: hits vs fallbacks, so an A/B of
        # KNURLOGIC_KV_KERNEL can see which path actually ran
        if _st.SERVED.get("kv_kernel") is not None:
            snap["kv_kernel"] = dict(_st.SERVED["kv_kernel"])
        # a split's every rank, from the per-step control exchange: the
        # followers serve no status of their own
        if _st.SERVED.get("ranks"):
            snap["ranks"] = [dict(x) for x in _st.SERVED["ranks"]]
        snap["preset"] = dict(r.preset)
        # what is running and what is waiting (scheduler.requests)
        from knurlogic.interfaces import http as _http
        snap["requests"] = _http.requests_now()
        # the host's own state: /status.json answers (with the artifact)
        # from the moment the server starts, long before the weights are in
        snap["load"] = _http.load_now()
        # what reasoning_effort does on the served model -- the MCP's
        # `models` answer plus the default the server actually renders
        try:
            snap["thinking"] = engine.thinking_status()
        # the status document must still answer; the error is in it
        except Exception as e:
            snap["thinking"] = {"error": f"{type(e).__name__}: {e}"}
        # Vision on the same contract page as drafting: spec, image store
        # size, encodes and pins -- the numbers that say whether images are
        # being reused or re-encoded.
        try:
            snap["vision"] = engine.vision_status()
        # the status document must still answer; the error is in it
        except Exception as e:
            snap["vision"] = {"error": f"{type(e).__name__}: {e}"}
        # `served_vision()` is the P0-frozen way to say whether the served
        # model sees images at all; the image store's own
        # size is P4's to expose (P4 owns the load path that creates it, and
        # the contract carries no accessor for a live store instance).
        # `engine.vision.store` is read defensively, by attribute, so this
        # keeps working -- reporting no size rather than crashing status --
        # whichever way P4 lands the accessor, or if it has not yet.
        from knurlogic.engine import vision as _vision
        spec = _vision.served_vision()
        image_store = None
        get_store = getattr(_vision, "served_image_store", None)
        if callable(get_store):
            try:
                store = get_store()
                if store is not None:
                    image_store = store.stats()
            except (AttributeError, TypeError, OSError, ValueError, RuntimeError):
                image_store = None
        snap["vision"] = {"served": spec.to_json() if spec else None,
                          "image_store": image_store}
        text = status.render_cluster(snap)
        if snap["wired"].get("known"):
            text += "\n\n" + wired.render(snap["wired"])
        return snap, text

    serving = dict(serving or {})
    pc_size, pc_bytes, pc_why = prompt_cache_policy(
        serving, world, ws, share if share is not None else a.bytes_on_disk)
    serving["prompt_cache_size"] = pc_size
    serving.pop("prompt_cache_bytes", None)
    if pc_bytes:
        serving["prompt_cache_bytes"] = pc_bytes
    print(f"prompt cache  {pc_why}")

    if world > 1 and int(ring["rank"]) > 0:
        # a follower: no HTTP, no scheduler -- rank 0's plans, until it stops
        from knurlogic.engine.runtime import tensor
        print(f"\nrank {ring['rank']}: following rank 0 (no HTTP here)",
              flush=True)
        tensor.serve_follower(
            str(a.path), link_kind=ring["link"],
            working_set=int(working_set_gib * GIB),
            prompt_cache_size=int((serving or {}).get("prompt_cache_size",
                                                      10)),
            completion_batch_size=int((serving or {}).get(
                "decode_concurrency", 32)),
            prefill_step_size=int(ring["prefill_chunk"]),
            executes_artifact_code=bool(a.model_file),
            split=ring.get("split", "tensor"), pipeline=ring.get("pipeline"),
            draft=draft, kv_bits=kv_bits, cross_chip=cross)
        return 0

    shown = host.split(",")[0]
    print(f"\nserving on http://{shown}:{port}/v1  (ctrl-c to stop)")
    print(f"open http://{shown}:{port}/ to see what loaded and try it")
    print("  /status (text) and /status.json for the same thing "
          "without a browser")
    print("  /settings.json - every knob, what it would be at another tune, "
          "and why it exists")
    from knurlogic.interfaces import connect
    print("\npoint a Claude-Messages harness at it:\n")
    print("  " + connect.claude_command(
        f"http://{shown}:{port}", a.path.name).replace("\n", "\n  "))
    print(f"\n  knurlogic connect --port {port} --model {a.path.name}"
          f"   for the other clients", flush=True)
    # The live environment, kept current as knobs are applied, so the panel
    # keeps telling the truth about what is RUNNING rather than about what
    # was resolved at startup.
    live_env = dict(r.env)

    def _apply(query, body):
        import json as _json

        try:
            want = _json.loads(body or b"{}")
        except ValueError:
            return {"error": "body must be JSON"}
        tune_name = want.get("tune")
        if tune_name:
            want = {k: v for k, v in _resolve_for(ws, tune_name).env.items()}
        # compaction's knobs are read per request by this server's HTTP
        # side (context_management/compaction): the environment is the setting
        # -- and they are knurlogic-wide (machine/preferences): a change
        # here is saved for every server on this machine, not this one's
        from knurlogic.machine import preferences
        from knurlogic.tuning.settings import COMPACT_KNOBS
        cur = preferences.compaction_env()
        compact = {k: str(v) for k, v in want.items()
                   if k in COMPACT_KNOBS
                   and str(v) != cur.get(k, COMPACT_KNOBS[k][0])}
        want = {k: str(v) for k, v in want.items()
                if k in engine.LIVE_KNOBS and str(v) != live_env.get(k)}
        why = documents.refuse_sets(a, {**want, **compact})
        if why:
            return {"error": why}
        if not want and not compact:
            return {"applied": {}, "note": "nothing to change on this server "
                                           "without a restart"}
        if compact:
            try:
                preferences.set(compact)
            except ValueError as e:
                return {"error": str(e)}
        done = {k: f"applied now ({v}; knurlogic-wide, read per request)"
                for k, v in compact.items()}
        if not want:
            return {"applied": done, "running": dict(live_env)}
        done.update(engine.apply_live(want))
        for k, v in want.items():
            if "applied" in done.get(k, "") or "set for" in done.get(k, ""):
                live_env[k] = v
        # on a ring the other ranks apply what took here (plan `set`)
        sched = http._CURRENT.get("scheduler")
        if sched is not None:
            sched.share_live({k: v for k, v in want.items()
                              if live_env.get(k) == v})
        return {"applied": done, "running": dict(live_env)}

    from knurlogic.interfaces import http

    # /v1/messages is served by the server itself (in-process over its
    # OpenAI surface), so it is not one of these routes.
    routes = documents.routes(
        status_fn=_status,
        settings_fn=documents.settings_document(
            a, live_env=live_env, live_tune=tune, live_working_set=ws,
            resolve_fn=_resolve_for, wired_advice=adv,
            live_knobs=engine.LIVE_KNOBS),
        models_fn=documents.models_document(serving=a.path.name),
        loaded_fn=documents.loaded_document(),
        load_fn=documents.load_action(
            artifact_for=lambda p: Artifact.load(p),
            resolve_fn=lambda art: resolve(art, ws, profile=profile, tune=tune),
            live_knobs=engine.LIVE_KNOBS,
            switch_fn=http.switch, unload_fn=http.unload),
        apply_fn=_apply)
    # A packed head is used because it is there. Nobody should have to know
    # an environment variable exists to run weights they already downloaded.
    head = mtp.find_head(a.path)
    if head is not None and draft:
        print(f"\ndrafting   {head.describe()}")
        print( "           multi-token prediction on"
              + ("; dynamic (drafts when it measures cheaper)"
                 if launch.get("mtp_dynamic", True) else
                 "; every step drafts (KNURLOGIC_MTP_DYNAMIC=off)")
              + "; KNURLOGIC_MTP=off turns it off")
    elif head is not None:
        print("\ndrafting   head present, off (KNURLOGIC_MTP=off or "
              "--no-draft)")
    if kv_bits is not None:
        print(f"kv cache   attention K/V stored at {kv_bits} bits")
    if not vision:
        from knurlogic.tuning.resolve import vision_freed_bytes
        freed = vision_freed_bytes(a, kv_bits)
        if freed:
            print(f"vision     off (KNURLOGIC_VISION=off): no tower, image "
                  f"store or image KV, {freed / GIB:.1f} GiB not held; "
                  f"image requests get a 400")
    print(f"cross-chip: {crosschip.describe(cross)}")

    # The knobs the ENGINE reads -- argv and a process-global mlx call --
    # from the environment as it finally stands, overrides included.
    from knurlogic.tuning.settings import engine_settings
    eng = engine_settings({**r.env, **forced})
    if eng:
        print("engine    " + "  ".join(f"{k}={v}" for k, v in sorted(eng.items())))

    from knurlogic.machine import allowance
    guard = int(working_set_gib * GIB) or (
        allowance.cap(wired.detected_working_set_bytes())
        if allowance.get() else 0)
    if guard:
        print(f"memory    the scheduler guards {guard / GIB:.1f} GiB ("
              + ("--working-set-gib" if working_set_gib else
                 f"the knurlogic allowance, {allowance.path()}") + ")")
    return http.serve(a, host, port, routes=routes,
                      settings={**eng, "cross_chip": cross, **(serving or {}),
                                **({"working_set_bytes": guard}
                                   if guard else {})}, draft=draft,
                      ring=ring if world > 1 else None)


def _ring_chips(text: str) -> list | None:
    """--ring-chips: a JSON list of {name, arch}; None if absent or bad."""
    import json
    try:
        v = json.loads(text) if text else None
    except ValueError:
        return None
    return [c for c in v if isinstance(c, dict)] \
        if isinstance(v, list) else None


def _ring_refusals(a: Artifact, ring: dict, working_set_gib: float,
                   overrides: dict) -> list:
    """Why this rank cannot join a tensor split, with the numbers."""
    from knurlogic.tuning.resolve import tensor_split_refusals
    why = []
    if ring.get("split") not in ("tensor", "pipeline"):
        why.append(f"--split {ring.get('split')!r}: this build splits "
                   f"'tensor' or 'pipeline'")
    if not 0 <= int(ring["rank"]) < int(ring["world"]):
        why.append(f"rank {ring['rank']} is outside a world of "
                   f"{ring['world']}")
    if working_set_gib <= 0:
        why.append("a ring needs --working-set-gib: every rank's memory "
                   "guard is stated, not detected")
    if not ring.get("prefill_chunk"):
        why.append("a ring needs --prefill-chunk: the prompt chunk is "
                   "ring-wide, never resolved per rank")
    if ring.get("link") == "ring" and not ring.get("hosts"):
        why.append("--link ring needs --hosts (one address:port per rank)")
    if ring.get("link") == "ring" and ring.get("hosts") and \
            len(ring["hosts"]) != int(ring["world"]):
        why.append(f"--hosts names {len(ring['hosts'])} ranks for a world "
                   f"of {ring['world']}")
    if ring.get("link") == "jaccl" and not (ring.get("ibv_devices")
                                            and ring.get("coordinator")):
        why.append("--link jaccl needs --ibv-devices and --coordinator")
    if ring.get("split") == "pipeline":
        from knurlogic.tuning.resolve import pipeline_refusals
        why += pipeline_refusals(a.raw_config, int(ring["world"]))
        n = ring.get("layers") or []
        if n and len(n) != int(ring["world"]):
            why.append(f"--layers names {len(n)} ranks for a world of "
                       f"{ring['world']}")
    else:
        why += tensor_split_refusals(a.path, int(ring["world"]),
                                     a.raw_config)
    return why


def _chip() -> str:
    """This machine's chip ("Apple M4 Max"), or "" when it does not say."""
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ring_env(ring: dict) -> None:
    """What mx.distributed.init reads, set before anything loads. The ring
    hostfile is written where the job's files live."""
    import json
    from pathlib import Path
    os.environ["MLX_RANK"] = str(int(ring["rank"]))
    if ring["link"] == "ring":
        d = Path.home() / ".cache" / "knurlogic" / "jobs" / ring["job"]
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"hostfile-rank{int(ring['rank'])}.json"
        f.write_text(json.dumps([[h] for h in ring["hosts"]]))
        os.environ["MLX_HOSTFILE"] = str(f)
    else:
        os.environ["MLX_IBV_DEVICES"] = str(ring["ibv_devices"])
        os.environ["MLX_JACCL_COORDINATOR"] = str(ring["coordinator"])


def _ring_marker(ring: dict) -> None:
    """A job the page started (its id is a nonce) gets a progress marker
    in its job dir, which the page watches (cluster/jobs.py). A ring
    started by hand has no page watching it and no marker."""
    from knurlogic.cluster import jobs
    if not jobs.JOB_RX.fullmatch(str(ring.get("job") or "")):
        return
    jobs.CURRENT["marker"] = jobs.Marker(ring["job"],
                                         int(ring["rank"])).start()


def _parse_sets(pairs) -> dict:
    out = {}
    for item in pairs or []:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    from knurlogic.tuning import settings as S
    p = argparse.ArgumentParser(prog="knurlogic serve",
                               description=__doc__.split("\n")[0])
    p.add_argument("artifact")
    p.add_argument("--host", default="127.0.0.1",
                   help="an address to bind (comma-separated for several: a "
                        "cluster job's leader binds loopback and its link "
                        "address), or `cluster`: every address "
                        "bound, answered on loopback and Thunderbolt only")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set; 0 asks the framework what "
                        "it may use")
    p.add_argument("--profile", default=None, choices=("v1.5", "v2"),
                   help="force a VQ numerics profile. Default: none -- "
                        "each model runs the numerics its own model.py "
                        "ships with. Forcing v1.5 on a v2 model changes "
                        "its outputs.")
    p.add_argument("--no-draft", action="store_true",
                   help="do not use a multi-token-prediction head even if "
                        "one is packed beside the weights (the same as "
                        "--set KNURLOGIC_MTP=off). Drafting preserves the "
                        "output distribution.")
    p.add_argument("--mtp-dynamic", choices=("on", "off"), default=None,
                   help="on: switch between drafting and plain steps by "
                        "their measured cost (default); off: draft every "
                        "step while a head is bound "
                        "(KNURLOGIC_MTP_DYNAMIC)")
    p.add_argument("--kv-bits", choices=("bf16", "8", "6", "4"),
                   default=None,
                   help="attention KV-cache precision (KNURLOGIC_KV_BITS): "
                        "bf16 by default; 8 is the recommended setting for "
                        "more context in the same memory, taken by every "
                        "family (6 and 4 where the family allows them)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", dest="sets",
                   help="force a setting, beating the resolver. Repeatable. "
                        "Most knobs are read at import, so this is the only "
                        "moment they can be chosen.")
    p.add_argument("--tune", "--preset", dest="tune", default="default",
                   type=S.preset_arg, metavar="{default,lean}",
                   help="launch preset: default (the measured settings) or "
                        "lean (8-bit KV where the family takes it, 512-token "
                        "prompt chunks, MTP off: most context and agents). "
                        "Explicit settings (--set, --kv-bits, a per-model "
                        "KNURLOGIC_PRESET) beat it.")
    p.add_argument("--decode-concurrency", type=int, default=32,
                   help="most requests decoding at once (the batch width)")
    p.add_argument("--prompt-cache-size", type=int, default=0,
                   help="prompt-cache entries kept (whole prompts and "
                        "segment checkpoints). Default: sized by memory on "
                        "one machine, 10 per concurrent agent on a ring")
    p.add_argument("--context-length", type=int, default=0,
                   help="the longest prompt + answer a request may use, in "
                        "tokens (KNURLOGIC_CONTEXT_LENGTH): a cap, nothing "
                        "reserved. Default: the model's own window")
    p.add_argument("--prompt-cache-gib", type=float, default=0.0,
                   help="cap the prompt cache's memory. Default: half of "
                        "what the working set leaves beside the model (one "
                        "machine); a ring's cache is count-based")
    p.add_argument("--allow-origin", action="append", default=[],
                   metavar="URL",
                   help="a web page origin (e.g. http://localhost:3000) "
                        "allowed to call this server from a browser. "
                        "Repeatable. By default only the server's own page "
                        "and non-browser clients are answered")
    p.add_argument("--allow-host", action="append", default=[],
                   metavar="NAME",
                   help="a DNS name this machine is reached by (e.g. "
                        "studio.tail1234.ts.net). Repeatable. localhost, IP "
                        "addresses, .local names and the hostname need none")
    p.add_argument("--max-request-mib", type=int, default=512,
                   help="largest request body accepted (413 above it)")
    p.add_argument("--image-store-gib", type=float, default=0.0,
                   help="memory for encoded images (default 0.25). A "
                        "request's images must fit it together")
    # Cluster plumbing: the cluster page passes these to each machine's
    # serve (tuning/resolve.rank_order picks the order). Hidden from --help:
    # nobody types a rank.
    hide = argparse.SUPPRESS
    p.add_argument("--rank", type=int, default=0, help=hide)
    p.add_argument("--world", type=int, default=1, help=hide)
    p.add_argument("--split", default="tensor", help=hide)
    p.add_argument("--link", default="ring", choices=("ring", "jaccl"),
                   help=hide)
    p.add_argument("--hosts", default="", help=hide)
    p.add_argument("--job", default="", help=hide)
    p.add_argument("--ibv-devices", default="", help=hide)
    p.add_argument("--coordinator", default="", help=hide)
    p.add_argument("--prefill-chunk", type=int, default=0, help=hide)
    p.add_argument("--prefill-why", default="", help=hide)
    p.add_argument("--decode-chunk", type=int, default=0, help=hide)
    # pipeline only: layers per rank (rank order; default: the resolver's
    # shares from every rank's working set and bandwidth), and this
    # machine's memory bandwidth when the chip table does not know it
    p.add_argument("--layers", default="", help=hide)
    p.add_argument("--bandwidth-gbs", type=float, default=0.0, help=hide)
    # every rank's {name, arch} as JSON, for KNURLOGIC_CROSS_CHIP=auto
    p.add_argument("--ring-chips", default="", help=hide)
    a = p.parse_args(argv)
    ring = None
    if a.world > 1:
        hosts = [h for h in a.hosts.split(",") if h]
        ring = {"rank": a.rank, "world": a.world, "split": a.split,
                "link": a.link, "hosts": hosts,
                "job": a.job or f"tensor-{a.port}",
                "ibv_devices": a.ibv_devices, "coordinator": a.coordinator,
                "prefill_chunk": a.prefill_chunk,
                "prefill_why": a.prefill_why,
                "decode_chunk": a.decode_chunk,
                "layers": [int(x) for x in a.layers.split(",") if x],
                "bandwidth_gbs": a.bandwidth_gbs or None,
                "chips": _ring_chips(a.ring_chips)}
    serving = {"decode_concurrency": a.decode_concurrency,
               "max_body": a.max_request_mib * 1024 * 1024,
               "allow_origins": a.allow_origin,
               "allow_hosts": a.allow_host,
               "prompt_cache_size": a.prompt_cache_size}
    if a.prompt_cache_gib > 0:
        serving["prompt_cache_bytes"] = int(a.prompt_cache_gib * GIB)
    if a.image_store_gib > 0:
        serving["image_store_bytes"] = int(a.image_store_gib * GIB)
    sets = _parse_sets(a.sets)
    if a.context_length > 0:
        sets["KNURLOGIC_CONTEXT_LENGTH"] = str(a.context_length)
    if a.mtp_dynamic:
        sets["KNURLOGIC_MTP_DYNAMIC"] = a.mtp_dynamic
    if a.kv_bits:
        sets["KNURLOGIC_KV_BITS"] = a.kv_bits
    return run(a.artifact, a.host, a.port, a.working_set_gib, a.profile,
               a.tune, sets, draft=not a.no_draft, serving=serving,
               ring=ring)


if __name__ == "__main__":
    raise SystemExit(main())
