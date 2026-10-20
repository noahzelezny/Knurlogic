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
import sys
from dataclasses import dataclass, field

from knurlogic.cluster import REFUSED_EXIT
from knurlogic.engine import arch, mtp
from knurlogic.engine import model as engine
from knurlogic.engine.mtp import binding
from knurlogic.interfaces import documents
from knurlogic.machine import identity, status
from knurlogic.machine.artifact import Artifact
from knurlogic.machine.memory import wired
from knurlogic.machine.processes import PROC_ERRORS
from knurlogic.machine.servers import MODEL_PORT
from knurlogic.machine.units import GIB, MIB, gib
from knurlogic.tuning import fit
from knurlogic.tuning.checks import refuse_sets, ring_refusals, settings_refusal
from knurlogic.tuning.live import LIVE_KNOBS
from knurlogic.tuning.resolve import Resolution, resolve

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

#: the prefix of every refusal line serve prints; the page reads it back
REFUSING = "REFUSING: "


class Refused(Exception):
    """A launch serve refuses: the REFUSING line(s) it prints before it
    exits with REFUSED_EXIT."""


@dataclass
class Launch:
    """What resolve_launch settled for one `knurlogic serve`: the artifact,
    the ring and this rank's share of it, the launch settings, the budget
    and the resolver's answer; `forced` is what apply_launch set over it."""
    a: Artifact
    ring: dict
    world: int
    share: int | None
    overrides: dict
    tune: str
    profile: str | None
    working_set_gib: float
    launch: dict
    kv_bits: int | None
    long_context: str
    draft: bool
    vision: bool
    rounding: dict
    ws: int
    adv: dict
    r: Resolution
    forced: dict = field(default_factory=dict)


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str | None, tune: str = "default",
        overrides: dict | None = None, draft: bool = True,
        serving: dict | None = None, ring: dict | None = None) -> int:
    """`knurlogic serve`: register the artifact, resolve its launch, apply
    and report it, then serve it (or follow rank 0). A refusal prints its
    REFUSING line(s) and returns REFUSED_EXIT."""
    a = Artifact.load(path)
    if a.embeds:
        return serve_embeddings(a, host, port, serving, ring)
    try:
        register_artifact(a)
        launch = resolve_launch(a, working_set_gib, profile, tune, overrides,
                                draft, ring)
        apply_launch(launch)
    except Refused as e:
        print(e, file=sys.stderr)
        return REFUSED_EXIT
    return serve_launch(launch, host, port, serving)


def serve_embeddings(a: Artifact, host: str, port: int,
                     serving: dict | None, ring: dict | None) -> int:
    """An embedding model: its own server (interfaces/http/embeddings),
    none of a chat model's launch settings. Refused split, or with no
    recipe for its model_type."""
    from knurlogic.engine.embeddings import refusal
    from knurlogic.interfaces.http import embeddings
    why = refusal(a.model_type) or (
        "an embedding model runs on one machine, not split"
        if ring and int(ring.get("world") or 1) > 1 else None)
    if why:
        print(f"{REFUSING}{why}", file=sys.stderr)
        return REFUSED_EXIT
    print(f"artifact  {a.path.name}  (embedding model, {a.model_type}, "
          f"{a.bytes_on_disk / MIB:.0f} MiB)")
    return embeddings.serve(a, host, port, settings=serving)


def register_artifact(a: Artifact) -> None:
    """Register the artifact's architecture -- the same registration a
    switch gets (interfaces/load_checks.py); Refused when it cannot load."""
    print(f"artifact  {a.path.name}  ({a.model_type}, {a.gib:.1f} GiB)")
    print(f"engine    {engine.describe()}")

    # the same registration a switch gets (interfaces/load_checks.py)
    from knurlogic.interfaces import load_checks
    needed = arch.modules_for_artifact(a)
    print(f"registered {needed} from knurlogic's vendored set" if needed
          else f"no mapping for {a.model_type!r}; relying on what the "
               f"engine ships")
    missing = load_checks.register(a)
    if missing:
        raise Refused(f"{REFUSING}no implementation for {missing}. This "
                      f"artifact cannot load, and starting a server that "
                      f"500s on every request helps nobody.")


def _ring_share(a: Artifact, ring: dict, world: int, working_set_gib: float,
                overrides: dict | None) -> tuple:
    """(this rank's share in bytes, the overrides with the ring-wide knobs)
    for a rank of a split: checked (Refused), its environment, marker and
    read plan set."""
    ring_why = ring_refusals(a, ring, working_set_gib)
    if ring_why:
        raise Refused("\n".join([f"REFUSING a {world}-rank "
                                 f"{ring.get('split')} split of "
                                 f"{a.path.name}:",
                                 *(f"  - {w}" for w in ring_why)]))
    if ring.get("split") == "pipeline":
        from knurlogic.tuning import pipeline_split
        per, other = pipeline_split.pipeline_layer_bytes(a)
        # rank 0 holds the tower only with vision on and the head only
        # with MTP on -- the ring-wide sets, never this rank's --no-draft,
        # so every rank computes the same split
        from knurlogic.tuning.knobs import canonical_sets, mtp_of, vision_of
        rs = canonical_sets(dict(overrides or {}))
        lead = pipeline_split.leader_bytes(a, vision=vision_of(rs), mtp=mtp_of(rs))
        bw = ring.get("bandwidth_gbs") or identity.chip_bandwidth_gbs(identity.chip())
        # every rank's working set and bandwidth are gathered once the
        # ring is up; the split is computed the same way on every rank
        ring["pipeline"] = {
            "layer_bytes": per, "other_bytes": other,
            "leader_bytes": lead,
            "reserve": fit.fit_reserve(a.raw_config),
            "working_set": int(working_set_gib * GIB),
            "bandwidth_gbs": bw, "counts": ring.get("layers") or None}
        share = pipeline_split.pipeline_share_bytes(per, other, int(ring["rank"]),
                                     world, ring.get("layers") or None,
                                     leader=lead)
        print(f"pipeline  rank {ring['rank']} of {world} over "
              f"{ring['link']}: {len(per)} layers, "
              f"{gib(sum(per))} split by layer + "
              f"{gib(other)} on every rank; memory bandwidth "
              + (f"{bw:g} GB/s" if bw else "unknown here"))
    else:
        from knurlogic.tuning.knobs import canonical_sets, mtp_of, vision_of
        from knurlogic.tuning.pipeline_split import leader_bytes
        from knurlogic.tuning.tensor_split import tensor_placement
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
    _read_plan(a)
    # what this rank holds once loaded (rank 0: its share plus the MTP
    # head and tower it alone holds): the page's load % is against it
    from knurlogic.engine.split import marker
    marker.progress(share_bytes=int(share or 0) or None)
    # the ring-wide knobs beat the resolver like any --set
    overrides = dict(overrides or {})
    overrides.pop("VQLAB_PREFILL_CHUNK", None)
    overrides["KNURLOGIC_PREFILL_CHUNK"] = str(int(ring["prefill_chunk"]))
    if ring.get("decode_chunk"):
        overrides["VQ_DECODE_CHUNK"] = str(int(ring["decode_chunk"]))
    return share, overrides


def resolve_launch(a: Artifact, working_set_gib: float, profile: str | None,
                   tune: str, overrides: dict | None, draft: bool,
                   ring: dict | None) -> Launch:
    """The launch settled: the model's launch settings, the budget and the
    resolver's answer for this artifact (and this rank's share of a
    split), each refusal raised as Refused with its reason."""
    ring = ring or {}
    world = int(ring.get("world") or 1)
    share = None
    if world > 1:
        share, overrides = _ring_share(a, ring, world, working_set_gib,
                                       overrides)

    # The model's own launch settings (MTP, its controller, KV precision):
    # read before the resolver, since the KV bits change what the context
    # costs, and refused here with the reason rather than at load.
    from knurlogic.tuning import context_window, knobs, presets
    from knurlogic.tuning.resolve import (
        apply_preset_overrides,
        kv_refusal,
        launch_settings,
    )
    # A launch preset IS the tune: a per-model KNURLOGIC_PRESET (Settings
    # -> Models, carried ring-wide like every launch set) picks it over
    # --tune; its launch values are defaults every explicit set beats.
    overrides = knobs.canonical_sets(overrides)
    # per-chip rounding is knurlogic-wide (Settings ->
    # Knurlogic, tuning/preferences), not a model's: saved, it beats the
    # preset's value; an explicit --set still beats it
    from knurlogic.tuning import preferences
    # a context past the native window turns long context on (or is lowered
    # to what the model reaches): a value never refuses a launch for that
    overrides, settled = context_window.settle_context(
        a.model_type, a.raw_config, knobs.canonical_sets(dict(overrides or {})))
    for n in settled:
        print(f"  note: {n}")
    why = settings_refusal(a, overrides)
    if why:
        raise Refused(f"{REFUSING}{why}")
    overrides = preferences.launch_sets(overrides)
    why = refuse_sets(a, overrides)
    if why:
        raise Refused(f"{REFUSING}{why}")
    try:
        tune = presets.preset_of(overrides.pop("KNURLOGIC_PRESET", None), tune)
    except ValueError as e:
        raise Refused(f"{REFUSING}{e}") from None
    try:
        launch = launch_settings(a, tune, overrides)
    except ValueError as e:
        raise Refused(f"{REFUSING}{e}") from None
    kv_bits = launch.get("kv_bits")
    why = kv_refusal(a, kv_bits)
    if why:
        raise Refused(f"{REFUSING}{why}")
    long_context = launch.get("long_context", "off")
    why = context_window.long_context_refusal(a.model_type, long_context)
    if why:
        raise Refused(f"{REFUSING}{why}")
    if launch.get("mtp") is False:
        draft = False
    vision = launch.get("vision", True)
    # per-chip rounding (engine/matmul_padding.py): on by default (each
    # chip its own kernel); off pads for identical rounding on every chip;
    # auto is off when this job's machines differ
    from knurlogic.engine import matmul_padding
    rounding = matmul_padding.resolve(launch.get("per_chip_rounding", "on"),
                              ring.get("chips") if world > 1 else None)

    ws = int(working_set_gib * GIB)
    if ws == 0:
        b = wired.load_budget()
        ws = b["bytes"]
        if ws:
            print(f"budget    {gib(ws)}, limited by {b['limited_by']} "
                  f"(working set {b['working_set_bytes'] / GIB:.1f}, "
                  f"available now {b['available_bytes'] / GIB:.1f}; "
                  f"--working-set-gib overrides)")
    if world == 1 and ws:
        # The weights, the MTP head and vision bytes that will be bound,
        # and the step margin: a load that fills the budget swaps
        # on its first request instead of failing here.
        from knurlogic.tuning.fit import single_fit_check
        chk = single_fit_check(a, ws, draft, kv_bits, vision)
        if chk["state"] == "cannot":
            raise Refused(f"{REFUSING}{chk['why']}")
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
        from knurlogic.tuning.context_window import long_context_room
        ctx = int(overrides.get("KNURLOGIC_CONTEXT_LENGTH")
                  or r.env.get("KNURLOGIC_CONTEXT_LENGTH") or 0)
        why = long_context_room(a, ws, a.bytes_on_disk if share is None
                                else share, ctx, kv_bits)
        if why:
            raise Refused(f"{REFUSING}KNURLOGIC_LONG_CONTEXT=yarn: {why}")
    print(f"preset    {tune}" + (
        f" (overridden: {', '.join(sorted(r.preset['overridden']))})"
        if r.preset.get("overridden") else ""))
    return Launch(a=a, ring=ring, world=world, share=share,
                  overrides=overrides, tune=tune, profile=profile,
                  working_set_gib=working_set_gib, launch=launch,
                  kv_bits=kv_bits, long_context=long_context, draft=draft,
                  vision=vision, rounding=rounding, ws=ws, adv=adv, r=r)


def apply_launch(L: Launch) -> None:
    """Set the resolved environment (an explicit --set over the resolver's
    value) and report it: every knob, the notes, the warnings and the
    runtime that will execute."""
    from knurlogic.tuning import knobs
    a, r, ring, world, overrides = L.a, L.r, L.ring, L.world, L.overrides
    # An explicit --set WINS over the resolver. Most of these knobs are read
    # at import and compiled into kernel source, so startup is the only
    # moment they can be chosen at all -- which makes "the resolver decides
    # and you may not" the wrong default for the one place it is possible.
    # Reported as overridden rather than applied quietly.
    forced = dict(overrides or {})
    # an explicit value under a knob's current name also reaches the old
    # name this artifact's bundled runtime reads
    forced.update(knobs.legacy_mirror(r.env, forced))
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
        from knurlogic.engine.model.loading import vq_without_runtime
        why = vq_without_runtime(a.path)
        if why:
            raise Refused(f"\n{REFUSING}{why}")
    L.forced = forced


def serve_launch(L: Launch, host: str, port: int,
                 serving: dict | None) -> int:
    """Serve the launch: the status, settings and load routes, the prompt
    cache's size, and the HTTP server (lifecycle.serve) -- or, on a rank
    of a split past 0, follow rank 0."""
    serving = _prompt_cache_serving(L, serving)
    if L.world > 1 and int(L.ring["rank"]) > 0:
        return _follow_rank_0(L, serving)
    _print_endpoints(L, host, port)
    routes = _routes(L)
    _print_features(L)
    eng = _engine_settings(L)
    guard = _guard_bytes(L)
    from knurlogic.interfaces.http import lifecycle
    return lifecycle.serve(L.a, host, port, routes=routes,
                           settings={**eng, "rounding": L.rounding, **serving,
                                     **({"working_set_bytes": guard}
                                        if guard else {})}, draft=L.draft,
                           ring=L.ring if L.world > 1 else None)


def _prompt_cache_serving(L: Launch, serving: dict | None) -> dict:
    """`serving` with the prompt cache's size and byte budget for this
    launch (fit.prompt_cache_policy); installs the fetcher that brings a
    saved entry's pieces this machine lacks from the page's peers."""
    serving = dict(serving or {})
    pc_size, pc_bytes, pc_why = fit.prompt_cache_policy(
        serving, L.world, L.ws,
        L.share if L.share is not None else L.a.bytes_on_disk)
    serving["prompt_cache_size"] = pc_size
    serving.pop("prompt_cache_bytes", None)
    if pc_bytes:
        serving["prompt_cache_bytes"] = pc_bytes
    print(f"prompt cache  {pc_why}")
    from knurlogic.interfaces.page import prompt_cache as page_prompt_cache
    page_prompt_cache.install_fetcher()
    return serving


def _follow_rank_0(L: Launch, serving: dict) -> int:
    """A follower rank: no HTTP, no scheduler -- rank 0's plans, until it
    stops."""
    from knurlogic.engine.split import follower as split_follower
    ring = L.ring
    print(f"\nrank {ring['rank']}: following rank 0 (no HTTP here)",
          flush=True)
    split_follower.serve_follower(
        str(L.a.path), link_kind=ring["link"],
        working_set=int(L.working_set_gib * GIB),
        prompt_cache_size=int(serving.get("prompt_cache_size", 10)),
        completion_batch_size=int(serving.get("decode_concurrency", 32)),
        prefill_step_size=int(ring["prefill_chunk"]),
        executes_artifact_code=bool(L.a.model_file),
        split=ring.get("split", "tensor"), pipeline=ring.get("pipeline"),
        draft=L.draft, kv_bits=L.kv_bits, rounding=L.rounding)
    return 0


def _print_endpoints(L: Launch, host: str, port: int) -> None:
    """Where the server answers, and how to point a client at it."""
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
        f"http://{shown}:{port}", L.a.path.name).replace("\n", "\n  "))
    print(f"\n  knurlogic connect --port {port} --model {L.a.path.name}"
          f"   for the other clients", flush=True)


def _resolve_for(L: Launch):
    """resolve() for this launch's artifact at another working set or tune."""
    def at(ws_bytes, tune_name):
        return resolve(L.a, ws_bytes, profile=L.profile, tune=tune_name,
                       holds_bytes=L.share, kv_bits=L.kv_bits,
                       long_context=L.long_context, vision=L.vision)
    return at


def _routes(L: Launch) -> dict:
    """The server's own routes: status, settings, models, loaded, load and
    the live-settings apply. /v1/messages is served by the server itself
    (in-process over its OpenAI surface), so it is not one of these."""
    from knurlogic.interfaces.http import lifecycle
    # The live environment, kept current as knobs are applied, so the panel
    # keeps telling the truth about what is RUNNING rather than about what
    # was resolved at startup.
    live_env = dict(L.r.env)
    return documents.routes(
        status_fn=_status_fn(L),
        settings_fn=documents.settings_document(
            L.a, live_env=live_env, live_tune=L.tune, live_working_set=L.ws,
            resolve_fn=_resolve_for(L), wired_advice=L.adv,
            live_knobs=LIVE_KNOBS),
        models_fn=documents.models_document(serving=L.a.path.name),
        loaded_fn=documents.loaded_document(),
        load_fn=documents.load_action(
            artifact_for=lambda p: Artifact.load(p),
            resolve_fn=lambda art: resolve(art, L.ws, profile=L.profile,
                                           tune=L.tune),
            live_knobs=LIVE_KNOBS,
            switch_fn=lifecycle.switch, unload_fn=lifecycle.unload),
        apply_fn=_apply_fn(L, live_env))


def _cached_memory_map(ttl_s: float = 4.0):
    """footprint.memory_map, cached for `ttl_s`: `top` plus `ps` costs about
    a third of a second and the page polls status every two, so a poll is
    free and a model load still shows up on the next one."""
    import time

    from knurlogic.machine.memory import footprint
    memo: dict = {"at": 0.0, "doc": None}

    def memory_map():
        now = time.time()
        if memo["doc"] is None or now - memo["at"] > ttl_s:
            try:
                memo["doc"] = footprint.memory_map()
            except (*PROC_ERRORS, ValueError, KeyError, AttributeError):
                memo["doc"] = None
            memo["at"] = now
        return memo["doc"]
    return memory_map


def _status_fn(L: Launch):
    """The /status.json document for this server: (snapshot, text)."""
    from knurlogic.engine import matmul_padding
    from knurlogic.engine.model import state as served
    from knurlogic.interfaces.http import lifecycle
    rows = arch.check(L.a.model_type)
    memory_map = _cached_memory_map()

    def status_doc(requests):
        # Served through `aggregate` even though there is exactly one node:
        # /status.json is the contract, and a client that learns the cluster
        # shape now does not get rewritten when a second node shows up.
        snap = status.aggregate([status.snapshot(
            artifact=L.a, arch_rows=rows, env=L.r.env, requests=requests,
            node="local", memory_map=memory_map(),
            memory_fn=engine.process_memory)])
        # The wired limit belongs here because this is where somebody looks
        # when a model will not load. Advice only -- knurlogic never sets it.
        snap["wired"] = wired.advise(L.a.bytes_on_disk)
        snap["drafting"] = binding.drafting_status()
        snap["per_chip_rounding"] = dict(
            served.SERVED.get("per_chip_rounding")
            or matmul_padding.public(L.rounding))
        # 8-bit KV decode kernel: hits vs fallbacks, so an A/B of
        # KNURLOGIC_KV_KERNEL can see which path actually ran
        if served.SERVED.get("kv_kernel") is not None:
            snap["kv_kernel"] = dict(served.SERVED["kv_kernel"])
        # a split's every rank, from the per-step control exchange: the
        # followers serve no status of their own
        if served.SERVED.get("ranks"):
            snap["ranks"] = [dict(x) for x in served.SERVED["ranks"]]
        snap["preset"] = dict(L.r.preset)
        # what is running and what is waiting (scheduler.requests)
        snap["requests"] = lifecycle.requests_now()
        # the host's own state: /status.json answers (with the artifact)
        # from the moment the server starts, long before the weights are in
        snap["load"] = lifecycle.load_now()
        # what reasoning_effort does on the served model -- the MCP's
        # `models` answer plus the default the server actually renders
        try:
            snap["thinking"] = engine.thinking_status()
        # the status document must still answer; the error is in it
        except Exception as e:
            snap["thinking"] = {"error": f"{type(e).__name__}: {e}"}
        snap["vision"] = _vision_status()
        text = status.render_cluster(snap)
        if snap["wired"].get("known"):
            text += "\n\n" + wired.render(snap["wired"])
        return snap, text
    return status_doc


def _vision_status() -> dict:
    """Whether the served model sees images (`served`: its VisionSpec, the
    key the page and clients read) and its image store, encodes and pins:
    the numbers that say whether images are reused or re-encoded."""
    from knurlogic.engine import vision
    spec = vision.served_vision()
    try:
        live = engine.vision_status()
    # the status document must still answer; the error is in it
    except Exception as e:
        live = {"error": f"{type(e).__name__}: {e}"}
    return {"served": spec.to_json() if spec else None,
            "image_store": live.get("store"),
            **{k: live[k] for k in ("encodes", "pinned", "error") if k in live}}


def _apply_fn(L: Launch, live_env: dict):
    """The live-settings route: apply what can change without a restart,
    keeping `live_env` the running truth."""
    import json

    from knurlogic.interfaces.http import lifecycle
    from knurlogic.tuning import preferences
    from knurlogic.tuning.groups import COMPACT_KNOBS, PROMPT_CACHE_KNOBS
    resolve_for = _resolve_for(L)

    def apply(query, body):
        try:
            want = json.loads(body or b"{}")
        except ValueError:
            return {"error": "body must be JSON"}
        tune_name = want.get("tune")
        if tune_name:
            want = dict(resolve_for(L.ws, tune_name).env)
        # compaction's knobs are read per request by this server's HTTP
        # side (context_management/compaction): the environment is the setting
        # -- and they are knurlogic-wide (tuning/preferences): a change
        # here is saved for every server on this machine, not this one's
        cur = preferences.compaction_env(preferences.prompt_cache_env())
        wide = {**COMPACT_KNOBS, **PROMPT_CACHE_KNOBS}
        compact = {k: str(v) for k, v in want.items()
                   if k in wide and str(v) != cur.get(k, wide[k][0])}
        want = {k: str(v) for k, v in want.items()
                if k in LIVE_KNOBS and str(v) != live_env.get(k)}
        why = refuse_sets(L.a, {**want, **compact})
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
        sched = lifecycle.CURRENT.get("scheduler")
        if sched is not None:
            sched.share_live({k: v for k, v in want.items()
                              if live_env.get(k) == v})
        return {"applied": done, "running": dict(live_env)}
    return apply


def _print_features(L: Launch) -> None:
    """What this launch runs beyond the weights: drafting, KV precision,
    vision off, per-chip rounding."""
    from knurlogic.engine import matmul_padding
    # A packed head is used because it is there. Nobody should have to know
    # an environment variable exists to run weights they already downloaded.
    head = mtp.find_head(L.a.path)
    if head is not None and L.draft:
        print(f"\ndrafting   {head.describe()}")
        print( "           multi-token prediction on"
              + ("; dynamic (drafts when it measures cheaper)"
                 if L.launch.get("mtp_dynamic", True) else
                 "; every step drafts (KNURLOGIC_MTP_DYNAMIC=off)")
              + "; KNURLOGIC_MTP=off turns it off")
    elif head is not None:
        print("\ndrafting   head present, off (KNURLOGIC_MTP=off or "
              "--no-draft)")
    if L.kv_bits is not None:
        print(f"kv cache   attention K/V stored at {L.kv_bits} bits")
    if not L.vision:
        from knurlogic.tuning.fit import vision_freed_bytes
        freed = vision_freed_bytes(L.a, L.kv_bits)
        if freed:
            print(f"vision     off (KNURLOGIC_VISION=off): no tower, image "
                  f"store or image KV, {gib(freed)} not held; "
                  f"image requests get a 400")
    print(f"per-chip rounding: {matmul_padding.describe(L.rounding)}")


def _engine_settings(L: Launch) -> dict:
    """The knobs the ENGINE reads -- argv and a process-global mlx call --
    from the environment as it finally stands, overrides included."""
    from knurlogic.tuning.knobs import engine_settings
    eng = engine_settings({**L.r.env, **L.forced})
    if eng:
        print("engine    " + "  ".join(f"{k}={v}" for k, v in sorted(eng.items())))
    return eng


def _guard_bytes(L: Launch) -> int:
    """The working set the scheduler guards: --working-set-gib, else the
    knurlogic allowance's cap, else 0 (none)."""
    from knurlogic.machine.memory import allowance
    guard = int(L.working_set_gib * GIB) or (
        allowance.cap(wired.detected_working_set_bytes(
            engine.memory().get("working_set_bytes") or 0))
        if allowance.get() else 0)
    if guard:
        print(f"memory    the scheduler guards {gib(guard)} ("
              + ("--working-set-gib" if L.working_set_gib else
                 f"the knurlogic allowance, {allowance.path()}") + ")")
    return guard


def _ring_chips(text: str) -> list | None:
    """--ring-chips: a JSON list of {name, arch}; None if absent or bad."""
    import json
    try:
        v = json.loads(text) if text else None
    except ValueError:
        return None
    return [c for c in v if isinstance(c, dict)] \
        if isinstance(v, list) else None


def _ring_env(ring: dict) -> None:
    """What mx.distributed.init reads, set before anything loads. The ring
    hostfile is written where the job's files live."""
    import json

    from knurlogic.cluster import jobs
    os.environ["MLX_RANK"] = str(int(ring["rank"]))
    if ring["link"] == "ring":
        f = jobs.job_dir(ring["job"]) / f"hostfile-rank{int(ring['rank'])}.json"
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
    from knurlogic.engine.split import marker
    if not jobs.JOB_RX.fullmatch(str(ring.get("job") or "")):
        return
    marker.CURRENT["marker"] = jobs.Marker(ring["job"],
                                         int(ring["rank"])).start()


def _read_plan(a) -> None:
    """This rank's entry of the job's read plan (KNURLOGIC_WEIGHTS, set by
    its page): when it reads, and the page whose disk it reads from
    (docs/design/weight-streaming.md). Absent: it reads as before."""
    from knurlogic.cluster import weights as CW
    from knurlogic.engine.model import weight_files
    from knurlogic.engine.split import weights
    from knurlogic.machine.artifact import identity
    plan, page, source = weights.plan_from_env(os.environ)
    weights.CURRENT["plan"] = plan
    if plan is None:
        return
    if page:
        weight_files.use_peer(weight_files.Peer(
            read=CW.reader(page, identity(a.path), a.path.name),
            source=source or "a peer", root=a.path))
    print("weights   "
          + (f"from {source or 'a peer'} over the link" if page
             else "from this machine's own copy")
          + (f", turn {plan.turn}" if plan.ordered else ""), flush=True)


def _parse_sets(pairs) -> dict:
    out = {}
    for item in pairs or []:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        out[k.strip()] = v.strip()
    return out


def _parser() -> argparse.ArgumentParser:
    """`knurlogic serve`'s flags (the hidden ones are a cluster rank's)."""
    from knurlogic.tuning import presets
    p = argparse.ArgumentParser(prog="knurlogic serve",
                               description=__doc__.split("\n")[0])
    p.add_argument("artifact")
    p.add_argument("--host", default="127.0.0.1",
                   help="an address to bind (comma-separated for several: a "
                        "cluster job's leader binds loopback and its link "
                        "address), or `cluster`: every address "
                        "bound, answered on loopback and Thunderbolt only")
    p.add_argument("--port", type=int, default=MODEL_PORT)
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
                        "output distribution but is not bit-identical to "
                        "plain decoding: a near-tie can go either way.")
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
                   type=presets.preset_arg,
                   metavar="{" + ",".join(presets.PRESETS) + "}",
                   help="launch preset: default (the measured settings), "
                        "lean (8-bit KV where the family takes it, 512-token "
                        "prompt chunks, MTP off: most context and agents) or "
                        "benchmark (reproducible runs: the same rounding on "
                        "every chip, MTP drafting every step, 512-token "
                        "chunks, no prompt-cache reuse). "
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
    # serve (tuning/rank_order.rank_order picks the order). Hidden from --help:
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
    # every rank's {name, arch} as JSON: named when per-chip rounding is off
    p.add_argument("--ring-chips", default="", help=hide)
    return p


def main(argv=None) -> int:
    """`knurlogic serve` from its command line: the ring, serving and
    launch settings the flags name, then run."""
    a = _parser().parse_args(argv)
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
               "max_body": a.max_request_mib * MIB,
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
