"""`knurlogic serve` -- an OpenAI-compatible endpoint that loads these models.

Knurlogic resolves the environment, registers the architecture, and serves
with its own server (interfaces/http over engine/runtime; docs/SERVER.md):
mlx-lm is the library underneath -- model classes, tokenizer, caches --
not the server. Its server was patched in ~34 places until 2026-09-25 and
is no longer used.

Point Cline, Continue, Zed, OpenWebUI or anything else that speaks OpenAI at
http://host:port/v1 -- which is the whole reason to prefer an endpoint over a
chat UI nobody asked for.

ENV IS SET BEFORE THE SERVER LOADS ANYTHING, and that ordering is not
incidental: a VQ artifact's bundled runtime reads its knobs AT IMPORT, and
the import happens inside the server's own model load. Setting them after
would silently do nothing -- the same class of bug as an env file sourced
after the one that overwrites it.

    knurlogic serve <artifact> [--host H] [--port P] [--working-set-gib N]

Serving across machines is knurlogic's own (cluster/: peers, Bonjour
discovery, and `--host cluster`); it does not drive exo.
"""

from __future__ import annotations

import argparse
import os
import sys

from knurlogic.engine import arch, serve as engine, mtp
from knurlogic.interfaces import web
from knurlogic.machine import status, wired
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str | None, tune: str = "balanced",
        overrides: dict | None = None, draft: bool = True,
        serving: dict | None = None) -> int:
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
        return 2

    ws = int(working_set_gib * GIB)
    if ws == 0:
        b = wired.load_budget()
        ws = b["bytes"]
        if ws:
            print(f"budget    {ws / GIB:.1f} GiB, limited by {b['limited_by']} "
                  f"(working set {b['working_set_bytes'] / GIB:.1f}, "
                  f"available now {b['available_bytes'] / GIB:.1f}; "
                  f"--working-set-gib overrides)")
    adv = wired.advise(a.bytes_on_disk)
    if adv.get("action") == "raise":
        print("\n" + wired.render(adv) + "\n")
    r = resolve(a, ws, profile=profile, tune=tune)
    # An explicit --set WINS over the resolver. Most of these knobs are read
    # at import and compiled into kernel source, so startup is the only
    # moment they can be chosen at all -- which makes "the resolver decides
    # and you may not" the wrong default for the one place it is possible.
    # Reported as overridden rather than applied quietly.
    forced = dict(overrides or {})
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
    for n in r.notes:
        print(f"  note: {n}")
    for w in r.warnings:
        print(f"  WARNING: {w}", file=sys.stderr)

    if a.model_file:
        from knurlogic.engine.vq import runtime as _vq
        if _vq.serves(a.path):
            print(f"\n{a.path.name} ships its own runtime ({a.model_file}); "
                  f"knurlogic's runtime serves it instead -- verified "
                  f"bit-identical to that file (G-VQ, rungs.json).")
        else:
            print(f"\n{a.path.name} ships its own runtime ({a.model_file}) "
                  f"and it WILL be executed -- that is where its kernels "
                  f"live (not yet verified against knurlogic's runtime).")

    rows = arch.check(a.model_type)

    def _resolve_for(ws_bytes, tune_name):
        return resolve(a, ws_bytes, profile=profile, tune=tune_name)

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
            except Exception:
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
        # what reasoning_effort does on the served model -- the MCP's
        # `models` answer plus the default the server actually renders
        try:
            snap["thinking"] = engine.thinking_status()
        except Exception as e:
            snap["thinking"] = {"error": f"{type(e).__name__}: {e}"}
        # Vision on the same contract page as drafting: spec, image store
        # size, encodes and pins -- the numbers that say whether images are
        # being reused or re-encoded.
        try:
            snap["vision"] = engine.vision_status()
        except Exception as e:
            snap["vision"] = {"error": f"{type(e).__name__}: {e}"}
        # `served_vision()` is the P0-frozen way to say whether the served
        # model sees images at all (critique C4); the image store's own
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
            except Exception:
                image_store = None
        snap["vision"] = {"served": spec.to_json() if spec else None,
                          "image_store": image_store}
        text = status.render_cluster(snap)
        if snap["wired"].get("known"):
            text += "\n\n" + wired.render(snap["wired"])
        return snap, text

    print(f"\nserving on http://{host}:{port}/v1  (ctrl-c to stop)")
    print(f"open http://{host}:{port}/ to see what loaded and try it")
    print(f"  /status (text) and /status.json for the same thing "
          f"without a browser")
    print(f"  /settings.json - every knob, what it would be at another tune, "
          f"and why it exists")
    from knurlogic.interfaces import connect
    print(f"\npoint a Claude-Messages harness at it:\n")
    print("  " + connect.claude_command(
        f"http://{host}:{port}", a.path.name).replace("\n", "\n  "))
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
        want = {k: str(v) for k, v in want.items()
                if k in engine.LIVE_KNOBS and str(v) != live_env.get(k)}
        if not want:
            return {"applied": {}, "note": "nothing to change on this server "
                                           "without a restart"}
        done = engine.apply_live(want)
        for k, v in want.items():
            if "applied" in done.get(k, "") or "set for" in done.get(k, ""):
                live_env[k] = v
        return {"applied": done, "running": dict(live_env)}

    from knurlogic.interfaces import http

    # /v1/messages is served by the server itself (in-process over its
    # OpenAI surface), so it is not one of these routes.
    routes = web.routes(
        status_fn=_status,
        settings_fn=web.settings_document(
            a, live_env=live_env, live_tune=tune, live_working_set=ws,
            resolve_fn=_resolve_for, wired_advice=adv,
            live_knobs=engine.LIVE_KNOBS),
        models_fn=web.models_document(serving=a.path.name),
        loaded_fn=web.loaded_document(),
        load_fn=web.load_action(
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
        print( "           multi-token prediction on; --no-draft turns it off")
    elif head is not None:
        print("\ndrafting   head present, disabled by --no-draft")

    # The knobs the ENGINE reads -- argv and a process-global mlx call --
    # from the environment as it finally stands, overrides included.
    from knurlogic.tuning.settings import engine_settings
    eng = engine_settings({**r.env, **forced})
    if eng:
        print("engine    " + "  ".join(f"{k}={v}" for k, v in sorted(eng.items())))

    return http.serve(a, host, port, routes=routes,
                      settings={**eng, **(serving or {})}, draft=draft)


def _parse_sets(pairs) -> dict:
    out = {}
    for item in pairs or []:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic serve",
                               description=__doc__.split("\n")[0])
    p.add_argument("artifact")
    p.add_argument("--host", default="127.0.0.1",
                   help="an address to bind, or `cluster`: every address "
                        "bound, answered on loopback and Thunderbolt only")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set; 0 asks the framework what "
                        "it may use")
    p.add_argument("--profile", default=None, choices=("v1.5", "v2"),
                   help="force a VQ numerics profile on every rung. Default: "
                        "none -- each rung runs the numerics it was PUBLISHED "
                        "with (engine/vq/rungs.json). Forcing v1.5 on a v2 "
                        "rung changes its outputs.")
    p.add_argument("--no-draft", action="store_true",
                   help="do not use a multi-token-prediction head even if "
                        "one is packed beside the weights. Troubleshooting: "
                        "drafting preserves the output distribution, so "
                        "there is nothing to trade away by leaving it on.")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", dest="sets",
                   help="force a setting, beating the resolver. Repeatable. "
                        "Most knobs are read at import, so this is the only "
                        "moment they can be chosen.")
    p.add_argument("--tune", default="balanced",
                   choices=("safe", "balanced", "fast"),
                   help="safe = lowest peak memory; fast = spend headroom "
                        "where it buys speed. Both are capped by what has "
                        "been measured.")
    p.add_argument("--decode-concurrency", type=int, default=32,
                   help="most requests decoding at once (the batch width)")
    p.add_argument("--prompt-cache-size", type=int, default=10,
                   help="prompt-cache entries kept (whole prompts and "
                        "segment checkpoints)")
    p.add_argument("--context-length", type=int, default=0,
                   help="the longest prompt + answer a request may use, in "
                        "tokens (KNURLOGIC_CONTEXT_LENGTH): a cap, nothing "
                        "reserved. Default: the model's own window")
    p.add_argument("--prompt-cache-gib", type=float, default=0.0,
                   help="cap the prompt cache's memory; 0 = entries only")
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
    a = p.parse_args(argv)
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
    return run(a.artifact, a.host, a.port, a.working_set_gib, a.profile,
               a.tune, sets, draft=not a.no_draft, serving=serving)


if __name__ == "__main__":
    raise SystemExit(main())
