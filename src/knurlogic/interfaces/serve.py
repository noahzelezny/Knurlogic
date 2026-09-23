"""`knurlogic serve` -- an OpenAI-compatible endpoint that loads these models.

This is an ADAPTER, not a server. mlx-lm already ships a complete
OpenAI-compatible server -- request schema, streaming, chat templates, stop
sequences -- and rewriting that would be the least valuable thing in this
package. What it does NOT do is get the environment right: the architecture
may be one mlx-lm does not ship, the artifact may carry its own runtime, and
the memory knobs that decide whether a long prompt survives are not exposed.

So: Knurlogic resolves and registers, mlx-lm serves.

Point Cline, Continue, Zed, OpenWebUI or anything else that speaks OpenAI at
http://host:port/v1 -- which is the whole reason to prefer an endpoint over a
chat UI nobody asked for.

ENV IS SET BEFORE THE SERVER LOADS ANYTHING, and that ordering is not
incidental: a VQ artifact's bundled runtime reads its knobs AT IMPORT, and
the import happens inside the server's own model load. Setting them after
would silently do nothing -- the same class of bug as an env file sourced
after the one that overwrites it.

    knurlogic serve <artifact> [--host H] [--port P] [--working-set-gib N]

One box is the default. `--cluster` serves the same artifact across nodes by
wrapping exo, which already places and shards; see cluster.py.
"""

from __future__ import annotations

import argparse
import os
import sys

from knurlogic.engine import arch, seam as engine, mtp, register
from knurlogic.interfaces import messages, web
from knurlogic.machine import status, wired
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str, passthrough: list, tune: str = "balanced",
        overrides: dict | None = None, draft: bool = True) -> int:
    a = Artifact.load(path)
    print(f"artifact  {a.path.name}  ({a.model_type}, {a.gib:.1f} GiB)")
    print(f"engine    {engine.describe()}")

    needed = arch.required_modules(a.model_type)
    if needed:
        done = register.register(*needed)
        print(f"registered {done or '(already imported)'} "
              f"from knurlogic's vendored set")
    else:
        print(f"no mapping for {a.model_type!r}; relying on what the engine "
              f"ships")

    missing = [r.module for r in arch.check(a.model_type) if not r.present]
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
        print(f"\n{a.path.name} ships its own runtime ({a.model_file}) and it "
              f"WILL be executed -- that is where its kernels live.")

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

    routes = web.routes(
        # `/v1/messages` so a harness pointed here with ANTHROPIC_BASE_URL
        # works. It is a translation over the engine's own OpenAI endpoint,
        # never a second inference path.
        messages_fn=messages.handler(
            f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
            f"/v1/chat/completions", model=a.path.name),
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
            live_knobs=engine.LIVE_KNOBS),
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

    return engine.serve(str(a.path), host, port,
                        executes_artifact_code=bool(a.model_file),
                        extra=passthrough, routes=routes, draft=draft,
                        settings=eng)


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
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set; 0 asks the framework what "
                        "it may use")
    p.add_argument("--profile", default="v1.5", choices=("v1.5", "v2"))
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
    p.add_argument("--cluster", action="store_true",
                   help="serve across nodes by wrapping exo: resolve settings "
                        "per node, proxy the OpenAI surface, aggregate /status")
    p.add_argument("--exo", default="http://127.0.0.1:52415",
                   help="the exo API to attach to (--cluster)")
    p.add_argument("--node", action="append", default=[], metavar="NAME:GIB",
                   help="declare a node and its usable working set, instead "
                        "of taking exo's system-RAM numbers (--cluster)")
    p.add_argument("--launch", action="store_true",
                   help="start exo rather than attaching to one, with this "
                        "node's settings in its environment (--cluster)")
    p.add_argument("--exo-cmd", default="",
                   help="the command that starts exo on this box")
    p.add_argument("--local", default=None,
                   help="which node name is this box (--cluster --launch)")
    a, rest = p.parse_known_args(argv)
    if a.cluster:
        from knurlogic.interfaces.cluster import run as run_cluster

        import shlex
        return run_cluster(a.artifact, a.host, a.port, a.profile, a.exo,
                           a.node, a.launch, shlex.split(a.exo_cmd), a.local,
                           a.tune, draft=not a.no_draft)
    return run(a.artifact, a.host, a.port, a.working_set_gib, a.profile, rest,
               a.tune, _parse_sets(a.sets), draft=not a.no_draft)


if __name__ == "__main__":
    raise SystemExit(main())
