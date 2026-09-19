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

from . import arch, engine, messages, register, status, web, wired
from .artifact import Artifact
from .resolve import resolve

GIB = 1 << 30


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str, passthrough: list, tune: str = "balanced") -> int:
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
        ws = wired.detected_working_set_bytes()
        if ws:
            print(f"working set {ws / GIB:.1f} GiB (detected; "
                  f"--working-set-gib overrides)")
    adv = wired.advise(a.bytes_on_disk)
    if adv.get("action") == "raise":
        print("\n" + wired.render(adv) + "\n")
    r = resolve(a, ws, profile=profile, tune=tune)
    for k, v in sorted(r.env.items()):
        os.environ[k] = v
        print(f"  {k}={v}")
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

    def _status(requests):
        # Served through `aggregate` even though there is exactly one node:
        # /status.json is the contract, and a client that learns the cluster
        # shape now does not get rewritten when a second node shows up.
        snap = status.aggregate([status.snapshot(
            artifact=a, arch_rows=rows, env=r.env, requests=requests,
            node="local")])
        # The wired limit belongs here because this is where somebody looks
        # when a model will not load. Advice only -- knurlogic never sets it.
        snap["wired"] = wired.advise(a.bytes_on_disk)
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
    print(f"\npoint a Claude-Messages harness at it with:")
    print(f"  ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_API_KEY=x \\")
    print(f"  ANTHROPIC_DEFAULT_SONNET_MODEL={a.path.name} claude",
          flush=True)
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
        apply_fn=_apply)
    return engine.serve(str(a.path), host, port,
                        executes_artifact_code=bool(a.model_file),
                        extra=passthrough, routes=routes)


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
        from .cluster import run as run_cluster

        import shlex
        return run_cluster(a.artifact, a.host, a.port, a.profile, a.exo,
                           a.node, a.launch, shlex.split(a.exo_cmd), a.local,
                           a.tune)
    return run(a.artifact, a.host, a.port, a.working_set_gib, a.profile, rest,
               a.tune)


if __name__ == "__main__":
    raise SystemExit(main())
