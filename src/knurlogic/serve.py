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
"""

from __future__ import annotations

import argparse
import os
import sys

from . import arch, engine, register, status
from .artifact import Artifact
from .resolve import resolve

GIB = 1 << 30


def run(path: str, host: str, port: int, working_set_gib: float,
        profile: str, passthrough: list) -> int:
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

    r = resolve(a, int(working_set_gib * GIB), profile=profile)
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

    def _status(requests):
        snap = status.snapshot(artifact=a, arch_rows=rows, env=r.env,
                               requests=requests)
        return snap, status.render(snap)

    print(f"\nserving on http://{host}:{port}/v1  (ctrl-c to stop)")
    print(f"what is loaded: http://{host}:{port}/status  "
          f"(/status.json for the machine-readable form)", flush=True)
    return engine.serve(str(a.path), host, port,
                        executes_artifact_code=bool(a.model_file),
                        extra=passthrough, status_fn=_status)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic serve",
                               description=__doc__.split("\n")[0])
    p.add_argument("artifact")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set; 0 leaves the memory knobs "
                        "at their defaults")
    p.add_argument("--profile", default="v1.5", choices=("v1.5", "v2"))
    a, rest = p.parse_known_args(argv)
    return run(a.artifact, a.host, a.port, a.working_set_gib, a.profile, rest)


if __name__ == "__main__":
    raise SystemExit(main())
