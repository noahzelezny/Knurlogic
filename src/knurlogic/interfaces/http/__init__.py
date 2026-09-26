"""knurlogic's own server: HTTP here, the model in engine/runtime.

    knurlogic serve <artifact> --server knurlogic

`serve()` builds the host and the scheduler, queues the load, and answers
HTTP at once: /status.json and the page work while the model loads, and
inference requests wait for it (they are queued, not refused).
"""

from __future__ import annotations

import logging
from pathlib import Path

#: the running server's scheduler, for the page's load/unload actions,
#: which are built before the server is
_CURRENT: dict = {}


def switch(path: str) -> dict:
    sched = _CURRENT["scheduler"]
    before = sched.host.path
    sched.load(str(path)).wait()
    if sched.host.state != "ready":
        raise RuntimeError(f"loading {Path(path).name} failed: "
                           f"{sched.host.error}")
    return {"loaded": str(path), "was": before}


def unload() -> dict:
    sched = _CURRENT["scheduler"]
    had = sched.host.path
    sched.unload().wait()
    return {"unloaded": had}


def scheduler_options(settings: dict) -> dict:
    """The resolved settings (tuning.settings.engine_settings plus the
    serve flags) as the scheduler takes them."""
    return {"completion_batch_size": int(settings.get("decode_concurrency",
                                                      32)),
            "prefill_step_size": int(settings.get("prefill_step_size", 2048)),
            "prompt_cache_size": int(settings.get("prompt_cache_size", 10)),
            "prompt_cache_bytes": settings.get("prompt_cache_bytes")}


def serve(artifact, host: str, port: int, *, routes: dict | None = None,
          settings: dict | None = None, draft: bool = True) -> int:
    from knurlogic.engine.runtime.host import ModelHost
    from knurlogic.engine.runtime.scheduler import Scheduler

    from . import scout
    from .server import App, make_server

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    settings = dict(settings or {})
    if "cache_limit_gb" in settings:
        from knurlogic.engine.serve import set_cache_limit
        print(f"cache limit {set_cache_limit(settings['cache_limit_gb'])}")
    mh = ModelHost(draft=draft,
                   executes_artifact_code=bool(artifact.model_file))
    sched = Scheduler(mh, **scheduler_options(settings)).start()
    sched.load(str(artifact.path))
    _CURRENT["scheduler"] = sched

    served = scout.served(artifact, mh)
    app = App(sched, served=served, routes=routes,
              image_limit=scout.image_limit,
              concurrency=lambda: scout.concurrency(sched),
              residency=lambda: scout.residency(artifact, mh, sched),
              ensure=lambda body: scout.ensure(body, artifact, mh, sched))
    httpd = make_server(app, host, port)
    print(f"knurlogic's own server on http://{host}:{port}/v1 "
          f"(loading {artifact.path.name})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        sched.stop()
    return 0
