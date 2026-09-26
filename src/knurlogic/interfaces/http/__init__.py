"""knurlogic's own server: HTTP here, the model in engine/runtime.

    knurlogic serve <artifact>

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


def switch(model: str, *, force: bool = False, wait: bool = True,
           timeout: float = 3600.0) -> dict:
    """Serve `model` (an id from /models.json, or the served one): the
    checks startup makes (interfaces/loading.prepare), then a load on the
    scheduler's thread. Idempotent. NotLoadable says why not; `force`
    switches even with requests running (they fail, saying so)."""
    from knurlogic.interfaces.loading import NotLoadable, prepare
    sched = _CURRENT["scheduler"]
    host = sched.host
    st = host.status()
    if not (model in ("", Path(host.path or "").name, host.path)
            and st["state"] in ("ready", "loading")):
        freed = int(st.get("memory_bytes") or 0)
        a = prepare(model, served=host.path, freed_bytes=freed)
        # compared resolved: the served model named through a symlink or
        # another store's path is the same model, not a switch
        same = host.path and Path(a.path).resolve() == \
            Path(host.path).resolve()
        if not same or st["state"] not in ("ready", "loading"):
            cmd = sched.load(str(a.path),
                             executes_artifact_code=bool(a.model_file),
                             force=force)
            cmd.started.wait()
            if cmd.error:
                raise NotLoadable(409, cmd.error, "model_busy")
    if wait:
        host.wait_ready(timeout)
    st = host.status()
    if st["state"] == "failed":
        raise NotLoadable(500, f"loading {Path(st['model'] or '').name} "
                               f"failed: {st['error']}", "load_failed")
    return {"model": Path(st["model"] or "").name, "state": st["state"],
            "memory_bytes": int(st.get("memory_bytes") or 0)}


def unload() -> dict:
    sched = _CURRENT["scheduler"]
    had = sched.host.path
    cmd = sched.unload(force=False)
    cmd.done.wait()
    if cmd.error:
        from knurlogic.interfaces.loading import NotLoadable
        raise NotLoadable(409, cmd.error, "model_busy")
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
                   executes_artifact_code=bool(artifact.model_file),
                   image_store_bytes=settings.get("image_store_bytes"))
    sched = Scheduler(mh, **scheduler_options(settings)).start()
    sched.load(str(artifact.path),
               executes_artifact_code=bool(artifact.model_file))
    _CURRENT["scheduler"] = sched

    served = scout.served(artifact, mh)
    from .server import DEFAULT_MAX_BODY
    app = App(sched, served=served, routes=routes,
              max_body=settings.get("max_body", DEFAULT_MAX_BODY),
              allow_origins=tuple(settings.get("allow_origins") or ()),
              allow_hosts=tuple(settings.get("allow_hosts") or ()),
              concurrency=lambda: scout.concurrency(sched),
              residency=lambda: scout.residency(mh, sched),
              ensure=scout.ensure)
    if host == "cluster":
        # every address bound, only loopback and Thunderbolt answered --
        # the same rule as the page's (cluster/links.Gate)
        from knurlogic.cluster import links
        app.gate, host = links.Gate(), "0.0.0.0"
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
