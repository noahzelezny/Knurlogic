"""knurlogic's own server: HTTP here, the model in engine/runtime.

    knurlogic serve <artifact>

`serve()` builds the host and the scheduler, queues the load, and answers
HTTP at once: /status.json and the page work while the model loads, and
inference requests wait for it (they are queued, not refused).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

#: the running server's scheduler, for the page's load/unload actions,
#: which are built before the server is
_CURRENT: dict = {}


def requests_now():
    """The running server's scheduler.requests(), or None before it runs."""
    sched = _CURRENT.get("scheduler")
    return sched.requests() if sched is not None else None


def load_now():
    """The running server's host state ({state, error}: loading, warming,
    ready, failed, empty), or None before it runs. /status.json carries it
    so a page does not call a model ready because its port answers."""
    sched = _CURRENT.get("scheduler")
    if sched is None:
        return None
    st = sched.host.status()
    return {"state": st.get("state"), "error": st.get("error") or ""}


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
            and st["state"] in ("ready", "loading", "warming")):
        freed = int(st.get("memory_bytes") or 0)
        a = prepare(model, served=host.path, freed_bytes=freed)
        # compared resolved: the served model named through a symlink or
        # another store's path is the same model, not a switch
        same = host.path and Path(a.path).resolve() == \
            Path(host.path).resolve()
        if not same or st["state"] not in ("ready", "loading", "warming"):
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
            "prompt_cache_bytes": settings.get("prompt_cache_bytes"),
            # what the memory guard counts against: --working-set-gib, else
            # the machine's knurlogic allowance (machine/allowance.py) under
            # the detected working set; None lets the scheduler detect it
            "working_set_bytes": settings.get("working_set_bytes")}


def watch_ring(sched, mh, exit_after: float = 1.5,
               stop_within: float = 15.0) -> None:
    """Rank 0 of a split model: its progress marker says whether work is
    in flight (the page tells idle from stalled by it), and a SIGTERM --
    the page tearing the job down because another rank died or stalled --
    answers every request in flight with a 503 before the process goes.
    The scheduler thread may be blocked in a collective that never
    returns, so the exit does not wait for it."""
    import os
    import signal
    import threading

    from knurlogic.cluster import jobs
    from knurlogic.engine.runtime.scheduler import RingFailed

    armed: list = []

    def probe():
        if not armed and getattr(mh, "state", "") == "ready":
            armed.append(1)
            jobs.after_load()
        return {"busy": bool(sched.busy)}
    m = jobs.CURRENT["marker"]
    if m is not None:
        m.probe = probe

    def fail_and_leave():
        sched.abort(RingFailed(
            "this model is split across machines and the cluster job is "
            "stopping (a rank exited or stalled, or it was unloaded); "
            "retry once it is loaded again"))
        import time
        time.sleep(exit_after)          # the 503s go out first
        # a failed ring leaves non-zero: the page relaunches it
        os._exit(1 if getattr(sched, "ring_failed", None) else 0)

    def leave_after_ring_failure(_exc):
        # The scheduler stopped at a step boundary on a failed collective
        # and answered everything with RingFailed: let it finish its
        # cleanup (the executor, the model, mx.synchronize) so nothing is
        # left on the GPU, then leave non-zero so the page's recovery
        # relaunches the job. A cleanup stuck behind a collective is not
        # waited for past stop_within.
        jobs.progress(phase="stopping")

        def go():
            import time
            if not sched.wait_stopped(stop_within):
                logging.getLogger(__name__).error(
                    "the scheduler did not finish its cleanup within %.0f s "
                    "after the ring failed; leaving anyway", stop_within)
            time.sleep(exit_after)          # the 503s go out first
            logging.shutdown()
            os._exit(1)
        threading.Thread(target=go, daemon=True).start()
    sched.on_ring_failed = leave_after_ring_failure

    def on_term(_sig, _frame):
        jobs.progress(phase="stopping")
        if getattr(mh, "state", "") in ("warming", "ready"):
            # An unload with the ring alive: stop at a step boundary (while
            # warming, once the warm-up's forwards are done). The
            # scheduler finishes the step it is in, fails what is in
            # flight, and sends the other ranks `stop`, so every rank
            # leaves between steps. Killed mid-step, a rank left its
            # peer's GPU waiting on a collective that never completed: the
            # stuck queue held that GPU at 100% and pinned the job's memory
            # past every process's exit, until a reboot (measured on M3 +
            # M4, tensor over TCP and RDMA, 4 requests in flight).
            # A step that never ends means a peer is gone: fail what is in
            # flight as before and leave.
            def stop_then_leave():
                if sched.stop_ring(timeout=stop_within):
                    os._exit(0)
                else:
                    fail_and_leave()
            threading.Thread(target=stop_then_leave, daemon=True).start()
            return
        # Loading: the read stops at its next batch boundary (host.LOAD_STOP)
        # rather than dying inside one, which left the GPU's utilization
        # counter stuck at 100% until a reboot.
        from knurlogic.engine.runtime import host as H
        H.LOAD_STOP.set()

        def after_load_stops():
            import time
            end = time.monotonic() + stop_within
            while getattr(mh, "state", "") == "loading" and \
                    time.monotonic() < end:
                time.sleep(0.05)
            # read past the flag into its warm-up: stop the ring as ready
            if getattr(mh, "state", "") in ("warming", "ready") and \
                    sched.stop_ring(timeout=stop_within):
                os._exit(0)
            else:
                fail_and_leave()
        threading.Thread(target=after_load_stops, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term)


def serve(artifact, host: str, port: int, *, routes: dict | None = None,
          settings: dict | None = None, draft: bool = True,
          ring: dict | None = None) -> int:
    """`ring`: this is rank 0 of a tensor or pipeline split
    (interfaces/serve.py's ring dict); the other ranks follow its
    scheduler."""
    from knurlogic.engine.runtime.host import ModelHost
    from knurlogic.engine.runtime.scheduler import Scheduler

    from . import residency as res_api
    from .server import App

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    settings = dict(settings or {})
    if "cache_limit_gb" in settings:
        from knurlogic.engine.serve import set_cache_limit
        print(f"cache limit {set_cache_limit(settings['cache_limit_gb'])}")
    tensor = shard = shard_config = agree = None
    pipe = bool(ring and ring.get("split") == "pipeline")
    if ring:
        from knurlogic.engine.runtime import tensor as T
        link = T.init(ring["link"])
        tensor = T.Ring(link, split=ring.get("split", "tensor"))
        if pipe:
            from knurlogic.engine.runtime import pipeline as PL
            shares = PL.agree(link.group, **ring["pipeline"])
            print(f"pipeline  rank 0: {shares['reason']}", flush=True)
            def shard(m):
                return PL.split(m, link.group, shares["bounds"])
        else:
            def shard(m):
                return T.shard(m, link.group)

            def shard_config(p):
                return T.load_config(p, link.group)
        # rank 0 drafts on either split and tells the others (agree_head)
        agree = T.agree_head(link)
    mh = ModelHost(draft=draft, head_agree=agree,
                   executes_artifact_code=bool(artifact.model_file),
                   image_store_bytes=settings.get("image_store_bytes"),
                   shard=shard, shard_config=shard_config,
                   vision=settings.get("vision", True),
                   load_wait_s=3600.0 if ring else 0.0,
                   kv_bits=settings.get("kv_bits"),
                   cross_chip=settings.get("cross_chip"))
    if ring:
        # the disk prompt cache's key: this rank's part of the split
        # (engine/serve/prompt_disk; the followers' in tensor.serve_follower)
        mh.cache_layout = {"split": ring.get("split", "tensor"),
                           "world": link.size, "rank": link.rank}
        if pipe:
            mh.cache_layout["bounds"] = [list(b) for b in shares["bounds"]]
    from knurlogic.engine.serve.load import gpu_in_use
    sched = Scheduler(mh, **scheduler_options(settings),
                      tensor=tensor, gpu_in_use=gpu_in_use).start()
    sched.load(str(artifact.path),
               executes_artifact_code=bool(artifact.model_file))
    _CURRENT["scheduler"] = sched
    if ring:
        watch_ring(sched, mh)

    served = res_api.served(artifact, mh)
    from .server import DEFAULT_MAX_BODY
    app = App(sched, served=served, routes=routes,
              max_body=settings.get("max_body", DEFAULT_MAX_BODY),
              allow_origins=tuple(settings.get("allow_origins") or ()),
              allow_hosts=tuple(settings.get("allow_hosts") or ()),
              concurrency=lambda: res_api.concurrency(sched),
              residency=lambda: res_api.residency(mh, sched, port),
              ensure=res_api.ensure)
    if host == "cluster":
        # every address bound, only loopback and Thunderbolt answered --
        # the same rule as the page's (cluster/links.Gate)
        from knurlogic.cluster import links
        app.gate, host = links.Gate(), "0.0.0.0"
    servers = bind_all(app, host, port)
    httpd, extra = servers[0], servers[1:]
    for srv in extra:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    where = ", ".join(f"http://{_url_host(s.server_address[0])}:{port}/v1"
                      for s in servers)
    print(f"knurlogic's own server on {where} "
          f"(loading {artifact.path.name})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.server_close()
        sched.stop()
    return 0


def _url_host(h: str) -> str:
    return f"[{h}]" if ":" in h else h


def bind_all(app, host: str, port: int, make=None) -> list:
    """One server per address in `host` (comma-separated: a cluster job's
    leader binds loopback and its link address). The first must bind;
    a later one that cannot (the cable's address gone) is reported and
    skipped -- the job still answers on loopback."""
    from .server import make_server
    make = make or make_server
    addrs = [h.strip() for h in host.split(",") if h.strip()] or [host]
    servers = [make(app, addrs[0], port)]
    for h in addrs[1:]:
        try:
            servers.append(make(app, h, port))
        except OSError as e:
            print(f"not answering on {h}:{port}: {e}", flush=True)
    return servers
