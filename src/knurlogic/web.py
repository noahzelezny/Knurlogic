"""The routes Knurlogic serves next to the engine's OpenAI surface.

One place, because two of them exist -- `serve` hands these to mlx-lm's
server and `serve --cluster` hands the same ones to its own front end, and a
settings page that differed between them would be worse than none.

WHY THERE IS A SETTINGS ROUTE AT ALL. A GUI that shows a green light and no
knobs is a status page wearing a costume; the knobs are the reason to open
it. But this cannot pretend a slider retunes a loaded model: the runtime
reads its environment AT IMPORT, and the import already happened. So the page
answers three separate questions and never blurs them --

    what is running now, what WOULD this setting give me, and how do I get it

-- which is more honest than a control that appears to work and does not, and
more useful than no control at all.
"""

from __future__ import annotations

import json
from pathlib import Path

GIB = 1 << 30
PAGE = Path(__file__).parent / "web" / "index.html"


def _json(obj) -> tuple:
    return json.dumps(obj, indent=1).encode(), "application/json"


def _text(s: str) -> tuple:
    return s.encode(), "text/plain; charset=utf-8"


def _connect_doc(artifact) -> dict:
    """How to point a client here -- the panel exo gets right."""
    from . import connect
    return {"model": artifact.path.name,
            "claude": connect.claude_command("__BASE__", artifact.path.name),
            "openai": connect.openai_snippet("__BASE__", artifact.path.name)}


def raw(fn):
    """Mark a handler that writes its own response.

    Everything else here returns one body and a content type, which cannot
    express an event stream: a harness reads tokens as they arrive, so the
    handler needs the socket rather than a return value.
    """
    fn.raw = True
    return fn


_MODELS: dict = {"at": 0.0, "rows": None}


def models_document(serving: str = "", ttl: float = 60.0):
    """`/models.json` -- what else is on this machine, and what is loaded.

    Cached for `ttl` seconds: a scan walks every store on every volume and
    takes about 1.4s here, which is fine once and not fine behind a status
    poll. The TTL rather than a permanent cache because models arrive while
    the server is up -- a download finishing should show up without a
    restart.

    Switching is NOT offered. A loaded model is loaded; what this can
    honestly hand over is the command that would serve another one.
    """
    def handler(_q: dict) -> dict:
        import time

        from . import discover
        now = time.time()
        if _MODELS["rows"] is None or now - _MODELS["at"] > ttl:
            try:
                _MODELS["rows"] = discover.find()
            except Exception:
                _MODELS["rows"] = []
            _MODELS["at"] = now
        out = []
        for f in _MODELS["rows"]:
            out.append({
                "name": f.name, "path": str(f.path), "store": f.store,
                "size_bytes": f.bytes_on_disk, "model_type": f.model_type,
                "is_vq": f.is_vq, "servable": f.servable, "why": f.why,
                "mtp": bool(f.extra.get("mtp_head")),
                "serving": bool(serving) and (f.name == serving
                                              or str(f.path) == serving),
            })
        return {"models": out, "serving": serving}
    return handler


_LOADED: dict = {"at": 0.0, "doc": None}


def loaded_document(ttl: float = 4.0):
    """`/loaded.json` -- every runtime on this box and what it holds.

    Short TTL, not the 60s the disk scan gets: residency changes the moment
    someone loads something, and a stale answer here is the answer being
    wrong rather than merely old. Four HTTP reads with short timeouts came
    back in 0.08s against a live exo, so the TTL is about coalescing a
    refresh burst, not about cost.
    """
    def handler(_q: dict) -> dict:
        import time

        from . import loaded
        now = time.time()
        if _LOADED["doc"] is None or now - _LOADED["at"] > ttl:
            try:
                _LOADED["doc"] = loaded.survey()
            except Exception as e:
                _LOADED["doc"] = {"resident": [], "runtimes": [],
                                  "bytes_resident": 0, "error": str(e)}
            _LOADED["at"] = now
        return _LOADED["doc"]
    return handler


def load_action(artifact_for, resolve_fn=None, live_knobs=()):
    """`POST /loaded.json` -- load, unload, or hand the job to exo.

    The reason this is knurlogic's job and not a link to exo's page: a model
    swapped into a running process gets the environment that process STARTED
    with. Knurlogic is the only thing here that knows what the incoming
    artifact would have resolved to, so it is the only thing that can say
    which of those settings did not survive the switch. Doing the load and
    staying quiet about that would be worse than not offering it.
    """
    from . import engine, loaded as L

    def _drift(path: str) -> dict:
        """Which resolved settings the running process cannot honour."""
        if resolve_fn is None:
            return {}
        try:
            a = artifact_for(path)
            want = resolve_fn(a).env
        except Exception:
            return {}
        import os
        stuck, applied = {}, {}
        for k, v in want.items():
            now = os.environ.get(k)
            if str(v) == str(now):
                continue
            (applied if k in live_knobs else stuck)[k] = {
                "wanted": str(v), "running": now}
        return {"applied_live": applied, "needs_restart": stuck}

    def handler(_q: dict, body=None) -> dict:
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        act, target = req.get("action"), req.get("target") or ""
        where = req.get("where") or ""
        try:
            if act == "load":
                r = engine.switch(target)
                r["settings"] = _drift(target)
                return r
            if act == "unload":
                return engine.unload()
            if act == "exo-load":
                return L.exo_load(where, target)
            if act == "exo-unload":
                return L.exo_unload(where, target)
            if act == "ollama-unload":
                return L.ollama_unload(where, target)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown action {act!r}"}
    return handler


def machine_settings():
    """`/settings.json` when no model is loaded.

    Most of what Settings shows belongs to an artifact -- the knobs are read
    by a bundled runtime and mean nothing without one. What is left is the
    setting that belongs to the MACHINE and is the same whether anything is
    loaded: how much of the GPU's memory macOS will let a process wire down.
    That is the one that decides whether a rung loads at all, and it is
    normally a sysctl somebody has to go and look up.

    knurlogic never runs it. It reads the current value, works out the
    ceiling, and hands over the exact line.
    """
    def handler(q: dict) -> dict:
        from . import wired

        want = q.get("wired_gib")
        if isinstance(want, list):
            want = want[0] if want else None
        adv = wired.advise(0)
        doc = {"knobs": [], "tunes": [], "unmanaged": [], "exports": "",
               "asked": {}, "wired": adv, "machine": wired.machine()}
        if adv.get("known"):
            cur = adv["limit_bytes"] / GIB
            ceil_ = adv["ceiling_bytes"] / GIB
            try:
                target = float(want) if want not in (None, "") else cur
            except (TypeError, ValueError):
                target = cur
            # `ceiling_bytes` is knurlogic's RECOMMENDATION -- installed
            # memory less a reserve for macOS -- and not a wall the OS
            # enforces. So it is a soft gate, as everywhere else here: past
            # it you get the command and a warning rather than a refusal.
            # The hard stop is 4 GiB from the top, where the machine stops
            # being able to run itself.
            hard = (adv["total_bytes"] - 4 * GIB) / GIB
            target = max(1.0, min(target, hard))
            doc["wired_target_gib"] = round(target, 1)
            doc["wired_recommended_gib"] = round(ceil_, 1)
            doc["wired_current_gib"] = round(cur, 1)
            doc["wired_installed_gib"] = round(adv["total_bytes"] / GIB, 1)
            doc["wired_command"] = (
                wired.command_for(int(target * GIB))
                if abs(target - cur) >= 0.05 else "")
            if target > ceil_ + 0.05:
                doc["wired_warn"] = (
                    f"above the {ceil_:.0f} GiB knurlogic recommends, which "
                    f"leaves {adv['total_bytes'] / GIB - target:.0f} GiB for "
                    f"macOS and everything else on the machine")
        return doc
    return handler


def routes(status_fn=None, settings_fn=None, apply_fn=None,
           messages_fn=None, models_fn=None, loaded_fn=None,
           load_fn=None) -> dict:
    """path -> handler(query: dict) -> (body, content_type).

    `status_fn(requests)` returns (snapshot, text). `settings_fn(query)`
    returns the settings document below.
    """
    r = {}

    if PAGE.is_file():
        def _page(_q, _n=0):
            return PAGE.read_bytes(), "text/html; charset=utf-8"
        r["/"] = r["/ui"] = _page

    if status_fn is not None:
        def _status(_q, n=0):
            return _text(status_fn(n)[1])

        def _status_json(_q, n=0):
            return _json(status_fn(n)[0])
        r["/status"] = _status
        r["/status.json"] = _status_json

    if settings_fn is not None:
        def _settings(q, _n=0):
            return _json(settings_fn(q))
        r["/settings.json"] = _settings
    if models_fn is not None:
        def _models(q, _n=0):
            return _json(models_fn(q))
        r["/models.json"] = _models
    if loaded_fn is not None:
        def _loaded(q, _n=0):
            return _json(loaded_fn(q))
        r["/loaded.json"] = _loaded
    if load_fn is not None:
        def _load(q, _n=0, body=None):
            _LOADED["doc"] = None       # residency just changed; do not
            return _json(load_fn(q, body))   # serve the cached answer
        r["POST /loaded.json"] = _load
    if messages_fn is not None:
        r["POST /v1/messages"] = raw(messages_fn)
    if apply_fn is not None:
        def _apply(q, _n=0, body=None):
            return _json(apply_fn(q, body))
        r["POST /settings.json"] = _apply
    return r


RESTART_WHY = ("read at import and compiled into the kernel, so it takes a "
               "restart")


def knob_reach(artifact, name: str, live_knobs, restart_why=RESTART_WHY):
    """(reach, why) for one knob on THIS artifact.

    Three outcomes, and keeping them apart is the point: it applies now, it
    needs a restart, or -- the one nobody checks -- the bundled runtime does
    not read it at all, so it will never do anything however it is set.
    """
    reads = artifact.reads_knob(name)
    if reads is False:
        return "no-effect", ("this artifact's bundled runtime never reads "
                             "this variable, so setting it does nothing")
    if name in live_knobs:
        return "live", "can be changed on the running server"
    return "restart", restart_why


def settings_document(artifact, live_env: dict, live_tune: str,
                      live_working_set: int, resolve_fn, wired_advice=None,
                      tunes=("safe", "balanced", "fast"),
                      live_knobs=(), restart_why=RESTART_WHY) -> callable:
    """Build the `/settings.json` handler.

    The document says, for every knob: the value RUNNING, the value this
    tune/working-set WOULD resolve to, whether those differ, and the sentence
    that explains why the knob exists at all. A panel built on this cannot
    show a number without its provenance, which is the whole complaint about
    settings UIs that show neither.
    """
    from . import settings as S

    def handler(q: dict) -> dict:
        tune = (q.get("tune") or [live_tune])[0]
        if tune not in S.TUNE_PROFILES:
            tune = live_tune
        try:
            ws = int(float((q.get("working_set_gib") or [0])[0]) * (1 << 30))
        except (TypeError, ValueError):
            ws = 0
        ws = ws or live_working_set

        r = resolve_fn(ws, tune)
        # A knob the ARTIFACT declares outranks anything scanned or hard
        # coded here: its config.json is the record of what shipped.
        declared = artifact.declared_knobs()
        headroom = max(ws - artifact.bytes_on_disk, 0)

        def _range(name):
            d = declared.get(name) or {}
            if d.get("values"):
                return list(d["values"]), d.get("unit", "")
            return S.KNOB_RANGE.get(name, (None, ""))

        def _cap(name, values):
            """Where the control stops, and why. The knob turns as far as the
            measurement allows and no further -- a control that lets you pick
            a setting the resolver would refuse is a control that lies."""
            if not values or "CACHE_LIMIT" not in name:
                return None, ""
            room = headroom / 2 / (1 << 30)
            usable = [v for v in values if v <= room] or [values[0]]
            if usable[-1] >= values[-1]:
                return None, ""
            return usable[-1], (f"{headroom / (1 << 30):.1f} GiB of headroom "
                                f"is all there is to hold it in")

        knobs = []
        for k in sorted(set(live_env) | set(r.env)):
            what, why = S.KNOB_DOC.get(k, ("", ""))
            reach, reach_why = knob_reach(artifact, k, live_knobs,
                                          restart_why)
            knobs.append({
                "tier": S.knob_tier(k),
                "name": k, "running": live_env.get(k),
                "would_be": r.env.get(k),
                "changed": live_env.get(k) != r.env.get(k),
                "what": what, "why": why,
                "reach": reach, "reach_why": reach_why,
            })
            vals, unit = _range(k)
            if vals:
                cap, cap_why = _cap(k, vals)
                knobs[-1].update(values=vals, unit=unit, cap=cap,
                                 cap_why=cap_why,
                                 doc=(declared.get(k) or {}).get("doc", ""))
        return {
            "artifact": artifact.path.name,
            "live": {"tune": live_tune,
                     "working_set_bytes": live_working_set},
            "asked": {"tune": tune, "working_set_bytes": ws},
            "knobs": knobs,
            "notes": r.notes,
            "warnings": r.warnings,
            "tunes": [{"name": t,
                       "why": S.TUNE_PROFILES[t].get("why", "")}
                      for t in tunes],
            # Per knob, because it is per knob: some apply now, some need a
            # restart, and some do nothing on this artifact at all. Saying
            # "restart" over all of them was true of most and wrong about the
            # two that matter most for not running out of memory.
            # The knobs this artifact's runtime reads that knurlogic has no
            # measured answer for. Not defaulted and not hidden: the runtime's
            # own defaults apply, and pretending the resolved list is the
            # whole environment is a quieter kind of overclaiming.
            "unmanaged": [
                {"name": n, "tier": S.knob_tier(n)}
                for n in artifact.knobs_read() if n not in r.env],
            "live_knobs": sorted(
                k["name"] for k in knobs if k["reach"] == "live"),
            "dead_knobs": sorted(
                k["name"] for k in knobs if k["reach"] == "no-effect"),
            "exports": r.as_exports(),
            "connect": _connect_doc(artifact),
            "wired": wired_advice or {},
        }
    return handler
