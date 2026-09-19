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

PAGE = Path(__file__).parent / "web" / "index.html"


def _json(obj) -> tuple:
    return json.dumps(obj, indent=1).encode(), "application/json"


def _text(s: str) -> tuple:
    return s.encode(), "text/plain; charset=utf-8"


def routes(status_fn=None, settings_fn=None, apply_fn=None) -> dict:
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
            "wired": wired_advice or {},
        }
    return handler
