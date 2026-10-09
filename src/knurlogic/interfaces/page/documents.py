"""The routes knurlogic serves next to the engine's OpenAI surface.

One place, because `serve` and the page `knurlogic ui` opens both use them.
The runtime reads its environment at import, so the settings route never
pretends a control retunes a loaded model; it answers three separate
questions: what is running now, what a setting would give, and how to get
it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from knurlogic.machine.disk_cache import DiskCache
from knurlogic.tuning.live import LIVE_KNOBS, RESTART_WHY, knob_reach

GIB = 1 << 30
ASSETS = Path(__file__).parent / "assets"
PAGE = ASSETS / "index.html"
# The page is plain static files the browser loads as they are (native ES
# modules, no build step); only these kinds are served.
ASSET_TYPES = {".html": "text/html; charset=utf-8",
               ".css": "text/css; charset=utf-8",
               ".js": "application/javascript; charset=utf-8",
               ".mjs": "application/javascript; charset=utf-8"}


def asset(name: str):
    """(bytes, content type) of the page file at `name` (a path relative to
    assets/, "/"-separated), or None: only a regular file of a served type
    whose real path is inside the assets directory -- no "..", no symlink
    that leads out of it."""
    parts = name.split("/")
    if "\\" in name or any(p in ("", ".", "..") for p in parts):
        return None
    ctype = ASSET_TYPES.get(Path(parts[-1]).suffix)
    if ctype is None:
        return None
    root = ASSETS.resolve()
    try:
        real = ASSETS.joinpath(*parts).resolve(strict=True)
    except OSError:
        return None
    if not real.is_relative_to(root) or not real.is_file():
        return None
    return real.read_bytes(), ctype


def asset_names() -> list:
    """Every page file asset() would serve, as its "/"-separated name."""
    if not ASSETS.is_dir():
        return []
    names = (p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob("*"))
    return sorted(n for n in names if asset(n) is not None)


def _json(obj) -> tuple:
    return json.dumps(obj, indent=1).encode(), "application/json"


def _text(s: str) -> tuple:
    return s.encode(), "text/plain; charset=utf-8"


def _connect_doc(artifact) -> dict:
    """How to point a client here -- the panel exo gets right."""
    from knurlogic.interfaces import connect
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


_MODELS: dict = {"rows": None}


def forget_models() -> None:
    """Drop the cached model scan: the next /models.json reads the stores
    again (a download finished, or a model was deleted)."""
    _MODELS.update(rows=None)


def models_document(serving: str = ""):
    """`/models.json` -- what else is on this machine, and what is loaded.

    The model folders are read only when someone acts: the first request,
    `?rescan=1` (the picker was opened), or after forget_models (a download
    finished). Never on a timer: a folder on a network share that stalls
    held the page, and a page that stalls gets a healthy job stopped.

    Switching is NOT offered. A loaded model is loaded; what this can
    honestly hand over is the command that would serve another one.
    """
    def handler(q: dict) -> dict:
        from knurlogic.engine.vision import registry as vision_registry
        from knurlogic.machine import discover
        rescan = (q or {}).get("rescan", [""])[0] in ("1", "true", "yes")
        if _MODELS["rows"] is None or rescan:
            try:
                _MODELS["rows"] = discover.find()
            except (OSError, ValueError, KeyError, AttributeError):
                _MODELS["rows"] = []
        out = []
        from knurlogic.machine.artifact import identity as artifact_identity
        from knurlogic.machine.memory import allowance, wired
        ws = allowance.cap(wired.detected_working_set_bytes())
        from knurlogic.interfaces.page import updates
        stale = updates.flagged([f.path for f in _MODELS["rows"]])
        for f in _MODELS["rows"]:
            out.append({
                "name": f.name, "path": str(f.path), "store": f.store,
                # what a peer is asked to load by (machine/artifact.py)
                "identity": artifact_identity(f.path),
                "size_bytes": f.bytes_on_disk, "model_type": f.model_type,
                "is_vq": f.is_vq, "servable": f.servable, "why": f.why,
                "mtp": bool(f.extra.get("mtp_head")),
                # a family for model_type and a tower in this config.json
                "vision": vision_registry.registered(f.model_type, f.path),
                # a vision family's conversion without its vision weights:
                # why the picker grays its vision switch ("" otherwise)
                "vision_why": vision_registry.unavailable_why(
                    f.model_type, f.path),
                "serving": bool(serving) and (f.name == serving
                                              or str(f.path) == serving),
                "room": _room(f, ws),
                **_splits(f),
                # the Hub has a newer revision of this Hugging Face model
                # (page/updates.py); False when unchecked or not from HF
                "update": str(f.path) in stale,
            })
        return {"models": out, "serving": serving}
    return handler


def _part_bytes(a) -> dict:
    """{"mtp_bytes", "vision_bytes"} of a loaded Artifact: what MTP off and
    vision off would free (tuning/fit.mtp_head_bytes,
    vision_freed_bytes); 0 when it has no such part or cannot be read.
    Only the ONE picked model's preview counts these -- the listing would
    read every artifact on every build (slow on a network store)."""
    from knurlogic.tuning.fit import mtp_head_bytes, vision_freed_bytes
    out = {"mtp_bytes": 0, "vision_bytes": 0}
    try:
        out["mtp_bytes"] = mtp_head_bytes(a)
        out["vision_bytes"] = vision_freed_bytes(a)
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        pass
    return out


def _room(f, ws: int):
    """A found model's room to talk (tuning/fit.context_room), for the
    picker -- None when it would not fit at all or cannot be read."""
    from knurlogic.tuning.fit import room_for
    if not (f.servable and ws and f.bytes_on_disk < ws):
        return None
    try:
        cfg = json.loads((Path(f.path) / "config.json").read_text())
        return room_for(f.bytes_on_disk, cfg, ws)
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return None


def splits_of(path, n: int = 2) -> list:
    """The cluster splits the artifact at `path` can take across `n`
    machines, in the order the picker offers them: the same refusals a
    launch runs (tuning/resolve, its config and its headers), so the picker
    never offers one a launch would refuse."""
    from knurlogic.tuning.pipeline_split import pipeline_refusals
    from knurlogic.tuning.tensor_split import tensor_split_refusals
    cfg = json.loads((Path(path) / "config.json").read_text())
    return [s for s, why in (
        ("tensor", lambda: tensor_split_refusals(path, n, cfg)),
        ("pipeline", lambda: pipeline_refusals(cfg, n))) if not why()]


#: {path: [identity, splits]} in <cache_dir>/splits.json: the answer reads
#: every shard's header, ~8 s for the whole library after a page restart
_SPLITS = DiskCache("splits.json", valid=lambda v: isinstance(v, dict))


def _splits(f) -> dict:
    """{"splits": `splits_of` a found model}, kept on disk under its identity (which
    changes with any shard, config or *.py); splits None when it cannot be
    read (the picker then offers both and the launch answers)."""
    from knurlogic.machine.artifact import identity
    if not f.servable:
        return {"splits": None}
    key, ident = str(f.path), identity(f.path)
    # the answer is the model's AND this build's rules: either changing
    # asks again (a reverted rule kept offering tensor from the cache)
    stamp = [ident, _rules_stamp()] if ident else None
    hit = _SPLITS.get(key, stamp) if stamp else None
    if hit is not None:
        return hit
    try:
        out = {"splits": splits_of(f.path)}
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return {"splits": None}
    if stamp:
        _SPLITS.put(key, stamp, out)
    return out


def tensor_bytes_of(a) -> dict | None:
    """What a tensor rank holds, for the picker to fit each picked machine
    (the picked model's preview: the listing loads no artifact): the split
    weights (divided by the ranks), the replicated ones, and what rank 0
    alone adds -- the MTP head and the vision tower, apart, since each
    counts only when that launch has it on. None: no tensor split."""
    from knurlogic.tuning import pipeline_split, tensor_split
    if tensor_split.tensor_refusals(a.raw_config, 2):
        return None
    try:
        pl = tensor_split.tensor_placement(a, 1)
    except (OSError, ValueError, KeyError):
        return None
    head = pipeline_split.leader_bytes(a, vision=False)
    return {"sharded": pl["sharded_bytes"], "replicated": pl["replicated_bytes"],
            "head": head, "tower": pipeline_split.leader_bytes(a, mtp=False)}


_RULES: list = []


def _rules_stamp() -> str:
    """sha256 of the source that decides a model's splits
    (tuning/tensor_split, tuning/pipeline_split,
    engine/split/tensor_rules)."""
    if not _RULES:
        import hashlib

        import knurlogic.engine.split.tensor_rules as tr
        import knurlogic.tuning.pipeline_split as ps
        import knurlogic.tuning.tensor_split as ts
        h = hashlib.sha256()
        for m in (ts, ps, tr):
            h.update(Path(m.__file__).read_bytes())
        _RULES.append(h.hexdigest()[:16])
    return _RULES[0]


_LOADED: dict = {"at": 0.0, "doc": None}


def loaded_document(ttl: float = 4.0):
    """`/loaded.json` -- every runtime on this box and what it holds.

    A short TTL: residency changes the moment
    someone loads something, and a stale answer here is the answer being
    wrong rather than merely old. Four HTTP reads with short timeouts came
    back in 0.08s against a live exo, so the TTL is about coalescing a
    refresh burst, not about cost.
    """
    def handler(_q: dict) -> dict:
        import time
        now = time.time()
        doc = _LOADED["doc"]   # read once: a POST may clear it meanwhile
        if doc is not None and now - _LOADED["at"] > ttl \
                and not _LOADED.get("refreshing"):
            # the last survey answers while a new one runs on its own
            # thread: it asks each server (a busy rank 0 answers slowly),
            # and a peer's Survey times out at 2.5 s
            _LOADED["refreshing"] = True

            def refresh():
                try:
                    survey_now(time.time())
                finally:
                    _LOADED["refreshing"] = False
            import threading
            threading.Thread(target=refresh, daemon=True,
                             name="knurlogic-loaded-survey").start()
            return doc
        if doc is None or now - _LOADED["at"] > ttl:
            doc = survey_now(now)
        return doc

    def survey_now(now: float) -> dict:
        from knurlogic.engine.vision import served_vision
        from knurlogic.machine import loaded
        try:
            doc = loaded.survey()
        # the survey is a page document that must still answer; the error is in it
        except Exception as e:
            doc = {"resident": [], "runtimes": [],
                   "bytes_resident": 0, "error": str(e)}
        # What the SERVED model sees, read fresh every time regardless
        # of the survey's own cache path -- a load/unload changes this
        # the moment it happens, and the chat panel's attach button
        # gates on this exact field.
        try:
            spec = served_vision()
            doc["vision"] = spec.to_json() if spec else None
        except (AttributeError, TypeError, ValueError):
            doc["vision"] = None
        _LOADED["doc"] = doc
        _LOADED["at"] = now
        return doc
    return handler


def load_action(artifact_for, resolve_fn=None, live_knobs=(),
                switch_fn=None, unload_fn=None):
    """`POST /loaded.json` -- load, unload, or ask ollama to let go.

    The reason this is knurlogic's job: a model
    swapped into a running process gets the environment that process STARTED
    with. Knurlogic is the only thing here that knows what the incoming
    artifact would have resolved to, so it is the only thing that can say
    which of those settings did not survive the switch. Doing the load and
    staying quiet about that would be worse than not offering it.
    """
    from knurlogic.machine import loaded as L

    def _drift(path: str) -> dict:
        """Which resolved settings the running process cannot honour."""
        if resolve_fn is None:
            return {}
        try:
            a = artifact_for(path)
            want = resolve_fn(a).env
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            return {}
        import os
        stuck: dict = {}
        applied: dict = {}
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
        except (ValueError, TypeError):
            req = {}
        act, target = req.get("action"), req.get("target") or ""
        where = req.get("where") or ""
        try:
            # the server's own: a load runs on the scheduler's thread,
            # which owns the MLX stream, never on this one
            if act in ("load", "unload") and (switch_fn is None or
                                              unload_fn is None):
                return {"error": "this page is not attached to a server "
                                 "that can load models"}
            if act == "load":
                r = switch_fn(target)
                r["settings"] = _drift(target)
                return r
            if act == "unload":
                return unload_fn()
            if act == "ollama-unload":
                return L.ollama_unload(where, target)
        # a load action's failure is the page's answer, not a dead handler
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
    def _one(q, key):
        v = q.get(key)
        if isinstance(v, list):
            v = v[0] if v else None
        return v

    def handler(q: dict) -> dict:
        from knurlogic.machine.memory import wired
        from knurlogic.tuning import presets

        # A PREVIEW for an artifact nobody has loaded. This is the point of
        # showing settings before a launch rather than after: nearly every
        # knob is read at import and compiled into kernel source, so once a
        # model is up they are facts, not settings. The only moment they can
        # be chosen is the moment being prepared here.
        art = _one(q, "artifact")
        if not art and _one(q, "identity"):
            # a peer page's picker asks by identity (never by path): this
            # machine's own copy, from its own model stores only
            from knurlogic.machine.artifact import AmbiguousIdentity, resolve_identity
            try:
                art = resolve_identity(_one(q, "identity"),
                                       name=str(_one(q, "name") or ""))
            except AmbiguousIdentity as e:
                return {"knobs": [], "error": str(e)}
            if not art:
                return {"knobs": [], "error": "this machine does not have "
                                              "that model"}
        if art:
            try:
                return _preview(art, presets.preset_or(_one(q, "tune"), "default"),
                                _one(q, "working_set_gib"),
                                kv_bits=_one(q, "kv_bits"),
                                long_context=_one(q, "long_context"))
            # a preview that cannot be built is the page's answer; the error is in it
            except Exception as e:
                return {"knobs": [], "error": f"{type(e).__name__}: {e}"}

        want = q.get("wired_gib")
        if isinstance(want, list):
            want = want[0] if want else None
        adv = wired.advise(0)
        doc = {"knobs": [], "tunes": [], "unmanaged": [], "exports": "",
               "asked": {}, "wired": adv, "machine": wired.machine(),
               # the defaults a model launched from this page starts with
               "compaction": compaction_document()}
        if adv.get("known"):
            cur = adv["limit_bytes"] / GIB
            ceil_ = adv["ceiling_bytes"] / GIB
            try:
                target = (float(want) if want is not None and want != ""
                          else cur)
            except (TypeError, ValueError):
                target = cur
            # `ceiling_bytes` is knurlogic's RECOMMENDATION -- installed
            # memory less a reserve for macOS -- and not a wall the OS
            # enforces. So it is a soft gate, as everywhere else here: past
            # it you get the command (the suggestion is shown beside the
            # field, neutrally), never a refusal.
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
        # read here too, so a peer's tab shows it through /peek -- which
        # reads /settings.json and nothing that can change anything
        doc["allowance"] = allowance_doc()
        doc["strategy"] = strategy_doc()
        doc["knurlogic"] = knurlogic_doc()
        return doc
    return handler


def allowance_doc() -> dict:
    """`GET /allowance.json`: the knurlogic allowance (machine/memory/allowance.py)
    and what it is lowering -- 0 means none, knurlogic takes the working
    set."""
    from knurlogic.machine.memory import allowance, wired
    ws = wired.detected_working_set_bytes()
    a = allowance.get()
    return {"allowance_gib": round(a / GIB, 1), "working_set_gib":
            round(ws / GIB, 1), "effective_gib": round(allowance.cap(ws)
                                                      / GIB, 1),
            "file": str(allowance.path())}


def set_allowance(body) -> dict:
    """`POST /allowance.json` {"gib": N}: remember it (0 clears). Only this
    machine's: a peer's is set on that peer's own page. Refused above the
    installed memory -- an allowance past it would not lower anything and
    reads as a mistake."""
    from knurlogic.machine.memory import allowance, wired
    try:
        gib = float(json.loads(body or b"{}").get("gib"))
    except (ValueError, TypeError, AttributeError):
        return {"error": "send {\"gib\": N}; 0 clears the allowance"}
    total = (wired.advise(0).get("total_bytes") or 0) / GIB
    if gib < 0 or (total and gib > total):
        return {"error": f"{gib:g} GiB is not between 0 and the "
                         f"{total:.0f} GiB installed"}
    allowance.set(int(gib * GIB))
    return {**allowance_doc(), "applied": {"knurlogic allowance":
            f"{gib:g} GiB" if gib else "none"}}


def strategy_doc() -> dict:
    """`GET /strategy.json`: this machine's knurlogic strategy -- the launch
    preset a launch takes when none is named (tuning/strategy.py) -- with
    every preset's values for the rows a custom set changes
    (tuning/presets.PRESET_ROWS)."""
    from knurlogic.tuning import presets, strategy
    return {"preset": strategy.get(), "default": presets.PRESET_DEFAULT,
            "presets": [{"name": n, "title": n.capitalize(),
                         "values": presets.preset_row_values(n)}
                        for n in presets.PRESETS],
            "rows": [{**r, "options": [{"v": v, "t": t}
                                       for v, t in r["options"]]}
                     for r in presets.PRESET_ROWS]}


def set_strategy(body) -> dict:
    """`POST /strategy.json` {"preset": name}: remember it for this
    machine; the next launch without a preset of its own takes it."""
    from knurlogic.tuning import strategy
    try:
        name = json.loads(body or b"{}").get("preset")
        strategy.set(name)
    except (ValueError, TypeError, AttributeError, OSError) as e:
        return {"error": str(e) if isinstance(e, ValueError)
                else "send {\"preset\": name}"}
    return {**strategy_doc(), "applied": {"knurlogic strategy":
                                          strategy.get()}}


def _preview(path: str, tune: str, working_set_gib=None,
             kv_bits=None, long_context=None) -> dict:
    """What this artifact WOULD resolve to, and which of those can still be
    chosen. Nothing is loaded and nothing is set: this only reads.
    `kv_bits`: the KV precision it would launch with ('bf16', '8', ...),
    for the room its context is counted in."""
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.memory import wired
    from knurlogic.tuning import context_window, knobs, presets
    from knurlogic.tuning.fit import room_for
    from knurlogic.tuning.resolve import kv_refusal as resolve_kv_refusal
    from knurlogic.tuning.resolve import resolve

    a = Artifact.load(path)
    try:
        ws = int(float(working_set_gib) * GIB) if working_set_gib else 0
    except (TypeError, ValueError):
        ws = 0
    budget = None
    if not ws:
        budget = wired.load_budget()
        ws = budget["bytes"]
    tune = presets.preset_or(tune, "default")
    if not kv_bits:
        # the room is counted at the preset's KV precision (lean: 8-bit)
        kv_bits = presets.preset_launch(tune, a.model_type)[0].get("kv_bits")
    bits = knobs.kv_bits_of(kv_bits)
    if resolve_kv_refusal(a, bits):
        bits = None
    try:
        lc = context_window.long_context_of(long_context)
    except ValueError:
        lc = "off"
    if context_window.long_context_refusal(a.model_type, lc):
        lc = "off"
    r = resolve(a, ws, tune=tune, kv_bits=bits, long_context=lc)

    rows = []
    for name, value in sorted(r.env.items()):
        # KNOB_DOC and friends are keyed by the EMITTED name, which is what
        # the resolver puts in `env`. (`default_alias` goes the other way --
        # logical to emitted -- and calling it here threw a KeyError on the
        # first artifact tried.)
        what, why = knobs.KNOB_DOC.get(name, ("", ""))
        reach, reach_why = knob_reach(a, name, LIVE_KNOBS)
        vals = knobs.KNOB_RANGE.get(name)
        if name in r.ranges:
            vals = (list(r.ranges[name]), vals[1] if vals else "")
        rows.append({
            "name": name, "value": str(value), "reach": reach,
            "reach_why": reach_why, "what": what, "why": why,
            "help": knobs.KNOB_HELP.get(name, ""),
            "tier": knobs.knob_tier(name),
            "values": vals[0] if isinstance(vals, tuple) else vals,
            **knob_limit(a, name),
        })
    return {
        "artifact": {"name": a.path.name, "path": str(a.path),
                     "model_type": a.model_type, "gib": round(a.gib, 1)},
        "tune": tune, "working_set_gib": round(ws / GIB, 1),
        "budget": ({"gib": round(ws / GIB, 1),
                    "limited_by": budget["limited_by"],
                    "working_set_gib": round(budget["working_set_bytes"] / GIB, 1),
                    "available_now_gib": round(budget["available_bytes"] / GIB, 1),
                    "headroom_gib": round((ws - a.bytes_on_disk) / GIB, 1)}
                   if budget else {"gib": round(ws / GIB, 1),
                                   "limited_by": "given by the caller"}),
        "knobs": rows, "notes": r.notes, "warnings": r.warnings,
        "wired": wired.advise(a.bytes_on_disk),
        # what the fit leaves to talk in, against the budget it is resolved
        # in: the working set given, or what this machine has room for now
        # (wired.load_budget: memory available now, so a model already
        # loaded here counts; the working set alone said "leaves 87 GiB" on
        # a 128 GiB Mac holding a 108 GiB model)
        "room": room_for(a.bytes_on_disk, a.raw_config, ws, kv_bits=bits),
        "preview": True,
        # what the Load model toggles free: the MTP head, and vision's
        # tower + image store + image KV allowance (0: no such part)
        **_part_bytes(a),
        "tensor_bytes": tensor_bytes_of(a),
    }


def routes(status_fn=None, settings_fn=None, apply_fn=None,
           messages_fn=None, models_fn=None, loaded_fn=None,
           load_fn=None) -> dict:
    """path -> handler(query: dict) -> (body, content_type).

    `status_fn(requests)` returns (snapshot, text). `settings_fn(query)`
    returns the settings document below.
    """
    r = {}

    def _connect(_q, _n=0):
        # Templates, not answers: which model the page is pointed at is the
        # page's to say (it knows what is running on every machine), so the
        # placeholders are filled in there.
        from knurlogic.interfaces import connect
        return _json({"endpoints": connect.endpoints("__BASE__",
                                                     "__MODEL__")})
    r["/connect.json"] = _connect

    # The page: index.html at / and /ui, every other file under assets/ at
    # its own path (/app.js, /views/chat.js ...). Only the files there when
    # the server starts are routes, and each is read, and checked again,
    # when it is asked for -- a restart shows new code.
    def _asset(name):
        def _serve(_q, _n=0):
            got = asset(name)
            return got if got is not None else _text("not found")
        return _serve
    for name in asset_names():
        if name == "index.html":
            r["/"] = r["/ui"] = _asset(name)
        else:
            r["/" + name] = _asset(name)

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


def knob_limit(artifact, name: str) -> dict:
    """{max, max_why} where the MODEL bounds a knob: the context length
    stops at its window (tuning/context_window.model_window). {} otherwise."""
    from knurlogic.tuning import context_window
    if name != "KNURLOGIC_CONTEXT_LENGTH":
        return {}
    cfg = getattr(artifact, "raw_config", None) or {}
    w, why = context_window.model_window(cfg)
    if not w:
        return {}
    top = context_window.context_ceiling(getattr(artifact, "model_type", ""), cfg)
    if top > w:
        # past the native window is long context: offered, and turned on
        # at launch (context_window.settle_context)
        return {"max": top, "max_why":
                f"Above {w:,} it uses YaRN scaling to reach up to "
                f"{top:,}, at a slight cost of quality."}
    return {"max": w, "max_why": f"This model's maximum is {w:,} tokens."}


def compaction_document(env=None, running: bool = True) -> dict:
    """Compaction's operator defaults (tuning/groups.COMPACT_KNOBS), each
    value and what it does. Knurlogic-wide (tuning/preferences): one set
    for every model, read per request by every running server, so each is
    `live`. `env`: an environment to read instead of this process's under
    the saved values; `running` is kept for callers and changes nothing."""
    from knurlogic.tuning import groups, knobs, preferences
    env = preferences.compaction_env(env)
    rows = []
    for name, (default, values, unit, what, why) in \
            groups.COMPACT_KNOBS.items():
        rows.append({
            "name": name, "running": env.get(name) or default,
            "would_be": default, "value": env.get(name) or default,
            "default": default,
            "changed": False, "tier": "reach",
            "what": what, "why": why, "help": knobs.KNOB_HELP.get(name, ""),
            "values": list(values), "unit": unit,
            "reach": "live",
            "reach_why": ("knurlogic-wide: every model server reads it for "
                          "every request, so a change applies to the next "
                          "one")})
    return {"knobs": rows,
            "effective": groups.compact_settings(env),
            "about": ("Harnesses ask with context_management (Anthropic's "
                      "compact_20260112, clear_tool_uses_20250919, "
                      "clear_thinking_20251015; the same object on "
                      "/v1/chat/completions); the server summarizes and "
                      "returns the summary for the client to resend. "
                      "Nothing is stored. usage.knurlogic.context reports "
                      "the prompt's tokens against the window.")}


def knurlogic_doc() -> dict:
    """The knurlogic-wide settings (tuning/preferences) on this machine:
    what is saved, identical results across chips with its trade-off, and
    compaction."""
    from knurlogic.tuning import knobs, preferences
    saved = preferences.get()
    what, why = knobs.KNOB_DOC[preferences.CROSS_CHIP]
    return {"saved": saved,
            "cross_chip": {"name": preferences.CROSS_CHIP,
                           "value": saved.get(preferences.CROSS_CHIP, ""),
                           "values": knobs.KNOB_RANGE[preferences.CROSS_CHIP][0],
                           "what": what, "why": why,
                           "help": knobs.KNOB_HELP[preferences.CROSS_CHIP]},
            "compaction": compaction_document(),
            "file": str(preferences.path())}


def set_knurlogic(body) -> dict:
    """`POST /knurlogic.json` {name: value, ...}: save knurlogic-wide
    settings on this machine ('' clears one). Compaction applies to the
    next request of every running server; identical results across chips
    to the next launch."""
    from knurlogic.tuning import preferences
    try:
        want = json.loads(body or b"{}")
        preferences.set(want)
    except (ValueError, TypeError, AttributeError, OSError) as e:
        return {"error": str(e) if isinstance(e, ValueError)
                else "send {name: value}"}
    return {**knurlogic_doc(), "applied": {
        k: (str(v) if str(v or "").strip() else "cleared")
        for k, v in want.items()}}


def settings_document(artifact, live_env: dict, live_tune: str,
                      live_working_set: int, resolve_fn, wired_advice=None,
                      tunes=None,
                      live_knobs=(), restart_why=RESTART_WHY) -> Callable:
    """Build the `/settings.json` handler.

    The document says, for every knob: the value RUNNING, the value this
    tune/working-set WOULD resolve to, whether those differ, and the sentence
    that explains why the knob exists at all. A panel built on this cannot
    show a number without its provenance, which is the whole complaint about
    settings UIs that show neither.
    """
    from knurlogic.tuning import knobs, presets
    tunes = tunes or presets.PRESETS

    def handler(q: dict) -> dict:
        tune = presets.preset_or((q.get("tune") or [live_tune])[0], live_tune)
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
            if name in r.ranges:
                return list(r.ranges[name]), knobs.KNOB_RANGE.get(
                    name, (None, ""))[1]
            return knobs.KNOB_RANGE.get(name, (None, ""))

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

        rows = []
        for k in sorted(set(live_env) | set(r.env)):
            what, why = knobs.KNOB_DOC.get(k, ("", ""))
            reach, reach_why = knob_reach(artifact, k, live_knobs,
                                          restart_why)
            rows.append({
                "tier": knobs.knob_tier(k),
                "name": k, "running": live_env.get(k),
                "would_be": r.env.get(k),
                "changed": live_env.get(k) != r.env.get(k),
                "what": what, "why": why, "help": knobs.KNOB_HELP.get(k, ""),
                "reach": reach, "reach_why": reach_why,
                **knob_limit(artifact, k),
            })
            vals, unit = _range(k)
            if vals:
                cap, cap_why = _cap(k, vals)
                rows[-1].update(values=vals, unit=unit, cap=cap,
                                cap_why=cap_why,
                                doc=(declared.get(k) or {}).get("doc", ""))
        return {
            "artifact": artifact.path.name,
            "live": {"tune": live_tune,
                     "working_set_bytes": live_working_set},
            "asked": {"tune": tune, "working_set_bytes": ws},
            "knobs": rows,
            "notes": r.notes,
            "warnings": r.warnings,
            "tunes": [{"name": t,
                       "why": presets.TUNE_PROFILES[t].get("why", "")}
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
                {"name": n, "tier": knobs.knob_tier(n)}
                for n in artifact.knobs_read() if n not in r.env],
            "live_knobs": sorted(
                k["name"] for k in rows if k["reach"] == "live"),
            "dead_knobs": sorted(
                k["name"] for k in rows if k["reach"] == "no-effect"),
            "exports": r.as_exports(),
            "compaction": compaction_document(),
            "connect": _connect_doc(artifact),
            "wired": wired_advice or {},
        }
    return handler
