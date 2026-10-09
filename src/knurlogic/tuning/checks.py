"""Every refusal of a setting, used by every surface (the page, `serve`,
the MCP, a cluster job), so a value is refused the same way everywhere:
one knob's value (check_knob), a request's knobs (clean_sets), a set of
values on one artifact (refuse_sets), and a whole launch before a process
starts (settings_refusal, launch_refusal, launch_fit).
"""

from __future__ import annotations

from knurlogic.tuning import context_window, groups, knobs, numerics, presets


def check_knob(name: str, value, window: int = 0):
    """None when `value` is one `name` may take, else the sentence saying
    why not. `window`: the model's (model_window), which caps the context
    length. Enumerated knobs are checked against their values; numeric ones
    for type and range; anything else is the runtime's own business."""
    s = str(value if value is not None else "").strip()
    if name in knobs.KNOB_BOUNDS:
        cast, lo, hi, unit = knobs.KNOB_BOUNDS[name]
        try:
            v = cast(s)
        except ValueError:
            return (f"{knobs.knob_title(name)}: {s!r} isn't a "
                    f"{'whole number' if cast is int else 'number'}")
        if name == "KNURLOGIC_CONTEXT_LENGTH" and window:
            hi = window
        if v < lo or (hi is not None and v > hi):
            where = (f"this model's maximum is {hi:,} tokens" if
                     name == "KNURLOGIC_CONTEXT_LENGTH" and window else
                     f"between {lo:g} and {hi:g}{' ' + unit if unit else ''}"
                     if hi is not None else f"at least {lo:g}")
            return f"{knobs.knob_title(name)}: {s} isn't allowed; {where}"
        return None
    if name in groups.COMPACT_KNOBS:
        return groups.check_compact_knob(name, value)
    if name in groups.PROMPT_CACHE_KNOBS:
        return groups.check_prompt_cache_knob(name, value)
    try:
        if name in ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC",
                    "KNURLOGIC_VISION"):
            if s.lower() not in ("", "on", "off", "1", "0", "true", "false",
                                 "yes", "no"):
                raise ValueError(f"{s!r} isn't on or off")
        elif name == "KNURLOGIC_KV_BITS":
            knobs.kv_bits_of(s)
        elif name == "KNURLOGIC_PRESET":
            presets.preset_of(s)
        elif name == presets.MTP_MODE and s not in presets.MTP_MODES:
            raise ValueError(f"{s!r} isn't {', '.join(presets.MTP_MODES)}")
        elif name == "KNURLOGIC_CROSS_CHIP":
            knobs.cross_chip_of(s)
        elif name == "KNURLOGIC_LONG_CONTEXT":
            context_window.long_context_of(s)
        elif name == "KNURLOGIC_THINKING_DEFAULT":
            knobs.thinking_default_of(s)
    except ValueError as e:
        m = str(e)
        return m if m.startswith(knobs.knob_title(name) + ":") \
            else f"{knobs.knob_title(name)}: {m}"
    return None


# --- what a launch request may carry ----------------------------------------
# The page, a forwarded load and a cluster job all check a request against
# these; they are settings facts, so they live here.

#: request keys that would name a place on disk; refused outright, never
#: ignored, so a coordinator that sends one learns it is wrong
PATH_KEYS = ("path", "target", "artifact", "where", "dir", "directory")


def launch_knobs() -> frozenset:
    """The knob names a forwarded load may set: the ones knurlogic documents
    (KNOB_DOC) and their aliases. Nothing else is passed on:
    `--set` puts it in the child's environment."""
    return frozenset(knobs.KNOB_DOC) | frozenset(
        n for v in knobs.KNOB_ALIASES.values() for n in v) | frozenset(
        numerics.NUMERICS_FLAGS)


def clean_sets(sets) -> tuple:
    """(allowed {name: value}, [refused names]). Values are short plain
    tokens: digits, letters, '.', '-', '_'."""
    import re
    ok, bad = {}, []
    allowed = launch_knobs()
    for k, v in (sets.items() if isinstance(sets, dict) else ()):
        v = str(v)
        if k in allowed and len(v) <= 64 and re.fullmatch(r"[\w.\-]*", v):
            ok[k] = v
        else:
            bad.append(str(k)[:64])
    return ok, bad


def refuse_sets(artifact, sets: dict):
    """None when every {name: value} in `sets` is one it may take on this
    artifact (check_knob, the context length against the
    model's window), else the first refusal. Used by a live apply and by a
    launch, so neither takes a value the other would refuse."""
    lc = (sets or {}).get("KNURLOGIC_LONG_CONTEXT")
    try:
        why = context_window.long_context_refusal(
            getattr(artifact, "model_type", ""), lc)
    except ValueError as e:
        why = f"KNURLOGIC_LONG_CONTEXT: {e}"
    if why:
        return why
    w, _ = context_window.model_window(context_window.with_long_context(
        getattr(artifact, "raw_config", None) or {}, lc))
    for k, v in (sets or {}).items():
        why = check_knob(k, v, w)
        if why:
            return why
    return None


def settings_refusal(a, overrides) -> str | None:
    """The first value a launch would use that its own setting refuses,
    named in the page's words and where to fix it: this model's settings
    (Settings -> Models) or the saved knurlogic-wide ones (Settings ->
    Knurlogic). Nothing is dropped silently. Launch knobs in the
    environment are not read at all (ignored_env says so), so they are not
    checked here."""
    from knurlogic.tuning import preferences
    # past the native window is long context where the family has YaRN
    # (context_window.settle_context turns it on), so the ceiling is the YaRN one
    w = context_window.context_ceiling(getattr(a, "model_type", ""),
                                       getattr(a, "raw_config", None) or {})
    why = next((m for m in (check_knob(k, v, w) for k, v in
                            knobs.canonical_sets(
                                dict(overrides or {})).items())
                if m), None)
    if why:
        return f"{why} (Settings \u2192 Models)"
    why = next((m for _, m in preferences.invalid()), None)
    return f"{why} (Settings \u2192 Knurlogic)" if why else None


def launch_refusal(a, overrides, tune: str = "default") -> str | None:
    """None when `overrides` (a launch's --set values) and the preset `tune`
    can start `a`, else why not -- the deterministic refusals `run` makes
    before it loads a thing, in the same words, so a page or the MCP can
    refuse the launch BEFORE a process (or a ring of them) is started."""
    from knurlogic.tuning import preferences
    from knurlogic.tuning.resolve import kv_refusal, preset_env
    # a context past the native window is settled (turned on / lowered)
    # before the values are checked, as run does
    sets, _ = context_window.settle_context(
        a.model_type, a.raw_config, knobs.canonical_sets(dict(overrides or {})))
    why = settings_refusal(a, sets)
    if why:
        return why
    sets = preferences.launch_sets(sets)
    why = refuse_sets(a, sets)
    if why:
        return why
    try:
        tune = presets.preset_of(sets.get("KNURLOGIC_PRESET"), tune)
        launch = knobs.engine_settings({**preset_env(a, tune),
                                        **{k: v for k, v in sets.items()
                                           if k in knobs.MODEL_KNOBS}})
    except ValueError as e:
        return str(e)
    return (kv_refusal(a, launch.get("kv_bits"))
            or context_window.long_context_refusal(
                a.model_type, launch.get("long_context", "off"))
            or None)


def launch_fit(a, overrides, tune: str = "default", draft: bool = True,
               budget_bytes: int | None = None) -> dict:
    """`tuning.fit.single_fit_check` for a launch's settings against
    `budget_bytes` (default: the load budget): the same check `run` makes,
    so the MCP and the page can refuse before a process starts."""
    from knurlogic.machine.memory import wired
    from knurlogic.tuning import preferences
    from knurlogic.tuning.fit import single_fit_check
    from knurlogic.tuning.resolve import preset_env
    if budget_bytes is None:
        budget_bytes = wired.load_budget()["bytes"]
    try:
        sets, _ = context_window.settle_context(
            a.model_type, a.raw_config,
            knobs.canonical_sets(dict(overrides or {})))
        sets = preferences.launch_sets(sets)
        tune = presets.preset_of(sets.get("KNURLOGIC_PRESET"), tune)
        launch = knobs.engine_settings({**preset_env(a, tune),
                                        **{k: v for k, v in sets.items()
                                           if k in knobs.MODEL_KNOBS}})
    except ValueError:
        launch = {}
    if launch.get("mtp") is False:
        draft = False
    return single_fit_check(a, budget_bytes, draft, launch.get("kv_bits"),
                            launch.get("vision", True))
