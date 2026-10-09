"""Knurlogic-wide knob groups read per request rather than at launch:
server-side compaction (context_management/compaction.py) and the prompt
cache on disk (engine/prompt_cache/disk). Each table is (default, values,
unit, what, why); each group has its own check.
"""

from __future__ import annotations

from knurlogic.tuning import knobs

# --- server-side context compaction (context_management/compaction.py) ----
# The operator's defaults for what a request's `context_management` leaves
# out, and whether the server compacts a request that asks for nothing.
# Read per request from the environment, so each applies live on a running
# server (POST /settings.json) and, set before a launch, from its start.
# Not measured: these are policy, ported from an earlier agent-loop
# compactor (keep_recent=6). There is no summary
# budget: a summary is only never longer than what it replaces.
COMPACT_KNOBS = {
    # name: (default, values, unit, what, why)
    "KNURLOGIC_COMPACT_AUTO": (
        "off", ["off", "on"], "",
        "compact a request that asks for nothing, once its prompt passes "
        "the trigger",
        "on keeps a client that never asks inside the window, at a cost: "
        "the request waits for a summary pass (one extra model call), and "
        "older turns survive only as that summary -- detail is lost. Off: "
        "only a harness that asks (context_management) is compacted; one "
        "that does not is refused past the window. The summary goes back "
        "in the response for the client to resend; nothing is kept here."),
    "KNURLOGIC_COMPACT_TRIGGER": (
        "0.8", ["0.5", "0.6", "0.7", "0.8", "0.9"], "of the window",
        "where compaction starts, as a share of the model's context window",
        "lower compacts earlier: shorter prompts, faster prefill and less "
        "KV memory, but more summary passes and detail lost sooner. Higher "
        "keeps more word for word, and runs closer to the window. Used by "
        "automatic compaction, and by a compact edit that names no trigger "
        "when the window is below the API's 150k default."),
    "KNURLOGIC_COMPACT_KEEP_TURNS": (
        "6", ["2", "4", "6", "8", "12", "16"], "messages",
        "the most recent messages kept word for word",
        "more keeps more recent work exact, at the cost of a longer "
        "compacted prompt (more prefill and KV memory). Fewer shrinks it "
        "further and leans harder on the summary, so more detail is lost. "
        "The kept tail never starts on a tool result (it is "
        "widened back to the call that asked for it); the first message "
        "and the goal turn are always kept."),
    "KNURLOGIC_COMPACT_TOOL_RESULTS": (
        "distill", ["distill", "clear"], "",
        "what becomes of a dropped tool result: a one-line finding, or "
        "nothing",
        "distill keeps what each dropped tool call established (a Grep for "
        "X -> 'X is defined at src/foo.py:120'), at the cost of more output "
        "from the same summary pass -- a slower compaction. clear is the "
        "cheapest, and the model loses what those calls found: it may run "
        "them again."),
}


def compact_settings(env: dict) -> dict:
    """The operator's compaction defaults from an environment, each
    falling back to its default when absent or unreadable:
    {auto, trigger, keep, distill}."""
    def get(name):
        v = str((env or {}).get(name, "") or "").strip()
        return v or COMPACT_KNOBS[name][0]

    def num(name, cast, lo, hi):
        try:
            v = cast(get(name))
        except ValueError:
            v = cast(COMPACT_KNOBS[name][0])
        return min(max(v, lo), hi)
    try:
        auto = knobs.on_off(get("KNURLOGIC_COMPACT_AUTO"), False)
    except ValueError:
        auto = False
    return {"auto": auto,
            "trigger": num("KNURLOGIC_COMPACT_TRIGGER", float, 0.05, 0.99),
            "keep": num("KNURLOGIC_COMPACT_KEEP_TURNS", int, 0, 1000),
            "distill": get("KNURLOGIC_COMPACT_TOOL_RESULTS") != "clear"}


def check_compact_knob(name: str, value):
    """None when `value` is one compaction knob `name` may take, else why
    not."""
    s = str(value if value is not None else "").strip()
    if name == "KNURLOGIC_COMPACT_AUTO":
        try:
            knobs.on_off(s)
            return None
        except ValueError as e:
            return f"{name}: {e}"
    if name == "KNURLOGIC_COMPACT_TOOL_RESULTS":
        return None if s in ("distill", "clear") else \
            f"{name}={s!r}: distill or clear"
    cast, lo, hi = {"KNURLOGIC_COMPACT_TRIGGER": (float, 0.05, 0.99),
                    "KNURLOGIC_COMPACT_KEEP_TURNS": (int, 0, 1000)}[name]
    try:
        v = cast(s)
    except ValueError:
        return f"{name}={s!r}: not a number"
    if not lo <= v <= hi:
        return f"{name}={s}: between {lo:g} and {hi:g}"
    return None


# --- the prompt cache on disk (engine/prompt_cache/disk) --------------------
# Knurlogic-wide, like compaction: one cache directory holds every model's
# saved prompt caches, so one budget and one TTL cover them all. Read at
# each save and load (tuning/preferences over the environment).
PROMPT_CACHE_KNOBS = {
    # name: (default, values, unit, what, why)
    "KNURLOGIC_PROMPT_CACHE_DISK": (
        "on", ["on", "off"], "",
        "let a client save prompt caches to disk (POST "
        "/v1/prompt-cache/save or /park) and restore them when that model "
        "loads again; nothing is saved unless a client asks",
        "on lets a coordinator turn a reload's re-prefill into a disk read "
        "(a 110k-token session took ~13.5 min to prefill at 136 tok/s; an "
        "SSD reads its cache in seconds), at the cost of disk space up to "
        "the budget. Off refuses those saves and reads nothing."),
    "KNURLOGIC_PROMPT_CACHE_DISK_GB": (
        "", ["", "8", "16", "32", "64", "128"], "GiB",
        "the most disk the saved prompt caches may take, every model "
        "together; the least recently used go first",
        "more keeps more sessions (and longer ones) restorable; less keeps "
        "the disk free. Unset: 20% of the disk's free space plus what the "
        "cache holds, at most 64 GiB."),
    "KNURLOGIC_PROMPT_CACHE_TTL_H": (
        "24", ["1", "6", "24", "72", "168"], "hours",
        "how long a saved prompt cache nobody used is kept",
        "longer keeps a session restorable after a longer break, holding "
        "its disk meanwhile; shorter frees it sooner."),
}


def check_prompt_cache_knob(name: str, value):
    """None when `value` is one prompt-cache knob `name` may take, else
    why not."""
    s = str(value if value is not None else "").strip()
    if name == "KNURLOGIC_PROMPT_CACHE_DISK":
        try:
            knobs.on_off(s)
            return None
        except ValueError as e:
            return f"{name}: {e}"
    if name == "KNURLOGIC_PROMPT_CACHE_DISK_GB" and not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return f"{name}={s!r}: not a number"
    if v < 0 or (name == "KNURLOGIC_PROMPT_CACHE_TTL_H" and v <= 0):
        return f"{name}={s}: must be more than 0"
    return None
