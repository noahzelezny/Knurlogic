"""The step plan: what rank 0 tells every other rank before each step of a
tensor-split model (docs/design/server.md, "Cluster: tensor split").

Rank 0 owns the scheduler, HTTP and ALL sampling. Before each step every
rank contributes one fixed-size control vector (an all_gather: over,
step, length), and when rank 0 has something to say the plan's bytes
follow (an all_sum in which every other rank contributes zeros). Ranks
>= 1 apply the plan's ops in order (admit, remove, insert, pop, reset,
set, stop), overwrite the batch's next tokens with the ones rank 0
sampled, and run the same step. They never sample for a live row and
never decide a prompt-cache hit or an eviction themselves.

The plan is JSON (never pickle: bytes from another process are data), and
this module is pure Python so the codec is tested without MLX. Field-level
format: docs/design/server.md (step plan).
"""

from __future__ import annotations

import json
from typing import List

#: one row per rank per step. ACTIVE / PEAK: that rank's own mlx active
#: and peak memory, so rank 0 can report every rank's (a follower serves
#: no /status.json of its own)
CONTROL_LEN = 5
OVER, STEP, LENGTH, ACTIVE, PEAK = range(CONTROL_LEN)

OPS = ("admit", "remove", "insert", "pop", "reset", "stop", "park", "set")
_FIELDS = {
    "admit": ("uid", "prompt", "segs", "hit", "max_tokens", "sampling",
              "penalties", "initial", "images", "refs"),
    "remove": ("uids",),
    "insert": ("uid", "event", "kind"),
    "pop": ("n",),
    "reset": (),
    "stop": (),
    "park": (),
    "set": ("name", "value"),
}
#: the live knobs (engine/serve/load.LIVE_KNOBS) that act on a rank's own
#: engine, so a change on rank 0 must reach every rank
SETS = ("VQ_DECODE_CHUNK", "VQ_CACHE_LIMIT_GB", "VQLAB_CACHE_LIMIT_GB",
        "KNURLOGIC_CACHE_LIMIT_GB")
EVENTS = ("checkpoint", "finished")


class PlanError(ValueError):
    """A plan that does not parse or does not say what a plan says: the
    ranks can no longer be trusted to be in step."""


def encode(plan: dict) -> bytes:
    check(plan)
    return json.dumps(plan, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def decode(data: bytes) -> dict:
    try:
        plan = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise PlanError(f"plan bytes do not parse: {e}") from None
    check(plan)
    return plan


def empty(plan: dict) -> bool:
    return not plan.get("ops") and not plan.get("tokens")


def check(plan) -> None:
    if not isinstance(plan, dict):
        raise PlanError("a plan is an object")
    extra = set(plan) - {"ops", "tokens"}
    if extra:
        raise PlanError(f"unknown plan fields {sorted(extra)}")
    ops = plan.get("ops", [])
    if not isinstance(ops, list):
        raise PlanError("ops is a list")
    for op in ops:
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise PlanError(f"not an op: {op!r}")
        want = set(_FIELDS[op["op"]]) | {"op"}
        if set(op) != want:
            raise PlanError(f"{op['op']} takes {sorted(want)}, got "
                            f"{sorted(op)}")
        if op["op"] == "insert" and op["event"] not in EVENTS:
            raise PlanError(f"insert event {op['event']!r}")
        if op["op"] == "pop" and (not isinstance(op["n"], int) or op["n"] < 1):
            raise PlanError(f"pop n must be a positive int, got {op['n']!r}")
        if op["op"] == "set" and (op["name"] not in SETS
                                  or not isinstance(op["value"], str)):
            raise PlanError(f"set takes a knob of {list(SETS)} and a string "
                            f"value, got {op['name']!r}={op['value']!r}")
        if op["op"] == "admit":
            _ints(op["prompt"], "admit prompt")
            if not isinstance(op["segs"], list):
                raise PlanError("admit segs is a list of lists of ints")
            if not isinstance(op["hit"], int) or isinstance(op["hit"], bool):
                raise PlanError(f"admit hit must be an int, got {op['hit']!r}")
            for s in op["segs"]:
                _ints(s, "admit segs")
            if not 0 <= op["hit"] <= len(op["prompt"]):
                raise PlanError(f"admit hit {op['hit']} outside the prompt")
            if sum(len(s) for s in op["segs"]) != len(op["prompt"]) - op["hit"]:
                raise PlanError("admit segs are not the prompt after the hit")
            _images(op, len(op["prompt"]))
    toks = plan.get("tokens")
    if toks is not None:
        if not isinstance(toks, list) or not all(
                isinstance(t, list) and len(t) == 2 for t in toks):
            raise PlanError("tokens is a list of [uid, token]")
        _ints([x for t in toks for x in t], "tokens")


def _images(op: dict, n: int) -> None:
    refs, spans = op["refs"], op["images"]
    if not isinstance(refs, list) or not all(
            isinstance(r, list) and len(r) == 4 and isinstance(r[0], str)
            and isinstance(r[1], str) and isinstance(r[2], int)
            and (r[3] is None or (isinstance(r[3], list) and len(r[3]) == 3))
            for r in refs):
        raise PlanError("admit refs is a list of [sha, proc_hash, n_tokens, "
                        "grid_thw or null]")
    if not isinstance(spans, list) or not all(
            isinstance(x, list) and len(x) == 4 for x in spans):
        raise PlanError("admit images is a list of [start, end, ref, k0]")
    _ints([v for x in spans for v in x], "admit images")
    for a, b, r, k0 in spans:
        if not (0 <= a < b <= n and 0 <= r < len(refs) and k0 >= 0
                and k0 + (b - a) <= refs[r][2]):
            raise PlanError(f"admit image run {[a, b, r, k0]} does not fit "
                            f"the prompt ({n}) or its image")


def key_to_wire(key: list, ref_of) -> tuple:
    """A cache key -> (ids, images, refs) for the admit op. `ref_of(sha,
    proc_hash)` -> (n_tokens, grid_thw or None). A key with no image:
    (the key, [], [])."""
    from knurlogic.engine.vision.key import image_spans, is_sentinel
    spans = image_spans(key)
    if not spans:
        return list(key), [], []
    ids = [-1 if is_sentinel(x) else int(x) for x in key]
    refs: List[list] = []
    at: dict = {}
    images = []
    for sp in spans:
        k = (sp.sha, sp.proc_hash)
        if k not in at:
            n, grid = ref_of(*k)
            at[k] = len(refs)
            refs.append([sp.sha, sp.proc_hash, int(n),
                         list(map(int, grid)) if grid else None])
        images.append([sp.start, sp.end, at[k], sp.k0])
    return ids, images, refs


def key_from_wire(ids: list, images: list, refs: list) -> list:
    """The admit op's (prompt, images, refs) -> the key rank 0 holds."""
    from knurlogic.engine.vision.key import TAG
    key = list(ids)
    for a, b, r, k0 in images:
        sha, ph = refs[r][0], refs[r][1]
        for j in range(a, b):
            key[j] = (TAG, sha, ph, k0 + j - a)
    if any(type(x) is int and x < 0 for x in key):
        raise PlanError("an image token in the admit prompt is not in any "
                        "image run")
    return key


def _ints(xs, what: str) -> None:
    if not isinstance(xs, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in xs):
        raise PlanError(f"{what} must be a list of ints")


def control(over: int, step: int, length: int, active: int = 0,
            peak: int = 0) -> List[int]:
    return [int(over), int(step), int(length), int(active), int(peak)]
