"""The step plan: what rank 0 tells every other rank before each step of a
tensor-split model (docs/SERVER.md, "Cluster: tensor split").

Rank 0 owns the scheduler, HTTP and ALL sampling. Before each step every
rank contributes one fixed-size control vector (an all_gather), and when
rank 0 has something to say the plan's bytes follow (an all_sum in which
every other rank contributes zeros). Ranks >= 1 apply the plan's ops in
order, overwrite the batch's next tokens with the ones rank 0 sampled, and
run the same step. They never sample for a live row and never decide a
prompt-cache hit or an eviction themselves.

The plan is JSON (never pickle: bytes from another process are data), and
this module is pure Python so the codec is tested without MLX.

Control vector, one int64 per slot, one row per rank:

    [0] over    this rank's active memory minus its limit (signed bytes)
    [1] step    the step counter (every rank counts; a mismatch is desync)
    [2] length  plan bytes that follow -- rank 0's slot only; 0 = none

Ops, applied in order (every field explicit):

    admit   uid, prompt (every token), segs (what is prefilled, after the
            prompt-cache hit and the lean decision), hit (tokens the prompt
            cache supplied: rank 0's fetch, repeated and checked), max_tokens,
            sampling (make_distribution kwargs + an assigned seed),
            penalties, initial (the control machine's start state)
    remove  uids
    insert  uid, event ("checkpoint" | "finished"), kind: store this
            rank's cache from that event of the last step in the prompt cache
    pop     n: evict the n least recently used prompt-cache entries
    reset   close the executor (rank 0 closed its own)
    stop    leave the loop

`tokens` (optional): [uid, token] for every row rank 0's batch holds at
the start of this step, in batch order.
"""

from __future__ import annotations

import json
from typing import List

CONTROL_LEN = 3
OVER, STEP, LENGTH = range(CONTROL_LEN)

OPS = ("admit", "remove", "insert", "pop", "reset", "stop")
_FIELDS = {
    "admit": ("uid", "prompt", "segs", "hit", "max_tokens", "sampling",
              "penalties", "initial"),
    "remove": ("uids",),
    "insert": ("uid", "event", "kind"),
    "pop": ("n",),
    "reset": (),
    "stop": (),
}
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
    toks = plan.get("tokens")
    if toks is not None:
        if not isinstance(toks, list) or not all(
                isinstance(t, list) and len(t) == 2 for t in toks):
            raise PlanError("tokens is a list of [uid, token]")
        _ints([x for t in toks for x in t], "tokens")


def _ints(xs, what: str) -> None:
    if not isinstance(xs, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in xs):
        raise PlanError(f"{what} must be a list of ints")


def control(over: int, step: int, length: int) -> List[int]:
    return [int(over), int(step), int(length)]
