"""The wired limit -- the knob that decides how much of the machine a model
may use.

On Apple Silicon `iogpu.wired_limit_mb` caps how much memory the GPU may
wire, and the framework's recommended working set follows it (measured:
86016 MB -> 84.0 GiB working set of 96 GiB installed). This module never
sets it: it needs root and is system-wide, so knurlogic works out the
number, prints the command and what it costs, and a person runs it. The
reserve left for macOS is a judgement, not a measurement, and is labelled
as one wherever it is used.

Design: docs/design/memory.md (wired limit).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from knurlogic.machine.identity import _sysctl
from knurlogic.machine.units import GIB, MIB, gib

#: Keys to try, in order. The name moved between macOS versions and the older
#: one is still present on current systems; reading only one answers "unknown"
#: on a box that has it.
SYSCTL_KEYS = ("iogpu.wired_limit_mb", "debug.iogpu.wired_limit")

#: Leave the OS at least this much. A JUDGEMENT: the fraction is what this
#: box was already tuned to by hand (96 GiB installed, 84 GiB wired = 12 GiB
#: left), which is corroboration, not proof.
RESERVE_FRACTION = 0.125
RESERVE_FLOOR_BYTES = 8 * GIB


def _sysctl_int(key: str) -> int | None:
    try:
        return int(_sysctl(key))
    except ValueError:
        return None


@dataclass
class Wired:
    total_bytes: int = 0
    limit_bytes: int = 0        # 0 when the system does not expose one
    key: str = ""

    @property
    def known(self) -> bool:
        return self.total_bytes > 0 and self.limit_bytes > 0

    @property
    def reserve_bytes(self) -> int:
        return max(int(self.total_bytes * RESERVE_FRACTION),
                   RESERVE_FLOOR_BYTES)

    @property
    def ceiling_bytes(self) -> int:
        """The most this module will ever suggest wiring."""
        return max(self.total_bytes - self.reserve_bytes, 0)

    @property
    def headroom_to_ceiling(self) -> int:
        return max(self.ceiling_bytes - self.limit_bytes, 0)


def read() -> Wired:
    """macOS's wired limit. Elsewhere there is none to read or set: the
    total alone, and `known` is False."""
    if sys.platform != "darwin":
        import psutil
        return Wired(total_bytes=int(psutil.virtual_memory().total))
    total = _sysctl_int("hw.memsize") or 0
    for key in SYSCTL_KEYS:
        mb = _sysctl_int(key)
        if mb:
            return Wired(total_bytes=total, limit_bytes=mb * MIB, key=key)
    return Wired(total_bytes=total)


def command_for(limit_bytes: int, key: str = SYSCTL_KEYS[0]) -> str:
    return f"sudo sysctl {key}={int(limit_bytes // MIB)}"


def advise(need_bytes: int = 0, w: Wired | None = None) -> dict:
    """What to do about the wired limit, if anything.

    `need_bytes` is what the artifact wants resident. The answer is one of:
    nothing to do, raising it would help by this much, or it does not fit even
    at the ceiling and no sysctl fixes that.
    """
    w = w or read()
    d = {"known": w.known, "key": w.key, "total_bytes": w.total_bytes,
         "limit_bytes": w.limit_bytes, "ceiling_bytes": w.ceiling_bytes,
         "reserve_bytes": w.reserve_bytes, "action": "unknown",
         "command": "", "note": ""}
    if not w.known:
        d["note"] = ("this system does not expose a wired limit, so the "
                     "working set is whatever the framework reports")
        return d

    if need_bytes and need_bytes > w.ceiling_bytes:
        d["action"] = "will-not-fit"
        d["note"] = (
            f"{gib(need_bytes)} wanted against a "
            f"{gib(w.ceiling_bytes)} ceiling "
            f"({gib(w.total_bytes, 0)} installed, leaving "
            f"{gib(w.reserve_bytes, 0)} for macOS). Raising the wired "
            f"limit will not fix this -- it needs a smaller rung or another "
            f"box.")
        return d

    if need_bytes and need_bytes > w.limit_bytes:
        target = min(w.ceiling_bytes, max(need_bytes, w.limit_bytes))
        d.update(action="raise", target_bytes=target,
                 command=command_for(target, w.key or SYSCTL_KEYS[0]))
        d["note"] = (
            f"the artifact wants {gib(need_bytes)} and the GPU is "
            f"allowed to wire {gib(w.limit_bytes)} of the "
            f"{gib(w.total_bytes, 0)} installed. It is not that the "
            f"model does not fit -- the machine has not been told it may use "
            f"its own memory. Raising it leaves "
            f"{gib(w.total_bytes - target)} for macOS; that "
            f"reserve is a judgement, and too little wedges the machine "
            f"rather than the model.")
        return d

    d["action"] = "ok"
    d["note"] = (f"the GPU may wire {gib(w.limit_bytes)} of "
                 f"{gib(w.total_bytes, 0)} installed"
                 + (f"; {gib(w.headroom_to_ceiling)} more is "
                    f"available under the reserve if a bigger rung needs it"
                    if w.headroom_to_ceiling >= GIB else ""))
    return d


def render(d: dict) -> str:
    L = ["wired limit"]
    if not d.get("known"):
        return f"wired limit  unknown -- {d.get('note', '')}"
    L.append(f"  {gib(d['limit_bytes'])} of "
             f"{gib(d['total_bytes'], 0)} installed  [{d['key']}]")
    if d.get("note"):
        L.append(f"  {d['note']}")
    if d.get("command"):
        L.append(f"  {d['command']}")
        L.append("  (resets at reboot; knurlogic will not run it for you -- "
                 "it is a system-wide setting and needs root)")
    return "\n".join(L)


def detected_working_set_bytes(fallback: int = 0) -> int:
    """What the framework says it may use, read where it comes from.

    `resolve()` still takes headroom as an INPUT -- that stance is what keeps
    the resolver testable and machine-independent. This is the COMMANDS
    filling that input in when the user did not, because the alternative is
    what shipped: forgetting a flag silently produced the roomy defaults,
    which is the footgun this package exists to remove.

    The number is the wired limit, and this module already reads it from
    sysctl: MLX's `max_recommended_working_set_size` is exactly
    `iogpu.wired_limit_mb` in bytes (measured agreeing; see
    the module docstring), and the engine's `memory()` asks the same
    sysctl through mlx. Reading it here instead keeps callers OUT of engine/
    -- the page process must never import mlx to report a number, and
    `memory()` would drag the whole framework in for one integer. Same
    value, no engine.

    `fallback`: the framework's own working set, from a caller that has the
    engine loaded, used where neither the wired limit nor Metal answers
    (off macOS).
    """
    limit = read().limit_bytes
    if limit:
        return limit
    # the sysctl left at its default reads 0: the default then is Metal's
    # own recommendedMaxWorkingSetSize, which is the number MLX reports
    # (measured equal), asked of Metal directly so no mlx is imported.
    # Elsewhere the caller that has the framework passes its number.
    return _metal_working_set_bytes() or int(fallback or 0)


_METAL_WS: dict = {}


def _metal_working_set_bytes() -> int:
    """MTLDevice.recommendedMaxWorkingSetSize of the default GPU, through
    the Objective-C runtime; 0 off macOS or where it cannot be read.
    Cached: it is fixed for the machine."""
    if sys.platform != "darwin":
        return 0
    if "ws" not in _METAL_WS:
        import ctypes
        try:
            metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/"
                                "Metal")
            objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
            metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
            objc.sel_registerName.restype = ctypes.c_void_p
            send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p,
                                    ctypes.c_void_p)(("objc_msgSend", objc))
            dev = metal.MTLCreateSystemDefaultDevice()
            _METAL_WS["ws"] = int(send(dev, objc.sel_registerName(
                b"recommendedMaxWorkingSetSize"))) if dev else 0
        except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
            _METAL_WS["ws"] = 0
    return _METAL_WS["ws"]


def available_now() -> int:
    """Bytes of memory macOS would hand this machine's next allocation now
    (free plus file cache and purgeable pages, not inactive anonymous ones:
    footprint.available_memory), 0 when it cannot be read. Read fresh each
    time, never cached: it is the difference between a working set the GPU
    is ALLOWED and memory that is actually there (a 96 GiB Mac's 84 GiB
    working set leaves the OS and every other program 12 GiB; a load that
    fills the working set swaps)."""
    from knurlogic.machine.memory.footprint import available_memory
    try:
        return int(available_memory().get("available_bytes") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def gpu_working_set(installed: int, wired_limit=None) -> int:
    """What a load may use on this machine: the smaller of `installed` and
    the GPU's wired limit (iogpu.wired_limit_mb, what Metal recommends),
    never the installed RAM alone -- the M3 Ultra's 96 GiB has 84 wired, and
    a 90 GiB share placed on RAM alone passes the check and fails at load.
    `wired_limit` None reads it."""
    if wired_limit is None:
        wired_limit = read().limit_bytes
    known = [b for b in (int(installed or 0), int(wired_limit or 0)) if b > 0]
    return min(known) if known else 0


def budget_of(m: dict) -> int:
    """What a load on a machine described by `m` (a cluster block:
    `working_set_bytes` under its allowance, `available_bytes` when it says
    it) may hold: the working set, lowered to the memory available now when
    both are known -- `load_budget`'s rule for a machine read over the
    wire (a peer that predates `available_bytes` says nothing)."""
    ws = int(m.get("working_set_bytes") or 0)
    av = int(m.get("available_bytes") or 0)
    return min(ws, av) if ws and av else ws


def load_budget() -> dict:
    """What a model loaded NOW could have: the smaller of the GPU working set
    and the memory macOS would hand over right now.

    One function, because three tools used to answer this three ways. `fit`
    measured against memory available now; `settings` and `serve` resolved
    against the working set alone. With exo holding 41 GiB on a 96 GiB box,
    that is 54 GiB against 84 -- so a 47.5 GiB model was "fitting with 6.8
    GiB to spare" in one answer and "36 GiB of headroom, roomy defaults" in
    the next, and the roomy defaults (2048-wide prefill, 8 prompts at once)
    are what OOM a box with 6.8 GiB to spare.
    """
    ws = detected_working_set_bytes()
    avail = available_now()
    # the knurlogic allowance (machine/memory/allowance.py) caps both: the most
    # this machine's owner lets knurlogic have, whatever the GPU could hold
    from knurlogic.machine.memory import allowance
    allow = allowance.get()
    known = [b for b in (ws, avail, allow) if b > 0]
    budget = min(known) if known else 0
    machine = min((b for b in (ws, avail) if b > 0), default=0)
    limited_by = ("nothing known" if not known else
                  "the knurlogic allowance" if allow and (not machine
                                                          or allow < machine)
                  else "memory available now" if budget == avail and avail != ws
                  else "the GPU working set")
    return {"bytes": budget, "working_set_bytes": ws,
            "available_bytes": avail, "allowance_bytes": allow,
            "limited_by": limited_by}
