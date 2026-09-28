"""The wired limit -- the knob that decides how much of the box a model may use.

On Apple Silicon `iogpu.wired_limit_mb` caps how much memory the GPU may
wire, and it is what the framework's "recommended working set" follows.
Measured on this box, which is the whole reason this module states it as a
fact rather than folklore:

    iogpu.wired_limit_mb: 86016        -> 84.0 GiB
    framework working set:                84.0 GiB   of 96 GiB installed

So an artifact that "does not fit" often fits perfectly well -- the machine
was simply never told it could use its own memory. That is the single most
common way somebody concludes local inference does not work on their Mac.

WHAT THIS MODULE WILL NOT DO. It does not set the value. Changing it needs
root, it is a system-wide setting, and a package that quietly raises how much
memory the GPU may wire out from under someone is not a package anyone should
install. Knurlogic works out the number, prints the command, says what it
costs, and the human runs it.

THE RESERVE IS A JUDGEMENT, NOT A MEASUREMENT, and is labelled as one
everywhere it is used. macOS still has to run: window server, browser, the
editor you are reading this in. Leaving too little does not OOM the model,
it wedges the machine.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

GIB = 1 << 30
MIB = 1 << 20

#: Keys to try, in order. The name moved between macOS versions and the older
#: one is still present on current systems; reading only one answers "unknown"
#: on a box that has it.
SYSCTL_KEYS = ("iogpu.wired_limit_mb", "debug.iogpu.wired_limit")

#: Leave the OS at least this much. A JUDGEMENT: the fraction is what this
#: box was already tuned to by hand (96 GiB installed, 84 GiB wired = 12 GiB
#: left), which is corroboration, not proof.
RESERVE_FRACTION = 0.125
RESERVE_FLOOR_BYTES = 8 * GIB


def _sysctl(key: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=5)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip())
    except Exception:
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
    total = _sysctl("hw.memsize") or 0
    for key in SYSCTL_KEYS:
        mb = _sysctl(key)
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
            f"{need_bytes / GIB:.1f} GiB wanted against a "
            f"{w.ceiling_bytes / GIB:.1f} GiB ceiling "
            f"({w.total_bytes / GIB:.0f} GiB installed, leaving "
            f"{w.reserve_bytes / GIB:.0f} GiB for macOS). Raising the wired "
            f"limit will not fix this -- it needs a smaller rung or another "
            f"box.")
        return d

    if need_bytes and need_bytes > w.limit_bytes:
        target = min(w.ceiling_bytes, max(need_bytes, w.limit_bytes))
        d.update(action="raise", target_bytes=target,
                 command=command_for(target, w.key or SYSCTL_KEYS[0]))
        d["note"] = (
            f"the artifact wants {need_bytes / GIB:.1f} GiB and the GPU is "
            f"allowed to wire {w.limit_bytes / GIB:.1f} GiB of the "
            f"{w.total_bytes / GIB:.0f} GiB installed. It is not that the "
            f"model does not fit -- the machine has not been told it may use "
            f"its own memory. Raising it leaves "
            f"{(w.total_bytes - target) / GIB:.1f} GiB for macOS; that "
            f"reserve is a judgement, and too little wedges the machine "
            f"rather than the model.")
        return d

    d["action"] = "ok"
    d["note"] = (f"the GPU may wire {w.limit_bytes / GIB:.1f} GiB of "
                 f"{w.total_bytes / GIB:.0f} GiB installed"
                 + (f"; {w.headroom_to_ceiling / GIB:.1f} GiB more is "
                    f"available under the reserve if a bigger rung needs it"
                    if w.headroom_to_ceiling >= GIB else ""))
    return d


def render(d: dict) -> str:
    L = ["wired limit"]
    if not d.get("known"):
        return f"wired limit  unknown -- {d.get('note', '')}"
    L.append(f"  {d['limit_bytes'] / GIB:.1f} GiB of "
             f"{d['total_bytes'] / GIB:.0f} GiB installed  [{d['key']}]")
    if d.get("note"):
        L.append(f"  {d['note']}")
    if d.get("command"):
        L.append(f"  {d['command']}")
        L.append("  (resets at reboot; knurlogic will not run it for you -- "
                 "it is a system-wide setting and needs root)")
    return "\n".join(L)


def detected_working_set_bytes() -> int:
    """What the framework says it may use, read where it comes from.

    `resolve()` still takes headroom as an INPUT -- that stance is what keeps
    the resolver testable and machine-independent. This is the COMMANDS
    filling that input in when the user did not, because the alternative is
    what shipped: forgetting a flag silently produced the roomy defaults,
    which is the footgun this package exists to remove.

    The number is the wired limit, and this module already reads it from
    sysctl: MLX's `max_recommended_working_set_size` is exactly
    `iogpu.wired_limit_mb` in bytes (this module's docstring measured the
    two agreeing on this box), and the engine's `memory()` asks the same
    sysctl through mlx. Reading it here instead keeps callers OUT of engine/
    -- the page process must never import mlx to report a number, and
    `memory()` would drag the whole framework in for one integer. Same
    value, no engine.
    """
    limit = read().limit_bytes
    if limit:
        return limit
    # the sysctl left at its default reads 0: only the framework knows the
    # default it applies, so it is asked -- on a box nobody has tuned
    try:
        from knurlogic.engine.serve import memory
        return int(memory().get("working_set_bytes") or 0)
    except Exception:
        return 0


#: Cached: a machine does not change model while the process runs, and the
#: lookup costs a subprocess.
_MACHINE = None


def _kind_from_identifier(model_id: str) -> str:
    """Last-resort guess from `hw.model`.

    It is a GUESS and only that. The old identifiers said the product in
    their name (`MacBookPro18,3`, `Macmini9,1`), but Apple dropped the
    product prefix: `Mac15,14` is a Mac Studio, `Mac16,7` a MacBook Pro, and
    nothing in either string says so. Anything without the old prefix reads
    as unknown rather than as a coin flip dressed up as a fact.
    """
    m = model_id.lower()
    if m.startswith(("macbookpro", "macbookair", "macbook")):
        return "laptop"
    if m.startswith("macmini"):
        return "mini"
    if m.startswith("macpro"):
        return "pro"
    if m.startswith("imac"):
        return "imac"
    return ""


def machine() -> dict:
    """What this box IS: {kind, model, model_id, chip}.

    `system_profiler SPHardwareDataType` is the channel that actually knows
    -- it prints "Model Name: Mac Studio" -- and it answers in about 0.13s,
    once per process. Deriving the product from `hw.model` cannot work on
    current hardware (this box is `Mac15,14` and is a Studio), so the
    identifier is a fallback for when the lookup fails, not the primary.
    """
    global _MACHINE
    if _MACHINE is not None:
        return _MACHINE
    model_id = ""
    try:
        model_id = subprocess.run(["sysctl", "-n", "hw.model"],
                                  capture_output=True, text=True,
                                  timeout=2).stdout.strip()
    except Exception:
        pass
    name = ""
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"],
                             capture_output=True, text=True, timeout=8).stdout
        for line in out.splitlines():
            if line.strip().startswith("Model Name:"):
                name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    low = name.lower()
    if "studio" in low:
        kind = "studio"
    elif "mini" in low:
        kind = "mini"
    elif "macbook" in low:
        kind = "laptop"
    elif "imac" in low:
        kind = "imac"
    elif "mac pro" in low:
        kind = "pro"
    else:
        kind = _kind_from_identifier(model_id)
    # The chip is what a person means by "which machine": M3 Ultra, M4 Max.
    chip = ""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True,
                              timeout=2).stdout.strip()
    except Exception:
        pass
    _MACHINE = {"kind": kind, "model": name, "model_id": model_id,
                "chip": chip.removeprefix("Apple ").strip()}
    return _MACHINE


def kind_from(name: str = "", model_id: str = "", product: str = "") -> dict:
    """Best honest guess at ANOTHER node's kind, from what it told us.

    A remote node cannot be asked -- `system_profiler` answers for THIS box,
    and labelling every node in a cluster with the local machine is the same
    bug as running a version check with bare `python3` inside a loop over
    envs: every iteration answers for the wrong thing.

    So two weak channels, in order, and neither pretends to be strong:

      1. the friendly name, because macOS seeds it from the product and
         people leave it ("Studio A", "Laptop B");
      2. the model identifier, which only says the product on pre-2022
         hardware -- see `_kind_from_identifier`.

    Nothing matching leaves `kind` empty, and the page draws a plain box.
    That is the correct outcome: an unknown machine should look unknown.
    """
    # A PRODUCT NAME, when the source has one, is not a weak channel at all.
    # exo reports `modelId: "Mac Studio"` / `"MacBook Pro"` -- measured
    # against the live daemon -- so when that is present it decides, and the
    # guessing below is only for sources that give nothing better.
    low = (product or name or "").lower()
    if "studio" in low:
        kind = "studio"
    elif "mini" in low:
        kind = "mini"
    elif "imac" in low:
        kind = "imac"
    elif "mac pro" in low or "macpro" in low:
        kind = "pro"
    elif "book" in low:        # MacBook, and the -book names people give them
        kind = "laptop"
    else:
        kind = _kind_from_identifier(model_id or "")
    return {"kind": kind, "model": product or "", "model_id": model_id or ""}


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
    from knurlogic.machine.loaded import available_memory
    ws = detected_working_set_bytes()
    try:
        avail = int(available_memory().get("available_bytes") or 0)
    except Exception:
        avail = 0
    # the knurlogic allowance (machine/allowance.py) caps both: the most
    # this machine's owner lets knurlogic have, whatever the GPU could hold
    from knurlogic.machine import allowance
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
