"""How hard this box is working, over the last few minutes.

Memory says what fits; these say why tokens are slow. GPU busy, CPU busy,
memory pressure, swap and thermal state are the five things that quietly
move decode speed, and each is readable WITHOUT sudo:

  gpu       `ioreg -c AGXAccelerator` PerformanceStatistics -- the same
            "Device Utilization %" Activity Monitor's GPU History draws
  cpu       host_statistics(HOST_CPU_LOAD_INFO) tick deltas, through ctypes
  swap      `sysctl vm.swapusage`
  thermal   NSProcessInfo.thermalState (nominal / fair / serious /
            critical), through the Objective-C runtime

Temperature in degrees is NOT here: it lives behind IOHID sensor services
that need a native helper, and a number from a guessed sensor is worse than
the OS's own four-step verdict, which is what throttling actually follows.

History is kept in this process, sampled when status is asked for and no
more often than MIN_INTERVAL_S, so a page that polls fast does not make the
box work harder to report how hard it is working. Every probe fails to
None, never to 0: a missing reading drawn as idle is a lie.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import time
from collections import deque

#: Samples kept. At the page's 2 s poll that is three minutes of line.
HISTORY = 90
MIN_INTERVAL_S = 1.5

THERMAL = ("nominal", "fair", "serious", "critical")

_hist: deque = deque(maxlen=HISTORY)
_last_ticks = None


def _cpu_pct():
    """Busy share since the previous call, from the kernel's tick counters.
    The first call has nothing to diff against and answers None."""
    global _last_ticks
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        ticks = (ctypes.c_uint * 4)()          # user, system, idle, nice
        count = ctypes.c_uint(4)
        lib.mach_host_self.restype = ctypes.c_uint
        rc = lib.host_statistics(lib.mach_host_self(), 3,   # HOST_CPU_LOAD_INFO
                                 ticks, ctypes.byref(count))
        if rc != 0:
            return None
        now = tuple(ticks)
    except Exception:
        return None
    prev, _last_ticks = _last_ticks, now
    if prev is None:
        return None
    d = [a - b for a, b in zip(now, prev)]
    total = sum(d)
    return round(100 * (total - d[2]) / total, 1) if total > 0 else None


_GPU = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_GPU_MEM = re.compile(r'"In use system memory"\s*=\s*(\d+)')


def _gpu():
    try:
        out = subprocess.run(["ioreg", "-rw0", "-c", "AGXAccelerator"],
                             capture_output=True, text=True,
                             timeout=2).stdout
    except Exception:
        return None, None
    u, m = _GPU.search(out), _GPU_MEM.search(out)
    return (int(u.group(1)) if u else None, int(m.group(1)) if m else None)


_UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def _swap():
    """Bytes of swap in use. `used = 7168.00M` in sysctl's own words."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True,
                             timeout=2).stdout
        m = re.search(r"used\s*=\s*([\d.]+)([KMG])", out)
        return int(float(m.group(1)) * _UNITS[m.group(2)]) if m else None
    except Exception:
        return None


def _thermal():
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/"
                    "Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        info = send(objc.objc_getClass(b"NSProcessInfo"),
                    objc.sel_registerName(b"processInfo"))
        state = send(info, objc.sel_registerName(b"thermalState"))
        state = int(state or 0)
        return THERMAL[state] if 0 <= state < len(THERMAL) else None
    except Exception:
        return None


def _pressure(memory_map):
    """Used share of installed memory, from the map the page already draws,
    so the line and the bar can never disagree."""
    mm = memory_map or {}
    total, used = mm.get("installed_bytes"), mm.get("used_bytes")
    return round(100 * used / total, 1) if total and used is not None else None


def sample(memory_map=None) -> dict:
    gpu, gpu_mem = _gpu()
    return {"t": round(time.time(), 1), "gpu_pct": gpu,
            "gpu_in_use_bytes": gpu_mem, "cpu_pct": _cpu_pct(),
            "memory_pct": _pressure(memory_map), "swap_bytes": _swap(),
            "thermal": _thermal()}


def metrics(memory_map=None) -> dict:
    """The latest sample plus the history, sampling only if the newest one
    is older than MIN_INTERVAL_S."""
    if not _hist or time.time() - _hist[-1]["t"] >= MIN_INTERVAL_S:
        _hist.append(sample(memory_map))
    return {"now": _hist[-1], "history": list(_hist),
            "interval_seconds": MIN_INTERVAL_S}
