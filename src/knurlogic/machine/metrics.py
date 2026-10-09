"""How hard this machine is working, over the last few minutes.

Memory says what fits; these say why tokens are slow. Each is readable
without sudo: gpu (`ioreg` AGXAccelerator utilization), cpu
(host_statistics ticks), swap (`sysctl vm.swapusage`), pressure
(`kern.memorystatus_vm_pressure_level`), thermal (NSProcessInfo) and
temp_c (IOHIDEventSystemClient, undocumented: reads None if it moves).

History is kept in-process, sampled on status requests no more often than
MIN_INTERVAL_S. Every probe fails to None, never to 0.

Design: docs/design/memory.md (metrics).
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
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
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
    except (OSError, subprocess.SubprocessError):
        return None, None
    u, m = _GPU.search(out), _GPU_MEM.search(out)
    return (int(u.group(1)) if u else None, int(m.group(1)) if m else None)


_UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def swap():
    """Bytes of swap in use. `used = 7168.00M` in sysctl's own words."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True,
                             timeout=2).stdout
        m = re.search(r"used\s*=\s*([\d.]+)([KMG])", out)
        return int(float(m.group(1)) * _UNITS[m.group(2)]) if m else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def _vm_pressure():
    """macOS's own memory pressure level: 1 normal, 2 warn, 4 critical."""
    try:
        out = subprocess.run(["sysctl", "-n",
                              "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True,
                             timeout=2).stdout
        return int(out.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


#: How far back swap growth is looked for.
SWAP_WINDOW_S = 60


def _swapping(now, hist) -> bool:
    """Whether pressure is pushing into swap right now. macOS keeps swap
    `used` long after the pressure that caused it has passed (6.6 GiB on a
    box 96% free), so a used figure alone is history: it counts only when
    the kernel says warn/critical, or swap grew within SWAP_WINDOW_S."""
    if (now.get("vm_pressure") or 0) >= 2:
        return True
    s = now.get("swap_bytes")
    if s is None:
        return False
    cutoff = now["t"] - SWAP_WINDOW_S
    base = next((h["swap_bytes"] for h in hist
                 if h["t"] >= cutoff and h.get("swap_bytes") is not None),
                None)
    return base is not None and s > base


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
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
        return None


_HID: dict = {}


def _hid():
    """The sensor client and its services, built once per process."""
    if _HID:
        return _HID
    C, vp = ctypes, ctypes.c_void_p
    io = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/"
                "CoreFoundation")
    for fn, res, args in (
            (cf.CFStringCreateWithCString, vp, [vp, C.c_char_p, C.c_uint32]),
            (cf.CFNumberCreate, vp, [vp, C.c_int, vp]),
            (cf.CFDictionaryCreate, vp, [vp, vp, vp, C.c_long, vp, vp]),
            (cf.CFArrayGetCount, C.c_long, [vp]),
            (cf.CFArrayGetValueAtIndex, vp, [vp, C.c_long]),
            (cf.CFStringGetCString, C.c_bool, [vp, C.c_char_p, C.c_long,
                                               C.c_uint32]),
            (cf.CFRelease, None, [vp]),
            (io.IOHIDEventSystemClientCreate, vp, [vp]),
            (io.IOHIDEventSystemClientSetMatching, None, [vp, vp]),
            (io.IOHIDEventSystemClientCopyServices, vp, [vp]),
            (io.IOHIDServiceClientCopyProperty, vp, [vp, vp]),
            (io.IOHIDServiceClientCopyEvent, vp, [vp, C.c_int64, C.c_int32,
                                                 C.c_int64]),
            (io.IOHIDEventGetFloatValue, C.c_double, [vp, C.c_int32])):
        fn.restype, fn.argtypes = res, args
    utf8 = 0x08000100

    def cfs(t):
        return cf.CFStringCreateWithCString(None, t.encode(), utf8)


    def num(v):
        i = C.c_int32(v)
        return cf.CFNumberCreate(None, 3, C.byref(i))       # kCFNumberSInt32
    # usage page 0xff00, usage 5: Apple's temperature sensors
    keys = (vp * 2)(cfs("PrimaryUsagePage"), cfs("PrimaryUsage"))
    vals = (vp * 2)(num(0xff00), num(5))
    kcb = vp.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
    vcb = vp.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
    match = cf.CFDictionaryCreate(None, keys, vals, 2, C.addressof(kcb),
                                  C.addressof(vcb))
    client = io.IOHIDEventSystemClientCreate(None)
    io.IOHIDEventSystemClientSetMatching(client, match)
    svcs = io.IOHIDEventSystemClientCopyServices(client)
    prod, sensors = cfs("Product"), []
    for i in range(cf.CFArrayGetCount(svcs) if svcs else 0):
        sv = cf.CFArrayGetValueAtIndex(svcs, i)
        nm, buf = io.IOHIDServiceClientCopyProperty(sv, prod), \
            C.create_string_buffer(128)
        name = (buf.value.decode(errors="replace")
                if nm and cf.CFStringGetCString(nm, buf, 128, utf8) else "")
        sensors.append((name, sv))
    # The die sensors, when the machine names them; every sensor otherwise.
    die = [s for s in sensors if "tdie" in s[0]]
    _HID.update(io=io, cf=cf, client=client, services=svcs,
                sensors=die or sensors)
    return _HID


def _temp_c():
    """The hottest sensor, in °C. Temperature event type 15, its value
    field 15 << 16."""
    try:
        h = _hid()
        io, cf, best = h["io"], h["cf"], None
        for _, sv in h["sensors"]:
            ev = io.IOHIDServiceClientCopyEvent(sv, 15, 0, 0)
            if not ev:
                continue
            t = io.IOHIDEventGetFloatValue(ev, 15 << 16)
            cf.CFRelease(ev)
            if 0 < t < 150 and (best is None or t > best):
                best = t
        return round(best, 1) if best is not None else None
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError, KeyError):
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
            "memory_pct": _pressure(memory_map), "swap_bytes": swap(),
            "vm_pressure": _vm_pressure(),
            "thermal": _thermal(), "temp_c": _temp_c()}


def metrics(memory_map=None) -> dict:
    """The latest sample plus the history, sampling only if the newest one
    is older than MIN_INTERVAL_S."""
    if not _hist or time.time() - _hist[-1]["t"] >= MIN_INTERVAL_S:
        now = sample(memory_map)
        now["swapping"] = _swapping(now, _hist)
        _hist.append(now)
    return {"now": _hist[-1], "history": list(_hist),
            "interval_seconds": MIN_INTERVAL_S}
