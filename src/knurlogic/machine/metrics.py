"""How hard this box is working, over the last few minutes.

Memory says what fits; these say why tokens are slow. GPU busy, CPU busy,
memory pressure, swap and thermal state are the five things that quietly
move decode speed, and each is readable WITHOUT sudo:

  gpu       `ioreg -c AGXAccelerator` PerformanceStatistics -- the same
            "Device Utilization %" Activity Monitor's GPU History draws
  cpu       host_statistics(HOST_CPU_LOAD_INFO) tick deltas, through ctypes
  swap      `sysctl vm.swapusage`
  thermal   NSProcessInfo.thermalState (nominal / fair / serious /
            critical), through the Objective-C runtime -- what throttling
            follows
  temp_c    the hottest die sensor, from IOHIDEventSystemClient -- the
            same sensors a compiled helper reads, reached through ctypes so
            nothing extra is installed. Undocumented API: if Apple moves it,
            this reads None and the line goes blank, it does not break.
            A °C line shows heat building before the state changes.

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


_HID = {}


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
    cfs = lambda t: cf.CFStringCreateWithCString(None, t.encode(), utf8)

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
            "thermal": _thermal(), "temp_c": _temp_c()}


def metrics(memory_map=None) -> dict:
    """The latest sample plus the history, sampling only if the newest one
    is older than MIN_INTERVAL_S."""
    if not _hist or time.time() - _hist[-1]["t"] >= MIN_INTERVAL_S:
        _hist.append(sample(memory_map))
    return {"now": _hist[-1], "history": list(_hist),
            "interval_seconds": MIN_INTERVAL_S}
