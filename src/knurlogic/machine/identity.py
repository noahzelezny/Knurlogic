"""Which machine this is, in a form that survives renames and re-addressing.

A node is its `id`, never its name: names collide (two "MacBook Pro"s) and
get changed, and addresses move when a Thunderbolt cable is replugged.

  id    first 12 hex of sha256(IOPlatformUUID). Stable across reboots,
        addresses and renames. Not the raw hardware UUID on the wire -- but
        it IS a persistent identifier for this machine on the local
        network, and is only sent to peers that ask for status.
  name  the ComputerName ("Noah's Mac Studio"), for display only.

Read through IOKit with ctypes, the same way the temperature sensors are:
no subprocess, nothing to install. A failure falls back to the hostname's
hash and says so in `id_source`, rather than inventing a UUID.
"""

from __future__ import annotations

import ctypes
import hashlib
import socket
import subprocess

_ID: dict = {}


def _platform_uuid() -> str:
    C, vp = ctypes, ctypes.c_void_p
    io = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/"
                "CoreFoundation")
    io.IOServiceMatching.restype, io.IOServiceMatching.argtypes = \
        vp, [C.c_char_p]
    io.IOServiceGetMatchingService.restype = C.c_uint32
    io.IOServiceGetMatchingService.argtypes = [C.c_uint32, vp]
    io.IORegistryEntryCreateCFProperty.restype = vp
    io.IORegistryEntryCreateCFProperty.argtypes = [C.c_uint32, vp, vp,
                                                   C.c_uint32]
    io.IOObjectRelease.argtypes = [C.c_uint32]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFStringCreateWithCString.argtypes = [vp, C.c_char_p, C.c_uint32]
    cf.CFStringGetCString.restype = C.c_bool
    cf.CFStringGetCString.argtypes = [vp, C.c_char_p, C.c_long, C.c_uint32]
    cf.CFRelease.argtypes = [vp]
    utf8 = 0x08000100
    # kIOMainPortDefault is 0; IOServiceGetMatchingService consumes the
    # matching dictionary, so it is not released here.
    svc = io.IOServiceGetMatchingService(
        0, io.IOServiceMatching(b"IOPlatformExpertDevice"))
    if not svc:
        return ""
    try:
        key = cf.CFStringCreateWithCString(None, b"IOPlatformUUID", utf8)
        val = io.IORegistryEntryCreateCFProperty(svc, key, None, 0)
        cf.CFRelease(key)
        if not val:
            return ""
        buf = C.create_string_buffer(64)
        ok = cf.CFStringGetCString(val, buf, 64, utf8)
        cf.CFRelease(val)
        return buf.value.decode() if ok else ""
    finally:
        io.IOObjectRelease(svc)


def _computer_name() -> str:
    try:
        out = subprocess.run(["scutil", "--get", "ComputerName"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return socket.gethostname().removesuffix(".local")


def identity() -> dict:
    """{id, name, id_source}, read once per process."""
    if _ID:
        return dict(_ID)
    try:
        uuid, source = _platform_uuid(), "IOPlatformUUID"
    except Exception:
        uuid = ""
    if not uuid:
        uuid, source = socket.gethostname(), "hostname"
    _ID.update(id=hashlib.sha256(uuid.encode()).hexdigest()[:12],
               name=_computer_name(), id_source=source)
    return dict(_ID)
