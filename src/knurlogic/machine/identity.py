"""Which machine this is: an id that survives renames and re-addressing,
and what hardware it is.

A node is its `id`, never its name: names collide (two "MacBook Pro"s) and
get changed, and addresses move when a Thunderbolt cable is replugged.

  id    first 12 hex of sha256 of the machine's own id: IOPlatformUUID
        on macOS, /etc/machine-id on Linux, MachineGuid on Windows. Stable
        across reboots, addresses and renames. Not the raw id on the wire
        -- but it IS a persistent identifier for this machine on the local
        network, and is only sent to peers that ask for status.
  name  the ComputerName on macOS ("Alex's Mac Studio"), the hostname
        elsewhere; for display only.

On macOS the id is read through IOKit with ctypes, the same way the
temperature sensors are, and the name from `scutil --get ComputerName`.
Nothing to install. A failure falls back to the hostname's hash and says so
in `id_source`, rather than inventing a UUID.

The hardware:

  chip()        "Apple M4 Max" (macOS); the GPU nvidia-smi names, else the
                processor, elsewhere
  machine()     this box's {kind, model, model_id, chip}
  kind_from()   another node's kind, from what it reported
  chip_generation(), chip_bandwidth_gbs()   what a chip name says
"""

from __future__ import annotations

import ctypes
import hashlib
import socket
import subprocess
import sys
from pathlib import Path

from knurlogic.machine.processes import PROC_ERRORS

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


#: Where Linux keeps the machine's id (systemd, then older D-Bus).
MACHINE_ID_FILES = ("/etc/machine-id", "/var/lib/dbus/machine-id")


def _machine_id() -> tuple:
    """(this machine's own id, what it is), ("", "") where none is read."""
    if sys.platform == "darwin":
        return _platform_uuid(), "IOPlatformUUID"
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Cryptography") as k:
            return str(winreg.QueryValueEx(k, "MachineGuid")[0]), \
                "MachineGuid"
    for f in MACHINE_ID_FILES:
        try:
            got = Path(f).read_text().strip()
        except OSError:
            continue
        if got:
            return got, "machine-id"
    return "", ""


def hostname() -> str:
    """This machine's host name, without mDNS's `.local`."""
    return socket.gethostname().removesuffix(".local")


def _computer_name() -> str:
    if sys.platform != "darwin":
        return hostname()
    try:
        out = subprocess.run(["scutil", "--get", "ComputerName"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except PROC_ERRORS:
        pass
    return hostname()


def identity() -> dict:
    """{id, name, id_source}, read once per process."""
    if _ID:
        return dict(_ID)
    try:
        uuid, source = _machine_id()
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
        uuid = ""
    if not uuid:
        uuid, source = socket.gethostname(), "hostname"
    _ID.update(id=hashlib.sha256(uuid.encode()).hexdigest()[:12],
               name=_computer_name(), id_source=source)
    return dict(_ID)


# ------------------------------------------------------------ the hardware

def _sysctl(key: str) -> str:
    """`sysctl -n key`, stripped; "" where it cannot be read."""
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=5)
    except PROC_ERRORS:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


_CHIP: dict = {}


def chip() -> str:
    """This machine's chip, as it names itself: macOS's
    machdep.cpu.brand_string ("Apple M4 Max"); elsewhere the GPU
    nvidia-smi names ("GB10"), else the processor. "" when nothing says.
    Read once per process."""
    if "chip" not in _CHIP:
        if sys.platform == "darwin":
            _CHIP["chip"] = _sysctl("machdep.cpu.brand_string")
        else:
            from knurlogic.machine import nvidia
            gpu = nvidia.query("name")
            _CHIP["chip"] = (gpu[0].removeprefix("NVIDIA ") if gpu
                             else _processor())
    return _CHIP["chip"]


def _processor() -> str:
    """The processor's marketing name: /proc/cpuinfo's on Linux, the
    platform's own string elsewhere."""
    import platform
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def chip_generation(chip_name) -> int:
    """"Apple M4 Max" -> 4; 0 when it does not say."""
    import re
    m = re.search(r"\bM(\d+)\b", str(chip_name or ""))
    return int(m.group(1)) if m else 0


#: memory bandwidth (GB/s) of chips that come in ONE bandwidth only. A Max
#: that is sold binned (M3 Max 300/400, M4 Max 410/546) is not here:
#: unknown is said, never guessed.
CHIP_BANDWIDTH_GBS = {
    "Apple M1 Max": 400.0, "Apple M1 Ultra": 800.0,
    "Apple M2 Max": 400.0, "Apple M2 Ultra": 800.0,
    "Apple M3 Ultra": 819.0, "Apple M4 Pro": 273.0,
}


def chip_bandwidth_gbs(chip_name) -> float | None:
    return CHIP_BANDWIDTH_GBS.get(str(chip_name or "").strip())


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
    current hardware (`Mac15,14` is a Mac Studio), so the
    identifier is a fallback for when the lookup fails, not the primary.
    """
    global _MACHINE
    if _MACHINE is not None:
        return _MACHINE
    if sys.platform != "darwin":
        _MACHINE = _other_machine()
        return _MACHINE
    model_id = _sysctl("hw.model")
    name = ""
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"],
                             capture_output=True, text=True, timeout=8).stdout
        for line in out.splitlines():
            if line.strip().startswith("Model Name:"):
                name = line.split(":", 1)[1].strip()
                break
    except PROC_ERRORS:
        pass
    # The chip is what a person means by "which machine": M3 Ultra, M4 Max.
    _MACHINE = {**kind_from(product=name, model_id=model_id), "model": name,
                "chip": chip().removeprefix("Apple ").strip()}
    return _MACHINE


#: The product name Linux reads from the firmware ("DGX Spark").
DMI_PRODUCT = "/sys/devices/virtual/dmi/id/product_name"


def _other_machine() -> dict:
    """{kind, model, model_id, chip} off macOS. `kind` stays empty, so the
    page draws a plain box."""
    import platform
    try:
        model = Path(DMI_PRODUCT).read_text().strip()
    except OSError:
        model = ""
    return {"kind": "", "model": model, "model_id": platform.machine(),
            "chip": chip()}


def kind_from(name: str = "", model_id: str = "", product: str = "") -> dict:
    """Best honest guess at ANOTHER node's kind, from what it told us.

    A remote node cannot be asked -- `system_profiler` answers for THIS box,
    and labelling every node in a cluster with the local machine is the same
    bug as running a version check with bare `python3` inside a loop over
    envs: every iteration answers for the wrong thing.

    So two weak channels, in order, and neither pretends to be strong:

      1. the friendly name, because macOS seeds it from the product and
         people leave it ("Alex's Mac Studio", "Alex's MacBook Pro");
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
