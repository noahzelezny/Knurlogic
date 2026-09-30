"""Bonjour: every knurlogic page on the network, found without being named.

DNS-SD service type `_knurlogic._tcp` through the `dns_sd.h` API in
libSystem; mDNSResponder owns the registration and the multicast. The page
(`ui`) registers one advertisement per machine (TXT: id, name, ver, schema,
role); every page browses, resolving on the interface each result was seen
on. The ctypes callbacks and DNSServiceRefs are owned by the Discovery
object while the daemon may call them, and a ref is never freed inside its
own callback. A failure is "nothing found", never a crash, and `status()`
says which.

Design: docs/design/discovery.md (Bonjour).
"""

from __future__ import annotations

import ctypes
import select
import socket
import threading
import time

SERVICE = "_knurlogic._tcp"
LOCAL_ONLY = 0xFFFFFFFF          # kDNSServiceInterfaceIndexLocalOnly
F_MORE, F_ADD = 0x1, 0x2          # kDNSServiceFlagsMoreComing / Add
IPV4 = 0x1                        # kDNSServiceProtocol_IPv4

vp, c_char_p, u32, u16, i32 = (ctypes.c_void_p, ctypes.c_char_p,
                               ctypes.c_uint32, ctypes.c_uint16,
                               ctypes.c_int32)
REG_CB = ctypes.CFUNCTYPE(None, vp, u32, i32, c_char_p, c_char_p, c_char_p,
                          vp)
BROWSE_CB = ctypes.CFUNCTYPE(None, vp, u32, u32, i32, c_char_p, c_char_p,
                             c_char_p, vp)
RESOLVE_CB = ctypes.CFUNCTYPE(None, vp, u32, u32, i32, c_char_p, c_char_p,
                              u16, u16, ctypes.POINTER(ctypes.c_ubyte), vp)
ADDR_CB = ctypes.CFUNCTYPE(None, vp, u32, u32, i32, c_char_p, vp, u32, vp)

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        lib.DNSServiceRegister.argtypes = [ctypes.POINTER(vp), u32, u32,
                                           c_char_p, c_char_p, c_char_p,
                                           c_char_p, u16, u16, vp, REG_CB, vp]
        lib.DNSServiceBrowse.argtypes = [ctypes.POINTER(vp), u32, u32,
                                         c_char_p, c_char_p, BROWSE_CB, vp]
        lib.DNSServiceResolve.argtypes = [ctypes.POINTER(vp), u32, u32,
                                          c_char_p, c_char_p, c_char_p,
                                          RESOLVE_CB, vp]
        lib.DNSServiceGetAddrInfo.argtypes = [ctypes.POINTER(vp), u32, u32,
                                              u32, c_char_p, ADDR_CB, vp]
        lib.DNSServiceRefSockFD.argtypes = [vp]
        lib.DNSServiceRefSockFD.restype = ctypes.c_int
        lib.DNSServiceProcessResult.argtypes = [vp]
        lib.DNSServiceProcessResult.restype = i32
        lib.DNSServiceRefDeallocate.argtypes = [vp]
        for f in ("DNSServiceRegister", "DNSServiceBrowse",
                  "DNSServiceResolve", "DNSServiceGetAddrInfo"):
            getattr(lib, f).restype = i32
        _LIB = lib
    return _LIB


def txt_encode(d: dict) -> bytes:
    out = b""
    for k, v in d.items():
        item = f"{k}={v}".encode()[:255]
        out += bytes([len(item)]) + item
    return out


def txt_decode(b: bytes) -> dict:
    out, i = {}, 0
    while i < len(b):
        n = b[i]
        item = b[i + 1:i + 1 + n].decode(errors="replace")
        k, _, v = item.partition("=")
        if k:
            out[k] = v
        i += 1 + n
    return out


def _label(s: str) -> bytes:
    """A DNS label: at most 63 BYTES, never cut inside a UTF-8 character."""
    b = s.encode()
    return b[:63].decode(errors="ignore").encode() if len(b) > 63 else b


def interface_of(ip: str) -> int:
    """The interface index an address is configured on, or 0 (all). Read
    from `ifconfig`, which is the one place that pairs the two."""
    if ip in ("", "0.0.0.0", "::"):
        return 0
    import subprocess
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True,
                             timeout=3).stdout
    except Exception:
        return 0
    iface = None
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":", 1)[0]
        elif iface and line.strip().startswith(f"inet {ip} "):
            try:
                return socket.if_nametoindex(iface)
            except OSError:
                return 0
    return 0


class Discovery:
    """One registration (optional) and one browse, served by one thread.

    `found` is {(instance, if_index): {name, if_index, host, port, txt}}
    for services that resolved to an IPv4 address; `on_change` is called
    (from the loop thread) whenever it changes."""

    def __init__(self, on_change=None, if_index: int = 0,
                 service: str = SERVICE):
        #: the DNS-SD type; tests use their own, so a live page on the
        #: same Mac never lists them as a machine
        self.service = service
        self.on_change = on_change
        self.if_index = if_index
        self.found: dict = {}
        self.errors: list = []
        self.started = 0.0
        self.registered_as = ""
        self._refs: list = []          # live refs the loop selects on
        self._free: list = []          # refs to deallocate outside callbacks
        # each ref's callback, alive exactly as long as the ref (the daemon
        # may call it until the ref is deallocated)
        self._keep: dict = {}
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    # -- starting ---------------------------------------------------------
    def _add_ref(self, ref, cb):
        with self._lock:
            self._refs.append(ref)
            self._keep[id(ref)] = (ref, cb)

    def register(self, instance: str, port: int, txt: dict) -> bool:
        lib = _lib()

        def cb(ref, flags, err, name, regtype, domain, ctx):
            if err:
                self.errors.append(f"register: error {err}")
            else:
                self.registered_as = (name or b"").decode(errors="replace")
        c = REG_CB(cb)
        ref = vp()
        rec = txt_encode(txt)
        err = lib.DNSServiceRegister(
            ctypes.byref(ref), 0, self.if_index, _label(instance),
            self.service.encode(), None, None, socket.htons(port), len(rec),
            ctypes.c_char_p(rec), c, None)
        if err:
            self.errors.append(f"register: error {err}")
            return False
        self._add_ref(ref, c)
        return True

    def browse(self) -> bool:
        lib = _lib()

        def cb(ref, flags, ifi, err, name, regtype, domain, ctx):
            if err:
                self.errors.append(f"browse: error {err}")
                return
            key = ((name or b"").decode(errors="replace"), ifi)
            if flags & F_ADD:
                self._resolve(key, regtype, domain)
            else:
                with self._lock:
                    gone = self.found.pop(key, None)
                if gone is not None and self.on_change:
                    self.on_change(self.snapshot())
        c = BROWSE_CB(cb)
        ref = vp()
        err = lib.DNSServiceBrowse(ctypes.byref(ref), 0, self.if_index,
                                   self.service.encode(), None, c, None)
        if err:
            self.errors.append(f"browse: error {err}")
            return False
        self._add_ref(ref, c)
        return True

    def _resolve(self, key, regtype, domain):
        lib = _lib()
        name, ifi = key
        holder = {}

        def rcb(ref, flags, ifi_, err, fullname, host, port, tlen, txt, ctx):
            # one answer is enough; a second callback in the same batch
            # (a TXT change, MoreComing) must not queue the ref again
            if not holder.get("queued"):
                holder["queued"] = True
                self._free.append(holder.get("ref"))
            if err:
                self.errors.append(f"resolve {name}: error {err}")
                return
            info = {"name": name, "if_index": ifi,
                    "port": socket.ntohs(port),
                    "txt": txt_decode(bytes(txt[:tlen])) if tlen else {}}
            self._addr(key, host, info)
        c = RESOLVE_CB(rcb)
        ref = vp()
        if lib.DNSServiceResolve(ctypes.byref(ref), 0, ifi,
                                 name.encode(), regtype, domain, c, None):
            return
        holder["ref"] = ref
        self._add_ref(ref, c)

    def _addr(self, key, host, info):
        lib = _lib()
        holder = {}

        def acb(ref, flags, ifi, err, hostname, addr, ttl, ctx):
            last = not flags & F_MORE
            if last and not holder.get("queued"):
                # the query is done with, answer or not: freed by the loop
                holder["queued"] = True
                self._free.append(holder.get("ref"))
            if err or not addr:
                return
            fam = ctypes.cast(addr, ctypes.POINTER(ctypes.c_ubyte))[1]
            if fam != socket.AF_INET:
                return
            raw = ctypes.string_at(addr, 8)
            ip = socket.inet_ntoa(raw[4:8])
            with self._lock:
                self.found[key] = {**info, "host": ip}
            if self.on_change:
                self.on_change(self.snapshot())
        c = ADDR_CB(acb)
        ref = vp()
        if lib.DNSServiceGetAddrInfo(ctypes.byref(ref), 0, info["if_index"],
                                     IPV4, host, c, None):
            return
        holder["ref"] = ref
        self._add_ref(ref, c)

    # -- the loop ---------------------------------------------------------
    def _loop(self):
        lib = _lib()
        while not self._stop:
            with self._lock:
                refs = [r for r in self._refs if r and r.value]
            fds = {}
            for r in refs:
                fd = lib.DNSServiceRefSockFD(r)
                if fd >= 0:
                    fds[fd] = r
            if not fds:
                time.sleep(0.2)
                continue
            try:
                ready, _, _ = select.select(list(fds), [], [], 0.5)
            except (OSError, ValueError):
                time.sleep(0.2)
                continue
            for fd in ready:
                lib.DNSServiceProcessResult(fds[fd])
            self._drain_free(lib)

    def _drain_free(self, lib) -> None:
        """Deallocate the refs callbacks asked to free -- here, never inside
        the callback, and only while still live: a ref leaves _refs exactly
        once, so it is deallocated exactly once, and its callback goes
        with it."""
        while self._free:
            r = self._free.pop()
            if r is None:
                continue
            with self._lock:
                live = any(x is r for x in self._refs)
                if live:
                    self._refs[:] = [x for x in self._refs if x is not r]
                    self._keep.pop(id(r), None)
            if live:
                lib.DNSServiceRefDeallocate(r)

    def start(self) -> "Discovery":
        if self._thread is None:
            self.started = time.time()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="knurlogic-bonjour")
            self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(2)
        lib = _lib()
        with self._lock:
            for r in self._refs:
                if r and r.value:
                    lib.DNSServiceRefDeallocate(r)
            self._refs.clear()

    # -- reading ----------------------------------------------------------
    def snapshot(self) -> list:
        with self._lock:
            return [dict(v) for v in self.found.values()]

    def status(self) -> dict:
        d = {"service": self.service, "registered_as": self.registered_as,
             "found": len(self.found), "errors": self.errors[-3:]}
        if self.started and not self.found and \
                time.time() - self.started > 10:
            d["hint"] = ("browsing has found no other knurlogic in "
                         f"{time.time() - self.started:.0f} s. Either none "
                         f"is advertising (a page bound to 127.0.0.1 does "
                         f"not), or macOS is blocking discovery: System "
                         f"Settings -> Privacy & Security -> Local Network "
                         f"-> allow the app you started knurlogic from (the terminal; "
                         f"under launchd or a script, the Python binary "
                         f"itself -- a rebuilt environment needs it "
                         f"granted again). "
                         f"`dns-sd -B {SERVICE}` shows what this machine "
                         f"can see.")
        return d
