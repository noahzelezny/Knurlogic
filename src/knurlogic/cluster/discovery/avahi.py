"""Discovery on Linux: Avahi, through libavahi-client (installed with
avahi-daemon, which owns the registration and the multicast, as
mDNSResponder does on macOS).

Every Avahi call is made on one thread: `register` and `browse` before
`start`, the rest inside the poll loop `start` runs, whose callbacks are
kept alive by this object. A resolver is freed in its own callback, as
Avahi's examples do. No daemon is "nothing found", with the reason in
`status()`.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
import time

from knurlogic.cluster.discovery import SERVICE, label

IF_UNSPEC, PROTO_INET = -1, 0
BROWSER_NEW, BROWSER_REMOVE, BROWSER_FAILURE = 0, 1, 4
RESOLVER_FOUND = 0
CLIENT_FAILURE = 100
GROUP_COLLISION, GROUP_FAILURE = 3, 4

vp, c_char_p, c_int = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
CLIENT_CB = ctypes.CFUNCTYPE(None, vp, c_int, vp)
GROUP_CB = ctypes.CFUNCTYPE(None, vp, c_int, vp)
BROWSE_CB = ctypes.CFUNCTYPE(None, vp, c_int, c_int, c_int, c_char_p,
                             c_char_p, c_char_p, c_int, vp)
RESOLVE_CB = ctypes.CFUNCTYPE(None, vp, c_int, c_int, c_int, c_char_p,
                              c_char_p, c_char_p, c_char_p, vp,
                              ctypes.c_uint16, vp, c_int, vp)

_LIB = None


def _load(name: str):
    return ctypes.CDLL(ctypes.util.find_library(name) or f"lib{name}.so.3")


def _lib():
    global _LIB
    if _LIB is None:
        c, common = _load("avahi-client"), _load("avahi-common")
        for lib, fn, res, args in (
                (common, "avahi_simple_poll_new", vp, []),
                (common, "avahi_simple_poll_get", vp, [vp]),
                (common, "avahi_simple_poll_iterate", c_int, [vp, c_int]),
                (common, "avahi_simple_poll_wakeup", None, [vp]),
                (common, "avahi_simple_poll_free", None, [vp]),
                (common, "avahi_strerror", c_char_p, [c_int]),
                (common, "avahi_string_list_new_from_array", vp,
                 [ctypes.POINTER(c_char_p), c_int]),
                (common, "avahi_string_list_free", None, [vp]),
                (common, "avahi_string_list_get_next", vp, [vp]),
                (common, "avahi_string_list_get_text", vp, [vp]),
                (common, "avahi_string_list_get_size", ctypes.c_size_t, [vp]),
                (common, "avahi_address_snprint", c_char_p,
                 [c_char_p, ctypes.c_size_t, vp]),
                (c, "avahi_client_new", vp,
                 [vp, c_int, CLIENT_CB, vp, ctypes.POINTER(c_int)]),
                (c, "avahi_client_free", None, [vp]),
                (c, "avahi_client_errno", c_int, [vp]),
                (c, "avahi_entry_group_new", vp, [vp, GROUP_CB, vp]),
                (c, "avahi_entry_group_add_service_strlst", c_int,
                 [vp, c_int, c_int, c_int, c_char_p, c_char_p, c_char_p,
                  c_char_p, ctypes.c_uint16, vp]),
                (c, "avahi_entry_group_commit", c_int, [vp]),
                (c, "avahi_entry_group_free", c_int, [vp]),
                (c, "avahi_service_browser_new", vp,
                 [vp, c_int, c_int, c_char_p, c_char_p, c_int, BROWSE_CB,
                  vp]),
                (c, "avahi_service_resolver_new", vp,
                 [vp, c_int, c_int, c_char_p, c_char_p, c_char_p, c_int,
                  c_int, RESOLVE_CB, vp]),
                (c, "avahi_service_resolver_free", c_int, [vp])):
            f = getattr(lib, fn)
            f.restype, f.argtypes = res, args
            setattr(c, fn, f)
        _LIB = c
    return _LIB


def _txt(lib, strlst) -> dict:
    out = {}
    while strlst:
        n = lib.avahi_string_list_get_size(strlst)
        raw = ctypes.string_at(lib.avahi_string_list_get_text(strlst), n)
        k, _, v = raw.decode(errors="replace").partition("=")
        if k:
            out[k] = v
        strlst = lib.avahi_string_list_get_next(strlst)
    return out


class Discovery:
    """One client, its registrations and one browse, served by one thread.
    The same interface as bonjour.Discovery."""

    def __init__(self, on_change=None, if_index: int = 0,
                 service: str = SERVICE):
        self.service = service
        self.on_change = on_change
        self.if_index = if_index
        self.found: dict = {}
        self.errors: list = []
        self.started = 0.0
        self.registered_as = ""
        self._lock = threading.Lock()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._poll = self._client = None
        # the C callbacks, alive as long as the client that calls them
        self._cbs = (CLIENT_CB(self._on_client), GROUP_CB(self._on_group),
                     BROWSE_CB(self._on_browse), RESOLVE_CB(self._on_resolve))

    # -- starting ---------------------------------------------------------
    def _connect(self) -> bool:
        if self._client:
            return True
        lib = _lib()
        self._poll = lib.avahi_simple_poll_new()
        err = c_int(0)
        self._client = lib.avahi_client_new(
            lib.avahi_simple_poll_get(self._poll), 0, self._cbs[0], None,
            ctypes.byref(err))
        if not self._client:
            self.errors.append(
                f"avahi: {lib.avahi_strerror(err.value).decode()}")
        return bool(self._client)

    def _ifindex(self) -> int:
        return self.if_index or IF_UNSPEC

    def register(self, instance: str, port: int, txt: dict) -> bool:
        try:
            if not self._connect():
                return False
        except OSError as e:
            self.errors.append(f"avahi: {e}")
            return False
        lib = _lib()
        group = lib.avahi_entry_group_new(self._client, self._cbs[1], None)
        items = [f"{k}={v}".encode()[:255] for k, v in txt.items()]
        strlst = lib.avahi_string_list_new_from_array(
            (c_char_p * len(items))(*items), len(items))
        name = label(instance)
        rc = group and lib.avahi_entry_group_add_service_strlst(
            group, self._ifindex(), PROTO_INET, 0, name,
            self.service.encode(), None, None, port, strlst)
        lib.avahi_string_list_free(strlst)
        if not group or rc or lib.avahi_entry_group_commit(group):
            self.errors.append("register: " + lib.avahi_strerror(
                lib.avahi_client_errno(self._client)).decode())
            if group:
                lib.avahi_entry_group_free(group)
            return False
        self.registered_as = name.decode(errors="replace")
        return True

    def browse(self) -> bool:
        try:
            if not self._connect():
                return False
        except OSError as e:
            self.errors.append(f"avahi: {e}")
            return False
        lib = _lib()
        if not lib.avahi_service_browser_new(
                self._client, self._ifindex(), PROTO_INET,
                self.service.encode(), None, 0, self._cbs[2], None):
            self.errors.append("browse: " + lib.avahi_strerror(
                lib.avahi_client_errno(self._client)).decode())
            return False
        return True

    # -- callbacks, on the loop thread ------------------------------------
    def _on_client(self, client, state, ud):
        if state == CLIENT_FAILURE:
            self.errors.append("avahi-daemon went away; nothing more is "
                               "found until the page restarts")

    def _on_group(self, group, state, ud):
        if state in (GROUP_COLLISION, GROUP_FAILURE):
            self.errors.append("register: " + (
                "the name is taken" if state == GROUP_COLLISION
                else "failed"))

    def _on_browse(self, b, ifi, proto, event, name, regtype, domain, flags,
                   ud):
        key = ((name or b"").decode(errors="replace"), ifi)
        if event == BROWSER_NEW:
            _lib().avahi_service_resolver_new(
                self._client, ifi, proto, name, regtype, domain, PROTO_INET,
                0, self._cbs[3], None)
        elif event == BROWSER_REMOVE:
            with self._lock:
                gone = self.found.pop(key, None)
            if gone is not None and self.on_change:
                self.on_change(self.snapshot())
        elif event == BROWSER_FAILURE:
            self.errors.append("browse: failed")

    def _on_resolve(self, r, ifi, proto, event, name, regtype, domain, host,
                    addr, port, txt, flags, ud):
        lib = _lib()
        if event == RESOLVER_FOUND and addr:
            buf = ctypes.create_string_buffer(64)
            ip = lib.avahi_address_snprint(buf, 64, addr).decode()
            key = ((name or b"").decode(errors="replace"), ifi)
            with self._lock:
                self.found[key] = {"name": key[0], "if_index": ifi,
                                   "host": ip, "port": port,
                                   "txt": _txt(lib, txt)}
            if self.on_change:
                self.on_change(self.snapshot())
        lib.avahi_service_resolver_free(r)

    # -- the loop ---------------------------------------------------------
    def _loop(self):
        lib = _lib()
        while not self._stop:
            if lib.avahi_simple_poll_iterate(self._poll, 500) != 0:
                break

    def start(self) -> Discovery:
        if self._thread is None and self._client:
            self.started = time.time()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="knurlogic-avahi")
            self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._client is None:
            return
        lib = _lib()
        lib.avahi_simple_poll_wakeup(self._poll)
        if self._thread:
            self._thread.join(2)
        lib.avahi_client_free(self._client)     # its groups and browsers too
        lib.avahi_simple_poll_free(self._poll)
        self._client = self._poll = None

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
                         f"not), or avahi-daemon is not running, or a "
                         f"firewall drops mDNS (UDP 5353). "
                         f"`avahi-browse -r {SERVICE}` shows what this "
                         f"machine can see.")
        elif not self._client and self.errors:
            d["hint"] = ("no avahi-daemon to ask: install and start it "
                         "(on Debian and Ubuntu, the avahi-daemon package) "
                         "for peers to be found; they can still be named "
                         "with --peer")
        return d
