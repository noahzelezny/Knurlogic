"""The guards in front of knurlogic's two servers (the model server and
the page): who may ask, and whether the asker is still there.

  is_loopback(address)     a client address is this machine's loopback
  browser_refusal(headers) why a browser's request is refused (Origin, and
                           the Host check against DNS rebinding), or None
  host_is_local(host)      a Host header names this machine
  hung_up(sock)            the client has closed its connection
"""

from __future__ import annotations

import ipaddress
import select
import socket


def is_loopback(address) -> bool:
    """`address` (a client_address host, maybe with a %zone) is loopback;
    anything unparseable is not."""
    try:
        return ipaddress.ip_address(str(address).split("%")[0]).is_loopback
    except ValueError:
        return False


def hung_up(sock) -> bool:
    """True when the peer has closed: readable with nothing to read (EOF)
    or an error. Peeks, so a request's bytes are never consumed; data
    waiting (a pipelined request) is not a hang-up."""
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except (OSError, ValueError):
        return True


def browser_refusal(headers, allow_origins=(), allow_hosts=()):
    """Why a request is refused, or None -- the browser guard both of
    knurlogic's servers (the model server and `knurlogic ui`) apply:

      Host    must name this machine, or be allowed (--allow-host): a
              foreign domain re-pointed at this address (DNS rebinding)
              would otherwise look same-origin to the browser.
      Origin  a browser marks its requests with one; answered only when it
              is this server itself or allowed (--allow-origin). Ordinary
              clients (SDKs, curl, harnesses) send none."""
    host = (headers.get("Host") or "").strip()
    if host and not host_is_local(host, allow_hosts):
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        return (f"Host {host!r} is not this machine (DNS-rebinding guard). "
                f"If you reach it by that name, start it with "
                f"--allow-host {name}.")
    o = (headers.get("Origin") or "").rstrip("/")
    if o and o not in (f"http://{host}", f"https://{host}") \
            and o not in {x.rstrip("/") for x in allow_origins}:
        return (f"requests from the web page at {o} are not answered: a "
                f"page in a browser must not be able to drive this server. "
                f"Start it with --allow-origin {o} to allow that page.")
    return None


def host_is_local(host_header: str, allow_hosts=()) -> bool:
    """Does a Host header name this machine: localhost, an IP literal, a
    `.local` / `.localhost` name, this machine's hostname, or a name the
    operator allowed?"""
    h = host_header.strip()
    if h.startswith("["):                       # [::1]:8080
        h = h[1:h.find("]")] if "]" in h else h
    elif h.count(":") == 1:
        h = h.split(":", 1)[0]
    h = h.lower().rstrip(".")
    if h in {x.lower() for x in allow_hosts}:
        return True
    if h in ("localhost",) or h.endswith((".localhost", ".local")):
        return True
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    # This machine's own name, exactly. Never "the first label matches":
    # my-mac.attacker.example would pass that, and a name an attacker
    # controls can be pointed at 127.0.0.1 (DNS rebinding) -- the Origin
    # check compares against this same Host, so it would pass too. A name
    # this machine is reached by (a tailnet, a LAN domain) is --allow-host.
    return h in _machine_names()


_NAMES: set = set()


def _machine_names() -> set:
    """This machine's hostname, once. Not getfqdn(): that is a DNS lookup
    (seconds on a network without reverse DNS), and `.local` names are
    accepted above anyway."""
    if not _NAMES:
        n = socket.gethostname().lower()
        _NAMES.update({n, n.split(".")[0]})
    return _NAMES
