"""Whether this machine takes load/unload requests from another machine's
page, and the secret that says a request came from a page it trusts.

A peer's id is what the peer SAYS it is (cluster/peers.py: the
introduction header is self-declared), so an allow-list of ids
authenticates nothing. What does: a random launch token this machine makes
for itself, shown only on its own page (Settings -> Cluster, reached over
loopback), and pasted once into the other machine's page, which sends it
with every forwarded load. Accepting is OFF until turned on here, on this
machine's own page.

Kept in ~/.config/knurlogic/cluster.json (XDG_CONFIG_HOME honoured), mode
0600: a few hundred bytes of settings, not data.

  launch_token     this machine's token (made on first read)
  accept_launches  false by default
  peer_tokens      {peer id: the token that peer showed on its page}
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
from pathlib import Path

#: the header a forwarded load carries its token in
HEADER = "X-Knurlogic-Launch-Token"

_LOCK = threading.Lock()


def path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME",
                               Path.home() / ".config")) / "knurlogic"
    return root / "cluster.json"


def _read() -> dict:
    try:
        d = json.loads(path().read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _write(d: dict) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(d, indent=1) + "\n")
    os.replace(tmp, p)


def _doc() -> dict:
    """The stored document, with a token made (and saved) if there is none."""
    d = _read()
    if not isinstance(d.get("launch_token"), str) or \
            len(d["launch_token"]) < 32:
        d["launch_token"] = secrets.token_urlsafe(24)
        d.setdefault("accept_launches", False)
        _write(d)
    return d


def token() -> str:
    with _LOCK:
        return _doc()["launch_token"]


def accepting() -> bool:
    return _read().get("accept_launches") is True


def set_accepting(on: bool) -> bool:
    with _LOCK:
        d = _doc()
        d["accept_launches"] = bool(on)
        _write(d)
        return d["accept_launches"]


def regenerate() -> str:
    """A new token; every page that had the old one must be given this."""
    with _LOCK:
        d = _doc()
        d["launch_token"] = secrets.token_urlsafe(24)
        _write(d)
        return d["launch_token"]


def check(presented) -> bool:
    """Is `presented` this machine's token? Constant-time."""
    if not isinstance(presented, str) or not presented:
        return False
    return hmac.compare_digest(presented.encode(), token().encode())


def peer_token(peer_id: str) -> str:
    t = (_read().get("peer_tokens") or {}).get(peer_id or "")
    return t if isinstance(t, str) else ""


def set_peer_token(peer_id: str, tok: str) -> None:
    """Remember `tok` for the peer `peer_id` ("" forgets it)."""
    if not isinstance(peer_id, str) or not peer_id or len(peer_id) > 64:
        raise ValueError("a peer id is needed")
    tok = (tok or "").strip()
    if len(tok) > 256:
        raise ValueError("that is not a launch token")
    with _LOCK:
        d = _doc()
        pt = d.get("peer_tokens") if isinstance(d.get("peer_tokens"),
                                                dict) else {}
        if tok:
            pt[peer_id] = tok
        else:
            pt.pop(peer_id, None)
        d["peer_tokens"] = pt
        _write(d)


def public(own: bool) -> dict:
    """What the page shows. The token itself only when `own` (the request
    came over loopback: this machine's own page)."""
    d = _read()
    out = {"accept_launches": d.get("accept_launches") is True,
           "peers_with_token": sorted((d.get("peer_tokens") or {}).keys()),
           "file": str(path())}
    if own:
        out["launch_token"] = token()
    return out
