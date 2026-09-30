"""The macOS menu-bar icon: a gear, a count of loaded models, a way to open or
quit the page.

It runs as a small CHILD process of `knurlogic ui` (`python -m
knurlogic.interfaces.menubar`): rumps wants the main thread's run loop, and a
child keeps the page's own shutdown untouched. The child talks to the page over
HTTP (/loaded.json), exits when the page does, and quits the page the way
Ctrl-C does (SIGINT). One icon per machine: the child holds a lock file.
The gear is the page logo's (views/logo.js), rendered to a template PNG.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

REFRESH_S = 5.0
ASSETS = Path(__file__).parent / "menubar_assets"


def lock_path() -> Path:
    env = os.environ.get("KNURLOGIC_MENUBAR_LOCK")
    if env:
        return Path(env)
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    return root / "menubar.lock"


def acquire_lock(path: Path | None = None):
    """The open lock file (keep it referenced: closing releases), or None when
    another icon already holds it."""
    path = path or lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def gui_available(env=None, platform: str | None = None, uid: int | None = None,
                  console_owner=None) -> bool:
    """A logged-in console session to draw in: macOS, not over ssh, and the
    console belongs to this user."""
    env = os.environ if env is None else env
    if (platform or sys.platform) != "darwin":
        return False
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return False
    uid = os.getuid() if uid is None else uid
    try:
        owner = (console_owner or (lambda: os.stat("/dev/console").st_uid))()
    except OSError:
        return False
    return bool(owner == uid)


def menu_model(doc: dict | None, host: str = "") -> dict:
    """{'title', 'models': [line, ...]} from a /loaded.json document; a page
    that does not answer (None) reads as zero."""
    rows = [r for r in ((doc or {}).get("resident") or [])
            if r.get("state", "loaded") == "loaded"]
    n = len(rows)
    lines = []
    for r in rows:
        where = r.get("machine") or host
        lines.append(f"{r.get('name', '?')} · {where}" if where
                     else str(r.get("name", "?")))
    # rumps keys menu items by title and drops a repeat, so two identical
    # lines would collapse into one: tell repeats apart by port (or a count)
    seen = {ln: lines.count(ln) for ln in lines}
    for k, r in enumerate(rows):
        if seen[lines[k]] > 1:
            port = str(r.get("where") or "").rpartition(":")[2]
            lines[k] += f" :{port}" if port.isdigit() else f" ({k + 1})"
    return {"title": f"Knurlogic — {n} model{'s' if n != 1 else ''} loaded",
            "models": lines}


def spawn(port: int, enabled: bool = True) -> bool:
    """Start the icon as a child of this page. False when skipped."""
    if not enabled or not gui_available():
        return False
    try:
        subprocess.Popen(
            [sys.executable, "-m", "knurlogic.interfaces.menubar",
             "--port", str(port), "--parent", str(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return False
    return True


def _fetch(port: int):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/loaded.json", timeout=2) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(port: int, parent: int) -> int:
    lock = acquire_lock()
    if lock is None:
        return 0
    try:
        import rumps
    except ImportError:
        return 0
    import socket
    host = socket.gethostname().split(".")[0]
    icon = ASSETS / "gear.png"
    app = rumps.App("Knurlogic", title=None if icon.exists() else "⚙",
                    icon=str(icon) if icon.exists() else None, template=True,
                    quit_button=None)
    state = {"models": 0}

    def open_page(_):
        subprocess.Popen(["open", f"http://127.0.0.1:{port}/"])

    def quit_page(_):
        if state["models"] and rumps.alert(
                title="Quit Knurlogic?",
                message="Quitting stops the page and the models it started.",
                ok="Quit", cancel="Cancel") != 1:
            return
        try:
            os.kill(parent, signal.SIGINT)    # what Ctrl-C does
        except OSError:
            pass
        rumps.quit_application()

    def refresh(_=None):
        if not _alive(parent):
            rumps.quit_application()
            return
        m = menu_model(_fetch(port), host)
        state["models"] = len(m["models"])
        items = [rumps.MenuItem(m["title"])]
        items += [rumps.MenuItem(x) for x in m["models"]]
        app.menu.clear()
        app.menu = [*items, None, rumps.MenuItem("Open Knurlogic", open_page),
                    None, rumps.MenuItem("Quit Knurlogic", quit_page)]

    refresh()
    rumps.Timer(refresh, REFRESH_S).start()
    app.run()
    lock.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic.interfaces.menubar")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--parent", type=int, required=True)
    a = p.parse_args(argv)
    return run(a.port, a.parent)


if __name__ == "__main__":
    sys.exit(main())
