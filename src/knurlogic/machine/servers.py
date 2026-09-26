"""Which knurlogic servers are running on this box -- the record.

A fact about the machine, not about any interface: the MCP, the page and the
CLI all read it, and `loaded.survey()` reads it to find servers on ports it
would never have guessed. Before it did, two models holding 33 GiB on ports
8092 and 8093 were invisible to the page and to `state` alike.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _cache_dir() -> Path:
    import os
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    root.mkdir(parents=True, exist_ok=True)
    return root


def serve_log(port: int) -> Path:
    """Where a server started from here writes. A child whose output went
    to /dev/null could crash on load and leave its caller holding
    'starting' forever, with no way to learn why."""
    return _cache_dir() / f"serve-{port}.log"


# --- servers outlive the session that started them --------------------------
# An agent's MCP session ends; the model it loaded should not, and neither
# should its ability to stop it. Measured: a server started through the MCP
# kept running after the session closed, and the next session could not
# unload it because the only record was a dict in the dead process. So the
# record is a file, and `_CHILDREN` only keeps the Popen for exit codes.

def registry_path() -> Path:
    return _cache_dir() / "servers.json"


def registry() -> dict:
    """port -> record, for records that name a port and a numeric pid; a
    hand-edited or half-written entry is skipped, not a KeyError later."""
    try:
        raw = json.loads(registry_path().read_text())
    except Exception:
        return {}
    out = {}
    for k, v in (raw.items() if isinstance(raw, dict) else ()):
        try:
            port = int(k)
            v = dict(v, pid=int(v["pid"]))
        except (TypeError, ValueError, KeyError):
            continue
        out[port] = v
    return out


def save_registry(reg: dict) -> None:
    registry_path().write_text(json.dumps(
        {str(k): v for k, v in sorted(reg.items())}, indent=1))


def is_our_server(pid: int) -> bool:
    """Alive AND still a knurlogic serve. A pid is reused once its process
    is gone; killing whatever inherited the number would be the bug."""
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return False
    return "knurlogic" in out and "serve" in out.split()




def listening_serves() -> dict:
    """port -> pid for every `knurlogic serve` on this box that is listening,
    whoever started it. The registry holds only what the page launched; a
    serve started by hand in a terminal is just as loaded, and leaving it
    out made the page say "nothing loaded" beside 110 GiB of model.
    Read-only (ps, lsof); an empty answer when either is unavailable."""
    try:
        ps = subprocess.run(["ps", "-axo", "pid=,command="],
                            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    pids = []
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if "knurlogic" in cmd and "serve" in cmd.split() and pid.isdigit():
            pids.append(pid)
    if not pids:
        return {}
    try:
        out = subprocess.run(["lsof", "-nP", "-a", "-iTCP", "-sTCP:LISTEN",
                              "-p", ",".join(pids), "-Fpn"],
                             capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return {}
    found, pid = {}, 0
    for line in out.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        elif line.startswith("n") and pid:
            port = line.rsplit(":", 1)[-1]
            if port.isdigit():
                found.setdefault(int(port), pid)
    return found
