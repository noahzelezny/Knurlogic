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
    try:
        return {int(k): v for k, v in
                json.loads(registry_path().read_text()).items()}
    except Exception:
        return {}


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
    return "knurlogic" in out and " serve " in f" {out} "


