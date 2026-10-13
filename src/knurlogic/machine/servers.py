"""Which knurlogic servers are running on this box -- the record.

A fact about the machine, not about any interface: the MCP, the page and the
CLI all read it, and `loaded.survey()` reads it to find servers on ports it
would never have guessed. Before it did, two models holding 33 GiB on ports
8092 and 8093 were invisible to the page and to `state` alike.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from knurlogic.machine import processes


def cache_dir() -> Path:
    import os
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    root.mkdir(parents=True, exist_ok=True)
    return root


def serve_argv(path: str, *args: str) -> list:
    """The `knurlogic serve` command line for a model at `path`: the one
    builder the page/MCP spawn (interfaces/spawn) and a cluster rank
    (cluster/launch.rank_argv) share. `args` follow the path as given."""
    return [sys.executable, "-m", "knurlogic", "serve", path, *args]


def new_instance() -> str:
    """A 16-hex id for a server this box starts -- what exo calls an
    instance id, so two servers, even one on this Mac and one on a peer
    both answering on :8080, are never confused with each other. A cluster
    job needs none of this: its job id already is its instance id."""
    import secrets
    return secrets.token_hex(8)


def serve_log(port: int) -> Path:
    """Where a server started from here writes. A child whose output went
    to /dev/null could crash on load and leave its caller holding
    'starting' forever, with no way to learn why."""
    return cache_dir() / f"serve-{port}.log"


# --- servers outlive the session that started them --------------------------
# An agent's MCP session ends; the model it loaded should not, and neither
# should its ability to stop it. Measured: a server started through the MCP
# kept running after the session closed, and the next session could not
# unload it because the only record was a dict in the dead process. So the
# record is a file, and `CHILDREN` only keeps the Popen for exit codes.

def registry_path() -> Path:
    return cache_dir() / "servers.json"


def registry() -> dict:
    """port -> record, for records that name a port and a numeric pid; a
    hand-edited or half-written entry is skipped, not a KeyError later."""
    try:
        raw = json.loads(registry_path().read_text())
    except (OSError, ValueError):
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
    out = processes.command(pid)
    return "knurlogic" in out and "serve" in out.split()


def listener_pid(port: int) -> int | None:
    """The pid listening on `port` on this machine, or None."""
    return processes.listener(port)


#: what marks a process a TEST started, never a real server: the flag the
#: suite's fake ranks carry (tests/cluster_fake_page.fake_argv), and
#: pytest's temp directories in its arguments. A real page leaving these
#: out is what keeps a leaked fake rank from being listed as a model.
TEST_MARKS = ("--knurlogic-test", "/pytest-of-")


def is_test_process(cmd: str) -> bool:
    return any(m in (cmd or "") for m in TEST_MARKS)


def listening_serves() -> dict:
    """port -> pid for every `knurlogic serve` on this box that is listening,
    whoever started it. The registry holds only what the page launched; a
    serve started by hand in a terminal is just as loaded, and leaving it
    out made the page say "nothing loaded" beside 110 GiB of model.
    Read-only (machine/processes)."""
    pids = [pid for pid, cmd in processes.commands().items()
            if "knurlogic" in cmd and "serve" in cmd.split()
            and not is_test_process(cmd)]
    return processes.listening(pids) if pids else {}


def port_free(port: int) -> bool:
    """Bindable here, and not held by a live knurlogic server of ours."""
    import socket
    rec = registry().get(port)
    if rec and is_our_server(int(rec["pid"])):
        return False
    for host in ("127.0.0.1", "0.0.0.0"):
        with socket.socket() as s:
            try:
                s.bind((host, int(port)))
            except OSError:
                return False
    return True


def free_port(start: int = 8080, taken=()) -> int:
    """The first port from `start` up that this machine can serve on."""
    for p in range(max(1024, int(start)), 65536):
        if p not in taken and port_free(p):
            return p
    raise OSError("no free port")
