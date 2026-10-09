"""The `knurlogic serve` children the page and the MCP start, poll and stop.

One home for both: the page's load button and the MCP's `load`/`unload`/
`state` start a model in its own process (`spawn`), list every server
knurlogic started with its phase (`children`, `loading`) and stop one
(`stop`, waiting until its memory is back). `SERVE_PORT` is the port a
model is launched on by default (and the page's own, "ui").
"""

from __future__ import annotations

import http.client
import subprocess
import sys
import time
from pathlib import Path

from knurlogic.machine.memory import footprint
from knurlogic.machine.servers import is_our_server, registry, save_registry, serve_log

#: Children started from the page: {port: (Popen, artifact path)}.
_CHILDREN: dict = {}


#: the page serves requests on threads: check-the-port-then-spawn is one step
_SPAWN_LOCK = __import__("threading").Lock()


#: The port a knurlogic on ANOTHER node is expected to answer on, which is
#: the one this page launches models on.
SERVE_PORT: dict = {"n": 8080, "ui": 8899}


#: A loading server whose log has said nothing for this long is reported as
#: stalled. Not killed -- a 400 GB rung read cold can be slow and silent --
#: but named, so an agent stops waiting and a person looks at the log.
STALL_QUIET_S = 180


#: A server holding less than this share of its weights is still warming.
WARM_FRACTION = 0.9


def _artifact_bytes(path: str) -> int:
    try:
        from knurlogic.machine.artifact import Artifact
        return int(Artifact.load(path).bytes_on_disk)
    except (OSError, ValueError, AttributeError):
        return 0


def _answers(port: int) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
        return True
    except (OSError, http.client.HTTPException):
        return False


def children() -> list:
    """Every server knurlogic started, from any session, with its PHASE.

    `alive` alone was the trap: loading, serving and hung all read `alive:
    true`, and an agent that cannot tell them apart either guesses or waits
    forever. So each server says:

      warming   the port answers but the weights are not resident yet
                (mlx maps them lazily); memory is still moving
      serving   answering, and holding its weights
      loading   alive, not answering yet; with seconds elapsed, the last log
                line, and how long the log has been quiet
      stalled   loading, and the log has been quiet for STALL_QUIET_S --
                stop waiting and read the log
      exited    gone; exit code when this process started it, log tail
    """
    now = time.time()
    try:
        pids = {r["pid"]: r["bytes"] for r in footprint.memory_map()["processes"]}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pids = {}
    out = []
    for port, rec in sorted(registry().items()):
        pid = int(rec["pid"])
        mine = _CHILDREN.get(port)
        code = mine[0].poll() if mine and mine[0].pid == pid else None
        alive = code is None and is_our_server(pid)
        log = Path(rec.get("log", ""))
        try:
            with log.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                lines = [ln for ln in f.read().decode(errors="replace")
                         .splitlines() if ln.strip()]
            quiet = now - log.stat().st_mtime
        except OSError:
            lines, quiet = [], None
        row = {"port": port, "artifact": rec.get("artifact"), "pid": pid,
               "log": str(log), "started": rec.get("started"),
               "seconds_since_start": (round(now - rec["t"]) if rec.get("t")
                                       else None)}
        if alive and _answers(port):
            held = pids.get(pid, 0)
            # the size the launch measured: a poll reads no model folder
            size = int(rec.get("bytes") or 0)
            row["bytes_resident"] = held
            # Answering is not loaded. mlx maps weights lazily: measured, a
            # 15.5 GiB model answered with 3 GiB resident and reached 15.0
            # seconds later. Until it holds its weights, memory is still
            # moving, and a fit taken now is stale.
            if size and held < WARM_FRACTION * size:
                row["phase"] = "warming"
                row["weights_resident_fraction"] = round(held / size, 2)
            else:
                row["phase"] = "serving"
        elif alive:
            stalled = quiet is not None and quiet > STALL_QUIET_S
            row["phase"] = "stalled" if stalled else "loading"
            row["log_quiet_seconds"] = round(quiet) if quiet is not None else None
            row["last_log_line"] = lines[-1][:200] if lines else ""
            row["bytes_resident"] = pids.get(pid, 0)
            if stalled:
                row["advice"] = (f"no log output for {round(quiet or 0)}s while "
                                 f"loading. Stop waiting; read {log}. "
                                 f"unload(port={port}) if it is hung.")
        else:
            row["phase"] = "exited"
            if code is not None:
                row["exit_code"] = code
            row["log_tail"] = lines[-15:]
            # a server that refused to start says why in its log: carried
            # up as `refused`, the page's and state()'s reason
            from knurlogic.cluster.recovery import refusal_line
            why = refusal_line("\n".join(lines[-60:]))
            if why:
                row["refused"] = why
        out.append(row)
    return out


def loading() -> list:
    """Servers knurlogic started whose memory is still in motion."""
    return [c for c in children()
            if c["phase"] in ("loading", "warming", "stalled")]


def spawn(*a, **k):
    """_spawn_unlocked under the lock: two loads for one port at once would
    both pass the registry check, and the second child's record would hide
    the first (still loading, holding its memory, invisible to unload)."""
    with _SPAWN_LOCK:
        return _spawn_unlocked(*a, **k)


def _spawn_unlocked(path: str, port: int, tune: str = "default",
           sets: dict | None = None, draft: bool = True) -> dict:
    """Start `knurlogic serve` for one artifact, on its own port.

    Deliberately a child process rather than an in-process load: the
    settings that matter are resolved and put in the ENVIRONMENT before the
    engine imports anything, and that cannot be done to a process that is
    already running. A fresh process is the only way the knobs are honoured
    in full -- which is the difference between this and switching a model
    inside a running server.
    """
    import os
    if not Path(path).exists():
        return {"error": f"no such artifact: {path}"}
    rec = registry().get(port)
    if rec and is_our_server(int(rec["pid"])):
        return {"error": f"port {port} is already serving "
                         f"{rec.get('artifact')} (pid {rec['pid']})"}
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--port", str(port), "--tune", tune]
    # Settings chosen at LAUNCH, which for most of these is the only moment
    # they can be chosen: they are read at import and compiled into kernel
    # source, so a running server cannot be told about them.
    for k, v in sorted((sets or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if not draft:
        cmd.append("--no-draft")
    log = serve_log(port)
    from knurlogic.machine.servers import new_instance
    instance = new_instance()
    try:
        with open(log, "w") as fh:
            # Its own session, so it is not taken down with the terminal or
            # the MCP client that asked for it.
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    start_new_session=True)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    _CHILDREN[port] = (proc, path)
    reg = registry()
    reg[port] = {"pid": proc.pid, "artifact": path, "log": str(log),
                 # measured once here (a launch is the user acting); the
                 # polls read it from this record, never the model folder
                 "bytes": _artifact_bytes(path),
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "t": time.time(), "instance": instance}
    save_registry(reg)
    return {"starting": path, "port": port, "pid": proc.pid,
            "log": str(log), "instance": instance,
            "note": "the model is loading in its own process; poll `state` "
                    "(started_here) or GET /v1/models on the port"}


def stop(port: int) -> dict:
    import os
    import signal

    from knurlogic.cluster import recovery
    recovery.cancel_port(port)          # asked for: never recovered
    reg = registry()
    rec = reg.get(port)
    if not rec:
        return {"error": f"knurlogic has no record of a server on port "
                         f"{port}; it stops only what it started"}
    if rec.get("job"):
        # rank 0 of a cluster job: the job stops, on every machine
        from knurlogic.cluster import launch
        out = launch.stop(rec["job"], reason="unloaded")
        return {**out, "stopped": rec.get("artifact"), "port": port}
    pid = int(rec["pid"])
    if not is_our_server(pid):
        reg.pop(port, None)
        save_registry(reg)
        _CHILDREN.pop(port, None)
        return {"error": f"the server on port {port} (pid {pid}) is already "
                         f"gone; record cleared", "log": rec.get("log")}
    os.kill(pid, signal.SIGTERM)
    # it finishes the step it is in and its scheduler's cleanup on the way
    # out: the same ceiling as its own (http.STOP_SAVE_S)
    from knurlogic.interfaces.http import STOP_SAVE_S
    end = time.monotonic() + STOP_SAVE_S + 5.0
    while time.monotonic() < end:
        time.sleep(0.25)
        if not is_our_server(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
    mine = _CHILDREN.get(port)
    # answer when the process is gone and its memory is back, not when the
    # signal was sent: a caller that loads next must see the memory free
    gone = _wait_exit(pid, mine[0] if mine else None)
    if not gone:
        return {"stopped": rec.get("artifact"), "port": port, "pid": pid,
                "exiting": [pid],
                "note": f"pid {pid} is still exiting after "
                        f"{EXIT_WAIT_S:.0f} s; its memory is not free yet"}
    _CHILDREN.pop(port, None)
    reg.pop(port, None)
    save_registry(reg)
    return {"stopped": rec.get("artifact"), "port": port, "pid": pid}


#: how long an unload waits for the server process to be gone
EXIT_WAIT_S = 60.0


def _wait_exit(pid: int, proc=None, wait_s: float | None = None) -> bool:
    """True once `pid` has exited (reaped when `proc` is ours), polling up
    to `wait_s` (EXIT_WAIT_S)."""
    end = time.time() + (EXIT_WAIT_S if wait_s is None else wait_s)
    while True:
        if proc is not None:
            if proc.poll() is not None:
                return True
        elif not is_our_server(pid):
            return True
        if time.time() >= end:
            return False
        time.sleep(0.1)
