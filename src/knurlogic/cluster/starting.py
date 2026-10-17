"""Start: one page starts the ranks it prepared.

The rank's command line, environment and bell (rank 0's address the
others ring), one start at a time, after the last job's ranks here are
gone.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from pathlib import Path

from knurlogic.cluster import here, recovery, tables, watching
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    Started,
    typed,
)
from knurlogic.machine import servers
from knurlogic.machine.artifact import bytes_on_disk
from knurlogic.machine.processes import PROC_ERRORS

#: how long start waits for a stopped rank's process on this machine to be
#: gone before it refuses: two jobs' shares in one working set is an OOM
#: that can reboot the machine (a 397B share loading beside the last job's
#: 50 GiB)
START_WAIT_S = 20.0


def rank_argv(path: str, spec: dict, files: dict) -> list:
    """`knurlogic serve` for one rank, with the hidden ring flags."""
    from knurlogic.machine.servers import serve_argv
    cmd = serve_argv(
        path,
        "--rank", str(spec["rank"]), "--world", str(spec["world"]),
        "--split", spec["split"], "--link", spec["link"],
        "--job", spec["job"],
        "--prefill-chunk", str(spec["prefill_chunk"]),
        "--prefill-why", str(spec.get("prefill_why") or ""),
        "--working-set-gib", f"{float(spec.get('working_set_gib') or 0):.3f}",
        "--tune", spec.get("tune") or "default")
    if spec.get("port"):
        cmd += ["--port", str(int(spec["port"]))]
    if spec.get("serve_hosts") and spec["rank"] == 0:
        # the leader answers the job's other machines on its link
        # address, and this one on loopback -- never every interface
        cmd += ["--host", ",".join(spec["serve_hosts"])]
    if spec["link"] == "ring":
        cmd += ["--hosts", ",".join(spec["hosts"])]
    else:
        cmd += ["--ibv-devices", files["ibv"],
                "--coordinator", spec["coordinator"]]
    if spec.get("layers"):
        cmd += ["--layers", ",".join(str(int(x)) for x in spec["layers"])]
    if spec.get("bandwidth_gbs"):
        cmd += ["--bandwidth-gbs", str(float(spec["bandwidth_gbs"]))]
    for k, v in sorted((spec.get("sets") or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if spec.get("chips"):
        # every rank's chip: per-chip rounding off names them in status
        # and the log (engine/matmul_padding.resolve)
        cmd += ["--ring-chips", json.dumps(spec["chips"])]
    return cmd


#: the argv builder; tests put a fake rank here
RANK_ARGV = [rank_argv]


#: the bell's port: rank 0's ring port + this (a slot's ranks take +0..+17
#: of it here, the jaccl coordinator +19)
BELL_OFFSET = 18


def bell_nonce() -> int:
    """A job's bell nonce (bell_address): random, never derived from the
    job id the page shows."""
    return secrets.randbits(62)


def bell_address(spec: dict) -> str:
    """"host:port:nonce:world" of rank 0's bell (engine/split/link.init):
    every rank connects to it over TCP BEFORE the ring is joined, so the
    ranks enter jaccl's first collectives together -- a rank that arrives
    seconds late (a slower chip, a longer start) can have rank 0's first
    RDMA message dropped. "" when the job has no ring hosts or more ranks
    than a slot leaves room for (the bell's port then comes over the ring,
    as before)."""
    hosts = spec.get("hosts") or []
    world = int(spec.get("world") or len(hosts))
    if not hosts or world < 2 or world > BELL_OFFSET:
        return ""
    nonce = spec.get("bell_nonce")
    if not isinstance(nonce, int) or nonce <= 0:
        return ""                 # a spec from a page without it: as before
    host, port = hosts[0].rsplit(":", 1)
    return f"{host}:{int(port) + BELL_OFFSET}:{nonce}:{world}"


def rank_env(spec: dict, files: dict, selfheal: bool) -> dict:
    # MLX_METAL_FAST_SYNCH: the GPU hands each collective to the CPU (and
    # takes it back) by a spinning shared event, not a command-buffer
    # completion. Measured M4 + M3 Ultra over Thunderbolt, 35B-A3B VQ split
    # two ways, one decode step (80 all_sums): jaccl 71 -> 18.5 ms, ring
    # 94 -> 25 ms. exo sets it for every runner.
    env = {"PYTHONUNBUFFERED": "1", "MLX_RANK": str(spec["rank"]),
           "MLX_METAL_FAST_SYNCH": "1"}
    bell = bell_address(spec)
    if bell:
        env["KNURLOGIC_BELL"] = bell
    from knurlogic.cluster import weights
    env.update(weights.rank_env(spec.get("read")))
    if spec["link"] == "ring":
        env["MLX_HOSTFILE"] = files["hostfile"]
    else:
        env["MLX_IBV_DEVICES"] = files["ibv"]
        env["MLX_JACCL_COORDINATOR"] = spec["coordinator"]
        if selfheal and spec.get("jaccl_timeout_ms"):
            # 0 while loading -- a cold read is not a hang -- and the
            # deadline once the model is in (engine/split/marker.after_load)
            env["JACCL_COLLECTIVE_TIMEOUT_MS"] = "0"
            env["KNURLOGIC_JACCL_TIMEOUT_MS"] = str(int(
                spec["jaccl_timeout_ms"]))
    return env


def start(job: str | None, *, spawn=None, wait_s: float | None = None,
          ring: dict | None = None) -> tuple:
    """A Start message: spawn this page's prepared rank -- once
    every stopped rank on this machine is gone (up to START_WAIT_S), and
    never beside another job's rank still loading."""
    key = str(job or "")
    with tables.LOCK:
        prep = tables.PREPARED.get(key)
    c = (ring or {}).get("prefill_chunk")
    if prep is not None and isinstance(c, int) and not isinstance(c, bool) \
            and c > 0:
        # the ring's chunk, known only once every rank answered Prepare
        prep["spec"] = dict(prep["spec"], prefill_chunk=c,
                            prefill_why=str((ring or {}).get("prefill_why")
                                            or "")[:120])
    reads = (ring or {}).get("reads")
    if prep is not None and isinstance(reads, list) and \
            len(reads) == prep["spec"]["world"]:
        # this rank's entry of the job's read plan (cluster/weights.py)
        prep["spec"] = dict(prep["spec"], read=reads[prep["spec"]["rank"]])
    if prep is None:
        return _started_already(key)
    mine: dict = prep
    with tables.START_LOCK:
        # The job stays prepared until _start has written its registry
        # record (or refused, or failed): a peer page asking JobState
        # in between sees it held, never "no rank" (docs/design/liveness.md)
        try:
            with tables.LOCK:
                if tables.PREPARED.get(key) is not mine:
                    # a retried Start that waited on the first one
                    return _started_already(key)
                mine["starting"] = True
            return _start(mine, spawn, START_WAIT_S if wait_s is None
                          else wait_s)
        finally:
            with tables.LOCK:
                if tables.PREPARED.get(key) is mine:
                    tables.PREPARED.pop(key)


def _started_already(job: str) -> tuple:
    """A retried Start (the answer was lost): the rank is already here, or
    the job is not prepared here."""
    again = _running_here(job)
    if again:
        return 200, typed(Started(job=job, rank=again[0], pid=again[1],
                                  log=again[2]))
    return 404, {"error": f"no prepared job {job[:40]!r} here"}


def _running_here(job: str):
    """(rank, pid, log) of this machine's live, not-stopping rank of `job`,
    or None."""
    for rec in J.by_job().get(job, []):
        try:
            if not rec.get("stopping") and here.alive(job, int(rec["pid"])):
                return int(rec["rank"]), int(rec["pid"]), rec.get("log")
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _start(prep: dict, spawn, wait_s: float) -> tuple:
    spec, path = prep["spec"], prep["path"]
    from knurlogic.machine import identity
    me = identity.identity().get("name") or "this machine"
    going = here.exiting(J.registry())
    left = J.wait_gone(list(going), wait_s,
                       alive=lambda p: here.alive(going[p], p)) if going else []
    if left:
        return 409, {"error": f"{me} refuses rank {spec['rank']}: "
                              + ", ".join(f"job {going[p]}'s rank (pid {p})"
                                          for p in left)
                              + f" was stopped and is still exiting after "
                                f"{wait_s:.0f} s; its memory is not free"}
    why = here.loading_elsewhere(spec["job"], J.registry())
    if why:
        return 409, {"error": f"{me} refuses rank {spec['rank']}: {why}"}
    conflict = here.slot_conflict(spec, J.registry())
    if conflict:
        j, r, pid = conflict
        left = J.wait_gone([pid], wait_s, alive=lambda p: here.alive(j, p))
        if left:
            return 409, {"error": f"{me} refuses rank {spec['rank']}: rank "
                                  f"{r} of job {j} is still exiting on "
                                  f"{me} and holds this job's ring port; "
                                  f"its memory is not free"}
    d = J.job_dir(spec["job"])
    files = {"hostfile": str(d / f"hostfile-rank{spec['rank']}.json"),
             "ibv": str(d / "ibv-devices.json")}
    if spec["link"] == "ring":
        Path(files["hostfile"]).write_text(json.dumps(
            [[h] for h in spec["hosts"]]))
    else:
        Path(files["ibv"]).write_text(json.dumps(spec["ibv_devices"]))
    log = d / f"rank{spec['rank']}.log"
    try:
        heal = bool(here.local_info().get("jaccl_selfheal"))
    except (OSError, ValueError, AttributeError, TypeError):
        heal = False
    cmd = RANK_ARGV[0](path, spec, files)
    env = {**os.environ, **rank_env(spec, files, heal)}
    try:
        with open(log, "w") as fh:
            proc = (spawn or subprocess.Popen)(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                start_new_session=True)
    except (*PROC_ERRORS, ValueError) as e:
        return 500, {"error": f"{type(e).__name__}: {e}"}
    rec = {"job": spec["job"], "rank": spec["rank"], "world": spec["world"],
           "pid": proc.pid, "artifact": path, "log": str(log),
           # the model by identity: each rank's folder name may differ
           "identity": spec.get("identity") or "",
           "split": spec["split"], "link": spec["link"],
           "machines": [n.get("name") for n in spec["nodes"]],
           # the machine THIS rank runs on, so a stop reason names it --
           # the reason is propagated to every page of the job
           "machine": spec["nodes"][spec["rank"]].get("name")
           if spec["rank"] < len(spec["nodes"]) else None,
           "leader": spec["nodes"][0].get("name"),
           # every machine of the job by id, so the watcher can ask their
           # pages -- after a page restart too (SPECS is in memory)
           "nodes": [{"rank": n.get("rank"), "id": n.get("id"),
                      "name": n.get("name")} for n in spec["nodes"]],
           # measured once at launch: the page's polls read this, not
           # the model folder
           "bytes": bytes_on_disk(path),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "t": time.time()}
    if spec.get("port") and spec["rank"] == 0:
        rec["port"] = int(spec["port"])
    try:
        rec["ring_port"] = int(spec["hosts"][spec["rank"]].rsplit(":", 1)[1])
    except (IndexError, ValueError, TypeError, AttributeError):
        pass
    with tables.LOCK:
        tables.PROCS[(spec["job"], spec["rank"])] = proc
        tables.SPECS[spec["job"]] = spec
        J.update_registry(lambda reg: reg.update(
            {f"{spec['job']}/{spec['rank']}": rec}))
        if "port" in rec:
            served = {"pid": proc.pid, "artifact": path, "log": str(log),
                      "job": spec["job"], "bytes": rec["bytes"],
                      "started": rec["started"], "t": rec["t"]}
            servers.update_registry(lambda sreg: sreg.update(
                {rec["port"]: served}))
    if "port" in rec:
        # this job's recovery row (a relaunch carries it; a launch by
        # somebody clears it), for rank 0's own /v1/residency
        recovery.write_port(rec["port"], spec.get("recovery"))
    watching.ensure_watcher()
    return 200, typed(Started(job=spec["job"], rank=spec["rank"],
                              pid=proc.pid, log=str(log)))
