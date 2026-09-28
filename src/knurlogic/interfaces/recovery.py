"""Auto-recovery: a model that died or stalled without being asked to stop
is brought back by the page that launched it -- bounded.

Who      the page that coordinated a cluster launch (cluster_jobs.launch
         registers the job here), and the page that started a one-Mac
         server (ui's Launch registers the port). A relaunch is the same
         launch again: the same identity, machines, rank order, split,
         link, port, tune and settings, through cluster_jobs.launch (its
         prepare checks fit, versions, links and one load at a time on
         every page; the cable failover still follows it) or mcp.load for
         one Mac (fit and memory-still-moving refusals).
Never    a requested stop (an unload from any page or the MCP, a page
         closing); a stop because the model does not fit or a machine ran
         out of memory -- relaunching into the same memory is what rebooted
         the M3 once -- which is `failed` at once, with the reason.
Waits    a machine that went away or stopped answering: the relaunch waits
         until every machine of the job answers its page again, within the
         window; else `failed`.
Limits   MAX_ATTEMPTS relaunches of a model within WINDOW_S, BACKOFF_S apart;
         after that the model is `failed`, with the last reason, until
         someone loads it again. A relaunch starts only once no rank of the
         old job is left on any of its machines (`knurlogic serve` for that
         job, by process), and a one-Mac server only once its old process
         is gone.
Switch   KNURLOGIC_RECOVER=off turns it off (on by default): failures stop
         the job and are reported, as before.

What is reported, per model: `view()` -> {attempts, last_reason, last_at,
next_at, state}, state recovering | recovered | failed; None when there is
nothing to report. The page writes each record also to this machine's
recovery.json under the serving port, and a relaunch carries it to the page
of the machine running rank 0, so that model's own /v1/residency row says
it too. What it takes to relaunch -- each tracked model's record, its
launch request and attempts -- is kept in recovery-models.json beside it,
so a page restarted mid-recovery picks up where it was (restore()).
Stdlib only: the page never imports mlx.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ENV = "KNURLOGIC_RECOVER"
#: relaunches of one model allowed within this window
WINDOW_S = 900.0
MAX_ATTEMPTS = 3
#: wait before the 1st, 2nd, 3rd relaunch
BACKOFF_S = (10.0, 30.0, 90.0)
#: how often the page looks
TICK_S = 2.0
#: a recovered model is reported as such this long, then nothing is
RECOVERED_S = WINDOW_S

#: stops somebody asked for: never recovered
REQUESTED = ("unloaded", "the page that started it closed", "refused",
             "a rank did not start", "stopped by another machine")
#: the model does not fit / the machine ran out of memory: never relaunched
MEMORY_RX = re.compile(
    r"out of memory|outofmemory|insufficient memory|does not fit|"
    r"will not fit|resource limit|unable to allocate|failed to allocate|"
    r"memoryerror|\boom\b|held now by", re.I)
#: a machine that went away or stopped answering: wait for it
MACHINE_RX = re.compile(
    r"has not answered|not a peer this page knows|is not a machine "
    r"answering|did not answer|no longer runs its rank", re.I)
#: a rank's or server's own output saying it ran out of memory
MEMORY_LINE_RX = re.compile(
    r"[^\n]*(?:out of memory|OutOfMemory|Insufficient Memory|"
    r"Resource limit|Unable to allocate|Failed to allocate|MemoryError)"
    r"[^\n]*", re.I)



def _no_load(**_):
    return {"error": "no page has set recovery.load_fn"}


# What recovery needs of the page that runs it, injected by that page at
# startup (interfaces/ui.py) so this module never imports it. Unset, there
# is nothing to relaunch with and nothing of the page's to look at.
#: () -> [peer record]: the page's PEERS store
peers_fn = list
#: port -> (Popen, artifact) | None: a server this page process started
child_fn = {}.get
#: port -> bool: that port's server answers
answers_fn = (lambda port: False)
#: (**load arguments) -> dict: start a single-Mac server (mcp.load)
load_fn = _no_load

#: key -> record (in this page process)
MODELS: dict = {}
_LOCK = threading.RLock()
_THREAD: list = []


def enabled() -> bool:
    return os.environ.get(ENV, "on").strip().lower() not in (
        "off", "0", "false", "no")


def kind(reason: str) -> str:
    """requested | memory | machine | failure."""
    r = str(reason or "").strip()
    # a peer's page relaying its own stop: "B stopped the job: unloaded"
    r = re.sub(r"^.{0,80}? stopped the job: ", "", r)
    if any(r == q or r.startswith(q + ";") or r.startswith(q + ":")
           for q in REQUESTED):
        return "requested"
    if MEMORY_RX.search(r):
        return "memory"
    if MACHINE_RX.search(r):
        return "machine"
    return "failure"


def memory_line(text: str) -> str:
    m = MEMORY_LINE_RX.search(text or "")
    return m.group(0).strip()[:240] if m else ""


# ------------------------------------------------------------ the file
# port -> view, on the machine serving that port: a server's /v1/residency
# reads its own port's row (interfaces/http/scout.residency).

def _path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    base.mkdir(parents=True, exist_ok=True)
    return base / "recovery.json"


def _models_path() -> Path:
    return _path().with_name("recovery-models.json")


#: launch arguments kept across a page restart; `post`/`follow` are the
#: tests' hooks and never written, `peers` is kept as what launch reads of
#: each (and fresher records from the peer store win on the relaunch)
_ARGS = ("me", "local_info", "ui_port", "serve_port")
_PEER = ("id", "name", "key", "state", "link", "node")


def _saved(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "args"}
    if "args" in rec:
        args = rec["args"] or {}
        out["args"] = {k: args[k] for k in _ARGS if k in args}
        out["args"]["peers"] = [{k: getattr(p, k, None) for k in _PEER}
                                for p in args.get("peers") or []]
    return out


def _peer(d: dict):
    from knurlogic.cluster.peers import Peer
    host, _, port = str(d.get("key") or "").rpartition(":")
    return Peer(host=host, port=int(port) if port.isdigit() else 0,
                id=str(d.get("id") or ""), name=str(d.get("name") or ""),
                state=str(d.get("state") or "not_answering"),
                link=str(d.get("link") or ""),
                node=d.get("node") if isinstance(d.get("node"), dict)
                else None)


#: what save() last wrote (the loop saves each tick; unchanged, it is not
#: written again)
_SAVED: dict = {}


def save() -> None:
    """Every tracked model's record, to this machine's recovery-models.json
    (what restore() reads back after a page restart)."""
    with _LOCK:
        try:
            doc = json.dumps({k: _saved(r) for k, r in MODELS.items()},
                             default=lambda o: None)
        except (TypeError, ValueError):
            return
        if doc == _SAVED.get("doc"):
            return
        tmp = _models_path().with_suffix(".tmp")
        try:
            tmp.write_text(doc)
            tmp.replace(_models_path())
            _SAVED["doc"] = doc
        except OSError:
            pass


def restore() -> list:
    """At page start: take back the models an earlier page process was
    tracking (not those this one already tracks). -> their keys."""
    try:
        d = json.loads(_models_path().read_text())
    except (OSError, ValueError):
        return []
    got = []
    with _LOCK:
        for k, r in (d.items() if isinstance(d, dict) else ()):
            if not isinstance(r, dict) or k in MODELS \
                    or r.get("kind") not in ("cluster", "single") \
                    or r.get("key") != k:
                continue
            if r["kind"] == "cluster":
                a = r.get("args") or {}
                r["args"] = dict(a, peers=[_peer(p) for p in
                                           a.get("peers") or []
                                           if isinstance(p, dict)])
            MODELS[k] = r
            got.append(k)
    if got:
        ensure_thread()
    return got


def read_file() -> dict:
    try:
        d = json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): v for k, v in (d.items() if isinstance(d, dict) else ())
            if isinstance(v, dict)}


def write_port(port, view) -> None:
    """Record (or with None, clear) the recovery row of `port` here."""
    try:
        port = int(port or 0)
    except (TypeError, ValueError):
        return
    if not port:
        return
    with _LOCK:
        d = read_file()
        if view:
            d[str(port)] = clean_view(view)
        else:
            d.pop(str(port), None)
        tmp = _path().with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(d))
            tmp.replace(_path())
        except OSError:
            pass


def clean_view(v) -> dict | None:
    """A view as another page may send it: those five fields, typed."""
    if not isinstance(v, dict):
        return None
    st = v.get("state")
    if st not in ("recovering", "recovered", "failed"):
        return None

    def num(x):
        return float(x) if isinstance(x, (int, float)) \
            and not isinstance(x, bool) else None
    a = v.get("attempts")
    return {"attempts": a if isinstance(a, int) and not isinstance(a, bool)
            and 0 <= a <= 100 else 0,
            "last_reason": str(v.get("last_reason") or "")[:600] or None,
            "last_at": num(v.get("last_at")), "next_at": num(v.get("next_at")),
            "state": st}


def served_view(port, ready: bool, now: float | None = None):
    """What a server on `port` reports: its row of this machine's file --
    recovering reads recovered once the server is ready; a recovered row
    older than RECOVERED_S is nothing."""
    now = time.time() if now is None else now
    v = clean_view(read_file().get(str(port)))
    if not v:
        return None
    if v["state"] == "recovering" and ready:
        v = dict(v, state="recovered", next_at=None)
    if v["state"] == "recovered" and v.get("last_at") \
            and now - v["last_at"] > RECOVERED_S:
        return None
    return v


# ------------------------------------------------------------ records

def view(rec: dict | None, now: float | None = None) -> dict | None:
    if not rec or not rec.get("state"):
        return None
    now = time.time() if now is None else now
    return {"attempts": len(_window(rec, now)),
            "last_reason": rec.get("last_reason"),
            "last_at": rec.get("last_at"),
            "next_at": rec.get("next_at") if rec.get("pending") else None,
            "state": rec["state"]}


def _window(rec: dict, now: float) -> list:
    return [t for t in rec.get("attempts") or [] if now - t <= WINDOW_S]


def cluster_key(identity: str, nodes) -> str:
    return f"cluster:{identity}:{','.join(sorted(map(str, nodes or [])))}"


def track_cluster(job: str, *, req: dict, args: dict, order: list,
                  port, leader_here: bool, previous: str | None = None) -> None:
    """A cluster launch succeeded on this (the coordinating) page. A
    launch by somebody starts a fresh record; the cable failover's
    relaunch (`previous`: the job it replaced) keeps the old one's."""
    key = cluster_key(req.get("identity"), req.get("nodes"))
    with _LOCK:
        rec = MODELS.get(key)
        if previous and rec and previous in (rec.get("job"),
                                             rec.get("ended_job")):
            rec.update(job=job, pending=False, ended_job=None)
            save()
            return
        MODELS[key] = {
            "key": key, "kind": "cluster", "job": job, "req": dict(req),
            "args": dict(args), "order": list(order), "port": port,
            "leader_here": leader_here, "name": req.get("identity"),
            "machines": [m.get("name") for m in order],
            "split": req.get("split"), "link": req.get("link"),
            "attempts": [], "state": None, "pending": False}
    _sync(key)
    save()
    ensure_thread()


def track_single(port: int, load: dict, pid: int | None = None) -> None:
    """The page started a one-Mac server on `port` (a launch by somebody:
    a fresh record)."""
    key = f"single:{int(port)}"
    with _LOCK:
        MODELS[key] = {"key": key, "kind": "single", "port": int(port),
                       "load": dict(load), "pid": pid, "state": None,
                       "name": Path(str(load.get("artifact") or "")).name,
                       "attempts": [], "pending": False, "leader_here": True}
    write_port(port, None)
    save()
    ensure_thread()


def cancel_job(job: str) -> None:
    """An unload of `job`: never recovered."""
    with _LOCK:
        for k in [k for k, r in MODELS.items() if r["kind"] == "cluster"
                  and job in (r.get("job"), r.get("ended_job"))]:
            _drop(k)


def cancel_port(port) -> None:
    with _LOCK:
        for k in [k for k, r in MODELS.items() if r.get("port") == port
                  and r["kind"] == "single"]:
            _drop(k)


def _drop(key: str) -> None:
    rec = MODELS.pop(key, None)
    if rec and rec.get("leader_here") and rec.get("port"):
        write_port(rec["port"], None)
    if rec:
        save()


def _sync(key: str) -> None:
    rec = MODELS.get(key)
    if rec and rec.get("leader_here") and rec.get("port"):
        write_port(rec["port"], view(rec))


def for_job(job: str):
    with _LOCK:
        rec = next((r for r in MODELS.values() if r["kind"] == "cluster"
                    and job in (r.get("job"), r.get("ended_job"))), None)
        return view(rec)


def for_port(port):
    """The view of the model tracked here that serves on `port` on THIS
    machine (a one-Mac server, or a job whose rank 0 is here)."""
    with _LOCK:
        rec = next((r for r in MODELS.values() if r.get("port") == port
                    and r.get("leader_here")), None)
        return view(rec)


def not_serving() -> list:
    """Models tracked here with no live server or job right now -- waiting
    to be relaunched, or failed -- for /loaded.json's `recovery`."""
    out = []
    with _LOCK:
        for r in MODELS.values():
            if not r.get("pending") and r.get("state") != "failed":
                continue
            out.append({"name": r.get("name"), "port": r.get("port"),
                        "machines": r.get("machines"),
                        "split": r.get("split"), "link": r.get("link"),
                        "job": r.get("ended_job") or r.get("job"),
                        "state": r["state"], "recovery": view(r)})
    return out


# ------------------------------------------------------------ the loop

def ensure_thread() -> None:
    with _LOCK:
        if _THREAD:
            return
        _THREAD.append(1)

    def loop():
        while True:
            time.sleep(TICK_S)
            try:
                tick()
            except Exception as e:
                print(f"recovery: {type(e).__name__}: {e}", file=sys.stderr,
                      flush=True)
    threading.Thread(target=loop, daemon=True,
                     name="knurlogic-recovery").start()


def tick(now: float | None = None) -> list:
    """One look at every tracked model. -> [(key, what happened)]."""
    if not enabled():
        return []
    out = []
    for key in list(MODELS):
        rec = MODELS.get(key)
        if rec is None:
            continue
        try:
            what = (_tick_cluster if rec["kind"] == "cluster"
                    else _tick_single)(rec, time.time() if now is None
                                       else now)
        except Exception as e:
            what = f"error: {type(e).__name__}: {e}"
        if what:
            out.append((key, what))
            _log(rec, what)
        if key in MODELS:
            _sync(key)
    save()
    return out


def _log(rec, what):
    print(f"recovery {rec.get('name')} "
          f"({', '.join(rec.get('machines') or []) or 'this Mac'}): {what}",
          file=sys.stderr, flush=True)


def _failed(rec: dict, now: float, why: str) -> str:
    rec.update(state="failed", pending=False, last_reason=why[:600],
               last_at=now, next_at=None)
    return f"failed: {why}"


def _schedule(rec: dict, now: float, why: str) -> str:
    """The model went down: plan the next relaunch, or give up."""
    k = kind(why)
    if k == "requested":
        _drop(rec["key"])
        return f"stopped on request ({why}); not recovered"
    if k == "memory":
        return _failed(rec, now, why)
    n = len(_window(rec, now))
    if n >= MAX_ATTEMPTS:
        return _failed(rec, now, f"{n} relaunches in {WINDOW_S / 60:.0f} min "
                                 f"did not hold; last: {why}")
    wait = BACKOFF_S[min(n, len(BACKOFF_S) - 1)]
    rec.update(state="recovering", pending=True, last_reason=why[:600],
               last_at=now, next_at=now + wait, down_at=now,
               wait_machines=k == "machine")
    return f"down ({why}); relaunch {n + 1} in {wait:.0f} s"


def _defer(rec: dict, now: float, why: str) -> str:
    """Not yet (a machine not answering, an old rank not gone): look again
    shortly -- within the window since it went down."""
    if now - float(rec.get("down_at", now)) > WINDOW_S:
        return _failed(rec, now, f"{why} for {WINDOW_S / 60:.0f} min")
    rec["next_at"] = now + TICK_S
    rec["waiting"] = why
    return ""


# ------------------------------------------------------------ cluster

def _tick_cluster(rec: dict, now: float) -> str:
    from knurlogic.interfaces import cluster_jobs as C
    if rec.get("state") == "failed":
        return ""
    if not rec.get("pending"):
        job = rec["job"]
        e = C.ENDED.get(job) or {}
        if e.get("relaunched"):           # the cable failover moved it
            rec["job"] = e["relaunched"]
            return f"followed the cable failover to job {e['relaunched']}"
        why = C._job_end(job, rec["order"], rec["args"].get("post")
                         or C._post)
        if why is None:
            if rec.get("state") == "recovering" and \
                    _cluster_phase(rec) == "ready":
                rec.update(state="recovered", last_at=now)
                return f"recovered: job {job} is ready"
            if rec.get("state") == "recovered" and \
                    now - float(rec.get("last_at") or now) > RECOVERED_S:
                rec.update(state=None, last_reason=None)
            return ""
        if C.link_init_failure(why) and job in C.FOLLOWING:
            return ""                     # the cable failover has it
        rec["ended_job"] = job
        return _schedule(rec, now, why)
    if now < float(rec.get("next_at") or 0):
        return ""
    old = rec.get("ended_job")
    e = C.ENDED.get(old) or {}
    if e.get("relaunched"):
        rec.update(job=e["relaunched"], pending=False, ended_job=None)
        return f"followed the cable failover to job {e['relaunched']}"
    why = _machines_down(rec)
    if why:
        return _defer(rec, now, why)
    left = _leftovers(rec, old)
    if left:
        return _defer(rec, now, f"job {old}'s ranks are not gone ({left})")
    rec.setdefault("attempts", []).append(now)
    n = len(_window(rec, now))
    view_now = {"attempts": n, "last_reason": rec.get("last_reason"),
                "last_at": now, "next_at": None, "state": "recovering"}
    args = dict(rec["args"], peers=_fresh_peers(rec))
    try:
        out = C.launch(dict(rec["req"]), recovering=view_now, **args)
    except Exception as ex:
        out = {"error": f"{type(ex).__name__}: {ex}"}
    if out.get("job"):
        rec.update(job=out["job"], pending=False, ended_job=None,
                   last_at=now, next_at=None)
        return f"relaunch {n}: job {out['job']} (from {old})"
    why = str(out.get("refused") or out.get("error") or "no answer")
    rec["pending"] = False
    msg = _schedule(rec, now, f"relaunch {n} refused: {why}")
    return msg


def _cluster_phase(rec: dict) -> str:
    from knurlogic.cluster import jobs as J
    from knurlogic.interfaces import cluster_jobs as C
    job = rec["job"]
    phases = []
    recs = J.by_job().get(job)
    if recs:
        phases.append(J.phase_of(job, recs))
    post = rec["args"].get("post") or C._post
    for m in rec["order"]:
        if not m.get("page"):
            continue
        try:
            doc = post(f"http://{m['page']}{C.JOB_PATH}", {"job": job})
        except Exception:
            return "unknown"
        phases.append(doc.get("phase") or "joining")
    return "ready" if phases and all(p == "ready" for p in phases) \
        else "loading"


def _fresh_peers(rec: dict) -> list:
    """The peers now, where their record carries the cluster block; the
    launch's own record of a peer otherwise."""
    stored = {getattr(p, "id", ""): p for p in rec["args"].get("peers") or []}
    try:
        now = {getattr(p, "id", ""): p for p in peers_fn()}
    except Exception:
        now = {}
    out = []
    for pid, p in stored.items():
        q = now.get(pid)
        out.append(q if q is not None and isinstance(
            (getattr(q, "node", None) or {}).get("cluster"), dict) else p)
    return out


def _machines_down(rec: dict) -> str:
    """"" when every machine of the job answers its page, else who does
    not."""
    from knurlogic.interfaces import cluster_jobs as C
    post = rec["args"].get("post") or C._post
    for m in rec["order"]:
        if not m.get("page"):
            continue
        try:
            doc = post(f"http://{m['page']}{C.JOB_PATH}",
                       {"job": rec.get("ended_job") or rec["job"]})
            if not isinstance(doc, dict) or "ranks_here" not in doc:
                raise ValueError("no job state")
        except Exception as e:
            return f"{m.get('name')} is not answering ({type(e).__name__})"
    return ""


def _leftovers(rec: dict, job: str) -> str:
    """"" when no rank of `job` is left on any of its machines (by record
    and by process), else which."""
    from knurlogic.cluster import jobs as J
    from knurlogic.interfaces import cluster_jobs as C
    if not job:
        return ""
    here = [int(r["pid"]) for r in J.by_job().get(job, [])] \
        + J.pids_of_job(job)
    if here:
        return f"this machine: pid {', '.join(map(str, sorted(set(here))))}"
    post = rec["args"].get("post") or C._post
    for m in rec["order"]:
        if not m.get("page"):
            continue
        doc = post(f"http://{m['page']}{C.JOB_PATH}", {"job": job})
        if doc.get("ranks_here") or doc.get("stopping") \
                or doc.get("processes"):
            return f"{m.get('name')}: ranks {doc.get('ranks_here')}, " \
                   f"processes {doc.get('processes')}"
    return ""


# ------------------------------------------------------------ one Mac

def _tick_single(rec: dict, now: float) -> str:
    from knurlogic.machine import servers
    port = rec["port"]
    if rec.get("state") == "failed":
        return ""
    if not rec.get("pending"):
        srec = servers.registry().get(port)
        if srec is None:
            _drop(rec["key"])             # unloaded (the record is gone)
            return "unloaded; not recovered"
        pid = int(srec["pid"])
        if rec.get("pid") is None:
            rec["pid"] = pid
        mine = child_fn(port)
        code = mine[0].poll() if mine and mine[0].pid == pid else None
        if code is None and servers.is_our_server(pid):
            if rec.get("state") == "recovering" and answers_fn(port):
                rec.update(state="recovered", last_at=now)
                return f"recovered: port {port} answers"
            if rec.get("state") == "recovered" and \
                    now - float(rec.get("last_at") or now) > RECOVERED_S:
                rec.update(state=None, last_reason=None)
            return ""
        why = f"the server on port {port} (pid {pid}) exited"
        if code is not None:
            why += f" with code {code}"
        tail = ""
        try:
            with open(srec.get("log") or servers.serve_log(port), "rb") as f:
                f.seek(0, 2)
                f.seek(max(f.tell() - (64 << 10), 0))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            pass
        line = memory_line(tail)
        if line:
            why += f": {line}"
        rec["old_pid"] = pid
        return _schedule(rec, now, why)
    if now < float(rec.get("next_at") or 0):
        return ""
    left = _serve_pids(port)
    if left:
        return _defer(rec, now, f"port {port}'s old server is not gone "
                                f"(pid {', '.join(map(str, left))})")
    rec.setdefault("attempts", []).append(now)
    n = len(_window(rec, now))
    ld = rec["load"]
    try:
        out = load_fn(artifact=ld.get("artifact") or "", port=port,
                          tune=ld.get("tune") or "balanced",
                          sets=ld.get("sets") or {}, force=False,
                          draft=ld.get("draft", True))
    except Exception as ex:
        out = {"error": f"{type(ex).__name__}: {ex}"}
    if out.get("pid"):
        rec.update(pid=out["pid"], pending=False, last_at=now, next_at=None)
        return f"relaunch {n}: pid {out['pid']}"
    rec["attempts"].pop()                 # nothing started
    why = str(out.get("refused") or out.get("error") or "no answer")
    if out.get("refused") == "memory is about to move":
        return _defer(rec, now, why)
    rec["pending"] = False
    if kind(why) == "memory":
        return _failed(rec, now, f"relaunch refused: {why}")
    rec["attempts"].append(now)
    return _schedule(rec, now, f"relaunch {n} refused: {why}")


def _serve_pids(port: int) -> list:
    """`knurlogic serve` processes on `port` on this machine."""
    try:
        out = subprocess.run(["pgrep", "-f",
                              f"knurlogic serve .*--port {port}( |$)"],
                             capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return []
    return [int(x) for x in out.split() if x.isdigit()
            and int(x) != os.getpid()]
