"""fake_cluster(n): n real knurlogic pages on this Mac, each its own process
(tests/support/cluster_fake_page.py: the REAL handler, liveness, cluster
steps and recovery), each with its own cache dir and identity, all knowing
all of the others, each spawning tests/support/cluster_fake_rank.py as its
rank. Only the machine facts are faked. Liveness clocks are short
(FAKE_FAST), so a page's death is seen in seconds.

    with fake_cluster(4, tmp_path) as c:
        out = c.launch(0)               # page 0 coordinates, all 4 run it
        c.kill_page(0)                  # SIGKILL it, and its rank with it
        c.wait(lambda: c.job_stopped_everywhere(out["job"]), 20)
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = str(HERE.parents[1] / "src")
GIB = 1 << 30
VERSIONS = {"knurlogic": "0.1.0.dev0", "mlx": "0.31.2",
            "build": "0123456789ab+mlx0.31.2"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def info(i: int, ws: int = 64 * GIB, **over) -> dict:
    return {"chip": "Apple M4 Max", "p_core_ghz": None,
            "bandwidth_gbs": None,
            "thunderbolt": [{"iface": f"en{i}", "ip": f"10.0.0.{i + 1}"}],
            "rdma": {"available": False, "reason": "off", "devices": [],
                     "active": []},
            "versions": dict(VERSIONS), "jaccl_selfheal": False,
            "working_set_bytes": ws, **over}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def wait(pred, t: float = 20.0, every: float = 0.1):
    end = time.time() + t
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    return pred()


class Page:
    def __init__(self, i: int, tmp: Path, env: dict, infos: dict):
        self.i, self.id, self.name = i, f"node{i}", f"P{i}"
        self.port = free_port()
        self.cache = tmp / f"cache{i}"
        self.info = infos
        self.env = env
        self.proc = None

    @property
    def addr(self) -> str:
        return f"127.0.0.1:{self.port}"

    def jobs_file(self) -> Path:
        return self.cache / "knurlogic" / "jobs" / "jobs.json"

    def registry(self) -> dict:
        try:
            d = json.loads(self.jobs_file().read_text())
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    def ranks(self, job: str | None = None) -> list:
        return [dict(v, key=k) for k, v in self.registry().items()
                if job is None or v.get("job") == job]

    def live_ranks(self, job: str | None = None) -> list:
        return [r for r in self.ranks(job) if alive(int(r["pid"]))]

    def start(self, peers: list) -> None:
        env = {**os.environ, "PYTHONPATH": SRC, **self.env,
               "XDG_CACHE_HOME": str(self.cache),
               "KNURLOGIC_HOME": str(self.cache / "home"),
               "FAKE_PEERS": json.dumps(peers)}
        self.cache.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "cluster_fake_page.py"),
             str(self.port), self.id, self.name, json.dumps(self.info)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        assert "up" in self.proc.stdout.readline(), "a fake page did not start"

    def kill(self) -> None:
        """The page's process dies (SIGKILL): no goodbye."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)

    def get(self, path: str, timeout: float = 10.0):
        with urllib.request.urlopen(f"http://{self.addr}{path}",
                                    timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path: str, doc: dict, timeout: float = 120.0):
        req = urllib.request.Request(
            f"http://{self.addr}{path}", data=json.dumps(doc).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def peer_state(self, other: "Page") -> str:
        """What THIS page's peer list says of `other`."""
        doc = self.get("/status.json?light=1")
        for p in doc.get("peers") or []:
            if p.get("id") == other.id:
                return p.get("state")
        return ""

    def peer_public(self, other: "Page") -> dict:
        for p in self.get("/status.json?light=1").get("peers") or []:
            if p.get("id") == other.id:
                return p
        return {}


class FakeCluster:
    def __init__(self, n: int, tmp_path: Path, env: dict | None = None,
                 page_env: dict | None = None, infos: dict | None = None,
                 knows: str = "all"):
        """`page_env`: {index: env} for single pages; `infos`: {index: the
        page's info overrides}."""
        self.tmp = Path(tmp_path)
        page_env = page_env or {}
        self.pages = [Page(i, self.tmp, {
            "FAKE_FAST": "1", "FAKE_SERVE_PORT": str(free_port()),
            **(env or {}), **page_env.get(i, {})},
            info(i, **(infos or {}).get(i, {}))) for i in range(n)]
        self.knows = knows

    def peers_of(self, p: Page) -> list:
        if p.env.get("FAKE_NO_PEERS"):
            return []
        return [[q.id, q.addr] for q in self.pages if q is not p]

    def start(self) -> "FakeCluster":
        for p in self.pages:
            p.start(self.peers_of(p))
        return self

    def restart_page(self, i: int) -> None:
        p = self.pages[i]
        p.kill()
        p.start(self.peers_of(p))

    def stop(self) -> None:
        for p in self.pages:
            for r in p.ranks():
                try:
                    os.kill(int(r["pid"]), signal.SIGKILL)
                except OSError:
                    pass
            p.kill()

    def __enter__(self) -> "FakeCluster":
        try:
            return self.start()
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *_a) -> None:
        self.stop()

    # -- driving ---------------------------------------------------------
    def wait_peers(self, t: float = 30.0) -> bool:
        """Every (live) page answering every other."""
        def ok():
            for p in self.pages:
                if p.proc.poll() is not None:
                    continue
                for q in self.pages:
                    if q is p or q.proc.poll() is not None:
                        continue
                    if p.peer_state(q) != "answering":
                        return False
            return True
        return bool(wait(ok, t, 0.3))

    def launch(self, coordinator: int = 0, split: str = "pipeline",
               link: str = "tcp", order: list | None = None,
               nodes: list | None = None, **extra) -> dict:
        ids = nodes or [p.id for p in self.pages]
        req = {"action": "load", "identity": "abc", "nodes": ids,
               "split": split, "link": link, **extra}
        if order is not None:
            req["order"] = order
        return self.pages[coordinator].post("/loaded.json", req)

    def running(self, job: str) -> bool:
        """Every page that is up runs a live rank of `job`."""
        return all(len(p.live_ranks(job)) == 1 for p in self.pages
                   if p.proc.poll() is None)

    def job_stopped_everywhere(self, job: str) -> bool:
        return all(not p.live_ranks(job) for p in self.pages
                   if p.proc.poll() is None)

    def jobs_running(self) -> set:
        out = set()
        for p in self.pages:
            for r in p.live_ranks():
                out.add(r["job"])
        return out

    wait = staticmethod(wait)
