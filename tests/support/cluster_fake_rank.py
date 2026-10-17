"""A fake rank for tests/test_cluster_jobs.py (not a test module): what a
page spawns in place of `knurlogic serve --rank r ...`. No model loads.

It writes the real progress marker (cluster/jobs.Marker) and walks the
real phases. Rank 0 also serves POST /v1/chat/completions on --port through
the REAL scheduler's submit/abort and the real SIGTERM path
(interfaces/http/lifecycle.watch_ring), answering with the real error mapping: a
request is held in flight until the job is torn down, then 503s.

The ranks join the REAL bell (engine/split/link.bell_early, as the
launcher sets it) and stand in for mlx's ring: a bell that closes -- a peer
rank's process ended -- makes this rank leave non-zero, as a collective
whose peer died raises "[ring] connection to a peer was lost".

argv: <artifact> knurlogic serve --rank R --job J [--port P] ...
env:  FAKE_STEPS=1  advance the step counter (a healthy, busy ring)
      FAKE_BAD_CABLE=<subnet>  rank 1 fails jaccl init on that cable
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def flag(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


#: what mlx's ring raises when a peer's process died (ring.cpp)
LOST = "[ring] connection to a peer was lost"


def ring_like(socks, lost):
    """Call `lost()` once any bell closes, as a collective whose peer died
    raises in mlx."""
    def watch(c):
        try:
            while c.recv(1):
                pass
        except OSError:
            pass
        lost()
    for c in socks:
        threading.Thread(target=watch, args=(c,), daemon=True).start()


#: a SIGTERM this follower deferred (follower._defer_sigterm)
TOLD = threading.Event()


def follower_lost():
    """A follower's bell closed. Told to stop (SIGTERM, deferred): rank 0
    left with the job's `stop`, and so does this rank, zero. Else its ring
    failed: one error line, non-zero (follower.py ring_error ->
    SystemExit(1))."""
    if TOLD.is_set():
        print("rank: leaving on rank 0's stop", flush=True)
        os._exit(0)
    print(f"rank: lost the ring to rank 0 ({LOST}); leaving", flush=True)
    os._exit(1)


def defer_sigterm():
    """As a real follower: a SIGTERM does not end it -- it leaves on rank
    0's stop (follower._defer_sigterm), else on the page's SIGKILL."""
    import signal
    signal.signal(signal.SIGTERM, lambda *_: TOLD.set())


def bell(job):
    """The real bell, on loopback: the fake machines' ring addresses are
    documentation addresses no socket can bind. Its port comes from the job
    id over a wide range, not the job's ring slot: parallel tests' pages
    do not see each other's slots, and nothing else here binds them."""
    from knurlogic.engine.split import link
    spec = os.environ.get("KNURLOGIC_BELL")
    if spec:
        _, _, rest = spec.split(":", 2)
        # below macOS's ephemeral range (49152-65535): a port the kernel
        # handed to some connection made the bind fail under load
        port = 20000 + int(job[:8], 16) % 29000
        os.environ["KNURLOGIC_BELL"] = f"127.0.0.1:{port}:{rest}"
    return link.bell_early() or []


def main():
    from knurlogic.cluster import jobs
    from knurlogic.engine.split import marker
    job, rank = flag("--job"), int(flag("--rank"))
    m = jobs.Marker(job, rank).start()
    marker.CURRENT["marker"] = m
    time.sleep(0.2)
    bad = os.environ.get("FAKE_BAD_CABLE")
    if bad and rank == 1 and flag("--cable") == bad:
        # what mlx's jaccl says when the queue pair cannot reach RTR
        print("ValueError: [jaccl] Changing queue pair to RTR failed with "
              "errno 96", flush=True)
        sys.exit(1)
    if bad and flag("--cable") == bad:
        # link init is collective: with rank 1's queue pair failed, no
        # other rank gets past it -- it waits here until the job is torn
        # down. (Walking on to "ready" raced the coordinator's failover,
        # which reads a job whose rank 0 is ready as serving.)
        while True:
            time.sleep(0.3)
    socks = bell(job)
    marker.progress(phase="loading")
    if rank != 0:
        defer_sigterm()
        ring_like(socks, follower_lost)
        marker.after_load()
        n = 0
        while True:
            time.sleep(0.3)
            if os.environ.get("FAKE_STEPS"):
                n += 1
                marker.progress(step=n)

    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime.scheduler import Job, Scheduler
    from knurlogic.interfaces.http import lifecycle
    from knurlogic.interfaces.http.openai import status_of

    class Host:
        state = "ready"

    sched = Scheduler(Host())             # never started: nothing steps
    # never at a step boundary either: 1 s, not 15, stands in for an idle
    # rank 0, which stops the ring at once (50 ms, measured) -- its
    # followers leave only then
    lifecycle.watch_ring(sched, Host(), exit_after=1.0, stop_within=1.0)
    # a peer's death fails rank 0's ring: the scheduler's own path (503s,
    # then a non-zero exit), as a collective raising in a real tick
    ring_like(socks, lambda: sched._ring_fatal(RuntimeError(LOST)))
    m.beat()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            out = json.dumps({"data": [{"id": "fake"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            j = sched.submit(Job(P.ChatRequest("text", "hi"),
                                 P.PromptArgs()))
            kind, val = j.outbox.get()
            err = status_of(val) if kind == "error" else None
            code = err.status if err else 200
            out = json.dumps(err.body() if err else {"ok": True}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", int(flag("--port"))), H)
    print("fake rank 0 serving", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
