"""A fake rank for tests/test_cluster_jobs.py (not a test module): what a
page spawns in place of `knurlogic serve --rank r ...`. No model loads.

It writes the real progress marker (cluster/jobs.Marker) and walks the
real phases. Rank 0 also serves POST /v1/chat/completions on --port through
the REAL scheduler's submit/abort and the real SIGTERM path
(interfaces/http.watch_ring), answering with the real error mapping: a
request is held in flight until the job is torn down, then 503s.

argv: <artifact> knurlogic serve --rank R --job J [--port P] ...
env:  FAKE_STEPS=1  advance the step counter (a healthy, busy ring)
      FAKE_BAD_CABLE=<subnet>  rank 1 fails jaccl init on that cable
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def flag(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def main():
    from knurlogic.cluster import jobs
    job, rank = flag("--job"), int(flag("--rank"))
    m = jobs.Marker(job, rank).start()
    jobs.CURRENT["marker"] = m
    time.sleep(0.2)
    bad = os.environ.get("FAKE_BAD_CABLE")
    if bad and rank == 1 and flag("--cable") == bad:
        # what mlx's jaccl says when the queue pair cannot reach RTR
        print("ValueError: [jaccl] Changing queue pair to RTR failed with "
              "errno 96", flush=True)
        sys.exit(1)
    jobs.progress(phase="loading")
    if rank != 0:
        jobs.after_load()
        n = 0
        while True:
            time.sleep(0.3)
            if os.environ.get("FAKE_STEPS"):
                n += 1
                jobs.progress(step=n)

    from knurlogic.engine.runtime import prompt as P
    from knurlogic.engine.runtime.scheduler import Job, Scheduler
    from knurlogic.interfaces import http
    from knurlogic.interfaces.http.openai import _status_of

    class Host:
        state = "ready"

    sched = Scheduler(Host())             # never started: nothing steps
    http.watch_ring(sched, Host(), exit_after=1.0)
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
            err = _status_of(val) if kind == "error" else None
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
