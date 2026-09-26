"""mlx-lm's server vs knurlogic's own, on one model: decode, prefill and a
concurrent batch, one server PROCESS per run (docs/SERVER.md, migration
step 3: "no slower on a measured decode/prefill comparison").

    python tools/server_bench.py <artifact> [--runs 3] [--port 8097]
        [--knurlogic /path/to/knurlogic] [--out result.json]

Per run: start `knurlogic serve <artifact> --server <arm>`, wait for it,
warm up, then
  decode   one request, greedy, streamed; tokens after the first / time
           after the first (3 reps, median)
  prefill  time to first token on a ~3000-token prompt with a fresh prefix
           each time, so the prompt cache cannot help (3 reps, median)
  batch    4 concurrent requests; total completion tokens / wall time
Arms alternate (mlx-lm, knurlogic, mlx-lm, ...) so drift hits both.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics as S
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

DECODE_Q = ("Write the numbers from 1 to 400 in words, one per line, "
            "and nothing else.")


def _post(url, body, stream=False):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def _stream_times(url, body):
    """(time to first content token, tokens after it, seconds after it)."""
    t0 = time.perf_counter()
    first, n = None, 0
    with _post(url, dict(body, stream=True,
                         stream_options={"include_usage": True})) as r:
        usage = None
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            c = json.loads(d)
            if c.get("usage"):
                usage = c["usage"]
            ch = (c.get("choices") or [{}])[0]
            dl = ch.get("delta") or {}
            if (dl.get("content") or dl.get("reasoning")) and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    toks = (usage or {}).get("completion_tokens", 0)
    return first - t0, toks - 1, end - first


def _wait(url, proc, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            b = {"model": "m", "max_tokens": 1, "reasoning_effort": "none",
                 "messages": [{"role": "user", "content": "hi"}]}
            with _post(url, b) as r:
                r.read()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError


def run_arm(kl, artifact, arm, port):
    url = f"http://127.0.0.1:{port}"
    log = open(f"bench-{arm}-{port}.log", "w")
    proc = subprocess.Popen([kl, "serve", artifact, "--port", str(port),
                             "--server", arm], stdout=log, stderr=log,
                            start_new_session=True)
    try:
        _wait(url, proc)
        base = {"model": "m", "temperature": 0, "reasoning_effort": "none"}
        _stream_times(url, dict(base, max_tokens=64, messages=[
            {"role": "user", "content": DECODE_Q}]))          # warm
        dec = []
        for _ in range(3):
            _, n, dt = _stream_times(url, dict(base, max_tokens=512,
                                               messages=[{"role": "user",
                                                          "content": DECODE_Q}]))
            dec.append(n / dt)
        pre = []
        filler = ("The quick brown fox jumps over the lazy dog. " * 300)
        for _ in range(3):
            ttft, _, _ = _stream_times(url, dict(base, max_tokens=2, messages=[
                {"role": "user", "content":
                 f"[{uuid.uuid4()}] {filler} Reply OK."}]))
            pre.append(ttft)
        out, t0 = [], time.perf_counter()

        def one(i):
            with _post(url, dict(base, max_tokens=256, messages=[
                    {"role": "user", "content":
                     f"Tell a short story about the number {i}."}])) as r:
                out.append(json.loads(r.read())["usage"]["completion_tokens"])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        batch = sum(out) / (time.perf_counter() - t0)
        return {"decode_tps": S.median(dec), "decode_all": dec,
                "prefill_ttft_s": S.median(pre), "prefill_all": pre,
                "batch4_tps": batch}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--port", type=int, default=8097)
    p.add_argument("--knurlogic", default="knurlogic")
    p.add_argument("--out", default="")
    a = p.parse_args()
    res = {"mlx-lm": [], "knurlogic": []}
    for i in range(a.runs):
        for arm in ("mlx-lm", "knurlogic"):
            r = run_arm(a.knurlogic, a.artifact, arm, a.port)
            res[arm].append(r)
            print(arm, i, json.dumps({k: round(v, 3) for k, v in r.items()
                                      if not k.endswith("_all")}),
                  flush=True)
    summary = {}
    for k in ("decode_tps", "prefill_ttft_s", "batch4_tps"):
        m = [r[k] for r in res["mlx-lm"]]
        o = [r[k] for r in res["knurlogic"]]
        summary[k] = {"mlx-lm": [round(x, 3) for x in m],
                      "knurlogic": [round(x, 3) for x in o],
                      "ratio_median": round(S.median(o) / S.median(m), 3)}
        print(k, json.dumps(summary[k]))
    if a.out:
        json.dump({"artifact": a.artifact, "runs": res,
                   "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
