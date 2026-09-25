"""Thinking levels, measured: reasoning tokens, accuracy and time per level.

RUN BY HAND on a free box, never from the test suite (it loads a real
model). One `knurlogic serve` process PER ARM -- a level is never measured
in a process another level warmed -- and every arm answers the same
questions `--runs` times at temperature 0.6, UNSEEDED, so the spread inside
an arm is visible next to the difference between arms. (Unseeded on
purpose: a seeded request to knurlogic serve samples identically whatever
the seed -- measured 2026-09-25, docs/PLAN.md -- and would fake zero
spread.)

    python tools/thinking_bench.py <artifact> --arms default,none,low,xhigh
        [--knurlogic PATH] [--port 8099] [--runs 3] [--out result.json]

Arms:
  default    the request says nothing about reasoning
  <level>    reasoning_effort=<level> (none minimal low medium high xhigh)
  closed     a candidate "off" for templates without one: the prompt at the
             lowest effort, with the think block already closed
             (`<think></think>`, the form the template itself writes for
             past turns), sent as a raw completion. Measured BEFORE any
             such level is offered.

Each question ends with a line the answer is checked against
("ANSWER: <number>"); a missing line counts as wrong, and a reply cut off
by max_tokens is counted as truncated, never as right.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

QUESTIONS = [
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
     "the ball. How many cents does the ball cost?", 5),
    ("If 5 machines take 5 minutes to make 5 widgets, how many minutes do "
     "100 machines take to make 100 widgets?", 5),
    ("What is the sum of the integers from 1 to 50 that are divisible by "
     "3?", 408),
    ("A train leaves at 14:40 and arrives at 17:25 the same day. How many "
     "minutes is the trip?", 165),
]
SUFFIX = "\n\nFinish with a final line of the form: ANSWER: <number>"


def _post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _answer(text: str):
    m = re.findall(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _closed_prompt(artifact: str, q: str) -> str:
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(Path(artifact))
    p = tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True,
                                reasoning_effort="low")
    start, end = getattr(tok, "think_start", "<think>"), \
        getattr(tok, "think_end", "</think>")
    if not p.rstrip().endswith(start):
        raise SystemExit(f"closed arm: the prompt does not end in {start!r};"
                         f" this template opens no think block to close")
    return p + end


def ask(base, arm, q, seed, max_tokens, artifact):
    t = time.time()
    if arm == "closed":
        r = _post(f"{base}/v1/completions", {
            "model": "m", "prompt": _closed_prompt(artifact, q + SUFFIX),
            "max_tokens": max_tokens, "temperature": 0.6})
        ch = r["choices"][0]
        text, reasoning = ch.get("text", ""), ""
    else:
        body = {"model": "m", "max_tokens": max_tokens, "temperature": 0.6,
                "messages": [{"role": "user", "content": q + SUFFIX}]}
        if arm != "default":
            body["reasoning_effort"] = arm
        r = _post(f"{base}/v1/chat/completions", body)
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = r.get("usage") or {}
    return {"seconds": round(time.time() - t, 1),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {})
            .get("reasoning_tokens"),
            "reasoning_chars": len(reasoning),
            "truncated": ch.get("finish_reason") == "length",
            "answer": _answer(text),
            "applied": ((u.get("knurlogic") or {}).get("thinking") or {})
            .get("applied")}


def _wait_up(base, proc, seconds=1800):
    t = time.time()
    while time.time() - t < seconds:
        if proc.poll() is not None:
            raise SystemExit(f"serve exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"{base}/v1/models", timeout=2)
            return round(time.time() - t, 1)
        except Exception:
            time.sleep(2)
    raise SystemExit("serve did not answer in time")


def run_arm(args, arm):
    base = f"http://127.0.0.1:{args.port}"
    log = open(Path(args.out).with_suffix(f".{arm}.serve.log"), "w")
    proc = subprocess.Popen([args.knurlogic, "serve", args.artifact,
                             "--port", str(args.port)],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        load_s = _wait_up(base, proc)
        rows = []
        lock = threading.Lock()
        jobs = [(qi, run) for run in range(args.runs)
                for qi in range(len(QUESTIONS))]

        def worker():
            while True:
                with lock:
                    if not jobs:
                        return
                    qi, run = jobs.pop(0)
                q, want = QUESTIONS[qi]
                try:
                    r = ask(base, arm, q, 1000 + run, args.max_tokens,
                            args.artifact)
                    r["right"] = (r["answer"] is not None
                                  and abs(r["answer"] - want) < 1e-6)
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"}
                r.update(question=qi, run=run)
                with lock:
                    rows.append(r)
                print(json.dumps({"arm": arm, **r}), flush=True)
        ts = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return {"arm": arm, "load_seconds": load_s, "rows": rows}
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def summarize(arm):
    rows = [r for r in arm["rows"] if "error" not in r]
    per_run = {}
    for r in rows:
        per_run.setdefault(r["run"], []).append(r)
    rt = [sum((x["reasoning_tokens"] or 0) for x in rs) / len(rs)
          for rs in per_run.values()]
    acc = [sum(x["right"] for x in rs) / len(rs) for rs in per_run.values()]
    return {"arm": arm["arm"],
            "applied": sorted({str(r.get("applied")) for r in rows}),
            "answered": len(rows), "errors": len(arm["rows"]) - len(rows),
            "truncated": sum(r["truncated"] for r in rows),
            "accuracy": round(sum(r["right"] for r in rows) / max(len(rows), 1), 3),
            "accuracy_per_run": [round(a, 2) for a in acc],
            "reasoning_tokens_mean": round(statistics.mean(rt), 1) if rt else None,
            "reasoning_tokens_per_run": [round(x, 1) for x in rt],
            "completion_tokens_mean": round(statistics.mean(
                [r["completion_tokens"] or 0 for r in rows]), 1) if rows else None,
            "seconds_mean": round(statistics.mean(
                [r["seconds"] for r in rows]), 1) if rows else None}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--arms", required=True)
    p.add_argument("--knurlogic", default="knurlogic")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="thinking_bench.json")
    args = p.parse_args(argv)
    arms = [run_arm(args, a) for a in args.arms.split(",")]
    out = {"artifact": Path(args.artifact).name, "runs": args.runs,
           "questions": len(QUESTIONS), "max_tokens": args.max_tokens,
           "concurrency": args.concurrency, "arms": arms,
           "summary": [summarize(a) for a in arms]}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for s in out["summary"]:
        print(json.dumps(s), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
