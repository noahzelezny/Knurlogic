"""Thinking: auto's onboarding: learn the served model's probe while the
server is idle.

Once a model's Thinking setting is auto and it has no probe, the server
works through the onboarding set (engine/probe/problems) as a client
would, one request at a time and only when nothing else is running: each
problem twice at the model's top level, keeping the tapped layers' states
at every paragraph break of the thinking (engine/probe/stop). Then each
of up to POINTS breaks per trace is labelled -- the thinking cut there,
the think block closed, "ANSWER:" written greedily -- and the probe is
fitted (engine/probe/fit) and saved for every quant of the base model.
Each finished trace is kept on disk, so a restart resumes rather than
starts over. Measured by hand before this was built: about 80 traces and
900 labels per model; hours on the larger models.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from knurlogic.engine.families import sampling_defaults
from knurlogic.engine.model import thinking
from knurlogic.engine.probe import binding, fit, problems, store

from . import openai as O

logger = logging.getLogger(__name__)

#: samples per problem: two gave every model's fit 80-82 traces
SAMPLES = 2
#: breaks labelled per trace, spread evenly: as in every measured fit
POINTS = 12
#: label requests in flight at once (each is a short prefill + 12 tokens)
LABEL_WIDTH = 8
#: how often the watcher looks (s), and waits while the server is busy
POLL_S = 20.0
ANSWER = re.compile(r"ANSWER:\s*\**\s*(-?[\d,]+)")


class Stale(Exception):
    """The model or its setting changed under onboarding: stop quietly."""


def watch(sched) -> threading.Thread:
    """Start the watcher: keeps auto bound to what the setting says and
    onboards a learning model when the server is idle."""
    t = threading.Thread(target=_loop, args=(sched,), daemon=True,
                         name="knurlogic-auto-onboarding")
    t.start()
    return t


def _loop(sched) -> None:
    while True:
        time.sleep(POLL_S)
        host = sched.host
        if host.state != "ready" or host.model is None:
            continue
        try:
            binding.refresh(host.model, host.path, host.shard is not None)
            if store.ACTIVE["state"] == "learning" and not sched.busy():
                learn(sched)
        except Stale:
            store.ACTIVE["onboarding"] = None
        except Exception:
            logger.exception("Thinking: auto onboarding failed; "
                             "trying again later")


def learn(sched) -> None:
    """Onboard the served model, start to finish (resuming kept traces)."""
    host = sched.host
    path, base = host.path, store.base_of(host.path)
    keep = store.probes_dir() / f"{base}.learning"
    keep.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for k in range(len(problems.PROBLEMS))
            for s in range(SAMPLES)]
    progress: dict[str, Any] = {"step": "traces", "done": 0, "of": len(jobs)}
    store.ACTIVE["onboarding"] = progress
    traces = []
    for k, s in jobs:
        f = keep / f"{k}_{s}.npz"
        if not f.is_file():
            _idle(sched, path)
            _save_trace(f, _trace(sched, path, k))
        traces.append(f)
        progress["done"] += 1
    progress.update(step="labels", done=0)
    for f in traces:
        _idle(sched, path)
        _label(sched, path, f)
        progress["done"] += 1
    pts = _points(traces)
    probe = fit.train(pts, base)
    store.save(probe)
    logger.info("Thinking: auto ready for %s: layer %d, threshold %.2f, "
                "AUC %.3f on %d traces", base, probe.layer, probe.threshold,
                probe.auc, probe.traces)
    store.ACTIVE["onboarding"] = None
    binding.refresh(host.model, path, host.shard is not None)


def _idle(sched, path: str) -> None:
    """Wait until nothing else runs; Stale if the model or setting moved."""
    while True:
        if sched.host.path != path or store.ACTIVE["state"] != "learning":
            raise Stale
        if not sched.busy():
            return
        time.sleep(POLL_S)


def _messages(k: int) -> list[dict]:
    return [{"role": "system", "content": problems.SYSTEM},
            {"role": "user", "content": problems.PROBLEMS[k][0]}]


def _run(sched, body: dict, *, chat: bool, auto: dict | None = None,
         path: str) -> tuple:
    """Submit one request as the server would; -> (content, job)."""
    job, _ctx = O.build_job(body, chat=chat,
                            translate=thinking.translate if chat else None,
                            sampling_defaults=sampling_defaults(path))
    job.auto = auto
    sched.submit(job)
    content = []
    while True:
        kind, val = job.outbox.get()
        if kind == "delta":
            content.append(val.content)
        elif kind == "error":
            raise val
        elif kind == "done":
            return "".join(content), job


def _trace(sched, path: str, k: int) -> dict:
    """Problem k at the top level, keeping the states at every break."""
    content, job = _run(sched, {"messages": _messages(k),
                                "reasoning_effort": thinking.AUTO},
                        chat=True, path=path,
                        auto={"keep_states": True, "report": {}})
    stop = job.auto["stop"]
    m = ANSWER.findall(content)
    got = int(m[-1].replace(",", "")) if m else None
    return {"k": k, "ok": got == problems.PROBLEMS[k][1],
            "tokens": np.array(stop.tokens, np.int64),
            "at": np.array([n for n, _ in stop.states], np.int64),
            "states": (np.stack([s for _, s in stop.states])
                       if stop.states else np.zeros((0, 0, 0), np.float32)),
            "layers": np.array(store.ACTIVE["tap"].layers, np.int64),
            "kwargs": json.dumps(job.args.chat_template_kwargs or {})}


def _save_trace(f: Path, t: dict) -> None:
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, **t)
    tmp.replace(f)


def _cuts(at: np.ndarray) -> list[int]:
    """Indices of up to POINTS breaks, spread evenly."""
    n = len(at)
    if n <= POINTS:
        return list(range(n))
    return sorted({round(i * (n - 1) / (POINTS - 1)) for i in range(POINTS)})


def _label(sched, path: str, f: Path) -> None:
    """Would an answer forced at each kept break be right? Saved into the
    trace's file; a file already labelled is left as it is."""
    with np.load(f) as d:
        t = dict(d)
    if "right" in t:
        return
    tok = sched.host.tokenizer
    kwargs = json.loads(str(t["kwargs"]))
    head = tok.apply_chat_template(_messages(int(t["k"])), tokenize=False,
                                   add_generation_prompt=True, **kwargs)
    end = getattr(tok, "think_end", "") or ""
    cuts = _cuts(t["at"])
    answer = problems.PROBLEMS[int(t["k"])][1]
    right = np.zeros(len(t["at"]), np.int8) - 1      # -1: not labelled
    done = threading.Semaphore(LABEL_WIDTH)
    lock = threading.Lock()

    def one(i: int) -> None:
        try:
            n = int(t["at"][i])
            prompt = (head + tok.decode(list(t["tokens"][:n])).rstrip()
                      + f"\n{end}\n\nANSWER:")
            text, _ = _run(sched, {"prompt": prompt, "max_tokens": 12,
                                   "temperature": 0}, chat=False, path=path)
            m = re.match(r"\s*\**\s*(-?[\d,]+)", text)
            got = int(m.group(1).replace(",", "")) \
                if m and m.group(1).strip("-,") else None
            with lock:
                right[i] = int(got == answer)
        finally:
            done.release()

    threads = []
    for i in cuts:
        done.acquire()
        th = threading.Thread(target=one, args=(i,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t["right"] = right
    _save_trace(f, t)


def _points(files: list[Path]) -> fit.Points:
    """The labelled breaks of every trace, as fit.train reads them."""
    cols: dict[str, list] = {k: [] for k in
                             ("trace", "problem", "frac", "right",
                              "final_ok", "states")}
    layers: list[int] = []
    for j, f in enumerate(files):
        with np.load(f) as d:
            layers = [int(x) for x in d["layers"]]
            right, at, states = d["right"], d["at"], d["states"]
            n_tok = max(1, len(d["tokens"]))
            for i in np.flatnonzero(right >= 0):
                cols["trace"].append(j)
                cols["problem"].append(int(d["k"]))
                cols["frac"].append(at[i] / n_tok)
                cols["right"].append(int(right[i]))
                cols["final_ok"].append(int(d["ok"]))
                cols["states"].append(states[i])
    return fit.Points(**{k: np.array(v) for k, v in cols.items()},
                      layers=layers)
