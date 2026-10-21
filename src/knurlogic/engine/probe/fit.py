"""Train a probe: pick the layer, fit the reader, pick the threshold.

Everything is scored out of fold BY PROBLEM (a problem's points never
train the reader that scores them), so neither the layer nor the threshold
is chosen on what the probe memorised. The threshold is the lowest one
whose simulated stop -- each trace stopped at its first break scoring over
it, answered with the answer forced there -- loses at most LOSS of the
answers thinking to the end gets right: the most thinking saved for that
price. numpy only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .store import DEFAULT_THRESHOLD, Probe

#: L2 strength of the reader; 10 was best on held-out problems for the
#: Flash, 27B and 35B (docs/design/auto-effort.md)
L2 = 10.0
#: answers the stop may lose against thinking to the end, as a fraction:
#: at 0.95 the measured models lost 0-5% (Flash 77/82 vs 80/82)
LOSS = 0.05
THRESHOLDS = (0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
FOLDS = 2
REPEATS = 5


@dataclass
class Points:
    """One row per paragraph break read: which trace and problem it is from,
    how far into the thinking (0-1), the label (an answer forced here is
    right), whether the trace's own final answer was right, and the states
    [N, L, D] at the tapped `layers`."""

    trace: np.ndarray
    problem: np.ndarray
    frac: np.ndarray
    right: np.ndarray
    final_ok: np.ndarray
    states: np.ndarray
    layers: list[int]


def logistic(X: np.ndarray, y: np.ndarray, lam: float = L2,
             steps: int = 500, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """L2 logistic regression by gradient descent on standardised X."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        g = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30))) - y
        w -= lr * (X.T @ g / len(y) + lam * w / len(y))
        b -= lr * float(g.mean())
    return w, b


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(a right point outscores a wrong one), ties half: Mann-Whitney."""
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _norm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(0), X.std(0) + 1e-6


def held_out(X: np.ndarray, y: np.ndarray, problem: np.ndarray) -> np.ndarray:
    """Each point's score from readers that never saw its problem."""
    ids = np.unique(problem)
    rng = np.random.default_rng(0)
    total, n = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(REPEATS):
        for part in np.array_split(rng.permutation(ids), FOLDS):
            te = np.isin(problem, part)
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            mu, sd = _norm(X[tr])
            w, b = logistic((X[tr] - mu) / sd, y[tr])
            z = np.clip((X[te] - mu) / sd @ w + b, -30, 30)
            total[te] += 1 / (1 + np.exp(-z))
            n[te] += 1
    return total / np.maximum(n, 1)


def simulate(pts: Points, score: np.ndarray, threshold: float) -> tuple[int, float]:
    """(answers right, the mean over traces of the fraction of its thinking
    kept) when each trace stops at its first break scoring >= threshold,
    or thinks to the end. Unweighted: a long trace stopped early saves
    more tokens than this says."""
    by_trace = defaultdict(list)
    for i, t in enumerate(pts.trace):
        by_trace[int(t)].append(i)
    right, kept = 0, 0.0
    for idx in by_trace.values():
        idx.sort(key=lambda i: pts.frac[i])
        stop = next((i for i in idx if score[i] >= threshold), None)
        if stop is None:
            right += int(pts.final_ok[idx[0]])
            kept += 1.0
        else:
            right += int(pts.right[stop])
            kept += float(pts.frac[stop])
    return right, kept / max(1, len(by_trace))


def train(pts: Points, base: str) -> Probe:
    """The probe for `base` from onboarding's points."""
    y = pts.right.astype(float)
    best = max(range(len(pts.layers)),
               key=lambda j: np.nan_to_num(auc(held_out(
                   pts.states[:, j], y, pts.problem), y), nan=0.0))
    X = pts.states[:, best]
    score = held_out(X, y, pts.problem)
    to_end, _ = simulate(pts, score, float("inf"))
    floor = to_end - LOSS * len(np.unique(pts.trace))
    threshold = next((t for t in THRESHOLDS
                      if simulate(pts, score, t)[0] >= floor),
                     DEFAULT_THRESHOLD)
    mu, sd = _norm(X)
    w, b = logistic((X - mu) / sd, y)
    return Probe(base=base, layer=pts.layers[best], mean=mu, scale=sd, w=w,
                 b=b, threshold=threshold,
                 traces=len(np.unique(pts.trace)), points=len(y),
                 auc=auc(score, y))
