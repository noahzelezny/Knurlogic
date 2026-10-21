"""engine/probe/fit: the layer that separates, a threshold that keeps the
answers, scored only on problems the reader never saw."""
import numpy as np

from knurlogic.engine.probe import fit


def _points(rng, problems=20, per=2, breaks=8, d=6):
    """Layer 1 holds the label in its first feature; layer 0 is noise. A
    trace becomes right after a random point and stays right."""
    rows = {k: [] for k in ("trace", "problem", "frac", "right",
                             "final_ok", "states")}
    t = 0
    for p in range(problems):
        for _ in range(per):
            turn = rng.integers(2, breaks)
            for i in range(breaks):
                right = int(i >= turn)
                good = rng.normal(size=d)
                good[0] = 3.0 * (2 * right - 1) + rng.normal() * 0.3
                rows["trace"].append(t)
                rows["problem"].append(p)
                rows["frac"].append((i + 1) / breaks)
                rows["right"].append(right)
                rows["final_ok"].append(1)
                rows["states"].append(np.stack([rng.normal(size=d), good]))
            t += 1
    return fit.Points(**{k: np.array(v) for k, v in rows.items()},
                      layers=[10, 20])


def test_the_layer_that_carries_the_label_is_picked():
    probe = fit.train(_points(np.random.default_rng(0)), "toy")
    assert probe.layer == 20
    assert probe.auc > 0.9
    assert probe.threshold in fit.THRESHOLDS
    assert probe.traces == 40 and probe.points == 320


def test_stopping_saves_thinking_and_keeps_answers():
    pts = _points(np.random.default_rng(1))
    y = pts.right.astype(float)
    score = fit.held_out(pts.states[:, 1], y, pts.problem)
    right_end, kept_end = fit.simulate(pts, score, float("inf"))
    right, kept = fit.simulate(pts, score, 0.8)
    assert kept_end == 1.0 and kept < 0.7
    assert right >= right_end - 0.05 * 40


def test_auc_is_half_for_noise_and_one_for_a_clean_split():
    y = np.array([0, 0, 1, 1])
    assert fit.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert fit.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
