"""A probe file, where it lives, and the served model's auto state.

One file per BASE model, not per artifact: every quant of a model reads the
same probe (AUC within 0.002 across four quants of a 35B). The base is the
language model's shape -- type, depth, width, vocabulary -- which every
quant of a model shares. Two fine-tunes of one base share it too; the
threshold check at onboarding is what would catch a probe that does not
fit (a guess: no fine-tune has been measured).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knurlogic.machine.artifact import Artifact
from knurlogic.machine.paths import state_dir

#: The process's auto state: the bound probe and tap, and what auto is
#: doing on the served model ("ready", "learning", "off") and why.
ACTIVE: dict = {"probe": None, "tap": None, "state": "off", "why": "",
                "path": None, "onboarding": None}

#: where a fitted probe stops, when onboarding has no better pick: at 0.95
#: the Qwen family and Gemma kept 33-43% of the thinking for at most a few
#: answers (docs/design/auto-effort.md, every model at its top level)
DEFAULT_THRESHOLD = 0.95


@dataclass
class Probe:
    """A logistic reader of one layer: p = sigmoid(((h - mean) / scale) . w + b)."""

    base: str
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray
    b: float
    threshold: float = DEFAULT_THRESHOLD
    #: what it was fitted on, for the page: traces, points, held-out AUC
    traces: int = 0
    points: int = 0
    auc: float = float("nan")

    def score(self, h: np.ndarray) -> np.ndarray:
        """[..., D] states -> [...] probabilities (numpy; the tap scores in mlx)."""
        z = ((np.asarray(h, np.float32) - self.mean) / self.scale) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def base_of(path) -> str:
    """The base model `path` is a quant of: its language model's type,
    depth, width and vocabulary, e.g. qwen3_5_moe-40x2048-248320."""
    cfg = Artifact.load(path).raw_config
    tc = cfg.get("text_config") or cfg
    kind = tc.get("model_type") or cfg.get("model_type") or "unknown"
    return (f"{kind}-{tc.get('num_hidden_layers', 0)}x"
            f"{tc.get('hidden_size', 0)}-{tc.get('vocab_size', 0)}")


#: probes fitted before release, shipped in the package as the chat
#: templates are: a model whose base is here is ready the moment auto is
#: chosen (docs/design/auto-effort.md, "Shipped probes")
SHIPPED = Path(__file__).with_name("shipped")


def probes_dir() -> Path:
    return state_dir() / "probes"


def path_of(base: str) -> Path:
    """Where this machine keeps the probe it fitted for `base`."""
    return probes_dir() / f"{base}.npz"


def find(base: str) -> Path | None:
    """The probe file for `base`: this machine's own fit first (it may be
    newer than the release's), else the shipped one, else None."""
    for p in (path_of(base), SHIPPED / f"{base}.npz"):
        if p.is_file():
            return p
    return None


def load(base: str) -> Probe | None:
    """The probe for `base`, or None when none has been fitted."""
    p = find(base)
    if p is None:
        return None
    with np.load(p) as d:
        return Probe(base=base, layer=int(d["layer"]), mean=d["mean"],
                     scale=d["scale"], w=d["w"], b=float(d["b"]),
                     threshold=float(d["threshold"]), traces=int(d["traces"]),
                     points=int(d["points"]), auc=float(d["auc"]))


def save(probe: Probe) -> Path:
    p = path_of(probe.base)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, layer=probe.layer, mean=probe.mean.astype(np.float32),
             scale=probe.scale.astype(np.float32), w=probe.w.astype(np.float32),
             b=probe.b, threshold=probe.threshold, traces=probe.traces,
             points=probe.points, auc=probe.auc)
    tmp.replace(p)
    return p


def status() -> dict:
    """What auto is doing on the served model, for /v1/models and usage."""
    probe = ACTIVE["probe"]
    out = {"state": ACTIVE["state"], "why": ACTIVE["why"]}
    if probe is not None:
        out.update(threshold=probe.threshold, layer=probe.layer,
                   traces=probe.traces)
    if ACTIVE["onboarding"]:
        out["onboarding"] = dict(ACTIVE["onboarding"])
    return out
