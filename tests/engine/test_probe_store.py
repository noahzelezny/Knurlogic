"""engine/probe/store: one probe file per base model, read back as written."""
import json

import numpy as np

from knurlogic.engine.probe import store


def _artifact(d, **text):
    d.mkdir()
    cfg = {"model_type": "wrap", "text_config": {
        "model_type": "toy", "num_hidden_layers": 4, "hidden_size": 8,
        "vocab_size": 100, **text}}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_every_quant_of_a_base_shares_its_name(tmp_path):
    a = _artifact(tmp_path / "toy-q4")
    b = _artifact(tmp_path / "toy-q8")
    c = _artifact(tmp_path / "toy-wider", hidden_size=16)
    assert store.base_of(a) == store.base_of(b) == "toy-4x8-100"
    assert store.base_of(c) != store.base_of(a)


def test_a_probe_reads_back_as_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path))
    rng = np.random.default_rng(0)
    p = store.Probe(base="toy-4x8-100", layer=2, mean=rng.normal(size=8),
                    scale=np.ones(8), w=rng.normal(size=8), b=-0.5,
                    threshold=0.9, traces=3, points=30, auc=0.8)
    assert store.load(p.base) is None
    store.save(p)
    q = store.load(p.base)
    assert q is not None and q.layer == 2 and q.threshold == 0.9
    h = rng.normal(size=(5, 8))
    assert np.allclose(q.score(h), p.score(h), atol=1e-5)
    assert ((q.score(h) > 0) & (q.score(h) < 1)).all()


def test_status_names_the_state_and_the_probe(monkeypatch):
    p = store.Probe(base="b", layer=1, mean=np.zeros(2), scale=np.ones(2),
                    w=np.zeros(2), b=0.0, threshold=0.97, traces=80)
    monkeypatch.setitem(store.ACTIVE, "probe", p)
    monkeypatch.setitem(store.ACTIVE, "state", "ready")
    st = store.status()
    assert st["state"] == "ready" and st["threshold"] == 0.97
    assert st["layer"] == 1 and st["traces"] == 80


def test_every_shipped_probe_reads_and_fits_its_base():
    """A shipped probe is ready the moment auto is chosen: it loads, its
    layer is inside the base's depth, its numbers are finite, and its
    threshold is one the fit can pick."""
    from knurlogic.engine.probe import fit
    files = sorted(store.SHIPPED.glob("*.npz"))
    assert files, "no probes shipped"
    for f in files:
        p = store.load(f.stem)
        depth = int(f.stem.rsplit("-", 2)[1].split("x")[0])
        assert p is not None and 0 <= p.layer < depth, f.stem
        assert p.mean.shape == p.scale.shape == p.w.shape, f.stem
        assert np.isfinite(p.w).all() and (p.scale > 0).all(), f.stem
        assert p.threshold in fit.THRESHOLDS and p.traces >= 38, f.stem


def test_this_machines_own_fit_wins_over_the_shipped_one(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path))
    base = min(store.SHIPPED.glob("*.npz")).stem
    assert store.find(base) == store.SHIPPED / f"{base}.npz"
    shipped = store.load(base)
    store.save(store.Probe(base=base, layer=1, mean=shipped.mean,
                           scale=shipped.scale, w=shipped.w, b=0.0,
                           threshold=0.97))
    assert store.find(base) == store.path_of(base)
    assert store.load(base).threshold == 0.97
