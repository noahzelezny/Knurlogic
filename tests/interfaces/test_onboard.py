"""interfaces/http/onboard: a learning model works through the onboarding
set, labels its breaks, fits a probe, saves it for its base, goes ready."""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from knurlogic.engine.probe import problems, store
from knurlogic.interfaces.http import onboard

D, BREAKS = 6, 8
INDEX = {q: k for k, (q, _) in enumerate(problems.PROBLEMS)}


class Tok:
    think_end = "</think>"
    chat_template = None

    def apply_chat_template(self, msgs, **kw):
        return f"<q{INDEX[msgs[1]['content']]}>"

    def decode(self, ids):
        return " ".join(map(str, ids))


class Sched:
    """Answers like a model that settles on its answer halfway through,
    and whose layer-2 state says so."""

    def __init__(self, path, model):
        self.host = types.SimpleNamespace(path=str(path), tokenizer=Tok(),
                                          model=model, shard=None,
                                          state="ready")
        self.rng = np.random.default_rng(0)

    def busy(self):
        return False

    def submit(self, job):
        if job.auto is not None:            # a trace at the top level
            k = INDEX[job.request.messages[1]["content"]]
            states = []
            for i in range(BREAKS):
                s = self.rng.normal(size=(2, D)).astype(np.float32)
                s[1, 0] = 3.0 if i >= BREAKS // 2 else -3.0
                states.append((10 * (i + 1), s))
            job.auto["stop"] = types.SimpleNamespace(
                tokens=list(range(10 * BREAKS)), states=states)
            text = f"ANSWER: {problems.PROBLEMS[k][1]}"
        else:                               # a label: right past halfway
            k = int(re.match(r"<q(\d+)>", job.request.prompt).group(1))
            n = len(job.request.prompt.split("</think>")[0].split())
            ans = problems.PROBLEMS[k][1]
            text = f" {ans if n > 10 * BREAKS // 2 else ans + 1}"
        job.outbox.put(("delta", types.SimpleNamespace(content=text)))
        job.outbox.put(("done", {}))


def _learning(tmp_path, monkeypatch):
    from test_probe_tap import Model

    from knurlogic.engine.model import thinking
    from knurlogic.engine.probe.tap import Tap
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(thinking.DEFAULT_ENV, "auto")
    art = tmp_path / "toy"
    art.mkdir(exist_ok=True)
    (art / "config.json").write_text(json.dumps(
        {"model_type": "toy", "num_hidden_layers": 4, "hidden_size": D,
         "vocab_size": 50}))
    model = Model()
    old = store.ACTIVE["tap"]
    if old is not None:
        old.remove()
    monkeypatch.setitem(store.ACTIVE, "tap", Tap(model, [1, 2]))
    monkeypatch.setitem(store.ACTIVE, "state", "learning")
    monkeypatch.setitem(store.ACTIVE, "path", str(art))
    return art, Sched(art, model)


def test_onboarding_fits_a_probe_and_goes_ready(tmp_path, monkeypatch):
    art, sched = _learning(tmp_path, monkeypatch)
    try:
        onboard.learn(sched)
        probe = store.load(store.base_of(art))
        assert probe is not None and probe.layer == 2
        assert probe.traces == len(problems.PROBLEMS) * onboard.SAMPLES
        assert probe.auc > 0.9
        assert store.ACTIVE["state"] == "ready"
        assert store.ACTIVE["onboarding"] is None
        assert store.ACTIVE["tap"].layers == [2]
    finally:
        store.ACTIVE["tap"].remove()


def test_a_restart_resumes_from_the_kept_traces(tmp_path, monkeypatch):
    art, sched = _learning(tmp_path, monkeypatch)
    onboard.learn(sched)
    store.path_of(store.base_of(art)).unlink()     # as if it never finished
    art, sched = _learning(tmp_path, monkeypatch)

    def nothing_runs(job):
        raise AssertionError("a kept trace or label was asked again")
    sched.submit = nothing_runs
    try:
        onboard.learn(sched)
        assert store.load(store.base_of(art)) is not None
    finally:
        store.ACTIVE["tap"].remove()


def test_cuts_spread_over_the_whole_thinking():
    assert onboard._cuts(np.arange(5)) == [0, 1, 2, 3, 4]
    c = onboard._cuts(np.arange(100))
    assert len(c) == onboard.POINTS and c[0] == 0 and c[-1] == 99
