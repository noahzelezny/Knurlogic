"""interfaces/loading.py: what a model must pass before it loads, at startup
and on every switch -- known to this machine, runnable, and fits -- and the
scheduler's refusal to switch under running requests."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.interfaces import loading as L  # noqa: E402

GIB = 1 << 30


def _artifact(root: Path, name: str, gib: float = 1.0,
              model_type: str = "qwen3_5") -> Path:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    return d


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Two artifacts in a store, one outside it; memory as asked."""
    from knurlogic.machine import discover, wired
    from knurlogic.machine.artifact import Artifact
    store = tmp_path / "store"
    store.mkdir()
    a = _artifact(store, "known-model")
    big = _artifact(store, "big-model")
    stray = _artifact(tmp_path, "stray-model")
    rows = [types.SimpleNamespace(name=p.name, path=p, servable=True,
                                  format="mlx") for p in (a, big)]
    monkeypatch.setattr(discover, "find", lambda *a, **k: rows)
    L._KNOWN.update(at=0.0, rows=None)
    sizes = {"known-model": 1 * GIB, "big-model": 90 * GIB,
             "stray-model": 1 * GIB}
    real = Artifact.load

    def load(p):
        art = real(p)
        art.bytes_on_disk = sizes[Path(p).name]
        return art
    monkeypatch.setattr(Artifact, "load", classmethod(
        lambda cls, p: load(p)))
    mem = {"working_set_bytes": 64 * GIB, "available_bytes": 40 * GIB}
    monkeypatch.setattr(wired, "load_budget", lambda: dict(
        mem, bytes=min(mem.values()), limited_by="test"))
    monkeypatch.setattr(L, "register", lambda art: [])
    return types.SimpleNamespace(a=a, big=big, stray=stray, mem=mem)


def test_a_known_artifact_by_id_or_path_is_prepared(machine):
    assert L.prepare("known-model").path == machine.a
    assert L.prepare(str(machine.a)).path == machine.a


def test_an_arbitrary_directory_is_not_loadable(machine):
    """A client names a model; it never names a directory, so a request
    cannot make the server run some other model.py."""
    for name in (str(machine.stray), "stray-model", "../store/../x", "/etc"):
        with pytest.raises(L.NotLoadable) as e:
            L.prepare(name)
        assert e.value.status == 404 and "/models.json" in str(e.value)


def test_a_model_that_does_not_fit_is_refused_with_the_numbers(machine):
    with pytest.raises(L.NotLoadable) as e:
        L.prepare("big-model")
    assert e.value.status == 507 and "90.0 GiB" in str(e.value)
    assert "/loaded.json" in str(e.value)
    # what unloading the current model frees counts, up to the working set
    machine.mem["working_set_bytes"] = 100 * GIB
    assert L.prepare("big-model", freed_bytes=60 * GIB).path == machine.big


def test_an_unsupported_architecture_is_a_422(machine, monkeypatch):
    monkeypatch.setattr(L, "register", lambda art: ["mlx_lm.models.nope"])
    with pytest.raises(L.NotLoadable) as e:
        L.prepare("known-model")
    assert e.value.status == 422 and "nope" in str(e.value)


def test_a_switch_under_running_requests_is_refused_on_the_scheduler():
    """force=False is decided on the scheduler thread when the command
    runs, so it cannot race the requests it would fail."""
    from knurlogic.engine.runtime.scheduler import Command, Job, Scheduler
    from knurlogic.engine.runtime import prompt as P

    class Host:
        state, path, error = "ready", "/m/a", ""
        loads = []

        def load(self, p, **k):
            self.loads.append(p)

    s = Scheduler(Host())
    s._waiting.append(Job(P.ChatRequest(), P.PromptArgs()))
    c = Command("load", "/m/b", force=False)
    s._commands.put(c)
    s._do_commands()
    assert c.done.is_set() and "running or queued" in c.error
    assert Host.loads == []
    c = Command("load", "/m/b", force=True)
    s._commands.put(c)
    s._do_commands()
    assert c.error == "" and Host.loads == ["/m/b"]


def test_a_failed_store_scan_is_named_and_not_cached(monkeypatch):
    """A store on an unmounted volume: the refusal says the scan failed
    (not that the model does not exist), and the next request scans again."""
    from knurlogic.machine import discover
    L._KNOWN.update(at=0.0, rows=None, error="")
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise OSError("volume is gone")
    monkeypatch.setattr(discover, "find", boom)
    for _ in range(2):
        with pytest.raises(L.NotLoadable) as e:
            L.resolve_name("some-model", None)
        assert e.value.status == 503 and "volume is gone" in str(e.value)
    assert len(calls) == 2


def test_the_knurlogic_allowance_caps_what_a_load_may_have(machine):
    """A model the working set would take is refused past the machine's
    allowance, and the refusal names it."""
    machine.mem.update(working_set_bytes=100 * GIB, available_bytes=100 * GIB)
    machine.mem["allowance_bytes"] = 50 * GIB
    with pytest.raises(L.NotLoadable) as e:
        L.prepare("big-model")
    assert e.value.status == 507 and "allowance 50.0" in str(e.value)
    machine.mem["allowance_bytes"] = 0
    assert L.prepare("big-model").path == machine.big
