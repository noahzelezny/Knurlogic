"""What is resident, across runtimes -- and the distinctions that matter.

The failure these pin is a page that says a model is loaded when it is not.
Every runtime here reports something slightly different, and flattening them
into "has a model" loses the only fact anybody opened the page for.
"""
import json

import pytest

from knurlogic.machine import loaded


class _Fake:
    """Stands in for the HTTP layer. Keyed by URL so one fixture can serve
    four runtimes at once, which is the case that actually gets rendered."""
    def __init__(self, table):
        self.table = table
        self.asked = []

    def __call__(self, url, timeout=1.5):
        self.asked.append(url)
        for k, v in self.table.items():
            if url.endswith(k):
                return v
        return None


@pytest.fixture
def fake(monkeypatch):
    def install(table):
        f = _Fake(table)
        monkeypatch.setattr(loaded, "_get", f)
        return f
    return install


def test_an_exo_instance_with_no_live_runner_is_not_loaded(fake):
    """Measured against the real daemon: it held 33 RunnerShuttingDown, and
    an instance map that still listed the model. Reading `instances` alone
    would have reported a loaded model that is on its way out."""
    fake({"/state": {
        "instances": {"i1": {"shard_assignments": {"model_id": "some/model"}}},
        "runners": {"r1": {"RunnerShuttingDown": {}},
                    "r2": {"RunnerShuttingDown": {}}}}})
    rows = loaded._exo("http://x")
    assert len(rows) == 1
    assert rows[0].state == "offered"
    assert rows[0].name == "some/model"
    assert "no runner up" in rows[0].detail


def test_exo_runners_up_without_an_instance_still_report(fake):
    """Also measured live: 2 WarmingUp + 1 Loading against an EMPTY instance
    map. Something is taking memory; saying "nothing loaded" is the wrong
    answer in the direction that matters."""
    fake({"/state": {"instances": {},
                     "runners": {"a": {"RunnerWarmingUp": {}},
                                 "b": {"RunnerWarmingUp": {}},
                                 "c": {"RunnerLoading": {}}}}})
    rows = loaded._exo("http://x")
    assert len(rows) == 1
    assert rows[0].state == "loading"
    assert "3 runners starting" in rows[0].name


def test_exo_reports_loaded_when_a_runner_is_actually_up(fake):
    fake({"/state": {
        "instances": {"i1": {"shardAssignments": {"modelId": "m/x"}}},
        "runners": {"r1": {"RunnerReady": {}}}}})
    r = loaded._exo("http://x")[0]
    # Read, never driven: knurlogic does not unload exo's instances.
    assert r.state == "loaded" and not r.can_unload and r.ident == "i1"


def test_ollama_reads_ps_not_tags(fake):
    """`/api/tags` is the disk and `/api/ps` is the memory. Reading tags here
    would list every model ever pulled as though it were resident."""
    f = fake({"/api/ps": {"models": [
        {"name": "qwen3:8b", "model": "qwen3:8b", "size_vram": 5 << 30,
         "details": {"parameter_size": "8B"}}]}})
    rows = loaded._ollama("http://x")
    assert [u for u in f.asked] == ["http://x/api/ps"]
    assert rows[0].bytes_resident == 5 << 30
    assert rows[0].runtime == "ollama" and rows[0].can_unload


def test_an_openai_port_is_offered_not_loaded(fake):
    """mlx-lm exposes no "what is loaded" anywhere: /v1/models is what it
    COULD serve. Calling that residency invents the fact being asked for."""
    fake({"/v1/models": {"data": [{"id": "some-model"}]}})
    rows = loaded._openai_port("http://x")
    assert rows[0].state == "offered"
    assert rows[0].bytes_resident == 0


def test_our_own_port_is_read_as_ours_not_as_an_anonymous_mlx_server(fake):
    fake({"/status.json": {"schema": 2,
                           "artifact": {"name": "A", "model_type": "qwen3_5",
                                        "path": "/p/A"},
                           "memory": {"active_bytes": 3 << 30}}})
    rows = loaded._knurlogic("http://x")
    assert rows[0].runtime == "knurlogic"
    assert rows[0].bytes_resident == 3 << 30
    assert rows[0].ident == "/p/A"


def test_render_never_prints_an_unreported_size_as_zero(fake):
    doc = {"resident": [{"runtime": "mlx", "state": "offered", "name": "m",
                         "bytes_resident": 0, "detail": ""}],
           "bytes_resident": 0}
    out = loaded.render(doc)
    assert "--" in out and "0.0G" not in out


def test_nothing_running_is_an_ordinary_answer(monkeypatch):
    """`survey()` shells out to `top` and `ps` for the memory map, so stubbing
    only the HTTP layer let this read the developer's actual machine -- the
    same leak the discovery tests had against the real disk."""
    monkeypatch.setattr(loaded, "_get", lambda *a, **k: None)
    monkeypatch.setattr(loaded, "memory_map", lambda *a, **k: {})
    # And the server record: the survey reads ~/.cache/knurlogic/servers.json
    # now, and this test first failed because a real model WAS running.
    from knurlogic.machine import servers
    monkeypatch.setattr(servers, "registry", lambda: {})
    doc = loaded.survey()
    assert doc["resident"] == []
    assert "nothing reports a loaded model" in loaded.render(doc)


# --- where the RAM went -----------------------------------------------------

def test_a_shell_in_a_project_directory_is_not_a_runtime():
    """Both of these were attributed on the first pass, because the match ran
    against the whole command line: a shell whose cwd was named after a
    project, and a tail following a log. Reported as runtimes holding
    memory."""
    assert loaded._runtime_of("-zsh") == ""
    assert loaded._runtime_of("/bin/zsh /Users/x/vqlab/run.sh") == ""
    assert loaded._runtime_of("tail -f /Users/x/exo/log.txt") == ""
    assert loaded._runtime_of("grep -r knurlogic src/") == ""


def test_a_runtime_is_read_from_its_executable_or_its_module():
    assert loaded._runtime_of(
        "/opt/anaconda3/envs/exo/bin/python3.13 run.py") == "exo"
    assert loaded._runtime_of("/usr/bin/python3 -m knurlogic.cli serve") \
        == "knurlogic"
    assert loaded._runtime_of("python3 -m mlx_lm.server --model x") == "mlx-lm"
    assert loaded._runtime_of("python -m mlx_vlm.server") == "mlx-vlm"
    assert loaded._runtime_of("/usr/local/bin/ollama serve") == "ollama"
    assert loaded._runtime_of("/usr/bin/python3 other.py") == ""


def test_an_idle_runtime_is_reported_below_the_floor(monkeypatch):
    """An idle exo holding 163 MiB is an ANSWER -- nothing is loaded. Dropped
    under a floor it looks identical to exo not running at all, which is the
    question the panel exists to settle."""
    monkeypatch.setattr(loaded, "_footprints", lambda: ({
        1: 160 << 20,          # exo, idle, under the floor
        2: 4 << 30,            # something else, over it
        3: 100 << 20}, {}))    # something else, under it
    monkeypatch.setattr(loaded, "_commands", lambda: {
        1: "/opt/anaconda3/envs/exo/bin/python3.13",
        2: "/Applications/Thing.app/Contents/MacOS/Thing",
        3: "/usr/bin/something-small"})
    m = loaded.memory_map(floor=256 << 20)
    pids = {r["pid"] for r in m["processes"]}
    assert pids == {1, 2}                 # 3 dropped, 1 kept despite the floor
    assert m["by_runtime"] == {"exo": 160 << 20}


def test_the_unattributed_remainder_is_reported(monkeypatch):
    """"Where did the RAM go" is not answered by a list that sums to less
    than the machine and does not say so."""
    monkeypatch.setattr(loaded, "_footprints", lambda: (
        {1: 2 << 30, 2: 6 << 30}, {}))
    monkeypatch.setattr(loaded, "_commands", lambda: {
        1: "/envs/exo/bin/python3", 2: "/Applications/Other"})
    m = loaded.memory_map(floor=1 << 30)
    assert m["runtime_bytes"] == 2 << 30
    assert m["other_bytes"] == 6 << 30
    assert m["seen_bytes"] == 8 << 30


def test_available_memory_counts_the_cache_macos_will_hand_over(monkeypatch):
    """Two wrong answers preceded this, in opposite directions: installed
    minus the footprints (arithmetic on the wrong quantity), then top's
    "unused" (only pages free this instant, which read 1.6 GiB on a box with
    70 GiB available). What a model can actually have is free + inactive --
    the file cache is handed over on demand -- and that is the number exo
    reports and the one this had to match."""
    monkeypatch.setattr(loaded, "_footprints", lambda: (
        {1: 4 << 30},
        {"available_bytes": 70 << 30, "free_bytes": 2 << 30,
         "cached_bytes": 68 << 30, "wired_bytes": 10 << 30}))
    monkeypatch.setattr(loaded, "_commands", lambda: {1: "/envs/exo/bin/python3"})
    m = loaded.memory_map(floor=1 << 30)
    inst = m["installed_bytes"]
    assert m["free_bytes"] == 70 << 30         # available, not "unused"
    assert m["truly_free_bytes"] == 2 << 30    # kept, but not the headline
    assert m["used_bytes"] == inst - (70 << 30)


def test_available_memory_reads_the_real_vm_stat():
    """Not mocked: the parse has to survive macOS's own wording."""
    d = loaded.available_memory()
    if not d:
        return
    assert d["available_bytes"] >= d["free_bytes"]
    assert d["available_bytes"] == d["free_bytes"] + d["cached_bytes"]


def test_a_missing_physmem_line_falls_back_rather_than_lying(monkeypatch):
    monkeypatch.setattr(loaded, "_footprints", lambda: ({1: 4 << 30}, {}))
    monkeypatch.setattr(loaded, "_commands", lambda: {1: "/envs/exo/bin/python3"})
    m = loaded.memory_map(floor=1 << 30)
    assert m["from_os"] is False
    assert m["used_bytes"] == 4 << 30


def test_a_registered_server_is_found_on_a_port_nobody_guessed(monkeypatch):
    """Two models holding 33 GiB on 8092 and 8093 were invisible: the survey
    only probed a fixed list of ports. It reads the record now."""
    from knurlogic.machine import servers
    monkeypatch.setattr(loaded, "_get", lambda *a, **k: None)
    monkeypatch.setattr(loaded, "memory_map", lambda *a, **k: {})
    monkeypatch.setattr(servers, "registry", lambda: {
        8092: {"pid": 1, "artifact": "/m/Qwen3.6-35B-A3B"}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    (r,) = loaded.survey()["resident"]
    assert r["runtime"] == "knurlogic" and r["state"] == "loading"
    assert r["where"].endswith(":8092")


def test_a_malformed_server_record_is_skipped_not_a_keyerror(tmp_path,
                                                            monkeypatch):
    import json as _j
    from knurlogic.machine import servers
    p = tmp_path / "servers.json"
    p.write_text(_j.dumps({"8080": {"pid": 12, "log": "x"},
                           "8081": {"log": "no pid"},
                           "8082": {"pid": "nope"}, "junk": {"pid": 1}}))
    monkeypatch.setattr(servers, "registry_path", lambda: p)
    assert list(servers.registry()) == [8080]


def test_knurlogics_own_store_is_scanned_and_movable(tmp_path, monkeypatch):
    import json as _j
    from knurlogic.machine import discover
    d = tmp_path / "Models" / "tiny"
    d.mkdir(parents=True)
    (d / "config.json").write_text(_j.dumps({"model_type": "qwen3_5"}))
    (d / "model.safetensors").write_bytes(b"\0" * 8)
    monkeypatch.setenv("KNURLOGIC_MODELS", str(tmp_path / "Models"))
    rows = discover.find(stores=["knurlogic"])
    assert [r.name for r in rows] == ["tiny"] and rows[0].store == "knurlogic"
