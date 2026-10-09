"""What is resident, across runtimes -- and the distinctions that matter.

The failure these pin is a page that says a model is loaded when it is not.
Every runtime here reports something slightly different, and flattening them
into "has a model" loses the only fact anybody opened the page for.
"""

import pytest

from knurlogic.machine import loaded
from knurlogic.machine.memory import footprint


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
    monkeypatch.setattr(footprint, "memory_map", lambda *a, **k: {})
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
    assert footprint._runtime_of("-zsh") == ""
    assert footprint._runtime_of("/bin/zsh /Users/x/vqlab/run.sh") == ""
    assert footprint._runtime_of("tail -f /Users/x/exo/log.txt") == ""
    # a console script run by its interpreter names itself only as the
    # script (`knurlogic serve` must not read as "everything else")
    assert footprint._runtime_of(
        "/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python."
        "framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python "
        "/Users/x/kl/venv/bin/knurlogic serve /m --port 8097") \
        == "knurlogic"
    assert footprint._runtime_of("python3 -u run.py") == ""
    assert footprint._runtime_of("grep -r knurlogic src/") == ""


def test_a_runtime_is_read_from_its_executable_or_its_module():
    assert footprint._runtime_of(
        "/opt/anaconda3/envs/exo/bin/python3.13 run.py") == "exo"
    assert footprint._runtime_of("/usr/bin/python3 -m knurlogic.cli serve") \
        == "knurlogic"
    assert footprint._runtime_of("python3 -m mlx_lm.server --model x") == "mlx-lm"
    assert footprint._runtime_of("python -m mlx_vlm.server") == "mlx-vlm"
    assert footprint._runtime_of("/usr/local/bin/ollama serve") == "ollama"
    assert footprint._runtime_of("/usr/bin/python3 other.py") == ""


def test_a_script_names_its_process_over_the_interpreters_env():
    # vqlab's benchmark run with exo's python read as exo on the page's
    # legend (M3, 2026-10-01)
    assert footprint._runtime_of(
        "/opt/anaconda3/envs/exo/bin/python /Users/x/.vqlab/queues/q/tree/"
        "src/vqlab/bench/speed_pair.py /Volumes/S/pin_Qwen") == "vqlab"


def test_an_idle_runtime_is_reported_below_the_floor(monkeypatch):
    """An idle exo holding 163 MiB is an ANSWER -- nothing is loaded. Dropped
    under a floor it looks identical to exo not running at all, which is the
    question the panel exists to settle."""
    monkeypatch.setattr(footprint, "_footprints", lambda: ({
        1: 160 << 20,          # exo, idle, under the floor
        2: 4 << 30,            # something else, over it
        3: 100 << 20}, {}))    # something else, under it
    monkeypatch.setattr(footprint, "_commands", lambda: {
        1: "/opt/anaconda3/envs/exo/bin/python3.13",
        2: "/Applications/Thing.app/Contents/MacOS/Thing",
        3: "/usr/bin/something-small"})
    m = footprint.memory_map(floor=256 << 20)
    pids = {r["pid"] for r in m["processes"]}
    assert pids == {1, 2}                 # 3 dropped, 1 kept despite the floor
    assert m["by_runtime"] == {"exo": 160 << 20}


def test_the_unattributed_remainder_is_reported(monkeypatch):
    """"Where did the RAM go" is not answered by a list that sums to less
    than the machine and does not say so."""
    monkeypatch.setattr(footprint, "_footprints", lambda: (
        {1: 2 << 30, 2: 6 << 30}, {}))
    monkeypatch.setattr(footprint, "_commands", lambda: {
        1: "/envs/exo/bin/python3", 2: "/Applications/Other"})
    m = footprint.memory_map(floor=1 << 30)
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
    monkeypatch.setattr(footprint, "_footprints", lambda: (
        {1: 4 << 30},
        {"available_bytes": 70 << 30, "free_bytes": 2 << 30,
         "cached_bytes": 68 << 30, "wired_bytes": 10 << 30}))
    monkeypatch.setattr(footprint, "_commands", lambda: {1: "/envs/exo/bin/python3"})
    m = footprint.memory_map(floor=1 << 30)
    inst = m["installed_bytes"]
    assert m["free_bytes"] == 70 << 30         # available, not "unused"
    assert m["truly_free_bytes"] == 2 << 30    # kept, but not the headline
    assert m["used_bytes"] == inst - (70 << 30)


def test_available_memory_reads_the_real_vm_stat():
    """Not mocked: the parse has to survive macOS's own wording."""
    d = footprint.available_memory()
    if not d:
        return
    assert d["available_bytes"] >= d["free_bytes"]
    assert d["available_bytes"] == (d["free_bytes"] + d["cached_bytes"]
                                    + d["purgeable_bytes"])


def test_a_missing_physmem_line_falls_back_rather_than_lying(monkeypatch):
    monkeypatch.setattr(footprint, "_footprints", lambda: ({1: 4 << 30}, {}))
    monkeypatch.setattr(footprint, "_commands", lambda: {1: "/envs/exo/bin/python3"})
    m = footprint.memory_map(floor=1 << 30)
    assert m["from_os"] is False
    assert m["used_bytes"] == 4 << 30


def test_a_registered_server_is_found_on_a_port_nobody_guessed(monkeypatch):
    """Two models holding 33 GiB on 8092 and 8093 were invisible: the survey
    only probed a fixed list of ports. It reads the record now."""
    from knurlogic.machine import servers
    monkeypatch.setattr(loaded, "_get", lambda *a, **k: None)
    monkeypatch.setattr(footprint, "memory_map", lambda *a, **k: {})
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


def test_available_memory_counts_file_backed_and_purgeable_once(monkeypatch):
    out = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             500.
Pages inactive:                           200.
Pages speculative:                         50.
Pages purgeable:                          300.
File-backed pages:                        150.
"""
    import subprocess as sp
    monkeypatch.setattr(sp, "run", lambda *a, **k: sp.CompletedProcess(
        a, 0, stdout=out))
    # free + file-backed + purgeable; speculative is already inside
    # file-backed (vm_stat: active+inactive+speculative == file+anonymous)
    # and inactive is not added
    assert footprint.available_memory()["available_bytes"] == 550 * 16384


def test_a_module_in_exos_env_is_that_module_not_exo():
    """vqlab run from exo's conda env read as exo: the env path matched
    before the `-m` module was asked."""
    env = "/opt/anaconda3/envs/exo/bin/python"
    assert footprint._runtime_of(f"{env} -m vqlab.cli pin /m") == "vqlab"
    assert footprint._runtime_of(f"{env} -m some.tool /m") == ""
    assert footprint._runtime_of(f"{env} -m exo.main") == "exo"
    assert footprint._runtime_of("/opt/anaconda3/envs/exo/bin/exo") == "exo"


def test_a_cluster_row_holds_every_ranks_bytes():
    d = {"ranks": [{"rank": 0, "active_bytes": 104 << 30},
                   {"rank": 1, "active_bytes": 75 << 30}]}
    assert loaded._held(d, {"active_bytes": 104 << 30}) == 179 << 30
    assert loaded._held({}, {"active_bytes": 5}) == 5


_M4_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               {free}.
Pages active:                             {act}.
Pages inactive:                           {inact}.
Pages speculative:                        {spec}.
Pages wired down:                         {wired}.
Pages purgeable:                          {purg}.
File-backed pages:                        {file}.
Anonymous pages:                          {anon}.
Pages occupied by compressor:             {comp}.
"""


def _m4_vm_stat(monkeypatch, **over):
    import subprocess as sp
    g = dict(free=1.1, act=60, inact=60, spec=0, wired=4.5, purg=0,
             file=1.8, anon=118.7, comp=1.0)
    g.update(over)
    out = _M4_VM_STAT.format(**{k: int(v * 65536) for k, v in g.items()})
    monkeypatch.setattr(sp, "run", lambda *a, **k: sp.CompletedProcess(
        a, 0, stdout=out))


def test_inactive_anonymous_pages_are_used_not_available(monkeypatch):
    """Live on a 128 GiB M4: 60 GiB of a MoE's idle experts sat in inactive
    ANONYMOUS pages, which macOS reclaims only by swapping. Counting them
    as available read 66.9 used when ~125 was."""
    _m4_vm_stat(monkeypatch)
    d = footprint.available_memory()
    gib = 1 << 30
    assert d["available_bytes"] == pytest.approx(2.9 * gib, abs=gib / 100)
    assert d["used_bytes"] == pytest.approx(124.2 * gib, abs=gib / 100)


def test_purgeable_and_speculative_are_available_once(monkeypatch):
    _m4_vm_stat(monkeypatch, spec=0.5, purg=2)
    d = footprint.available_memory()
    gib = 1 << 30
    # free + file-backed + purgeable (speculative is inside file-backed,
    # counted once); purgeable leaves used
    assert d["available_bytes"] == pytest.approx(
        (1.1 + 1.8 + 2) * gib, abs=gib / 100)
    assert d["used_bytes"] == pytest.approx(
        (116.7 + 4.5 + 1.0) * gib, abs=gib / 100)


def test_the_map_takes_used_from_the_os_and_splits_swap(monkeypatch):
    from knurlogic.machine import metrics
    gib = 1 << 30
    monkeypatch.setattr(footprint, "_footprints", lambda: (
        {1: 109 * gib, 2: 3 * gib},
        {"available_bytes": 3 * gib, "free_bytes": gib,
         "cached_bytes": 2 * gib, "wired_bytes": 4 * gib,
         "used_bytes": 100 * gib}))
    monkeypatch.setattr(footprint, "_commands", lambda: {
        1: "/v/bin/python -m knurlogic serve", 2: "/Applications/Other"})
    monkeypatch.setattr(metrics, "swap", lambda: 12 * gib)
    m = footprint.memory_map(floor=1 << 30)
    assert m["used_bytes"] == 100 * gib
    # footprints (112) exceed used (100): the excess is swapped, and the
    # runtime's row is its resident part, so the rows add up
    assert m["swapped_by_runtime"] == {"knurlogic": 12 * gib}
    assert m["by_runtime"] == {"knurlogic": 97 * gib}
    assert m["footprint_by_runtime"] == {"knurlogic": 109 * gib}
    assert m["other_bytes"] == 3 * gib and m["swap_bytes"] == 12 * gib


def test_an_instance_card_reads_the_same_sample_and_says_its_swap():
    mm = {"processes": [{"pid": 7, "bytes": 109 << 30}],
          "footprint_by_runtime": {"knurlogic": 109 << 30},
          "swapped_by_runtime": {"knurlogic": 12 << 30}}
    r = loaded.Resident(runtime="knurlogic", name="m",
                        where="http://127.0.0.1:8100", bytes_resident=5)
    loaded._from_the_map([r], {8100: {"pid": 7}}, mm)
    assert r.bytes_resident == 109 << 30 and r.swapped_bytes == 12 << 30


def test_a_port_that_answers_before_the_weights_are_in_is_loading(monkeypatch):
    doc = {"schema": 1, "artifact": {"name": "m", "size_bytes": 100},
           "memory": {"active_bytes": 40}, "load": {"state": "loading"}}
    monkeypatch.setattr(loaded, "_get", lambda *a, **k: doc)
    (r,) = loaded._knurlogic("http://127.0.0.1:1")
    assert r.state == "loading" and "40%" in r.detail
    doc["load"] = {"state": "ready"}
    assert loaded._knurlogic("http://127.0.0.1:1")[0].state == "loaded"
    doc["load"] = {"state": "warming"}
    assert loaded._knurlogic("http://127.0.0.1:1")[0].state == "warming"


def test_the_row_says_mtp_and_vision_only_when_they_run(fake):
    base = {"schema": 2, "artifact": {"name": "A", "model_type": "qwen3_5",
                                      "path": "/p/A"}, "memory": {}}
    fake({"/status.json": {**base, "drafting": {"drafts_now": True},
                           "vision": {"served": True}}})
    assert loaded._knurlogic("http://x")[0].detail == "qwen3_5 · VISION · MTP"
    fake({"/status.json": {**base, "drafting": {"drafts_now": False},
                           "vision": {"served": False}}})
    assert loaded._knurlogic("http://x")[0].detail == "qwen3_5"


def test_our_server_busy_generating_keeps_its_card(monkeypatch):
    """A server busy generating missed /status.json's 1.5 s read while
    /v1/models still answered, and the card turned into an anonymous
    "mlx ... served on demand . offered" until it caught up. It keeps the
    last card it had, marked busy."""
    from knurlogic.machine import servers
    status = {"schema": 2, "artifact": {"name": "GLM", "path": "/m/GLM"},
              "memory": {"active_bytes": 3 << 30}}
    up = {"ok": True}

    def get(url, timeout=1.5):
        if url.endswith("/status.json"):
            return status if up["ok"] else None
        if url.endswith("/v1/models"):
            return {"data": [{"id": "GLM"}]}
        return None
    monkeypatch.setattr(loaded, "_get", get)
    monkeypatch.setattr(footprint, "memory_map", lambda *a, **k: {})
    monkeypatch.setattr(loaded, "_LAST", {})
    monkeypatch.setattr(servers, "registry", lambda: {
        8080: {"pid": 1, "artifact": "/m/GLM"}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    listen = {8080: 1}
    monkeypatch.setattr(servers, "listening_serves", lambda: dict(listen))

    def ours():
        (r,) = [r for r in loaded.survey()["resident"]
                if r["where"].endswith(":8080")]
        return r
    r = ours()
    assert r["runtime"] == "knurlogic" and r["state"] == "loaded"
    up["ok"] = False
    r = ours()
    assert r["runtime"] == "knurlogic" and r["state"] == "loaded"
    assert r["detail"].endswith("busy")
    # its port closed while the process lives: an unload on its way out,
    # not a busy model (a READY · busy card for a server that was gone)
    listen.clear()
    r = ours()
    assert r["state"] == "stopping" and r["can_unload"] is False
