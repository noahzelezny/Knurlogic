"""The Settings audit (2026-09-28): what each knob claims against what the
code does -- the context length's cap, the prompt chunk's one name, the
dead prompt concurrency, and a peer's model's settings reaching the peer."""

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from knurlogic.interfaces.page import server as ui
from knurlogic.interfaces.page import documents as web
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import settings as S
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30

#: Qwen3.8-Flash-Next's text config: 262144, rope_type default, no YaRN
FLASH_NEXT = {"model_type": "qwen3_5_moe", "text_config": {
    "max_position_embeddings": 262144,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25}}}


def _art(cfg=None, model_type="qwen3_5"):
    return Artifact(path=Path("/nonexistent"), model_type=model_type,
                    model_file=None, bytes_on_disk=20 * GIB, hidden_size=4096,
                    moe_intermediate_size=1024, vq_other={},
                    raw_config=cfg or {})


# --- the context length -----------------------------------------------------

def test_the_window_is_max_position_embeddings_without_yarn():
    w, why = S.model_window(FLASH_NEXT)
    assert w == 262144 and "no YaRN" in why


def test_yarn_rope_scaling_extends_the_window():
    cfg = {"max_position_embeddings": 262144, "rope_scaling": {
        "rope_type": "yarn", "factor": 4.0,
        "original_max_position_embeddings": 262144}}
    w, why = S.model_window(cfg)
    assert w == 1048576 and "YaRN" in why


def test_a_context_past_the_models_window_is_refused():
    """1048576 was taken for a 262144-token model: the engine computes rope
    for any position, so it would just run past its trained length."""
    why = S.check_knob("KNURLOGIC_CONTEXT_LENGTH", "1048576", 262144)
    assert why and "maximum is 262,144" in why
    assert S.check_knob("KNURLOGIC_CONTEXT_LENGTH", "262144", 262144) is None
    assert "whole number" in S.check_knob("KNURLOGIC_CONTEXT_LENGTH", "1e6")
    a = _art(FLASH_NEXT)
    assert "262,144" in web.refuse_sets(
        a, {"KNURLOGIC_CONTEXT_LENGTH": "1048576"})
    assert web.refuse_sets(a, {"KNURLOGIC_CONTEXT_LENGTH": "131072"}) is None


def test_the_control_stops_at_the_window_and_says_so():
    a = _art(FLASH_NEXT)
    r = resolve(a, 96 * GIB)
    assert r.env["KNURLOGIC_CONTEXT_LENGTH"] == "262144"
    steps = r.ranges["KNURLOGIC_CONTEXT_LENGTH"]
    assert steps[-1] == 262144 and 1048576 not in steps
    lim = web.knob_limit(a, "KNURLOGIC_CONTEXT_LENGTH")
    assert lim["max"] == 262144 and "262,144" in lim["max_why"]
    assert web.knob_limit(a, "VQ_DECODE_CHUNK") == {}


def test_the_context_length_is_the_engines_and_live(tmp_path):
    """It was hidden as 'no-effect' on an artifact whose bundled runtime
    does not read it -- but the scheduler reads it at every admission."""
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py"}')
    (tmp_path / "model.py").write_text("import os\n")
    a = Artifact.load(tmp_path)
    assert web.knob_reach(a, "KNURLOGIC_CONTEXT_LENGTH",
                          ("KNURLOGIC_CONTEXT_LENGTH",))[0] == "live"


def test_other_knobs_are_checked_for_type_and_range():
    assert S.check_knob("VQ_DECODE_CHUNK", "64")          # capped at 32
    assert S.check_knob("VQ_DECODE_CHUNK", "16") is None
    assert S.check_knob("KNURLOGIC_CACHE_LIMIT_GB", "40")
    assert S.check_knob("KNURLOGIC_MTP", "maybe")
    assert S.check_knob("KNURLOGIC_KV_BITS", "3")
    assert S.check_knob("KNURLOGIC_PRESET", "turbo")
    assert S.check_knob("KNURLOGIC_CROSS_CHIP", "auto") is None
    assert S.check_knob("VQ_GEMMSEG_PIPE", "anything") is None  # not ours


# --- the prompt chunk: one knob, one name -----------------------------------

def test_the_prompt_chunk_is_emitted_under_knurlogics_own_name():
    """VQLAB_PREFILL_CHUNK is read by no bundled runtime -- only the engine,
    under either name -- so it was the general prompt chunk filed under
    VQ. The cache limit keeps its legacy name: runtimes read that one."""
    assert S.default_alias("prefill_chunk") == "KNURLOGIC_PREFILL_CHUNK"
    assert S.default_alias("cache_limit_gb") == "VQLAB_CACHE_LIMIT_GB"
    env = resolve(_art(), 96 * GIB).env
    assert "KNURLOGIC_PREFILL_CHUNK" in env
    assert "VQLAB_PREFILL_CHUNK" not in env
    assert "KNURLOGIC_PREFILL_CHUNK" in S.KNOB_DOC
    assert "VQLAB_PREFILL_CHUNK" not in S.KNOB_DOC


def test_the_legacy_name_is_still_accepted_and_beats_the_resolver():
    sets = S.canonical_sets({"VQLAB_PREFILL_CHUNK": "2048"})
    assert sets == {"KNURLOGIC_PREFILL_CHUNK": "2048"}
    env = {**resolve(_art(), 96 * GIB).env, **sets}
    assert S.engine_settings(env)["prefill_step_size"] == 2048
    ok, bad = ui.clean_sets({"VQLAB_PREFILL_CHUNK": "1024"})
    assert ok and not bad
    both = S.canonical_sets({"VQLAB_PREFILL_CHUNK": "2048",
                             "KNURLOGIC_PREFILL_CHUNK": "1024"})
    assert both == {"KNURLOGIC_PREFILL_CHUNK": "1024"}


from test_cluster_jobs import cache  # noqa: E402,F401 (a fixture)


def test_a_cluster_job_runs_the_saved_prompt_chunk(cache, monkeypatch):
    """The ring's chunk was 512 whatever was saved, and serve puts the
    ring's over any --set: a saved launch setting shown, never run."""
    from test_cluster_jobs import jaccl_launch, C
    monkeypatch.setattr(C, "BAD_CABLES", {})
    out, got = jaccl_launch(monkeypatch, {
        "cable": "127.0.1.x", "sets": {"KNURLOGIC_PREFILL_CHUNK": "2048"}})
    spec = next(d for u, d in got if u.endswith(C.PREPARE_PATH))
    assert spec["prefill_chunk"] == 2048, out
    out, got = jaccl_launch(monkeypatch, {"cable": "127.0.1.x"})
    spec = next(d for u, d in got if u.endswith(C.PREPARE_PATH))
    assert spec["prefill_chunk"] == C.PREFILL_CHUNK
    out, _ = jaccl_launch(monkeypatch, {
        "cable": "127.0.1.x", "sets": {"KNURLOGIC_PREFILL_CHUNK": "9999"}})
    assert "between" in out.get("error", ""), out


# --- prompt concurrency: read by nothing ------------------------------------

def test_prompt_concurrency_is_not_offered():
    """knurlogic's engine admits one row per step; the knob went with
    mlx-lm's server and was shown as 'needs reload' over nothing."""
    for tune in S.PRESETS:
        env = resolve(_art(), 24 * GIB, tune=tune).env
        assert "KNURLOGIC_PROMPT_CONCURRENCY" not in env, tune
    assert "KNURLOGIC_PROMPT_CONCURRENCY" not in S.KNOB_DOC
    assert "KNURLOGIC_PROMPT_CONCURRENCY" not in S.ENGINE_KNOB_NAMES
    # a launch setting saved before still passes, so it cannot fail a launch
    assert ui.clean_sets({"KNURLOGIC_PROMPT_CONCURRENCY": "1"})[0]


# --- a peer's model: read AND changed on the peer ---------------------------

def test_a_peers_settings_go_through_its_page_by_port(monkeypatch):
    monkeypatch.setitem(ui._PEER_TARGETS, "http://192.0.2.2:8080",
                        {"machine": "M4", "relay": "http://192.0.2.2:8899"})
    assert ui.upstream("http://192.0.2.2:8080", "/settings.json") == \
        "http://192.0.2.2:8899/peer/settings.json?port=8080"
    # chat still goes by model name
    assert ui.upstream("http://192.0.2.2:8080", "/v1/models") == \
        "http://192.0.2.2:8899/peer/v1/models"
    sent = []
    monkeypatch.setattr(ui, "known_target", lambda b: True)
    code, doc = ui.apply_settings(
        "http://192.0.2.2:8080", b'{"VQ_DECODE_CHUNK": "16"}',
        post=lambda u, d, t: sent.append(u) or (200, b'{"applied": {}}'))
    assert code == 200 and sent == [
        "http://192.0.2.2:8899/peer/settings.json?port=8080"]


def test_peer_settings_only_reaches_a_server_this_machine_started(
        monkeypatch):
    from knurlogic.machine import servers
    monkeypatch.setattr(ui, "registry", lambda: {8080: {"pid": 1}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    calls = []

    def call(u, d, t):
        calls.append((u, d))
        return 200, b'{"ok": 1}'
    assert ui.peer_settings("GET", {"port": ["9"]}, b"", call)[0] == 404
    assert ui.peer_settings("GET", {}, b"", call)[0] == 400
    assert ui.peer_settings("POST", {"port": ["8080"]}, b"[1]", call)[0] \
        == 400
    code, doc = ui.peer_settings("GET", {"port": ["8080"],
                                         "tune": ["fast"]}, b"", call)
    assert code == 200 and calls[-1] == (
        "http://127.0.0.1:8080/settings.json?tune=fast", None)
    code, _ = ui.peer_settings("POST", {"port": ["8080"]},
                               b'{"VQ_DECODE_CHUNK": "16"}', call)
    assert code == 200 and calls[-1] == (
        "http://127.0.0.1:8080/settings.json", b'{"VQ_DECODE_CHUNK": "16"}')


def test_a_peers_live_knob_reaches_the_peers_model_end_to_end(monkeypatch):
    """This page -> the peer page's /peer/settings.json -> the model server
    on the peer's loopback. Before, /peek and /apply went to the peer's
    loopback address from HERE (502) and the peer's model fell back to a
    preview, every knob a launch setting."""
    from test_ui_peer_relay import _serve, Peers
    from http.server import BaseHTTPRequestHandler
    from knurlogic.machine import servers
    seen = []

    class M(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, doc):
            out = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            seen.append(("GET", self.path))
            self._json({"knobs": [], "live": {"tune": "balanced"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(("POST", self.path, json.loads(self.rfile.read(n))))
            self._json({"applied": {"VQ_DECODE_CHUNK": "applied"}})

    model, mport = _serve(M)
    monkeypatch.setattr(ui, "registry", lambda: {mport: {"pid": 1}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    peer_page, pport = _serve(ui.make_handler({}))
    here, hport = _serve(ui.make_handler({}))
    try:
        p = SimpleNamespace(name="M4", host="127.0.0.1", port=pport,
                            state="answering", key=f"127.0.0.1:{pport}",
                            id="m4", found_by={"bonjour"})
        monkeypatch.setattr(ui, "PEERS", Peers(p))
        row = {"runtime": "knurlogic", "name": "m", "state": "ready",
               "where": f"http://127.0.0.1:{mport}"}
        ui.peer_residency(ui.PEERS, fetch=lambda url, t: {"resident": [row]})
        base = f"http://127.0.0.1:{mport}"
        assert base in ui._PEER_TARGETS
        with urllib.request.urlopen(
                f"http://127.0.0.1:{hport}/peek?where={base}"
                f"&path=/settings.json&tune=fast", timeout=5) as r:
            assert json.loads(r.read())["live"]["tune"] == "balanced"
        assert seen[-1] == ("GET", "/settings.json?tune=fast")
        req = urllib.request.Request(
            f"http://127.0.0.1:{hport}/apply?where={base}",
            data=b'{"VQ_DECODE_CHUNK": "16"}', method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert "VQ_DECODE_CHUNK" in json.loads(r.read())["applied"]
        assert seen[-1] == ("POST", "/settings.json",
                            {"VQ_DECODE_CHUNK": "16"})
    finally:
        for s in (model, peer_page, here):
            s.shutdown()
        ui._PEER_TARGETS.clear()
