"""Finding models in four tools' stores -- and not overclaiming about them.

Every test passes `include_defaults=False`. Without it they scanned the real
machine and one of them asserted against 54 actual models -- a test that
depends on what happens to be on the developer's disk is not a test.

Four tools keep four stores and none of them looks at the others, which is
why a person with 5 TB of weights on disk still sees an empty list. The trap
is that FINDING a model is not being able to RUN it: ollama and most of LM
Studio hold GGUF, and this engine loads safetensors. Checked, not assumed --
`mlx_lm.gguf` exposes `convert_to_gguf` and no loader.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.machine import discover

GIB = 1 << 30


def _mlx_model(d: Path, model_type="qwen3_5", size=4096):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    (d / "model.safetensors").write_bytes(b"\0" * size)
    return d


def _gguf_model(d: Path, size=2048):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.gguf").write_bytes(b"\0" * size)
    return d


def test_it_reads_a_flat_store_and_an_hf_cache(tmp_path):
    _mlx_model(tmp_path / "store" / "SomeOrg--Some-Model")
    _mlx_model(tmp_path / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "abc")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store", tmp_path / "hub"])
    names = {r.name for r in rows}
    assert "SomeOrg--Some-Model" in names
    # An HF cache entry's real name lives in the models--org--repo directory.
    assert "Qwen/Qwen3-8B" in names


def test_a_gguf_model_is_found_and_reported_as_not_servable(tmp_path):
    """A menu of entries that 500 on click is the same 'why did it fail' that
    doctor exists to end."""
    _gguf_model(tmp_path / "store" / "llama-3.1-8b-q4")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store"])
    g = next(r for r in rows if r.name == "llama-3.1-8b-q4")
    assert g.format == "gguf" and g.servable is False
    assert "safetensors" in g.why


def test_an_embedder_is_found_but_not_listed_as_a_servable_chat_model(tmp_path):
    """`models/fit` (P5, vision v2) must stop listing non-chat models --
    embedders, whisper, siglip, background removers -- as servable chat
    models. This has an mlx runtime and safetensors weights, so before this
    check it was `servable=True` and showed up in the picker next to real
    chat models, where loading it produces a server that 400s on the first
    /v1/chat/completions."""
    _mlx_model(tmp_path / "store" / "bge-embedder", model_type="bert")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store"])
    e = next(r for r in rows if r.name == "bge-embedder")
    assert e.servable is False
    assert "not a chat model" in e.why


def test_a_chat_model_type_stays_servable(tmp_path):
    """The mutation check for the test above: break the filter (treat every
    model_type as chat-servable) and this must go red, which it does --
    confirmed by hand while writing the fix, restored here."""
    _mlx_model(tmp_path / "store" / "some-chat-model", model_type="qwen3_5")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store"])
    c = next(r for r in rows if r.name == "some-chat-model")
    assert c.servable is True and c.why == ""


def test_it_does_not_descend_into_an_artifact(tmp_path):
    """A model directory holds shards, subfolders and sometimes a nested
    snapshot; walking into it would report one model several times."""
    d = _mlx_model(tmp_path / "store" / "Model-A")
    _mlx_model(d / "inner")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store"])
    assert [r.name for r in rows] == ["Model-A"]


def test_the_same_path_reached_twice_is_reported_once(tmp_path):
    _mlx_model(tmp_path / "store" / "Model-A")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store", tmp_path / "store"])
    assert len(rows) == 1


def test_render_keeps_found_loadable_and_fits_apart(tmp_path):
    """Three counts for three different questions: is it here, can this
    engine read it, will it fit."""
    _mlx_model(tmp_path / "store" / "Small", size=1024)
    _mlx_model(tmp_path / "store" / "Huge", size=8192)
    _gguf_model(tmp_path / "store" / "Ggufy")
    rows = discover.find(include_defaults=False, extra=[tmp_path / "store"])
    out = discover.render(rows, working_set_bytes=4096)
    assert "3 found" in out and "2 in a format this engine loads" in out
    assert "1 that fit one box" in out
    assert "1 are GGUF" in out
    assert "needs more than this box" in out


def test_an_empty_machine_says_where_it_looked(tmp_path):
    out = discover.render([])
    assert "no models found" in out and "Looked in" in out


def test_exo_model_dirs_follow_exos_own_resolution():
    """On macOS exo's data home is ~/.exo, not XDG -- the default that went
    missing and hid a whole external store."""
    from knurlogic.machine.discover import exo_model_dirs
    assert exo_model_dirs({}, "darwin", "/h") == [Path("/h/.exo/models")]
    assert exo_model_dirs({}, "linux", "/h") == [
        Path("/h/.local/share/exo/models")]
    assert exo_model_dirs({"XDG_DATA_HOME": "/x"}, "linux", "/h") == [
        Path("/x/exo/models")]
    got = exo_model_dirs({"EXO_DEFAULT_MODELS_DIR": "/d",
                          "EXO_MODELS_DIRS": "/a:/b",
                          "EXO_MODELS_READ_ONLY_DIRS": "/r"}, "darwin", "/h")
    assert got == [Path("/d"), Path("/a"), Path("/b"), Path("/r")]
