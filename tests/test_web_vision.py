"""P5's changes: the Anthropic->OpenAI image shim (`interfaces/http/messages.py`),
and the `vision` field on `/models.json` and `/loaded.json`.

No mlx, no PIL, no real model: `interfaces/page/documents.py` reads
`served_vision()` and `registry.registered()`, both stdlib-only per the
frozen contract (`docs/design/vision-contracts.md`), and this file proves it
stays that way by never importing anything that would drag mlx in.
"""
import json

from knurlogic.interfaces.http import messages
from knurlogic.interfaces.page import documents


# --- anthropic_images_to_openai --------------------------------------------

def test_base64_image_block_becomes_a_data_url_image_part():
    content = [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": "QUJD"}},
        {"type": "text", "text": "what is this?"},
    ]
    out = messages.anthropic_images_to_openai(content)
    assert out[0] == {"type": "image_url",
                      "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert out[1] == {"type": "text", "text": "what is this?"}


def test_url_source_block_passes_the_url_through():
    content = [{"type": "image", "source": {"type": "url",
                                            "url": "https://example/x.png"}}]
    out = messages.anthropic_images_to_openai(content)
    assert out == [{"type": "image_url",
                    "image_url": {"url": "https://example/x.png"}}]


def test_already_openai_shaped_content_is_unchanged():
    content = [{"type": "image_url", "image_url": {"url": "data:..."}}]
    assert messages.anthropic_images_to_openai(content) == content


def test_plain_string_content_passes_through():
    assert messages.anthropic_images_to_openai("just text") == "just text"


def test_unrecognised_block_passes_through_rather_than_dropping():
    """A text block, or an image block shaped unexpectedly, must not
    silently vanish -- that is a worse bug than one left for the engine's
    own (clear) refusal."""
    content = [{"type": "text", "text": "hello"},
              {"type": "image", "source": {"type": "base64"}}]  # no data
    out = messages.anthropic_images_to_openai(content)
    assert out[0] == {"type": "text", "text": "hello"}
    assert out[1] == content[1]


def test_default_media_type_when_missing():
    content = [{"type": "image",
               "source": {"type": "base64", "data": "AAA="}}]
    out = messages.anthropic_images_to_openai(content)
    assert out[0]["image_url"]["url"] == "data:image/png;base64,AAA="


# --- vision field on /models.json and /loaded.json -------------------------

def test_models_document_reports_vision_capable(monkeypatch, tmp_path):
    from knurlogic.machine.discover import Found

    p = tmp_path / "m"
    p.mkdir()
    monkeypatch.setattr(
        "knurlogic.machine.discover.find",
        lambda *a, **k: [Found(name="m", path=p, store="given",
                               format="mlx", bytes_on_disk=1 << 20,
                               model_type="qwen3_5", servable=True)])
    monkeypatch.setattr(
        "knurlogic.engine.vision.registry.registered", lambda mt: True)
    doc = documents.models_document()({})
    assert doc["models"][0]["vision"] is True


def test_models_document_false_for_a_non_vision_family(monkeypatch, tmp_path):
    from knurlogic.machine.discover import Found

    p = tmp_path / "m"
    p.mkdir()
    monkeypatch.setattr(
        "knurlogic.machine.discover.find",
        lambda *a, **k: [Found(name="m", path=p, store="given",
                               format="mlx", bytes_on_disk=1 << 20,
                               model_type="llama", servable=True)])
    monkeypatch.setattr(
        "knurlogic.engine.vision.registry.registered", lambda mt: False)
    doc = documents.models_document()({})
    assert doc["models"][0]["vision"] is False


def _reset_loaded_cache():
    """`web._LOADED` is a module-level TTL cache (deliberately, so a status
    poll is free) -- a test that does not clear it sees the PREVIOUS test's
    served-vision answer, not the one it just set up."""
    documents._LOADED["doc"] = None
    documents._LOADED["at"] = 0.0


def test_loaded_document_carries_served_vision_spec(monkeypatch):
    from knurlogic.engine.vision import VisionSpec

    _reset_loaded_cache()
    spec = VisionSpec(family="gemma4", image_token_id=7, patch=14,
                      merge=None, min_pixels=1, max_pixels=2,
                      fixed_tokens=256, proc_hash="ab" * 16)
    monkeypatch.setattr("knurlogic.machine.loaded.survey",
                        lambda: {"resident": [], "runtimes": []})
    monkeypatch.setattr("knurlogic.engine.vision.served_vision",
                        lambda: spec)
    doc = documents.loaded_document()({})
    assert doc["vision"]["family"] == "gemma4"
    assert doc["vision"]["fixed_tokens"] == 256


def test_loaded_document_vision_none_when_nothing_served(monkeypatch):
    _reset_loaded_cache()
    monkeypatch.setattr("knurlogic.machine.loaded.survey",
                        lambda: {"resident": [], "runtimes": []})
    monkeypatch.setattr("knurlogic.engine.vision.served_vision",
                        lambda: None)
    doc = documents.loaded_document()({})
    assert doc["vision"] is None


def test_web_module_never_imports_mlx_or_pil():
    """Consuming served_vision() and registry.registered() must not drag
    mlx or PIL into `interfaces/` -- only `engine/` may import mlx (a repo
    wide test enforces it); this is the P5-local half of that guarantee."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from knurlogic.interfaces.page import documents\n"
        "documents.models_document()({})\n"
        "documents.loaded_document()({})\n"
        "assert 'mlx' not in sys.modules, sorted(sys.modules)\n"
        "assert 'PIL' not in sys.modules\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_chat_proxy_only_reaches_models_this_page_knows(monkeypatch):
    """The control page serves no model, so its chat is proxied to the one a
    person clicked. The target comes from the request, so it is checked
    against a fixed list -- servers knurlogic started -- or the page
    would forward anything to any address."""
    import io
    from knurlogic.interfaces.page import server as page_server
    monkeypatch.setattr(page_server, "chat_targets",
                        lambda: {"http://127.0.0.1:8080"})
    sent = {}

    class H:
        wfile = io.BytesIO()
        def send_response(self, code): sent["code"] = code
        def send_header(self, *a): pass
        def end_headers(self): pass
    page_server.proxy_chat(H(), "http://169.254.169.254", b"{}")
    assert sent["code"] == 403


def test_the_page_asset_ships_with_the_package():
    """interfaces/page/assets/index.html is package data: a wheel or a move
    that loses it must fail here, not as a blank page."""
    from knurlogic.interfaces.page.documents import PAGE
    assert PAGE.exists(), PAGE
