"""The Anthropic->OpenAI image shim (`interfaces/http/messages.py`),
and the `vision` field on `/models.json` and `/loaded.json`.

No mlx, no PIL, no real model: `interfaces/page/documents.py` reads
`served_vision()` and `registry.registered()`, both stdlib-only per the
frozen contract (`docs/design/vision-contracts.md`), and this file proves it
stays that way by never importing anything that would drag mlx in.
"""
from pathlib import Path

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
        "knurlogic.engine.vision.registry.registered", lambda mt, path=None: True)
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
        "knurlogic.engine.vision.registry.registered", lambda mt, path=None: False)
    doc = documents.models_document()({})
    assert doc["models"][0]["vision"] is False


def test_models_document_says_why_a_conversion_has_no_vision(monkeypatch,
                                                              tmp_path):
    """A vision family's conversion without its vision weights: `vision`
    false and `vision_why` the reason, which the picker grays its vision
    switch with; a model that is not a vision model has no reason (its
    switch stays hidden)."""
    import json

    from knurlogic.machine.discover import Found
    rows = []
    for name, cfg in (("text", {"model_type": "deepseek_v4",
                                "vision_n_layers": 32}),
                      ("flash", {"model_type": "deepseek_v4"})):
        p = tmp_path / name
        p.mkdir()
        (p / "config.json").write_text(json.dumps(cfg))
        (p / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"model.norm.weight": "model.safetensors"}}))
        rows.append(Found(name=name, path=p, store="given", format="mlx",
                          bytes_on_disk=1 << 20, model_type="deepseek_v4",
                          servable=True))
    monkeypatch.setattr("knurlogic.machine.discover.find",
                        lambda *a, **k: rows)
    documents.forget_models()
    doc = {m["name"]: m for m in documents.models_document()({})["models"]}
    assert doc["text"]["vision"] is False
    assert doc["text"]["vision_why"] == "this conversion has no vision weights"
    assert doc["flash"]["vision"] is False and doc["flash"]["vision_why"] == ""
    # the picker shows that row's switch disabled, off, with the reason
    js = (Path(documents.__file__).parent / "assets" / "views"
          / "picker.js").read_text()
    assert "m.vision_why" in js and "b.disabled=off" in js


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


def test_every_page_file_ships_and_every_reference_resolves():
    """The page is many static files now: each one under assets/ must be
    matched by pyproject's package-data, and every file index.html links
    and every module imports must be there -- a missing module is a blank
    page with one console error, not a failing request anyone sees."""
    import re
    try:
        import tomllib
    except ModuleNotFoundError:            # 3.10: pytest depends on tomli
        import tomli as tomllib
    from knurlogic.interfaces.page.documents import ASSETS, asset_names
    root = Path(__file__).resolve().parents[2]
    pkg = root / "src" / "knurlogic"
    patterns = tomllib.loads((root / "pyproject.toml").read_text())[
        "tool"]["setuptools"]["package-data"]["knurlogic"]
    shipped = {p for pat in patterns for p in pkg.glob(pat) if p.is_file()}
    files = [p for p in ASSETS.rglob("*") if p.is_file()]
    assert {"index.html", "page.css", "app.js"} <= set(asset_names())
    for f in files:
        assert f in shipped, f"{f} is not package data"
    html = (ASSETS / "index.html").read_text()
    refs = [(ASSETS, r) for r in re.findall(r'(?:href|src)="/([^"]+)"', html)]
    assert {"page.css", "app.js"} <= {r for _, r in refs}
    for js in ASSETS.rglob("*.js"):
        for r in re.findall(r"^(?:import|export)\b[^;]*?from '([^']+)'|"
                            r"^import '([^']+)'", js.read_text(), re.M):
            refs.append((js.parent, r[0] or r[1]))
    assert len(refs) > 10
    for base, r in refs:
        target = (base / r).resolve()
        assert target.is_file() and target.is_relative_to(ASSETS.resolve()), \
            f"{r} (from {base}) is missing"


def _page_server():
    import threading
    from http.server import ThreadingHTTPServer

    from knurlogic.interfaces.page import documents as web
    from knurlogic.interfaces.page import server as ui
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              ui.make_handler(web.routes()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_the_page_serves_its_modules_with_a_module_type():
    srv, base = _page_server()
    try:
        code, h, body = _get(base + "/app.js")
        assert code == 200 and h["Content-Type"].startswith(
            "application/javascript")
        assert h["Cache-Control"] == "no-cache"
        assert b"import" in body
        code, h, _ = _get(base + "/views/chat.js")
        assert code == 200 and h["Content-Type"].startswith(
            "application/javascript")
        code, h, _ = _get(base + "/page.css")
        assert code == 200 and h["Content-Type"].startswith("text/css")
        code, h, body = _get(base + "/")
        assert code == 200 and h["Content-Type"].startswith("text/html")
        assert b'type="module"' in body
    finally:
        srv.shutdown()


def test_the_static_route_refuses_traversal_and_unknown_files():
    srv, base = _page_server()
    try:
        for path in ("/nope.js", "/views/nope.js", "/../documents.py",
                     "/views/../../documents.py", "/%2e%2e/documents.py",
                     "/documents.py", "/server.py"):
            code, _, body = _get(base + path)
            assert code == 404, path
            assert b"import json" not in body
    finally:
        srv.shutdown()


def test_asset_stays_inside_the_assets_directory(tmp_path, monkeypatch):
    from knurlogic.interfaces.page import documents as web
    assets = tmp_path / "assets"
    (assets / "views").mkdir(parents=True)
    (assets / "app.js").write_text("export {}")
    (assets / "views" / "a.js").write_text("export {}")
    (tmp_path / "secret.js").write_text("secret")
    (assets / "out.js").symlink_to(tmp_path / "secret.js")
    (assets / "notes.txt").write_text("not a page file")
    monkeypatch.setattr(web, "ASSETS", assets)
    assert web.asset("app.js") == (b"export {}",
                                   "application/javascript; charset=utf-8")
    assert web.asset("views/a.js")[0] == b"export {}"
    for bad in ("../secret.js", "views/../../secret.js", "out.js",
                "/app.js", "views//a.js", "./app.js", "views\\a.js",
                "notes.txt", "missing.js", "views"):
        assert web.asset(bad) is None, bad
    assert web.asset_names() == ["app.js", "views/a.js"]
    r = web.routes()
    assert "/app.js" in r and "/views/a.js" in r and "/out.js" not in r


def test_vendored_pdfjs_ships_and_is_served():
    """The chat's PDF attachments load pdf.js from the page itself (the page
    works offline): both modules must be there and served as JavaScript."""
    from knurlogic.interfaces.page import documents as web
    for name in ("vendor/pdfjs/pdf.min.mjs", "vendor/pdfjs/pdf.worker.min.mjs"):
        assert name in web.asset_names(), name
    assert (web.ASSETS / "vendor/pdfjs/LICENSE").is_file()
    srv, base = _page_server()
    try:
        for name in ("pdf.min.mjs", "pdf.worker.min.mjs"):
            code, h, body = _get(f"{base}/vendor/pdfjs/{name}")
            assert code == 200 and h["Content-Type"].startswith(
                "application/javascript") and len(body) > 100_000
    finally:
        srv.shutdown()
    js = (web.ASSETS / "views" / "chat.js").read_text()
    assert "import('../vendor/pdfjs/pdf.min.mjs')" in js
    assert "not supported yet" not in js
