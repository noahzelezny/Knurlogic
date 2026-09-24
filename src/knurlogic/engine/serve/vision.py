"""Images through mlx-lm's server (design D3, D5, D6; docs/design/vision.md).
"""

from __future__ import annotations

from . import state


# Images reach the model through three wraps of mlx-lm's server and nothing
# else:
#
#   APIHandler.do_POST          (HTTP thread) images sent to a model with no
#                               vision -> 400. Nothing else happens here.
#   ResponseGenerator.generate  (HTTP thread) a request with image parts is
#                               TAGGED (args._knurlogic_images); no decoding.
#   ResponseGenerator._is_batchable   a tagged request is always batchable,
#                               so it goes through the batch engine (D5).
#   ResponseGenerator._tokenize (GENERATOR thread) all image work, returns
#                               the cache key as the prompt (D3).
#
# plus the BatchGenerator factory (drafting.install_batch), which hands a vision model
# MTPBatchGenerator whether or not it has a head.
#
# Every wrapped method is resolved BY NAME with an assertion (design D2): a
# pinned mlx-lm that renames one fails at install, loudly, instead of a
# wrap that silently never runs. tests/test_pins.py pins server.py's digest.

#: What is served: the VisionServe (family + store + pins) and the model it
#: is bound to. Only bind_vision / clear_vision write here.

#: (owner attribute on srv, method name) for every mlx-lm method wrapped
#: for vision -- one list, so the install and the tests agree on it.
VISION_WRAPS = (("APIHandler", "do_POST"),
                ("ResponseGenerator", "generate"),
                ("ResponseGenerator", "_is_batchable"),
                ("ResponseGenerator", "_tokenize"))

#: Paths whose body is a chat request that may carry images.
_CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")


def _method(srv, owner: str, name: str):
    cls = getattr(srv, owner, None)
    fn = getattr(cls, name, None) if cls is not None else None
    assert callable(fn), (
        f"mlx_lm.server.{owner}.{name} is gone: the vision serve path wraps "
        f"it by name. The pinned mlx-lm changed; re-read server.py before "
        f"re-pinning (tests/test_pins.py).")
    return cls, fn


def set_spec(spec) -> None:
    from knurlogic.engine.vision import set_served_vision
    set_served_vision(spec)


def served_vision():
    """The served model's VisionSpec, or None. The same answer as
    engine.vision.served_vision(), which interfaces/ reads without
    importing this module (critique C4)."""
    from knurlogic.engine.vision import served_vision as _sv
    return _sv()


def vision_status() -> dict:
    """For /status.json: the spec, the store and the pins, or why not."""
    v = state.VISION.get("serve")
    out = {"on": v is not None, "error": state.VISION.get("error", "")}
    if v is not None:
        out.update(spec=v.spec.to_json(), store=v.store.stats(),
                   encodes=v.encodes, pinned=v.pinned_count())
    return out


def bind(model_path: str, provider, *, store_bytes: int | None = None):
    """Build the loaded model's Family through the registry and serve it.

    Called after every load. model_type and config come from the artifact's
    config.json; `registry.build` answers None for a model without vision
    (unregistered type, no vision_config, family package absent), and that
    is the ordinary case, not an error. The store is keyed by mlx-lm's
    `model_key`, the same key its prompt cache uses."""
    import json
    from pathlib import Path

    from knurlogic.engine.vision import registry
    from knurlogic.engine.vision.request import VisionServe
    from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES, ImageStore

    clear()
    cfg_path = Path(str(model_path)) / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    fam = registry.build(config.get("model_type", ""), str(model_path),
                         provider.model, config)
    state.VISION.update(model=provider.model, error="")
    if fam is None:
        return None
    n = fam.load_weights(str(model_path))
    serve = VisionServe(fam, ImageStore(store_bytes or DEFAULT_MAX_BYTES),
                        provider.model_key)
    serve.tensors = int(n)
    state.VISION["serve"] = serve
    set_spec(fam.spec)
    return serve


def clear() -> None:
    """Drop the family and its store (features AND refs: the prompt cache
    they index dies with the model), and say no vision is served."""
    v = state.VISION.get("serve")
    if v is not None:
        v.store.clear()
    state.VISION.update(serve=None, model=None, error="")
    set_spec(None)


def _refuse(handler, code: int, msg: str) -> None:
    import json

    body = json.dumps({"error": msg}).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def install(srv) -> None:
    """Wrap mlx-lm's server for images. Idempotent; a server whose model
    has no vision behaves exactly as before -- every wrap passes straight
    through when no image part is present (G5)."""
    import io
    import json

    from knurlogic.engine.vision import VisionError
    from knurlogic.engine.vision import request as vreq

    Handler, real_post = _method(srv, "APIHandler", "do_POST")
    RG, real_generate = _method(srv, "ResponseGenerator", "generate")
    _, real_batchable = _method(srv, "ResponseGenerator", "_is_batchable")
    _, real_tokenize = _method(srv, "ResponseGenerator", "_tokenize")
    if getattr(real_tokenize, "_knurlogic_vision", False):
        return

    def do_POST(self):
        # Design D3: `_post` only refuses. The body is read to look for
        # image parts and put back for mlx-lm to read again.
        if self.path.split("?")[0] not in _CHAT_PATHS:
            return real_post(self)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return real_post(self)           # mlx-lm answers that one
        raw = self.rfile.read(n) if n else b""
        self.rfile = io.BytesIO(raw)
        try:
            body = json.loads(raw.decode() or "null")
        except (ValueError, UnicodeDecodeError):
            body = None
        if (isinstance(body, dict)
                and vreq.has_images(body.get("messages"))
                and state.VISION.get("serve") is None):
            why = state.VISION.get("error") or "the served model has no vision"
            return _refuse(self, 400, f"this request has images but "
                                      f"{why}; send text only")
        return real_post(self)

    def generate(self, request, generation_args, *a, **kw):
        if vreq.has_images(getattr(request, "messages", None)):
            vreq.tag(generation_args)
        return real_generate(self, request, generation_args, *a, **kw)

    def _is_batchable(self, args):
        # D5: a request with an image is served by the batch engine even
        # when it carries a seed (which mlx-lm serves sequentially): the
        # sequential path chunks prefill with no hook and cannot read a key.
        if vreq.tagged(args):
            return True
        return real_batchable(self, args)

    def _tokenize(self, tokenizer, request, args):
        from knurlogic.engine.vision import cachehook
        from knurlogic.engine.vision import key as K
        # Release pins a request took and never handed to the batch engine
        # (it raised between tokenize and insert). Every request, text too.
        cachehook.sweep()
        if not vreq.has_images(getattr(request, "messages", None)):
            return real_tokenize(self, tokenizer, request, args)
        v = state.VISION.get("serve")
        if v is None:
            # Raced a switch to a text-only model after _post let it in.
            raise VisionError("the served model has no vision")
        out = v.tokenize(real_tokenize, self, tokenizer, request, args)
        cachehook.pending(v, K.images_in(out[0]))
        return out

    for fn in (do_POST, generate, _is_batchable, _tokenize):
        fn._knurlogic_vision = True
    # The pin refcount, installed HERE with the _tokenize wrap that feeds it
    # and never separately: pending() without install_admit() lets a later
    # sweep release pins a queued row still owns. Images a cached
    # conversation references stay pinned (Flash-Next review point 1).
    from knurlogic.engine.vision import cachehook
    cachehook.install(srv, lambda: (state.VISION["serve"].store
                                    if state.VISION.get("serve") else None))
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    cachehook.install_admit(MTPBatchGenerator)
    Handler.do_POST = do_POST
    RG.generate = generate
    RG._is_batchable = _is_batchable
    RG._tokenize = _tokenize
    from . import drafting
    drafting.install_batch(srv)
