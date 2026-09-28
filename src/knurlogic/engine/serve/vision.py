"""The served model's vision: bind a Family at load, clear it at unload,
and say what is served (design D3, D5, D6; docs/design/vision.md). Images
reach the model through the scheduler's tokenize (engine/runtime)."""

from __future__ import annotations

from . import state


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


def bind(model_path: str, provider, *, store_bytes: int | None = None,
         tower: bool = True):
    """Build the loaded model's Family through the registry and serve it.

    Called after every load. model_type and config come from the artifact's
    config.json; `registry.build` answers None for a model without vision
    (unregistered type, no vision_config, family package absent), and that
    is the ordinary case, not an error. The store is keyed by mlx-lm's
    `model_key`, the same key its prompt cache uses.

    `tower=False`: a follower rank of a split model -- the family without
    its tower or a store (engine.vision.request.MirrorVision); rank 0
    encodes and ships the image rows."""
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
    if not tower:
        from knurlogic.engine.vision.request import MirrorVision
        serve = MirrorVision(fam)
        state.VISION["serve"] = serve
        set_spec(fam.spec)
        return serve
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
        store = getattr(v, "store", None)
        # a follower's MirrorVision has no store, only refs
        (store if store is not None else v).clear()
    state.VISION.update(serve=None, model=None, error="")
    set_spec(None)
