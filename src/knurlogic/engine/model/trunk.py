"""Where things sit inside a loaded model: the trunk core, a layer behind a
pipeline stage wrapper, and dotted attribute paths.

Heads, MTP and vision families ask these questions; they live here so none
of them imports the pipeline split just to find the layers.
"""

from __future__ import annotations

from mlx import nn

_MISSING = object()


def text_model(model):
    """The text model under a multimodal wrapper (`model.language_model`),
    or `model` itself."""
    return getattr(model, "language_model", model)


def core_of(model):
    """The trunk core holding `layers` (qwen3_5 / glm5_next:
    model.language_model.model; qwen4_exp / deepseek_v4: model.model)."""
    return text_model(model).model


class _Wrap(nn.Module):
    """A module that stands in for `inner` and reads through to it."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "inner" not in self:
                raise
            return getattr(self["inner"], name)


def unwrap(layer):
    """The layer a stage end wraps (`Recv` / `Send`, possibly both on a
    one-layer stage), or `layer` itself. Attribute reads already see through
    a wrapper; `type(...)` and `isinstance(...)` do not, so code that asks
    what class a layer is asks this first."""
    while isinstance(layer, _Wrap):
        layer = layer["inner"]
    return layer


def attr_path(obj, path: str, default=_MISSING):
    """`obj.a.b.c` for `path` "a.b.c"; `default` when a step is missing (or
    None, when `default` is given) -- AttributeError when no default."""
    for part in path.split("."):
        nxt = getattr(obj, part, None)
        if nxt is None:
            if default is _MISSING:
                raise AttributeError(f"{type(obj).__name__} has no {part!r} "
                                     f"(in {path!r})")
            return default
        obj = nxt
    return obj


def first_attr(obj, paths, accept=None):
    """The first of the dotted `paths` present on `obj` (and passing
    `accept`, when given); AttributeError when none is."""
    for path in paths:
        found = attr_path(obj, path, None)
        if found is not None and (accept is None or accept(found)):
            return found
    raise AttributeError(f"none of {', '.join(paths)} on "
                         f"{type(obj).__name__}")
