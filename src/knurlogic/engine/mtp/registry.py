"""Per-family registration for MTP speculative decoding.

Adding a family's head is a `head` entry in its manifest
(engine/families/<family>/__init__.py) plus the head module in its heads/.
Nothing in this package names an architecture. A `FamilySpec` says:

  head             "module:Class"; exposes `from_sidecar(model, arch, path)`
                   and `draft_logits(h_row, next_ids, cache)`.
  capture          dotted path of the submodule whose INPUT is the
                   pre-lm_head activation (wrapped for the generation's
                   duration, capture.py).
  draft_cache      the attribute that builds the head's own KV cache.
  cache_semantics  "reassign" or "copy" (caches.py); new families start at
                   "copy" until `caches.check_snapshot_semantics` passes.

A family without a measured acceptance number is not registered.
Design: docs/design/drafting.md (registry).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    name: str
    head: str
    capture: str
    draft_cache: str
    sidecar_name: str = "mtp-head-q6.safetensors"
    cache_semantics: str = "copy"
    #: the sidecar's top-level tensor prefixes (engine/mtp/_artifacts)
    layout: tuple = ()

    def head_cls(self):
        mod, _, attr = self.head.partition(":")
        return getattr(importlib.import_module(mod), attr)

    def arch_module(self, model):
        """The module the trunk's classes were defined in. Artifacts ship a
        `model.py` that subclasses the registry arch, so walk to the core.

        Vision-capable artifacts (the 397B's `custom_model.Model`, the GLM
        VLM wrapper) have no `.model` of their own -- the core hangs off
        `.language_model` -- so walk that first when it is there. The bound
        text model is what every head class binds to anyway."""
        text = getattr(model, "language_model", model)
        core = getattr(text, "model", None)
        if core is None:
            raise RuntimeError(
                f"family {self.name}: {type(model).__name__} exposes neither "
                f"`.model` nor `.language_model.model`; cannot resolve the "
                f"architecture module")
        return importlib.import_module(type(core).__module__)

    def make_draft_cache(self, arch):
        try:
            return getattr(arch, self.draft_cache)()
        except AttributeError as e:
            raise RuntimeError(
                f"family {self.name}: architecture module {arch.__name__} has "
                f"no {self.draft_cache!r}; the registry entry is stale against "
                f"the installed mlx-lm") from e


FAMILIES: dict[str, FamilySpec] = {}


def load_head(model, sidecar=None, family: str | None = None,
              model_path=None):
    """Load a drafting head for `model` -> (head, spec).

    `sidecar` may be a file; if omitted, the family's sidecar name is looked
    for in `model_path`. The head is optional by construction: sidecars are
    named outside mlx-lm's `model*.safetensors` glob, so a model directory
    carrying one still loads normally through the stock loader.
    """
    import pathlib

    spec = resolve(model, family)
    if sidecar is None:
        if model_path is None:
            raise ValueError("pass either sidecar= or model_path=")
        sidecar = pathlib.Path(model_path) / spec.sidecar_name
    sidecar = pathlib.Path(sidecar)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no MTP sidecar at {sidecar}; build one with `vqlab mtp-pack`. "
            f"The head is optional -- without it the model decodes normally.")
    arch = spec.arch_module(model)
    return spec.head_cls().from_sidecar(model, arch, sidecar), spec


def register(spec: FamilySpec, *, replace: bool = False) -> FamilySpec:
    if spec.name in FAMILIES and not replace:
        raise ValueError(f"family already registered: {spec.name}")
    if spec.cache_semantics not in ("reassign", "copy"):
        raise ValueError(f"cache_semantics must be 'reassign' or 'copy', "
                         f"got {spec.cache_semantics!r}")
    FAMILIES[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    FAMILIES.pop(name, None)


def model_type_of(model) -> str | None:
    """mlx-lm keeps the resolved config on `model.args`; multimodal configs
    nest the text half, and the top-level type is the one that names the
    architecture module."""
    for obj in (getattr(model, "args", None), model):
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def resolve(model, family: str | None = None) -> FamilySpec:
    if family is not None:
        if family not in FAMILIES:
            raise KeyError(f"unknown MTP family {family!r}; registered: "
                           f"{sorted(FAMILIES)}")
        return FAMILIES[family]
    mt = model_type_of(model)
    if mt in FAMILIES:
        return FAMILIES[mt]
    raise KeyError(
        f"no MTP family registered for model_type {mt!r}; registered: "
        f"{sorted(FAMILIES)}. Adding one is a head entry in "
        f"its family's manifest (knurlogic/engine/families/) plus a head "
        f"module.")


# ------------------------------------------------------------------ builtins
# Every family's heads, from its manifest (engine/families/<family>/): the
# capture point, draft cache and cache semantics are per-architecture facts
# and live there, each beside the measurement that set it.
def _register_builtins() -> None:
    from knurlogic.engine import families
    for name, h in families.build_maps()["heads"].items():
        register(FamilySpec(name=name, **h))


_register_builtins()
