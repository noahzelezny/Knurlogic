"""Per-family registration for MTP speculative decoding.

Adding a family's head is a `head` entry in its manifest
(engine/families/<family>/__init__.py) plus the head module in its heads/.
Nothing in this package names an architecture; the loop, the caches and the sampler are all
family-agnostic.

A `FamilySpec` says four things, and every one of them is a place where
architectures genuinely differ:

  head             where the drafting head lives ("module:Class"). The class
                   must expose `from_sidecar(model, arch, path)` and
                   `draft_logits(h_row, next_ids, cache)`.
  capture          dotted path, relative to the trunk core, of the submodule
                   whose INPUT is the pre-lm_head activation the head drafts
                   from. There is no public mlx-lm hook for this, so we wrap
                   that one module for the duration of the generation (see
                   capture.py) rather than monkeypatching the class.
  draft_cache      the attribute on the architecture module that builds the
                   head's own KV cache.
  cache_semantics  "reassign" or "copy" — see caches.py. qwen4_exp reassigns
                   its recurrent cache slots rather than mutating them, which
                   makes snapshots free; that is an implementation accident of
                   that arch, NOT a contract, so new families start at "copy"
                   and only move to "reassign" once
                   `caches.check_snapshot_semantics` has been run against them.

Families that ship an MTP head upstream but are NOT registered here, because
nothing in this repo can test them today:

  deepseek_v3 DeepSeek's MTP module is a different shape again (its own
              embed/norm/head rather than a shared lm_head).

(glm5_next graduated 2026-09-03, vendored from VQLab: the old blocker was
"mlx-lm has no glm5_next class", but the trunk runs under mlx_vlm's class —
which exo also uses — so the head binds to THAT arch module.)

Registering either means writing its head module and running
`caches.check_snapshot_semantics` plus the acceptance probe first. A table
entry without a measured acceptance number is not evidence of anything.
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
