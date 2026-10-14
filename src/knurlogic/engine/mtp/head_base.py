"""What every family's drafting head shares, so each head module holds only
its own wiring.

  DraftHead       `advance` (seed the head's cache, no vocabulary projection)
                  over the head's `_trunk(h_row, nxt_id, cache)`
  GraftHead       a head built from an upstream `mtp.*` graft: `draft_logits`
                  through the shared lm_head / tied embedding, the graft's
                  fused expert split, the "every graft key has a slot" check,
                  and the `vqlab_mtp` sidecar (its recipe read and written)
  mixed_predicate one nn.quantize pass carrying two bit widths (experts and
                  the rest), some layers kept full precision
  PackedSidecar   a head whose sidecar's recipe is read off the packed tensors
                  (deepseek_v4's one-token head and DSpark): bind, load, save

The sidecar file formats and their metadata keys (`vqlab_mtp`, written by
`vqlab mtp-pack` too) are persisted data: nothing here changes them.
"""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from knurlogic.engine.model import packed, weight_files

#: the sidecar metadata key holding a graft head's recipe (VQ Lab's name)
RECIPE_KEY = "vqlab_mtp"
#: the routed expert stack's module, in mlx-lm's SwitchGLU layout
EXPERT_SUBSTR = "switch_mlp"


def mixed_predicate(bits, expert_bits, group_size, keep_fp=()):
    """An nn.quantize class_predicate: the expert stack (`EXPERT_SUBSTR`) at
    `expert_bits`, every other quantizable layer at `bits`, and layers whose
    path ends with one of `keep_fp` left full precision. nn.quantize hands
    the predicate every submodule, norms included, so it gates on
    `to_quantized`, not on names."""
    keep_fp = tuple(keep_fp)

    def predicate(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        if keep_fp and path.endswith(keep_fp):
            return False
        b = expert_bits if EXPERT_SUBSTR in path else bits
        return {"group_size": group_size, "bits": b}
    return predicate


def prefixed(w: dict, prefix: str) -> list:
    """[(key without `prefix`, value)] for the keys of `w` under it."""
    return [(k[len(prefix):], v) for k, v in w.items() if k.startswith(prefix)]


def fill(module, w: dict, prefix: str) -> None:
    """Fill `module` from the tensors of `w` under `prefix`."""
    module.update(tree_unflatten(prefixed(w, prefix)))


def read_recipe(path) -> dict:
    """The recipe a graft head's sidecar was saved with ({} without one)."""
    return json.loads(weight_files.metadata(path).get(RECIPE_KEY, "{}"))


def save_sidecar(path, flat: dict, recipe: dict, **meta) -> dict:
    """Write a graft head's sidecar: `flat` tensors, the recipe under
    RECIPE_KEY, and any further metadata."""
    mx.save_safetensors(str(path), flat, metadata={
        "format": "mlx", **meta, RECIPE_KEY: json.dumps(recipe)})
    return flat


class DraftHead:
    """A head that drafts through `_trunk(h_row, nxt_id, cache)`."""

    def advance(self, h_row, nxt_id, cache):
        """Fill the head's cache for these positions without projecting to
        the vocabulary -- over a whole prompt the lm_head matmul would be
        the single largest cost of seeding."""
        self._trunk(h_row, nxt_id, cache)
        return cache


class GraftHead(DraftHead):
    """A head built from an upstream bf16 `mtp.*` graft and saved as a
    `vqlab_mtp` sidecar. Needs `core` (the trunk core), `tie` and
    `lm_head`."""

    def draft_logits(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> logits for token t+2."""
        out = self._trunk(h_row, nxt_id, cache)
        return (self.core.embed_tokens.as_linear(out) if self.tie
                else self.lm_head(out))

    @staticmethod
    def split_experts(k: str, v, out: dict) -> bool:
        """Put a fused expert tensor of the graft into `out` in the
        SwitchGLU layout -- the split the model's own sanitize applies to
        trunk layers. -> whether `k` was one."""
        if k.endswith("mlp.experts.gate_up_proj"):
            mid = v.shape[-2] // 2
            out[f"mlp.{EXPERT_SUBSTR}.gate_proj.weight"] = v[..., :mid, :]
            out[f"mlp.{EXPERT_SUBSTR}.up_proj.weight"] = v[..., mid:, :]
            return True
        if k.endswith("mlp.experts.down_proj"):
            out[f"mlp.{EXPERT_SUBSTR}.down_proj.weight"] = v
            return True
        return False

    @staticmethod
    def check_slots(slots: set, keys) -> list:
        """Refuse graft keys with no parameter slot. -> the slots the graft
        leaves unfilled, sorted."""
        keys = set(keys)
        unmatched = sorted(keys - slots)
        if unmatched:
            raise ValueError(f"{len(unmatched)} graft keys found no "
                             f"parameter slot, e.g. {unmatched[:4]}")
        return sorted(slots - keys)

    @staticmethod
    def slots(module, prefix: str = "") -> set:
        """The parameter paths of `module`, under `prefix`."""
        return {prefix + k for k, _ in tree_flatten(module.parameters())}


class PackedSidecar:
    """A head held as one module `m` whose sidecar is read off its packed
    tensors (no recipe metadata): `bind` shapes `m` to them and checks they
    match exactly. Subclasses name themselves (`LABEL`) and map keys between
    the sidecar and `m` (`to_module`, `to_sidecar`)."""

    LABEL = "head"

    @staticmethod
    def own_names(own: dict, k: str, *, back: bool = False) -> str:
        """Rename `k` by `own` {sidecar name: module path} (or back)."""
        if back:
            return next((src for src, dst in own.items() if k == dst), k)
        return own.get(k, k)

    def to_module(self, k: str) -> str:
        raise NotImplementedError

    def to_sidecar(self, k: str) -> str:
        raise NotImplementedError

    def _check_keys(self, w: dict) -> None:
        """Refuse sidecar keys this head cannot hold (before renaming)."""

    def _prepare(self, mw: dict) -> None:
        """Swap modules the sidecar packs in a non-mlx format (before the
        quantize pass)."""

    def bind(self, w: dict) -> dict:
        """Shape this head to sidecar tensors and check it matches them
        exactly: every name, every shape. `w` maps sidecar keys to anything
        with a `.shape` -- arrays, or a header's shapes, so a file can be
        checked without loading it. Returns the same values under this
        module's parameter paths."""
        self._check_keys(w)
        mw = {self.to_module(k): v for k, v in w.items()}
        self._prepare(mw)

        def pred(path, mod):
            return packed.recipe(mw, path, mod) or False

        nn.quantize(self.m, class_predicate=pred)
        slots = dict(tree_flatten(self.m.parameters()))
        missing = sorted(set(slots) - set(mw))
        extra = sorted(set(mw) - set(slots))
        if missing or extra:
            raise ValueError(
                f"{self.LABEL} does not match the sidecar: missing "
                f"{missing[:4]} ({len(missing)}), unexpected {extra[:4]} "
                f"({len(extra)})")
        wrong = [(k, tuple(v.shape), tuple(slots[k].shape))
                 for k, v in mw.items()
                 if tuple(v.shape) != tuple(slots[k].shape)]
        if wrong:
            raise ValueError(f"{self.LABEL}: shape mismatch {wrong[:4]}")
        return mw

    def load_weights(self, w: dict):
        """Fill from sidecar tensors."""
        mw = self.bind(w)
        self.m.load_weights(list(mw.items()), strict=True)
        mx.eval(self.m.parameters())
        return self

    def tensors(self) -> dict:
        """This head's parameters under their sidecar names."""
        return {self.to_sidecar(k): v
                for k, v in tree_flatten(self.m.parameters())}

    def save(self, path) -> dict:
        flat = self.tensors()
        mx.save_safetensors(str(path), flat, metadata={"format": "mlx"})
        return flat

    @classmethod
    def from_sidecar(cls, model, arch, path):
        return cls(model, arch).load_weights(weight_files.load(path))
