"""Multi-token prediction: what an artifact declares, and what it actually has.

Two independent channels -- config.json for the declaration, the
safetensors HEADERS for the tensors. They disagree: the MLX converters drop
`mtp.*` at CONVERSION, so most rungs declare MTP in config yet ship no head
weights. MTP here is therefore "the head is a separate artifact": grafted
once from the one checkpoint that carries it, quantized, and written beside
each rung as a sidecar (`mtp-head-q6.safetensors`).

The sidecar is deliberately outside `model*.safetensors`, so a loader's
glob and an index-only scan never see it: read the FILES, not the index.
Nothing here imports an engine; the recipe is in the sidecar's metadata.
Design: docs/design/drafting.md (artifacts).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30

#: What `vqlab mtp-pack` names its sidecar. Other names are read too -- the
#: glob is the contract, this is only the default.
SIDECAR_GLOB = "mtp-head*.safetensors"

def head_layouts() -> list:
    """(top-level tensor prefixes, head name) per head the family manifests
    declare (engine/families/<family>, head `layout`), most specific first
    so a superset layout is matched before its subset. The layout
    identifies the family when a sidecar's metadata does not say."""
    from knurlogic.engine import families
    seen, out = set(), []
    for name, h in families.build_maps()["heads"].items():
        lay = frozenset(h.get("layout") or ())
        if lay and lay not in seen:
            seen.add(lay)
            out.append((lay, name))
    return sorted(out, key=lambda x: -len(x[0]))


@dataclass
class Head:
    """A built drafting head found beside an artifact."""
    path: Path
    bytes_on_disk: int
    tensors: int
    prefixes: list = field(default_factory=list)
    recipe: dict = field(default_factory=dict, repr=False)

    @property
    def gib(self) -> float:
        return self.bytes_on_disk / GIB

    @property
    def family(self) -> str:
        """From the metadata when it says, else from the module tree.

        The layout is the stronger channel of the two: metadata is a string
        someone wrote, the tree is what the head IS. It is the fallback only
        because the metadata is cheaper to trust when present and matching.
        """
        declared = self.recipe.get("family")
        for keys, fam in head_layouts():
            if keys <= set(self.prefixes):
                return fam
        return declared or ""

    @property
    def bits(self):
        return self.recipe.get("bits")

    def describe(self) -> str:
        b = self.bits
        rec = f"{b}-bit" if b else "unquantized"
        eb = self.recipe.get("expert_bits")
        if eb:
            rec += f" ({eb}-bit experts)"
        fam = self.family or "unknown family"
        return f"{self.path.name}  {self.gib:.2f} GiB, {rec}, {fam}"


def _header(f: Path) -> dict:
    """The JSON header of a safetensors file. No tensor data is read."""
    with f.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return {}
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > (1 << 28):     # a header is kilobytes, not gigabytes
            return {}
        return json.loads(fh.read(n))


def find_head(path) -> Head | None:
    """The built head beside this artifact, if one is there."""
    d = Path(path)
    for f in sorted(d.glob(SIDECAR_GLOB)):
        try:
            hdr = _header(f)
        except Exception:
            continue
        meta = hdr.pop("__metadata__", {}) or {}
        if not hdr:
            continue
        recipe = {}
        for key in ("vqlab_mtp", "knurlogic_mtp"):
            if key in meta:
                try:
                    recipe = json.loads(meta[key])
                except Exception:
                    recipe = {}
                break
        return Head(f, f.stat().st_size, len(hdr),
                    sorted({k.split(".")[0] for k in hdr}), recipe)
    return None


def graft_weights(path) -> int:
    """How many upstream `mtp.*` tensors the TRUNK carries.

    These are what a head is built FROM -- bf16, unquantized, inside the
    model glob. One artifact on this disk has them.
    """
    d = Path(path)
    idx = d / "model.safetensors.index.json"
    keys: set = set()
    if idx.is_file():
        try:
            keys = set(json.loads(idx.read_text()).get("weight_map", {}))
        except Exception:
            keys = set()
    if not keys:
        for f in sorted(d.glob("model*.safetensors")):
            try:
                keys |= set(_header(f))
            except Exception:
                continue
    keys.discard("__metadata__")
    return sum(1 for k in keys if k.startswith("mtp.") or ".mtp." in k)


#: The three states, and they are genuinely different situations for whoever
#: downloaded the thing.
BUILT = "built"          # a head is here; it can draft once a loop loads it
GRAFTABLE = "graftable"  # raw mtp.* in the trunk; a head can be built from it
DECLARED = "declared"    # config says MTP, nothing shipped. The common case.
NONE = "none"

#: A SIDECAR IS A VQLAB PRODUCT. Measured across every artifact here: 11 of
#: 11 built heads sit beside a VQ artifact and not one community rung has
#: one. That is not a coincidence to be reported neutrally -- vqlab builds
#: models, knurlogic runs them, and a head is something the build step made.
#:
#: So the absence of a head means two different things. On a community rung
#: it means nothing at all: the `mtp` key is inherited from the upstream
#: config and no publisher ships the weights, so there is no defect and
#: nothing anybody can do. On a VQ artifact it means the head was not packed,
#: which is a vqlab question and still not knurlogic's to answer.


@dataclass
class Status:
    state: str
    head: Head | None = None
    graft_tensors: int = 0
    #: Whether a head could ever have been packed for this artifact by the
    #: thing that builds them.
    is_vq: bool = False

    def render(self) -> str:
        if self.state == BUILT:
            return ("multi-token-prediction head PRESENT and built\n"
                    f"           {self.head.describe()}\n"
                    "           It is outside the model glob, so nothing has "
                    "loaded it: it costs\n           no memory until a "
                    "drafting loop asks for it.")
        if self.state == GRAFTABLE:
            return (f"this artifact carries {self.graft_tensors} raw `mtp.*` "
                    "tensors -- the upstream\n           head, unquantized. A "
                    "drafting head can be BUILT from these; the\n           "
                    "architecture that loads the trunk discards them.")
        if self.state == DECLARED:
            if self.is_vq:
                return ("config declares a multi-token-prediction head and "
                        "none is packed beside\n           these weights. "
                        "Packing one is vqlab's job -- it built this "
                        "artifact.")
            return ("config declares a multi-token-prediction head, which is "
                    "inherited from\n           the upstream config. No "
                    "publisher ships those weights and MLX conversion\n"
                    "           drops them, so nothing is missing here and "
                    "there is nothing to do.")
        return ""


def status(artifact) -> Status:
    """Which of the three MTP situations this artifact is actually in."""
    vq = bool(getattr(artifact, "is_vq", False))
    head = find_head(artifact.path)
    if head is not None:
        return Status(BUILT, head=head, is_vq=vq)
    n = graft_weights(artifact.path)
    if n:
        return Status(GRAFTABLE, graft_tensors=n, is_vq=vq)
    if artifact.has_mtp:
        return Status(DECLARED, is_vq=vq)
    return Status(NONE, is_vq=vq)
