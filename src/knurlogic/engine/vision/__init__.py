"""engine/vision/ -- images as real context: the contracts every family and
the serve path build against (ImageRef, EncodedImage, VisionSpec, the
Family protocol, the errors, served_vision()). docs/design/vision-contracts.md
is the prose home of what is written here, with the data shapes.

  key.py        the cache key: token ids with image sentinels
  store.py      encoded images, kept so an image is encoded once
  images.py     request bytes -> a bounded RGB image and its pixel hash
  request.py    a chat request with images -> the cache key
  scatter.py    image features into text embeddings, uncached span only
  cachehook.py  the prompt cache pins the images it references
  registry.py   model_type -> the family package that serves its images
  quant.py      quantize a vision module to match the checkpoint
  _base.py      the three mlx-vlm helpers the vendored towers import

Each family's tower lives under engine/families/<family>/vision/.

This front door is stdlib only: the page and the MCP read `VisionSpec` and
`served_vision()` without paying for mlx or PIL (tests/test_resolve.py).
Design: docs/design/vision.md.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "ImageRef", "EncodedImage", "VisionSpec", "Family", "RefLookup",
    "FeatureLookup", "VisionError", "KeyMismatch", "ImageEvicted",
    "ImageRejected", "NoVision", "proc_hash", "served_vision",
    "set_served_vision",
]


# --- errors ------------------------------------------------------------------
# Every one of these is loud on purpose: the failure this build exists to
# prevent is SILENT (fluent text grounded on the wrong image), so anything
# that could produce it raises instead of degrading.

class VisionError(Exception):
    """Base for every vision failure; the serve path maps it to HTTP 400
    (a client's image) or 500 (ours)."""


class KeyMismatch(VisionError):
    """Token ids and image refs disagree: a run of image tokens with no
    image, an image with no run, or a run of the wrong length. The common
    real cause is a USER typing the family's pad token as text -- the
    tokenizer turns it into image_token_id. That must be a 400, never a
    guess at which run is which image."""


class ImageEvicted(VisionError, KeyError):
    """An image's features were needed but the store no longer has them.
    Pin entries between tokenize and admit (store.pinned) so this cannot
    happen inside one request."""


class ImageRejected(VisionError, ValueError):
    """The request's image could not be used: undecodable, too many bytes,
    or over the decompression-bomb pixel limit. A client error (400)."""


class ImageTooLarge(ImageRejected):
    """An image the server will not decode: over the encoded-bytes limit or
    the decompression-bomb pixel limit, judged before any pixel is decoded.
    (Anything under them is downscaled -- first to MAX_DECODE_PIXELS, then
    by the family's own processor to what the model takes -- never
    refused.) The server answers 413, naming the limit."""


class ImagesOverBudget(ImageTooLarge):
    """A request's images do not fit the image store's memory budget
    together -- they must all be resident at admission. 413, naming both
    numbers."""


class NoVision(VisionError):
    """An image was sent to a model with no vision tower (400)."""


# --- data --------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRef:
    """One image, as the prompt and the positions see it. Small, immutable
    and NEVER evicted with the features (store.ref):
    Qwen's positions for a text turn after an image need that image's
    grid, and a key restored from the prompt cache must not depend on
    features the LRU may already have dropped.

    sha        images.pixel_sha(): sha256 of the normalised decoded pixels,
               so the same picture re-encoded by a client still hits
    proc_hash  VisionSpec.proc_hash at preprocess time; a processor change
               (a max_pixels clamp) can then never hit a stale entry
    n_tokens   length of the image's run of image tokens in the prompt,
               after merge/pool -- the number of sentinels
    grid_thw   (t, h, w) in patches, before merge, for Qwen and GLM
               positions; None for gemma
    """
    sha: str
    proc_hash: str
    n_tokens: int
    grid_thw: tuple[int, int, int] | None = None


@dataclass
class EncodedImage:
    """What the store holds per image: the tower's output, already projected
    into the text model's embedding space.

    feats   mx.array [n_tokens, text_hidden]. Row k is the embedding for
            sentinel k. Family-specific scaling (gemma's embed_scale) is the
            family's business and is documented in its package.
    extras  further per-image arrays a family needs at embed time (e.g.
            deepstack features); counted in nbytes like feats.

    Evaluate the arrays (mx.eval) BEFORE putting them in the store: a lazy
    graph would keep pixel_values and every tower activation alive behind
    a number that claims a few MB.
    """
    ref: ImageRef
    feats: Any
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        """Bytes of every array held. Read from the arrays (shape x
        itemsize), never estimated: the store's bound and the memory budget
        are computed from this number."""
        n = int(getattr(self.feats, "nbytes", 0))
        for v in self.extras.values():
            n += int(getattr(v, "nbytes", 0))
        return n


@dataclass(frozen=True)
class VisionSpec:
    """What a served model's vision is, for the page, the MCP and the fit.

    family          registry key ("qwen3_5", "gemma4", ...)
    image_token_id  the one token id an image occupies n_tokens copies of
    patch, merge    patch size in pixels; spatial merge (None if the family
                    pools instead, as gemma does)
    min_pixels, max_pixels   the processor's resize bounds
    fixed_tokens    tokens per image if the family always spends the same
                    count, else None. None until a family's processor was
                    read and shows it (gemma's count is aspect-dependent)
    proc_hash       proc_hash() of every processor setting that changes the
                    features; part of every sentinel and store key
    """
    family: str
    image_token_id: int
    patch: int
    merge: int | None
    min_pixels: int
    max_pixels: int
    fixed_tokens: int | None
    proc_hash: str

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def proc_hash(settings: dict[str, Any]) -> str:
    """The hash that goes into VisionSpec.proc_hash: every processor setting
    that changes an image's features, canonical JSON, sha256, 16 hex.

    16 hex (64 bits) because it sits in every sentinel of every image token
    in the prompt trie; a collision needs two processor configs of the SAME
    model on the SAME image, so 64 bits is far past enough."""
    blob = json.dumps(settings, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


#: (sha, proc_hash) -> ImageRef, never evicted (store.ImageStore.ref).
RefLookup = Callable[[str, str], ImageRef]
#: (sha, proc_hash) -> EncodedImage, raises ImageEvicted (store.features).
FeatureLookup = Callable[[str, str], EncodedImage]


class Family(Protocol):
    """One vision family. Built by `registry.build`; the serve path
    calls these and nothing else. Every method is called on the GENERATOR
    thread -- none of them may be reached from an HTTP thread.

    Key and positions below mean the full cache key (key.py): token ids with
    each image token replaced by a sentinel ("img", sha, proc_hash, k).
    """

    spec: VisionSpec

    def load_weights(self, model_path: str) -> int:
        """Read the vision tensors (sidecar or main shards, filtered by the
        index weight_map) into a STANDALONE tower not attached to the trunk
        -- the trunk's sanitize keeps dropping
        vision keys. Returns the number of tensors loaded, for the per-rung
        tensor-count bound."""

    def preprocess(self, img: Any, sha: str) -> tuple[dict[str, Any], ImageRef]:
        """PIL image (already clamped and RGB, images.decode) -> the tower's
        inputs and the image's ref. `sha` is images.pixel_sha(img); the ref
        carries it and spec.proc_hash."""

    def encode(self, pixels: dict[str, Any], ref: ImageRef) -> EncodedImage:
        """Run the tower once. feats has exactly ref.n_tokens rows. The only
        place the tower runs, so wrapping it from outside counts G6."""

    def placeholder_text(self, ref: ImageRef) -> str:
        """Text that replaces one image part in the message before the chat
        template, and tokenizes to EXACTLY ONE image_token_id (plus whatever
        framing ids the family needs, e.g. Qwen's vision_start/end or
        gemma's boi/eoi, which stay ordinary ids). key.expand_pads then
        widens that one id to n_tokens -- the only expansion there is, so
        segments expand with the same rule."""

    def embed(self, model: Any, key: list[Any], start: int,
              features: FeatureLookup) -> dict[str, Any]:
        """Inputs for the trunk over key[start:] -- the uncached span only.

        Returns {"input_embeddings": mx [1, len(key)-start, D], **extras},
        extras being the trunk's own keyword names (position_ids,
        per_layer_inputs, a mask, ...). Image rows come from the store by
        sentinel (scatter.merge), so an image cut by the prefix hit takes
        rows k..n-1 with no global feature index to recompute."""

    def positions(self, key: list[Any], refs: RefLookup) -> tuple[Any, int]:
        """(position ids for the WHOLE key or None, rope_delta). Pure in the
        key: called on every prefill and decode of a row whose key
        holds an image, whether or not the new suffix does. None means the
        trunk's own 1D positions are right (gemma, GLM); Qwen returns
        mx [3, 1, len(key)] and the delta decode adds to the offset."""

    def chunk_boundaries(self, key: list[Any]) -> list[tuple[int, int]]:
        """[start, end) spans a prefill chunk edge must not fall strictly
        inside. gemma: every image span (bidirectional attention within
        an image block). Causal families: []."""


# --- what is being served ------------------------------------------------------
# The page and the MCP must say whether the served model sees images
# without importing engine/model. The serve path sets this when a vision
# family is built and clears it on unload; interfaces only read it.

_SERVED_SPEC: VisionSpec | None = None


def served_vision() -> VisionSpec | None:
    """The served model's VisionSpec, or None when it has no vision (or
    nothing is served)."""
    return _SERVED_SPEC


def set_served_vision(spec: VisionSpec | None) -> None:
    """Serve path only: record (or clear, with None) what is served."""
    global _SERVED_SPEC
    _SERVED_SPEC = spec
