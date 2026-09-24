"""A chat request with images -> the cache key, on the generator thread.

Design D3. mlx-lm's server tokenizes on its generator thread
(`ResponseGenerator._tokenize`, which has `request.messages`), and generation
runs on that same thread. v1 encoded images on the HTTP thread: two threads
on one GPU, two uncoordinated allocations on a shared host (critique B2). So
ALL image work happens here, called from the seam's `_tokenize` wrap:

    image parts -> decode + clamp + pixel hash (images.load)
                -> store hit, or preprocess + encode + put (the ONLY tower
                   call; G6 counts it from outside)
                -> each part replaced by Family.placeholder_text(ref)
                -> the real _tokenize (template, segments, thinking state)
                -> key.expand_segments: one pad per image widened to
                   n_tokens, each image token a sentinel
                -> (key, segment keys, types, state) back to the server

The server then does its own prefix arithmetic on the key -- it is the same
length as the KV (key.py) -- and hands it to the batch generator, which turns
it back into ids and embeddings (mtp/batch_generator.py). No side table
carries images between threads; the key names them.

PINS (vision-contracts.md, "For P4"). Between this tokenize and the batch
admit, other requests may be tokenized and a small store could evict this
one's images. Every image of the key is pinned here and released by the
generator when the row is admitted (or removed unadmitted). Pins are counted
per image, so two queued requests on one image hold it twice.

Stdlib only at import, like the rest of engine/vision's front door: PIL
and mlx are reached through images.load and the family.
"""
from __future__ import annotations

import copy
import dataclasses
import threading
from contextlib import ExitStack
from typing import Any, Callable, Dict, Hashable, List, Tuple

from . import ImageRejected, key as K

#: OpenAI chat content part types that carry an image. `image_url` is the
#: Chat Completions form; `input_image` the Responses form; `image` what
#: several clients (and P5's Anthropic translation, if it keeps the name)
#: send. A part of any other non-text type is left for mlx-lm to refuse, as
#: it does today.
IMAGE_TYPES = ("image_url", "input_image", "image")


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in IMAGE_TYPES


def image_source(part: Dict[str, Any]) -> Any:
    """The bytes-or-URL string of one image part, as images.decode takes it.
    Raises ImageRejected (a 400) for a part that names no image."""
    for k in ("image_url", "url", "image", "data"):
        v = part.get(k)
        if isinstance(v, dict):
            v = v.get("url") or v.get("data")
        if isinstance(v, (str, bytes)) and v:
            return v
    raise ImageRejected(f"image part of type {part.get('type')!r} carries "
                        f"no url or data")


def image_parts(messages: Any) -> List[Dict[str, Any]]:
    """Every image part of every message, in prompt order."""
    out: List[Dict[str, Any]] = []
    for m in messages or ():
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            out.extend(p for p in c if _is_image_part(p))
    return out


def has_images(messages: Any) -> bool:
    """Cheap: a scan of the parts, no decoding. Safe on the HTTP thread --
    it is what `_post` uses to refuse images to a text-only model."""
    return bool(image_parts(messages))


def with_placeholders(messages: List[Dict[str, Any]],
                      texts: List[str]) -> List[Dict[str, Any]]:
    """A copy of the messages with the i-th image part replaced by a text
    part holding texts[i]. mlx-lm then joins a message's text parts with ""
    (server.process_message_content), so the placeholder lands exactly
    where the image was. The caller's messages are not touched: mlx-lm
    rewrites content in place, and the request may be read again."""
    it = iter(texts)
    out = []
    for m in messages:
        m = copy.copy(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = [{"type": "text", "text": next(it)}
                            if _is_image_part(p) else p for p in c]
        out.append(m)
    rest = list(it)
    assert not rest, f"{len(rest)} placeholder texts left over"
    return out


class VisionServe:
    """The served model's vision, as the serve path uses it: its Family,
    its image store and the model key the store is partitioned by. One per
    loaded vision model; the seam builds it at load and drops it (and the
    store) at unload.

    `encodes` counts tower runs made through here, for /status.json; G6
    does NOT read it (critique 4: a counter the code under test increments
    proves nothing) -- the test wraps the family's tower from outside."""

    def __init__(self, family: Any, store: Any, model_key: Hashable, *,
                 allow_paths: bool = False):
        self.family = family
        self.store = store
        self.model_key = model_key
        self.allow_paths = allow_paths
        self.encodes = 0
        self._lock = threading.Lock()
        self._pins: Dict[Tuple[str, str], List[ExitStack]] = {}

    @property
    def spec(self):
        return self.family.spec

    def lookup(self):
        """(FeatureLookup, RefLookup) for Family.embed / Family.positions."""
        return self.store.lookup(self.model_key)

    # --- pins ------------------------------------------------------------------

    def _pin(self, images: List[Tuple[str, str]]) -> None:
        for im in images:
            st = ExitStack()
            st.enter_context(self.store.pinned(self.model_key, [im]))
            with self._lock:
                self._pins.setdefault(im, []).append(st)

    def release(self, images: List[Tuple[str, str]]) -> None:
        """Drop one pin per image (as taken by one tokenize). Unknown images
        are ignored: a row restored wholly from the cache pinned nothing
        extra, and a double release must not unpin someone else's."""
        for im in images:
            with self._lock:
                stack = self._pins.get(im)
                st = stack.pop() if stack else None
                if stack is not None and not stack:
                    del self._pins[im]
            if st is not None:
                st.close()

    def pinned_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pins.values())

    # --- the tokenize wrap -------------------------------------------------------

    def ensure(self, src: Any):
        """One image source -> its ImageRef, encoding on a store miss.

        Hashing is of the NORMALISED pixels (images.load: EXIF transpose,
        RGB, clamp), so the same picture resent by a client in another
        container format still hits (Flash-Next 4.4's review, point 2)."""
        from . import images

        img, sha = images.load(src, allow_paths=self.allow_paths)
        ph = self.spec.proc_hash
        mk = self.model_key
        # Pin BEFORE the lookup, so a hit cannot be evicted by another put
        # between the lookup and the request's own pin.
        self._pin([(sha, ph)])
        try:
            enc = self.store.get(mk, sha, ph)
            if enc is not None:
                return enc.ref
            pixels, ref = self.family.preprocess(img, sha)
            if ref.sha != sha or ref.proc_hash != ph:
                raise AssertionError(
                    f"family preprocess returned ref ({ref.sha[:12]}, "
                    f"{ref.proc_hash}) for ({sha[:12]}, {ph}); the key would "
                    f"name features the store does not hold")
            enc = self.family.encode(pixels, ref)
            self.encodes += 1
            self.store.put(mk, enc)
            return ref
        except BaseException:
            self.release([(sha, ph)])
            raise

    def tokenize(self, real: Callable, gen: Any, tokenizer: Any,
                 request: Any, args: Any):
        """The `_tokenize` wrap's body for a request with images.

        Returns what mlx-lm's `_tokenize` returns, with the prompt and the
        segments replaced by the key and the segment keys. Every image of
        the key stays pinned (one pin per image occurrence) until the batch
        generator admits or removes the row -- on any failure here the pins
        are dropped before the exception reaches the server."""
        parts = image_parts(request.messages)
        refs: List[Any] = []
        before = self.encodes      # tokenize runs on the one generator thread
        try:
            for p in parts:
                refs.append(self.ensure(image_source(p)))
            texts = [self.family.placeholder_text(r) for r in refs]
            req = dataclasses.replace(
                request, messages=with_placeholders(request.messages, texts))
            # The handler holds the ORIGINAL request; the cache report for
            # this one must reach it (engine/cachereport.attach).
            req._knurlogic_origin = request
            # Tower runs THIS request caused (store misses), for its report.
            req._knurlogic_encoded = self.encodes - before
            try:                    # and on the original, whichever the
                request._knurlogic_encoded = req._knurlogic_encoded
            except Exception:       # cache hook ends up holding
                pass
            prompt, segments, types, state = real(gen, tokenizer, req, args)
            key, seg_keys = K.expand_segments(segments, refs,
                                              self.spec.image_token_id)
        except BaseException:
            self.release([(r.sha, r.proc_hash) for r in refs])
            raise
        # expand_segments assigned refs in the order the pads appear, which
        # is the order of the parts: the template does not reorder content.
        # Pins were taken per part; the generator releases images_in(key),
        # one per image run -- the same multiset.
        return key, seg_keys, types, state


def tagged(args: Any) -> bool:
    """Was this request marked as carrying images on the HTTP thread?"""
    return bool(getattr(args, "_knurlogic_images", False))


def tag(args: Any) -> None:
    setattr(args, "_knurlogic_images", True)


__all__ = ["IMAGE_TYPES", "image_parts", "image_source", "has_images",
           "with_placeholders", "VisionServe", "tag", "tagged"]

