"""Encoded images, kept so an image is encoded once per conversation.

Keyed PER IMAGE by (model_key, sha, proc_hash): adding a second image to a
conversation is one miss, not a re-encode of the first; two models'
features are different spaces; a processor change cannot serve stale
features. BYTE-bounded, not count-bounded (a GLM image can be ~65 MB of
features, a small gemma one 1.4 MB); `tuning/fit.py` counts
DEFAULT_MAX_BYTES before a load.

Refs are never evicted (positions for a later text turn need the grid);
they die with the store on unload. `pinned()` holds entries past the bound
between tokenize and admit; the overshoot is visible in stats(). Stdlib
only; thread-safe (the generator thread writes, /status.json reads).
Design: docs/design/vision.md (image store).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from . import EncodedImage, ImageEvicted, ImageRef

#: Default bound for encoded features, bytes. 256 MiB:
#: ~4 of the largest GLM images or ~180 gemma images -- a conversation's worth
#: on any family, small next to a model on a 128 GB box. Also the number the
#: memory budget adds per served vision model (`budget_bytes`).
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

Key = tuple[Hashable, str, str]


def estimate_nbytes(n_tokens: int, text_hidden: int,
                    dtype_bytes: int = 2) -> int:
    """Features for one image, before it exists: n_tokens x hidden x
    itemsize. For the page and `fit` to say what an image will cost; the
    store itself only ever counts real arrays (EncodedImage.nbytes)."""
    return int(n_tokens) * int(text_hidden) * int(dtype_bytes)


class ImageStore:
    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self._lock = threading.RLock()
        self._lru: OrderedDict[Key, tuple[EncodedImage, int]] = OrderedDict()
        self._refs: dict[Key, ImageRef] = {}
        self._pins: dict[Key, int] = {}
        self._max = int(max_bytes)
        self._nbytes = 0
        self.hits = self.misses = self.evictions = 0

    # --- the bound -----------------------------------------------------------

    @property
    def max_bytes(self) -> int:
        return self._max

    @max_bytes.setter
    def max_bytes(self, n: int) -> None:
        """Resize; shrinking evicts now, not on the next put."""
        with self._lock:
            self._max = int(n)
            self._evict()

    @property
    def nbytes(self) -> int:
        """Bytes of features held now (pinned included)."""
        return self._nbytes

    def budget_bytes(self) -> int:
        """What the memory budget must reserve for this store: the bound,
        or more if pins currently hold it over. tuning/fit.py counts
        this (or DEFAULT_MAX_BYTES before a store exists)."""
        return max(self._max, self._nbytes)

    # --- lookups -------------------------------------------------------------

    def get(self, model_key: Hashable, sha: str,
            proc_hash: str) -> EncodedImage | None:
        """The encoded image, or None. A hit moves it to the fresh end."""
        k = (model_key, sha, proc_hash)
        with self._lock:
            v = self._lru.get(k)
            if v is None:
                self.misses += 1
                return None
            self._lru.move_to_end(k)
            self.hits += 1
            return v[0]

    def features(self, model_key: Hashable, sha: str,
                 proc_hash: str) -> EncodedImage:
        """As get(), but a miss raises ImageEvicted -- for embed time, where a
        missing image is a bug (it should have been pinned), not a miss."""
        v = self.get(model_key, sha, proc_hash)
        if v is None:
            raise ImageEvicted(f"image {sha[:12]} ({proc_hash}) is not in the "
                               f"store at embed time; pin it across admit")
        return v

    def ref(self, model_key: Hashable, sha: str,
            proc_hash: str) -> ImageRef | None:
        """The image's ref, even after its features were evicted."""
        return self._refs.get((model_key, sha, proc_hash))

    def lookup(self, model_key: Hashable):
        """(features, refs) callables bound to one model -- the
        FeatureLookup and RefLookup a Family's embed()/positions() take."""
        def feats(sha: str, ph: str) -> EncodedImage:
            return self.features(model_key, sha, ph)

        def refs(sha: str, ph: str) -> ImageRef:
            r = self.ref(model_key, sha, ph)
            if r is None:
                raise ImageEvicted(f"no ref for image {sha[:12]} ({ph})")
            return r
        return feats, refs

    def entry_nbytes(self, model_key: Hashable, sha: str,
                     proc_hash: str) -> int:
        """Bytes one stored image's features take (0 if not held)."""
        v = self._lru.get((model_key, sha, proc_hash))
        return v[1] if v is not None else 0

    def __contains__(self, k: Key) -> bool:
        return k in self._lru

    def __len__(self) -> int:
        return len(self._lru)

    # --- writes --------------------------------------------------------------

    def put(self, model_key: Hashable, enc: EncodedImage) -> bool:
        """Store one encoded image. Returns False if it could not stay: an
        unpinned image bigger than the whole bound is evicted at once (the
        caller still has `enc` for this request; pin before put to keep it).
        Evaluate the arrays before calling (EncodedImage's docstring)."""
        r = enc.ref
        k = (model_key, r.sha, r.proc_hash)
        n = enc.nbytes
        with self._lock:
            self._refs[k] = r
            old = self._lru.pop(k, None)
            if old is not None:
                self._nbytes -= old[1]
            self._lru[k] = (enc, n)
            self._nbytes += n
            self._evict()
            return k in self._lru

    def _evict(self) -> None:
        # Oldest unpinned first; pinned entries are skipped, not evicted, so
        # the store can sit above the bound while pins hold it there.
        if self._nbytes <= self._max:
            return
        for k in list(self._lru):
            if self._nbytes <= self._max:
                break
            if self._pins.get(k):
                continue
            _, n = self._lru.pop(k)
            self._nbytes -= n
            self.evictions += 1

    @contextmanager
    def pinned(self, model_key: Hashable,
               images: Iterable[tuple[str, str]]) -> Iterator[None]:
        """Hold these (sha, proc_hash) entries past the bound for the block.
        Pinning an image not yet put is allowed (pin, encode, put): the pin
        is what keeps it once it arrives. Reentrant (counted)."""
        keys = [(model_key, s, p) for s, p in images]
        with self._lock:
            for k in keys:
                self._pins[k] = self._pins.get(k, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                for k in keys:
                    c = self._pins.get(k, 0) - 1
                    if c > 0:
                        self._pins[k] = c
                    else:
                        self._pins.pop(k, None)
                self._evict()

    def clear(self, model_key: Hashable | None = None) -> None:
        """Forget everything (or one model's entries), refs included. Call on
        unload -- the prompt cache whose keys the refs serve dies with it."""
        with self._lock:
            for k in list(self._lru):
                if model_key is None or k[0] == model_key:
                    self._nbytes -= self._lru.pop(k)[1]
            for k in list(self._refs):
                if model_key is None or k[0] == model_key:
                    del self._refs[k]

    # --- what /status.json and the page show ---------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"entries": len(self._lru), "nbytes": self._nbytes,
                    "max_bytes": self._max,
                    "over_bound_by_pins": max(0, self._nbytes - self._max),
                    "pinned": sum(1 for c in self._pins.values() if c),
                    "refs": len(self._refs), "hits": self.hits,
                    "misses": self.misses, "evictions": self.evictions}
