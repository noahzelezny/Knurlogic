"""Images in use are never evicted: the prompt cache pins what it references.

Flash-Next review point 1. A prompt-cache entry whose key holds image
sentinels (key.py) is KV computed from those images. While it lives, a turn
that extends it may re-read the images -- a partial hit that cuts into an
image span re-embeds the rest of that span from the store. So each image
stays in the ImageStore while ANY live entry references its sha: one store
pin per (entry, image run), taken when the entry is inserted, dropped when
it leaves the cache by any path.

HOW. mlx-lm's LRUPromptCache (mlx_lm/models/cache.py) removes entries on
four paths: insert_cache's replacement of an equal key, its pop_prefixes of
shorter keys, its size/bytes LRU pops, and trim_to. Intercepting each is
brittle; instead the two MUTATING methods are wrapped by name (with
assertions -- a pinned mlx-lm that renames one fails at install, loudly),
and after either returns the hook RECONCILES: the live entries are read off
`_lru._lrus` (the deques of (model, tokens) every path keeps in step with the
trie), new ones are pinned, gone ones unpinned. Entries are tracked by the
identity of their `tokens` list (the hook holds a reference, so the id is
stable); a replacement of an equal key is a new list, so the old entry's pins
go and the new one's come.

THE ADMIT GAP (P4's leak). VisionServe.tokenize pins a request's images
until the batch generator admits the row. If mlx-lm's `_generate` raises
between `_tokenize` and `insert_segments` (in `_make_state_machine`, say),
nothing admits the row and the pins stay forever. `pending()` records the
pins a tokenize took; `claim()` is called once `insert_segments` has queued
the row (from then on the generator's admit/remove releases them);
`sweep()` releases whatever was never claimed. engine/serve/vision.py calls sweep at the
top of every `_tokenize` -- the generator thread is sequential, so a pending
entry still unclaimed when the next request is tokenized was abandoned.
`install_admit(cls)` wraps a batch generator class's insert_segments to
claim; `admit_guard()` is the same protocol as a context manager.

Stdlib only; no mlx import (the wrapped objects are mlx-lm's, passed in).
"""
from __future__ import annotations

import threading
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Tuple

from . import key as K

#: The LRUPromptCache methods that add or remove entries, wrapped by name.
MUTATORS = ("insert_cache", "trim_to")

StoreLike = Any
StoreGetter = Callable[[], Optional[StoreLike]]

_STATE_ATTR = "_knurlogic_cachehook"


class _Refs:
    """Per-cache bookkeeping: id(tokens) -> (tokens, the pins it holds)."""

    def __init__(self, store: StoreGetter):
        self.store = store
        self.lock = threading.Lock()
        self.entries: Dict[int, Tuple[Any, ExitStack]] = {}

    def reconcile(self, cache: Any) -> None:
        lrus = cache._lru._lrus
        live: Dict[int, Tuple[Hashable, Any]] = {}
        for dq in lrus.values():
            for model, tokens in dq:
                live[id(tokens)] = (model, tokens)
        with self.lock:
            gone = [i for i in self.entries if i not in live]
            dropped = [self.entries.pop(i)[1] for i in gone]
            for i, (model, tokens) in live.items():
                if i in self.entries or not K.has_image(tokens):
                    continue
                st = ExitStack()
                s = self.store()
                if s is not None:
                    st.enter_context(s.pinned(model, K.images_in(tokens)))
                self.entries[i] = (tokens, st)
        for st in dropped:
            st.close()

    def release_all(self) -> None:
        with self.lock:
            stacks = [st for _, st in self.entries.values()]
            self.entries.clear()
        for st in stacks:
            st.close()

    def count(self) -> int:
        with self.lock:
            return len(self.entries)


def _cache_class(target: Any):
    cls = getattr(target, "LRUPromptCache", None)
    if cls is None:
        cls = target if isinstance(target, type) else type(target)
    for name in MUTATORS:
        assert callable(getattr(cls, name, None)), (
            f"mlx_lm LRUPromptCache.{name} is gone: the image refcount wraps "
            f"it by name. The pinned mlx-lm changed; re-read "
            f"mlx_lm/models/cache.py before re-pinning.")
    return cls


def _refs(cache: Any) -> Optional[_Refs]:
    return cache.__dict__.get(_STATE_ATTR)


def install(target: Any, store: StoreGetter | StoreLike) -> None:
    """Make every LRUPromptCache pin the images its entries reference.

    `target` is mlx_lm.server (or any module exposing LRUPromptCache), the
    class itself, or an instance. The class's mutators are wrapped once
    (idempotent); each instance keeps its own refcount. `store` is the
    ImageStore, or a zero-argument callable returning the current one (or
    None when nothing with vision is served) -- serve/vision.py passes a getter,
    since the store changes with every load."""
    getter: StoreGetter = store if callable(store) and not hasattr(
        store, "pinned") else (lambda s=store: s)
    cls = _cache_class(target)
    cls._knurlogic_store_getter = staticmethod(getter)
    if not isinstance(target, type) and hasattr(target, "_lru") and (
            type(target) is cls or isinstance(target, cls)):
        target.__dict__.setdefault(_STATE_ATTR, _Refs(getter))
    for name in MUTATORS:
        fn = getattr(cls, name)
        if getattr(fn, "_knurlogic_cachehook", False):
            continue
        setattr(cls, name, _wrap(fn))


def _wrap(real):
    def wrapped(self, *a, **kw):
        out = real(self, *a, **kw)
        lru = getattr(self, "_lru", None)
        assert lru is not None and hasattr(lru, "_lrus"), (
            "LRUPromptCache no longer keeps its order in `_lru._lrus`; the "
            "image refcount reads live entries there. Re-read "
            "mlx_lm/models/cache.py.")
        refs = self.__dict__.get(_STATE_ATTR)
        if refs is None:
            refs = self.__dict__.setdefault(
                _STATE_ATTR, _Refs(type(self)._knurlogic_store_getter))
        refs.reconcile(self)
        return out
    wrapped._knurlogic_cachehook = True
    wrapped.__name__ = real.__name__
    wrapped.__wrapped__ = real
    return wrapped


def pinned_entries(cache: Any) -> int:
    """How many live entries of this cache hold image pins."""
    r = _refs(cache)
    return r.count() if r is not None else 0


def release(cache: Any) -> None:
    """Drop every pin this cache holds -- for unload, with store.clear()."""
    r = _refs(cache)
    if r is not None:
        r.release_all()


# --- the admit gap -----------------------------------------------------------

_PENDING = threading.local()


def _pending_list() -> List[Tuple[Any, List[Tuple[str, str]]]]:
    lst = getattr(_PENDING, "items", None)
    if lst is None:
        lst = _PENDING.items = []
    return lst


def pending(serve: Any, images: List[Tuple[str, str]]) -> None:
    """Record that a tokenize on this thread left these images pinned on
    `serve` (a VisionServe: its release() drops one pin per image), to be
    claimed by the admit or swept."""
    if images:
        _pending_list().append((serve, list(images)))


def claim() -> int:
    """The row was queued: its pins now belong to the batch generator,
    which releases them at admit or remove. Returns how many requests'
    pins were handed over."""
    lst = _pending_list()
    n = len(lst)
    lst.clear()
    return n


def sweep() -> int:
    """Release the pins of every tokenize on this thread that never reached
    insert_segments. Returns the number of images released."""
    lst = _pending_list()
    items = list(lst)
    lst.clear()
    n = 0
    for serve, images in items:
        serve.release(images)
        n += len(images)
    return n


def outstanding() -> int:
    """Unclaimed tokenize pins on this thread (for tests and status)."""
    return sum(len(i) for _, i in _pending_list())


@contextmanager
def admit_guard() -> Iterator[None]:
    """Everything between a tokenize and its insert: a clean exit claims,
    an exception sweeps. For callers that own that span of code."""
    try:
        yield
    except BaseException:
        sweep()
        raise
    claim()


def install_admit(cls: type) -> None:
    """Wrap a batch generator class's insert_segments (by name, asserted)
    so a successful insert claims this thread's pending pins."""
    real = getattr(cls, "insert_segments", None)
    assert callable(real), (
        f"{cls.__name__}.insert_segments is gone: the admit claim wraps it "
        f"by name.")
    if getattr(real, "_knurlogic_cachehook", False):
        return

    def insert_segments(self, *a, **kw):
        out = real(self, *a, **kw)
        claim()
        return out
    insert_segments._knurlogic_cachehook = True
    insert_segments.__wrapped__ = real
    cls.insert_segments = insert_segments


__all__ = ["MUTATORS", "install", "pinned_entries", "release", "pending",
           "claim", "sweep", "outstanding", "admit_guard", "install_admit"]
