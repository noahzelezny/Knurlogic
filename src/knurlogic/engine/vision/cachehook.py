"""Images in use are never evicted: the prompt cache pins what it references.

A prompt-cache entry whose key holds image sentinels (key.py) is KV
computed from those images, and a turn that extends it may re-read them.
So each image stays in the ImageStore while ANY live entry references its
sha: one store pin per (entry, image run). The two MUTATING methods of
mlx-lm's LRUPromptCache are wrapped by name (asserted, so a rename fails
loudly at install) and after either returns the hook RECONCILES the live
entries against its pins.

The admit gap: tokenize pins a request's images until the batch generator
admits the row. `pending()` records those pins, `claim()` hands them to the
generator, and `sweep()` releases any a failed admission abandoned; the
scheduler wraps tokenize-to-insert in `admit_guard()`. Stdlib only.
Design: docs/design/vision.md (cache pins).
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




__all__ = ["MUTATORS", "install", "pinned_entries", "release", "pending",
           "claim", "sweep", "outstanding", "admit_guard"]
