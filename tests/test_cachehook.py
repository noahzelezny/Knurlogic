"""Images in use are never evicted (Flash-Next review point 1), and a pin
taken at tokenize cannot leak when the server raises before insert (P4).

The prompt cache is mlx-lm's real LRUPromptCache (the pinned install), with
stand-in KV entries: the cache only reads `.nbytes` off them. No model runs.
"""
from __future__ import annotations

import pytest

from knurlogic.engine.vision import EncodedImage, ImageRef
from knurlogic.engine.vision import cachehook
from knurlogic.engine.vision import key as K
from knurlogic.engine.vision.request import VisionServe
from knurlogic.engine.vision.store import ImageStore

MK = ("model", None, None)
PH = "ph0"


class _Arr:
    def __init__(self, n):
        self.nbytes = n


class _KV:
    nbytes = 10

    def is_trimmable(self):          # so insert_cache pops shorter prefixes
        return True


def _ref(sha, n=3):
    return ImageRef(sha=sha, proc_hash=PH, n_tokens=n)


def _put(store, sha, nbytes=100):
    store.put(MK, EncodedImage(ref=_ref(sha), feats=_Arr(nbytes)))


def _key(*shas):
    """[text, image run, text, ...] as the tokenize wrap builds it."""
    out = [1, 2]
    for s in shas:
        r = _ref(s)
        out += [K.sentinel(r, k) for k in range(r.n_tokens)] + [5]
    return out


@pytest.fixture
def cache_cls():
    # A subclass, so the class-level wrap never leaks into mlx-lm itself.
    from mlx_lm.models.cache import LRUPromptCache

    class C(LRUPromptCache):
        pass
    return C


def _pressure(store, n=5):
    for i in range(n):
        _put(store, f"filler{i}")


def test_referenced_image_survives_pressure_then_goes_with_its_entry(
        cache_cls):
    store = ImageStore(max_bytes=250)          # two images' worth
    cachehook.install(cache_cls, lambda: store)
    pc = cache_cls(max_size=1)
    _put(store, "used")
    _put(store, "unused")
    pc.insert_cache(MK, _key("used"), [_KV()])
    assert cachehook.pinned_entries(pc) == 1
    _pressure(store)
    assert (MK, "used", PH) in store             # pinned by the live entry
    assert (MK, "unused", PH) not in store       # nobody references it
    # max_size=1: inserting another key evicts the first entry, which must
    # drop its pin; the next pressure then evicts the image.
    pc.insert_cache(MK, [9, 9, 9], [_KV()])
    assert cachehook.pinned_entries(pc) == 0
    _pressure(store)
    assert (MK, "used", PH) not in store
    assert store.stats()["pinned"] == 0


def test_trim_and_replacement_unpin(cache_cls):
    store = ImageStore(max_bytes=250)
    cachehook.install(cache_cls, lambda: store)
    pc = cache_cls(max_size=10)
    k = _key("a", "b")
    pc.insert_cache(MK, k, [_KV()])
    pc.insert_cache(MK, list(k), [_KV()])      # equal key: a replacement
    assert cachehook.pinned_entries(pc) == 1
    assert store.stats()["pinned"] == 2         # one pin per image, not two
    pc.trim_to(n_sequences=0)
    assert cachehook.pinned_entries(pc) == 0
    assert store.stats()["pinned"] == 0


def test_prefix_pop_moves_the_pin_to_the_longer_entry(cache_cls):
    store = ImageStore(max_bytes=150)
    cachehook.install(cache_cls, lambda: store)
    pc = cache_cls(max_size=10)
    k = _key("a")
    pc.insert_cache(MK, k, [_KV()], cache_type="user")
    pc.insert_cache(MK, k + [7, 7], [_KV()])   # pops the shorter prefix
    assert len(pc) == 1 and cachehook.pinned_entries(pc) == 1
    _put(store, "a")
    _pressure(store)
    assert (MK, "a", PH) in store
    assert store.stats()["pinned"] == 1


def test_install_asserts_by_name():
    class Renamed:
        def insert(self):
            pass
    with pytest.raises(AssertionError, match="insert_cache"):
        cachehook.install(Renamed, lambda: None)


# --- the admit gap -----------------------------------------------------------

def _server_step(vs, fail_before_insert, inserted):
    """mlx-lm _generate's shape between _tokenize and insert_segments, with
    the seam's _tokenize wrap as reported: sweep, tokenize, pending."""
    cachehook.sweep()
    imgs = [("x", PH)]
    vs._pin(imgs)                               # what tokenize leaves pinned
    cachehook.pending(vs, imgs)
    if fail_before_insert:
        raise RuntimeError("_make_state_machine blew up")
    inserted.append(imgs)
    cachehook.claim()                           # the insert_segments wrap


def test_no_leak_when_the_server_raises_between_tokenize_and_insert():
    store = ImageStore(max_bytes=150)
    vs = VisionServe(family=None, store=store, model_key=MK)
    inserted = []
    with pytest.raises(RuntimeError):
        _server_step(vs, True, inserted)
    assert vs.pinned_count() == 1               # held until the next tokenize
    _server_step(vs, False, inserted)           # the next request sweeps it
    assert vs.pinned_count() == 1               # only the admitted row's pin
    # the batch generator releases at admit, as it does today
    vs.release(inserted.pop())
    assert vs.pinned_count() == 0 and store.stats()["pinned"] == 0
    assert cachehook.outstanding() == 0


def test_admit_guard_and_insert_wrap():
    store = ImageStore(max_bytes=150)
    vs = VisionServe(family=None, store=store, model_key=MK)
    imgs = [("y", PH)]
    with pytest.raises(ValueError):
        with cachehook.admit_guard():
            vs._pin(imgs)
            cachehook.pending(vs, imgs)
            raise ValueError
    assert vs.pinned_count() == 0

    class Gen:
        def insert_segments(self, **kw):
            return [1]
    cachehook.install_admit(Gen)
    vs._pin(imgs)
    cachehook.pending(vs, imgs)
    Gen().insert_segments(segments=[])
    assert cachehook.outstanding() == 0
    assert cachehook.sweep() == 0
    assert vs.pinned_count() == 1               # the row owns it now
    vs.release(imgs)
