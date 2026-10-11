"""A split rank's read of its share: from a peer's page instead of a network
mount, and in turn (docs/design/weight-streaming.md).

While the model is built lazily, `recording` remembers every array mx.load
hands out (its shard file and tensor name) and the tensor split's cut
records what it derives from them (`derived`). `read_share` then waits for
the rank's turn, fills each array the rank still holds from the peer
(`fill`), evaluates what is left (read from the mount as before), and lets
the next turn go.

The reader comes from whoever starts the rank (interfaces/serve.py puts a
`ReadPlan` in CURRENT): engine never imports cluster. Outside a job with a
plan, every function here is a no-op.
"""
from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .marker import progress

logger = logging.getLogger(__name__)

GIB = 1 << 30


@dataclass
class ReadPlan:
    """This rank's entry of the job's read plan. `read(file, offset,
    length, into)` fills `into` (a writable memoryview of `length` bytes)
    with that range of the shard `file` (a file name) from the peer, or
    raises; None reads the mount. `turn` 0 reads at once, 1 after every
    turn-0 rank; `ordered` false: no waiting at all."""
    read: Callable | None = None
    source: str = ""
    turn: int = 0
    ordered: bool = False


#: plan: this rank's ReadPlan (None: read as before); link: the ring's
#: Link once joined (engine/split/link.init), whose bell carries the turn
CURRENT: dict = {"plan": None, "link": None}

#: id(array) -> (array, shard path, tensor name): what mx.load handed out
_SOURCES: dict = {}
#: id(out) -> (out, source array, fn): out == fn(source), lazily
_DERIVED: dict = {}


def _fetching() -> bool:
    plan = CURRENT["plan"]
    return plan is not None and plan.read is not None


@contextmanager
def recording():
    """Remember every array mx.load returns while the model is built (only
    when this rank fetches from a peer)."""
    if not _fetching():
        yield
        return
    import mlx.core as mx
    real = mx.load

    def load(f, *a, **kw):
        out = real(f, *a, **kw)
        if isinstance(f, (str, Path)) and str(f).endswith(".safetensors") \
                and isinstance(out, dict):
            for k, v in out.items():
                _SOURCES[id(v)] = (v, str(f), k)
        return out
    mx.load = load
    try:
        yield
    finally:
        mx.load = real


def derived(out, src, fn) -> None:
    """`out` is fn(src): rebuilt from src's bytes when src is fetched."""
    if _fetching() and out is not src:
        _DERIVED[id(out)] = (out, src, fn)


def forget() -> None:
    _SOURCES.clear()
    _DERIVED.clear()


def header(path: str) -> dict:
    """{tensor name: absolute byte offset} of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        doc = json.loads(f.read(n))
    return {k: 8 + n + int(v["data_offsets"][0]) for k, v in doc.items()
            if k != "__metadata__"}


def _origin(a):
    """(source array, path, name, fn) for an array the rank holds, or None
    when no record covers it (it is then read from the mount)."""
    fns = []
    seen = 0
    while id(a) not in _SOURCES:
        d = _DERIVED.get(id(a))
        if d is None or seen > 8:
            return None
        fns.append(d[2])
        a = d[1]
        seen += 1
    src, path, name = _SOURCES[id(a)]

    def fn(x):
        for f in reversed(fns):
            x = f(x)
        return x
    return src, path, name, (fn if fns else None)


#: libraries whose objects never hold the model's arrays: a walk does not
#: go into them (a transformers processor's lazy auto-mappings import whole
#: model families when touched, and can raise)
_FOREIGN = ("transformers", "tokenizers", "huggingface_hub", "jinja2",
            "PIL", "numpy", "torch", "threading", "logging", "http",
            "socket", "concurrent")


def _slots(root, limit: int = 2_000_000):
    """(container, key, array) for every mx.array slot under `root`: the
    places held_arrays (runtime/model_host) finds arrays, writable ones
    only (a tuple's are left alone, and read from the mount)."""
    import mlx.core as mx
    seen: set = set()
    stack = [root]
    while stack and len(seen) < limit:
        o = stack.pop()
        if isinstance(o, (str, bytes, int, float, bool, type(None), type,
                          bytearray, mx.array)) or id(o) in seen:
            continue
        seen.add(id(o))
        if type(o).__module__.split(".")[0] in _FOREIGN:
            continue
        items = []
        try:
            if isinstance(o, dict):
                items += [(o, k, v) for k, v in dict.items(o)]
            elif isinstance(o, list):
                items += [(o, i, v) for i, v in enumerate(o)]
            elif isinstance(o, (tuple, set, frozenset)):
                stack.extend(o)
            if hasattr(o, "__dict__") and not isinstance(o, type):
                items += [(o, ("attr", k), v) for k, v in vars(o).items()]
        except Exception:   # an object that will not be walked holds no share
            continue
        for c, k, v in items:
            if isinstance(v, mx.array):
                yield c, k, v
            else:
                stack.append(v)


def _put(c, k, v) -> None:
    if isinstance(k, tuple) and k and k[0] == "attr":
        object.__setattr__(c, k[1], v)
    else:
        c[k] = v


#: concurrent range requests: the holder's array reads cold at ~0.5 GB/s
#: one request at a time and ~2 GB/s with 8 in flight (measured over the
#: Thunderbolt 4 link, 64 MiB pieces)
PARALLEL = 8
#: a tensor is fetched in pieces of at most this
PIECE = 64 << 20
#: fetched bytes waiting to be built, beyond the tensor being built: the
#: memory bound of a fill (a larger tensor still goes, alone)
BUDGET = 1 << 30


class _Fetch:
    """The tensors of a fill, fetched ahead in order by PARALLEL workers,
    PIECE at a time, at most BUDGET bytes waiting; `take(i)` gives tensor
    i's buffer once all its pieces are in (or raises the first error)."""

    def __init__(self, items: list, read: Callable):
        self.items, self.read = items, read          # [(file, off, n)]
        self.bufs: dict = {}
        self.left: dict = {}
        self.cv = threading.Condition()
        self.held = 0
        self.err: BaseException | None = None
        self.closed = False
        self.tasks: list = []
        self.next = 0                    # the next tensor to schedule
        self.ts = [threading.Thread(target=self._work, daemon=True)
                   for _ in range(min(PARALLEL, max(len(items), 1)))]
        for t in self.ts:
            t.start()

    def _schedule(self) -> None:
        """(under cv) give buffers to tensors while the budget allows."""
        while self.next < len(self.items) and (
                self.held == 0 or
                self.held + self.items[self.next][2] <= BUDGET):
            i = self.next
            f, off, n = self.items[i]
            self.bufs[i] = bytearray(n)
            self.held += n
            ps = [(i, o, min(PIECE, n - o)) for o in range(0, n, PIECE)] \
                or [(i, 0, 0)]
            self.left[i] = len(ps)
            self.tasks.extend(ps)
            self.next += 1
        self.cv.notify_all()

    def _work(self) -> None:
        while True:
            with self.cv:
                while not self.tasks and not self.closed:
                    if self.err is None:
                        self._schedule()
                    if self.tasks:
                        break
                    self.cv.wait()
                if self.closed:
                    return
                i, o, n = self.tasks.pop(0)
                buf = self.bufs[i]
            try:
                if n:
                    f, off, _ = self.items[i]
                    self.read(f, off + o, n, memoryview(buf)[o:o + n])
            except BaseException as e:  # the mount reads what the peer did not
                with self.cv:
                    self.err = self.err or e
                    self.tasks.clear()
                    self.cv.notify_all()
                continue
            with self.cv:
                self.left[i] -= 1
                self.cv.notify_all()

    def take(self, i: int) -> bytearray:
        with self.cv:
            while self.err is None and self.left.get(i, 1) > 0:
                self._schedule()
                self.cv.wait()
            if self.err is not None:
                raise self.err
            buf = self.bufs.pop(i)
            self.held -= len(buf)
            self.left.pop(i)
            self._schedule()
            return buf

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.tasks.clear()
            self.cv.notify_all()
        for t in self.ts:
            t.join()


def _groups(items: list) -> list:
    """Tensors (file, offset, n) in file order, neighbours in one file
    merged up to PIECE: [(file, offset, n, [(index, rel, n)])]. The array
    reads a file front to back, and a layer's small tensors (scales,
    norms) go as one request, not one each."""
    out: list = []
    for i in sorted(range(len(items)), key=lambda i: items[i][:2]):
        f, off, n = items[i]
        g = out[-1] if out else None
        if g and g[0] == f and g[1] + g[2] == off and g[2] + n <= PIECE:
            g[2] += n
            g[3].append((i, off - g[1], n))
        else:
            out.append([f, off, n, [(i, 0, n)]])
    return [tuple(g) for g in out]


def fill(model, read: Callable, source: str = "", stop=None) -> dict:
    """Put fetched arrays in place of the lazy ones `model` (or each of a
    list of roots) holds, fetched in file order, in parallel pieces, at
    most BUDGET waiting. -> {fetched, left, failed}: bytes fetched,
    arrays not from the peer."""
    import mlx.core as mx
    import numpy as np
    heads: dict = {}
    roots = model if isinstance(model, list) else [model]
    todo = [(c, k, a, _origin(a)) for r in roots for c, k, a in _slots(r)]
    left = sum(1 for *_, o in todo if o is None)
    t0 = time.monotonic()
    big = sorted(((int(a.nbytes), k[1] if isinstance(k, tuple) else k)
                  for _, k, a, o in todo if o is None), reverse=True)[:6]
    if big:
        logger.info("weights: largest arrays not fetched: %s", ", ".join(
            f"{k} {n / GIB:.2f} GiB" for n, k in big))
    order: dict = {}                     # id(array) -> index among uniques
    uniq: list = []
    for _c, _k, a, o in todo:
        if o is not None and id(a) not in order:
            order[id(a)] = len(uniq)
            uniq.append((a, o))
    items = []
    for _a, (src, path, name, _) in uniq:
        if path not in heads:
            heads[path] = header(path)
        items.append((Path(path).name, heads[path][name], src.nbytes))
    groups = _groups(items)
    built: dict = {}
    got, failed, last = 0, "", 0
    f = _Fetch([g[:3] for g in groups], read)
    try:
        for gi, (_, _, _, members) in enumerate(groups):
            if stop is not None and stop():
                break
            try:
                buf = f.take(gi)
            except BaseException as e:
                failed = f"{type(e).__name__}: {e}"
                logger.warning("weights: fetching from %s failed (%s); the "
                               "rest is read from the mount",
                               source or "the peer", failed)
                break
            for i, rel, n in members:
                a, (src, _, _, fn) = uniq[i]
                arr = mx.array(np.frombuffer(buf, dtype=np.uint8, count=n,
                                             offset=rel)).view(
                    src.dtype).reshape(src.shape)
                if fn is not None:
                    arr = fn(arr)
                mx.eval(arr)
                built[id(a)] = arr
                got += n
            del buf
            if got - last >= GIB:
                last = got
                progress(fetched_bytes=got, reading_from=source)
    finally:
        f.close()
    for c, k, a, _ in todo:
        if id(a) in built:
            _put(c, k, built[id(a)])
    progress(fetched_bytes=got, reading_from="")
    return {"fetched": got, "left": left + len(uniq) - len(built),
            "failed": failed, "seconds": time.monotonic() - t0}


def fill_bound(roots: list) -> None:
    """After the weights: what was bound beside them (the drafting head,
    the vision tower) under `recording`, filled from the peer too."""
    plan = CURRENT["plan"]
    try:
        if plan is not None and plan.read is not None and _SOURCES:
            from knurlogic.engine.runtime.model_host import LOAD_STOP
            out = fill([r for r in roots if r is not None], plan.read,
                       plan.source, stop=LOAD_STOP.is_set)
            logger.info("weights: head and tower %.1f GiB from %s over the "
                        "link in %.1fs%s", out["fetched"] / GIB,
                        plan.source or "the peer", out["seconds"],
                        f" ({out['failed']})" if out["failed"] else "")
    except Exception:  # only slower: what is left lazy is read from the mount
        logger.exception("weights: the head and tower were not fetched; "
                         "they are read from the mount")
    finally:
        forget()


def read_share(model, evaluate: Callable) -> None:
    """The rank's read: wait its turn, fill from the peer, evaluate the
    rest, let the next turn go. Without a plan: evaluate(model)."""
    plan = CURRENT["plan"]
    if plan is None:
        evaluate(model)
        return
    link = CURRENT["link"]
    if plan.ordered and plan.turn == 1 and link is not None:
        progress(turn="waiting")
        logger.info("weights: waiting for the ranks that read remotely")
        link.align()
        progress(turn="reading")
    try:
        if plan.read is not None:
            from knurlogic.engine.runtime.model_host import LOAD_STOP
            try:
                out = fill(model, plan.read, plan.source,
                           stop=LOAD_STOP.is_set)
            except Exception as e:  # only slower: the mount reads it all
                logger.exception("weights: not fetched; the share is read "
                                 "from the mount")
                out = {"fetched": 0, "left": 0,
                       "failed": f"{type(e).__name__}: {e}"}
            logger.info("weights: %.1f GiB from %s over the link in %.1fs; "
                        "%d arrays not from the peer (read here)%s",
                        out["fetched"] / GIB, plan.source or "the peer",
                        out.get("seconds", 0.0), out["left"],
                        f" ({out['failed']})" if out["failed"] else "")
        evaluate(model)
    finally:
        forget()
    if plan.ordered and plan.turn == 0 and link is not None:
        link.align()


def plan_from_env(env, reader: Callable) -> ReadPlan | None:
    """The ReadPlan in KNURLOGIC_WEIGHTS (cluster/weights.rank_env), with
    `reader(page)` -> read() for a peer page; None without one."""
    raw = env.get("KNURLOGIC_WEIGHTS")
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    page = str(d.get("from") or "")
    return ReadPlan(read=reader(page) if page else None,
                    source=str(d.get("source") or ""),
                    turn=1 if d.get("turn") == 1 else 0,
                    ordered=bool(d.get("ordered")))
