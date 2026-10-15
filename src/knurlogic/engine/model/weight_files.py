"""The one place an artifact's tensor files are opened.

Every load of a model's safetensors goes through `load` (the trunk, through
load_model; a family's drafting head and vision tower, directly). Nothing
else calls `mx.load` on an artifact's files. That keeps where the bytes come
from in one home:

- normally, from the file itself: `mx.load`, lazy, exactly as before;
- on a split rank whose copy is on a network mount, from the page of a
  machine holding the files (`peer` set by engine/split/weights.py): a
  family's open (`load`) is fetched over the link at once, in parallel
  pieces; the trunk's open (`load(..., deferred=True)`) stays lazy and is
  recorded, so the split fills only the share it keeps (`origin`). What the
  model's sanitize makes from those tensors (a fused projection, say) is
  in no file: `sanitize` records it, and `remake` makes it again from the
  consumed tensors' fetched bytes (`made`).

A fetch that fails falls back to the file, logged. KeyboardInterrupt and
SystemExit are never caught. Design: docs/design/weight-streaming.md.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from knurlogic.machine.artifact import safetensors_layout
from knurlogic.machine.units import GIB, gib

logger = logging.getLogger(__name__)


#: range requests in flight. Measured from the M4 against an M3's SoftRAID
#: array (cold, 64 MiB pieces): 0.50 GB/s with one, 2.04 GB/s with eight
PARALLEL = 8
#: a range request's size: one tensor goes in several, small neighbours
#: share one
PIECE = 64 << 20
#: fetched bytes waiting to be built: the memory bound of a fetch (a
#: larger tensor still goes, alone)
BUDGET = 4 << 30
#: the most `remake` fetches; above it (a sanitize that stacks every
#: expert, say) what sanitize made is read from the files
REMAKE_LIMIT = 2 << 30


# --------------------------------------------------------------- headers

def header(path) -> dict:
    """{tensor name: (absolute offset, length, dtype, shape)} of a
    safetensors file, and its metadata under "__metadata__"."""
    doc, start = safetensors_layout(path)
    out: dict = {"__metadata__": doc.pop("__metadata__", None) or {}}
    for k, v in doc.items():
        a, b = v["data_offsets"]
        out[k] = (start + int(a), int(b) - int(a), v["dtype"],
                  tuple(v["shape"]))
    return out


def names(path) -> list:
    """The tensor names in a safetensors file, read from its header."""
    return [k for k in header(path) if k != "__metadata__"]


def metadata(path) -> dict:
    """A safetensors file's metadata (its header's __metadata__)."""
    return dict(header(path)["__metadata__"])


# ------------------------------------------------------- an artifact's map

INDEX = "model.safetensors.index.json"


def weight_map(model_path, pattern: str = "*.safetensors") -> dict:
    """{tensor name: file name} of the artifact at `model_path`: its index's
    weight_map, else every `pattern` file's header. Raises OSError /
    ValueError when they cannot be read."""
    import json
    root = Path(str(model_path))
    index = root / INDEX
    if index.is_file():
        wmap = json.loads(index.read_text()).get("weight_map") or {}
        if wmap:
            return dict(wmap)
    return {k: f.name for f in sorted(root.glob(pattern)) for k in names(f)}


def files_with(model_path, accept: Callable[[str], bool]) -> dict:
    """{file path: [tensor names]} of the artifact's tensors `accept` takes,
    in the order the map gives them."""
    root = Path(str(model_path))
    out: dict = {}
    for k, f in weight_map(root).items():
        if accept(k):
            out.setdefault(root / f, []).append(k)
    return out


def tensors_under(model_path, prefixes) -> dict:
    """{name: array} of every tensor of the artifact whose name starts with
    one of `prefixes`, each file opened through `load`."""
    prefixes = tuple(prefixes)
    out: dict = {}
    for f, keys in files_with(model_path,
                              lambda k: k.startswith(prefixes)).items():
        out.update(load(f, keys=keys))
    return out


# ------------------------------------------------------------ the source

@dataclass
class Peer:
    """Where a split rank's tensor bytes come from instead of its files:
    `read(file name, offset, length, into)` fills `into` from the page of
    `source` (a machine holding the artifact `root`), or raises."""
    read: Callable
    source: str
    root: Path


#: the peer while a split rank loads (engine/split/weights.py sets it)
_PEER: dict = {"peer": None}
#: id(array) -> (array, file path, tensor name): the deferred opens
_OPENED: dict = {}
#: the model's sanitize over the deferred opens: fn (the bound sanitize),
#: inputs {name: (file path, tensor name)} it consumed, made
#: {id(array): (array, name)} the arrays it made
_SANITIZED: dict = {"fn": None, "inputs": {}, "made": {}}
_LOCK = threading.Lock()


def use_peer(peer: Peer | None) -> None:
    _PEER["peer"] = peer
    if peer is None:
        forget()


def peer() -> Peer | None:
    return _PEER["peer"]


def origin(array):
    """(file path, tensor name) a deferred open gave `array`, or None."""
    rec = _OPENED.get(id(array))
    return (rec[1], rec[2]) if rec is not None and rec[0] is array else None


def made(array) -> str | None:
    """The name `array` has in the output of the sanitize that made it
    (`sanitize`), or None."""
    rec = _SANITIZED["made"].get(id(array))
    return rec[1] if rec is not None and rec[0] is array else None


def forget() -> None:
    with _LOCK:
        _OPENED.clear()
        _SANITIZED.update(fn=None, inputs={}, made={})


def _streamable(path: Path, p: Peer) -> bool:
    """A file of the artifact being streamed (directly in its folder: the
    serving page takes a bare file name)."""
    try:
        return path.resolve().parent == p.root.resolve()
    except OSError:
        return False


# --------------------------------------------------------------- opening

def load(path, *, keys: Iterable[str] | None = None,
         deferred: bool = False) -> dict:
    """{name: array} of a safetensors file (only `keys`, when given).

    Without a peer: `mx.load`, lazy. With one: fetched over the link now,
    or, `deferred`, opened lazily and recorded for the split's fill."""
    import mlx.core as mx
    path = Path(str(path))
    p = peer()
    want = None if keys is None else set(keys)
    if p is None or not _streamable(path, p):
        out = mx.load(str(path))
        return out if want is None else {k: out[k] for k in out if k in want}
    if deferred:
        out = mx.load(str(path))
        if want is not None:
            out = {k: out[k] for k in out if k in want}
        with _LOCK:
            for k, v in out.items():
                _OPENED[id(v)] = (v, str(path), k)
        return out
    todo = [k for k in names(path) if want is None or k in want]
    t0 = time.monotonic()
    try:
        out, got = fetch({k: (str(path), k) for k in todo}, p)
        logger.info("weights: %s, %d tensors, %.2f GiB from %s over the "
                    "link in %.1fs", path.name, len(out), got / GIB,
                    p.source, time.monotonic() - t0)
        return out
    except Exception as e:  # only slower: the file is read instead
        logger.warning("weights: %s not fetched from %s (%s: %s); read "
                       "from the file", path.name, p.source,
                       type(e).__name__, e)
        out = mx.load(str(path))
        return {k: out[k] for k in todo}


# ------------------------------------------------------------- sanitize

def sanitize(model, weights: dict) -> dict:
    """model.sanitize(weights) (load_model's call). While a peer is set it
    also records what that did to the deferred opens: the tensors it
    consumed (gone from its output) and the arrays it made, for `remake`."""
    given = dict(weights)                # a sanitize may pop from weights
    out = model.sanitize(weights)
    if peer() is None:
        return out
    kept = {id(v) for v in out.values()}
    inputs = {k: o for k, v in given.items()
              if id(v) not in kept and (o := origin(v)) is not None}
    known = {id(v) for v in given.values()}
    with _LOCK:
        _SANITIZED.update(
            fn=model.sanitize, inputs=inputs,
            made={id(v): (v, k) for k, v in out.items() if id(v) not in known})
    return out


def remake(stop: Callable | None = None) -> tuple:
    """The recorded sanitize run again on the tensors it consumed, fetched
    from the peer. -> ({name: array} it makes, bytes fetched). Raises when
    the inputs pass REMAKE_LIMIT, or the fetch or the sanitize fails."""
    p = peer()
    assert p is not None                # asked only with one (split/weights)
    got, n = fetch(_SANITIZED["inputs"], p, stop=stop, limit=REMAKE_LIMIT)
    return _SANITIZED["fn"](got), n


# --------------------------------------------------------------- fetching

_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32",
           "F64": "float64", "U8": "uint8", "I8": "int8", "U16": "uint16",
           "I16": "int16", "U32": "uint32", "I32": "int32", "U64": "uint64",
           "I64": "int64", "BOOL": "bool_"}


def fetch(tensors: dict, p: Peer, *, each: Callable | None = None,
          stop: Callable | None = None, limit: int | None = None) -> tuple:
    """Fetch `tensors` {key: (file path, tensor name)}, from any files of
    the artifact, over the link. -> ({key: array}, bytes fetched); with
    `each`, each(key, array) takes every array as it is built instead.
    Raises ValueError before fetching when the bytes pass `limit`, and on
    a failed range."""
    heads: dict = {}
    keys = list(tensors)
    items = []
    for k in keys:
        path, name = tensors[k]
        if path not in heads:
            heads[path] = header(path)
        off, n, _, _ = heads[path][name]
        items.append((Path(path).name, off, n))
    total = sum(n for _, _, n in items)
    if limit is not None and total > limit:
        raise ValueError(f"{gib(total, 2)} to fetch, over the limit "
                         f"({gib(limit, 0)})")
    out: dict = {}

    def build(i: int, view) -> None:
        path, name = tensors[keys[i]]
        _, _, dt, shape = heads[path][name]
        a = _to_array(view, _mx_dtype(dt), shape)
        if each is None:
            out[keys[i]] = a
        else:
            each(keys[i], a)
    return out, ranges(items, p.read, build, stop=stop)


def _mx_dtype(name: str):
    """The mx dtype of a safetensors dtype name ("BF16")."""
    import mlx.core as mx
    return getattr(mx, _DTYPES[name])


def _to_array(view, dtype, shape):
    """An evaluated mx.array of `dtype` and `shape` from raw bytes."""
    import mlx.core as mx
    import numpy as np
    a = mx.array(np.frombuffer(view, dtype=np.uint8)).view(dtype)
    a = a.reshape(shape)
    mx.eval(a)
    return a


def ranges(items: list, read: Callable, build: Callable,
           stop: Callable | None = None) -> int:
    """Fetch byte ranges `items` [(file name, offset, length)] with `read`
    and call build(index, memoryview) for each, in order. Neighbouring
    ranges of one file go in one request (up to PIECE); a range is read in
    PIECE pieces, PARALLEL at a time, at most BUDGET waiting. -> bytes."""
    order = sorted(range(len(items)), key=lambda i: items[i][:2])
    groups: list = []                    # [file, offset, length, [(i, rel)]]
    for i in order:
        f, off, n = items[i]
        g = groups[-1] if groups else None
        if g and g[0] == f and g[1] + g[2] == off and g[2] + n <= PIECE:
            g[3].append((i, g[2]))
            g[2] += n
        else:
            groups.append([f, off, n, [(i, 0)]])
    pool = _Pool([(g[0], g[1], g[2]) for g in groups], read)
    got = 0
    try:
        for gi, (_, _, _, members) in enumerate(groups):
            if stop is not None and stop():
                raise InterruptedError("the load was stopped")
            buf = pool.take(gi)
            mv = memoryview(buf)
            for i, rel in members:
                n = items[i][2]
                build(i, mv[rel:rel + n])
                got += n
            del mv, buf
    finally:
        pool.close()
    return got


class _Pool:
    """Ranges fetched ahead in order by PARALLEL workers, PIECE at a time,
    at most BUDGET bytes waiting; `take(i)` gives range i's bytes once all
    its pieces are in, or raises the first error."""

    def __init__(self, ranges_: list, read: Callable):
        self.ranges, self.read = ranges_, read
        self.bufs: dict = {}
        self.left: dict = {}
        self.cv = threading.Condition()
        self.held = 0
        self.err: Exception | None = None
        self.closed = False
        self.tasks: deque = deque()
        self.next = 0
        self.workers = [threading.Thread(target=self._work, daemon=True)
                        for _ in range(min(PARALLEL, max(len(ranges_), 1)))]
        for t in self.workers:
            t.start()

    def _schedule(self) -> None:
        """(under cv) give buffers to ranges while the budget allows."""
        while self.next < len(self.ranges) and (
                self.held == 0 or
                self.held + self.ranges[self.next][2] <= BUDGET):
            i = self.next
            n = self.ranges[i][2]
            self.bufs[i] = bytearray(n)
            self.held += n
            ps = [(i, o, min(PIECE, n - o)) for o in range(0, n, PIECE)]
            self.left[i] = len(ps)       # 0 for an empty tensor
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
                i, o, n = self.tasks.popleft()
                view = memoryview(self.bufs[i])[o:o + n]
            f, off, _ = self.ranges[i]
            try:
                self.read(f, off + o, n, view)
            except Exception as e:  # the taker raises it; the load decides
                with self.cv:
                    self.err = self.err or e
                    self.tasks.clear()
                    self.cv.notify_all()
                continue
            finally:
                view.release()
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
        for t in self.workers:
            t.join()
