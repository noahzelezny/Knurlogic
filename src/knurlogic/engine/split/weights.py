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
        items = []
        if isinstance(o, dict):
            items += [(o, k, v) for k, v in o.items()]
        elif isinstance(o, list):
            items += [(o, i, v) for i, v in enumerate(o)]
        elif isinstance(o, (tuple, set, frozenset)):
            stack.extend(o)
        if hasattr(o, "__dict__") and not isinstance(o, type):
            items += [(o, ("attr", k), v) for k, v in vars(o).items()]
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


def fill(model, read: Callable, source: str = "", stop=None) -> dict:
    """Put fetched arrays in place of the lazy ones `model` holds, one
    source tensor in flight (and the next being fetched). -> {fetched,
    left, failed}: bytes fetched, arrays left lazy (read from the mount)."""
    import mlx.core as mx
    import numpy as np
    heads: dict = {}
    todo, done = [], {}
    for c, k, a in _slots(model):
        o = _origin(a)
        todo.append((c, k, a, o))
    left = sum(1 for *_, o in todo if o is None)

    def fetch(o):
        src, path, name, _ = o
        if path not in heads:
            heads[path] = header(path)
        off = heads[path][name]
        buf = bytearray(src.nbytes)
        read(Path(path).name, off, len(buf), memoryview(buf))
        return buf

    jobs = [(c, k, a, o) for c, k, a, o in todo if o is not None]
    got, failed, last = 0, "", 0
    nxt: dict = {}

    def ahead(i):
        if i >= len(jobs) or id(jobs[i][2]) in done:
            return
        box: dict = {}

        def run():
            try:
                box["buf"] = fetch(jobs[i][3])
            except Exception as e:  # the mount reads what the peer did not
                box["err"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        nxt[i] = (t, box)
    ahead(0)
    for i, (c, k, a, o) in enumerate(jobs):
        if id(a) in done:
            _put(c, k, done[id(a)])
            continue
        if failed:
            continue
        t, box = nxt.pop(i, (None, None))
        if t is None:
            ahead(i)
            t, box = nxt.pop(i)
        t.join()
        if stop is not None and stop():
            break
        if "err" in box:
            failed = f"{type(box['err']).__name__}: {box['err']}"
            logger.warning("weights: fetching from %s failed (%s); the rest "
                           "is read from the mount", source or "the peer",
                           failed)
            continue
        j = i + 1
        while j < len(jobs) and id(jobs[j][2]) in done:
            j += 1
        ahead(j)
        src, _, _, fn = o
        buf = box.pop("buf")
        arr = mx.array(np.frombuffer(buf, dtype=np.uint8)).view(
            src.dtype).reshape(src.shape)
        del buf
        if fn is not None:
            arr = fn(arr)
        mx.eval(arr)
        done[id(a)] = arr
        _put(c, k, arr)
        got += src.nbytes
        if got - last >= GIB:
            last = got
            progress(fetched_bytes=got, reading_from=source)
    for t, _ in nxt.values():
        t.join()
    progress(fetched_bytes=got, reading_from="")
    return {"fetched": got, "left": left + (len(jobs) - len(done) if failed
                                            else 0), "failed": failed}


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
            out = fill(model, plan.read, plan.source, stop=LOAD_STOP.is_set)
            logger.info("weights: %.1f GiB from %s over the link; %d arrays "
                        "not from the peer (read here)%s", out["fetched"] / GIB,
                        plan.source or "the peer", out["left"],
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
