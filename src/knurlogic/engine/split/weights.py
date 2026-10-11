"""A split rank's read of its share: from a peer's page instead of a network
mount, and in turn (docs/design/weight-streaming.md).

The rank's weight files are opened by engine/model/weight_files.py, which
records what each lazy array is (file, tensor) while this rank has a peer.
After the model is cut to the rank's share, `read_share` waits for the
rank's turn, fetches the bytes of every parameter the model still holds
(`model.parameters()`; a tensor split's cut is redone on the fetched bytes,
`derived`), puts them in place with `model.update`, and evaluates what is
left from the files. `done` ends the rank's turn once its drafting head and
vision tower are in too.

Whoever starts the rank sets the plan (interfaces/serve.py): engine never
imports cluster. Without a plan, every function here is a no-op.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from knurlogic.engine.model import weight_files

from .marker import progress

logger = logging.getLogger(__name__)

GIB = 1 << 30


@dataclass
class ReadPlan:
    """This rank's entry of the job's read plan: `turn` 0 reads at once, 1
    after every turn-0 rank; `ordered` false: no waiting at all. Where the
    bytes come from is weight_files.peer()."""
    turn: int = 0
    ordered: bool = False


#: plan: this rank's ReadPlan (None: read as before); link: the ring's
#: Link once joined (engine/split/link.init), whose bell carries the turn
CURRENT: dict = {"plan": None, "link": None}

#: id(out) -> (out, source array, fn): out == fn(source), a tensor split's
#: cut of a parameter (engine/split/tensor.py)
_DERIVED: dict = {}


def derived(out, src, fn) -> None:
    """`out` is fn(src): rebuilt from src's bytes when src is fetched."""
    if weight_files.peer() is not None and out is not src:
        _DERIVED[id(out)] = (out, src, fn)


def _origin(a):
    """(file path, tensor name, source array, fn) for a parameter, or None
    when it is not a file's tensor or a cut of one (read from the files)."""
    fns = []
    for _ in range(8):
        o = weight_files.origin(a)
        if o is not None:
            return o[0], o[1], a, (_chain(fns[::-1]) if fns else None)
        d = _DERIVED.get(id(a))
        if d is None or d[0] is not a:
            return None
        fns.append(d[2])
        a = d[1]
    return None


def _chain(fns: list) -> Callable:
    """x -> fns[-1](...fns[0](x))."""
    def run(x):
        for f in fns:
            x = f(x)
        return x
    return run


def fill(model, stop: Callable | None = None) -> dict:
    """Fetch every parameter `model` holds that a file's tensor (or a cut of
    one) gives, and put it in place. -> {fetched, left, left_bytes,
    largest_left, seconds}: what came over the link, and what (by name and
    size) is read from the files."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    p = weight_files.peer()
    t0 = time.monotonic()
    names: dict = {}                     # id(array) -> its parameter names
    found: dict = {}                     # id(array) -> its _origin
    left: list = []                      # (bytes, name): read from the files
    for name, a in tree_flatten(model.parameters()):
        if id(a) not in names:
            o = _origin(a)
            if o is None:
                left.append((int(a.nbytes), name))
            else:
                found[id(a)] = o
        names.setdefault(id(a), []).append(name)
    heads: dict = {}
    keys, items = [], []
    for k, (path, tensor, _, _) in found.items():
        if path not in heads:
            heads[path] = weight_files.header(path)
        off, n, _, _ = heads[path][tensor]
        keys.append(k)
        items.append((path.rsplit("/", 1)[-1], off, n))
    built: dict = {}
    took = [0]

    def build(i: int, view) -> None:
        k = keys[i]
        path, tensor, _, fn = found[k]
        _, _, dt, shape = heads[path][tensor]
        a = weight_files.to_array(view, weight_files.mx_dtype(dt), shape)
        if fn is not None:
            a = fn(a)
            mx.eval(a)
        built[k] = a
        took[0] += len(view)
        if len(built) % 64 == 0:
            progress(fetched_bytes=took[0], reading_from=p.source)
    got = weight_files.ranges(items, p.read, build, stop=stop)
    model.update(tree_unflatten([(n, built[k]) for k in built
                                 for n in names[k]]))
    progress(fetched_bytes=got, reading_from="")
    left.sort(reverse=True)
    return {"fetched": got, "left": len(left),
            "left_bytes": sum(n for n, _ in left), "largest_left": left[:4],
            "seconds": time.monotonic() - t0}


def read_share(model, evaluate: Callable) -> None:
    """The rank's read: wait its turn, fetch its share from the peer,
    evaluate the rest from the files. Without a plan: evaluate(model)."""
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
    p = weight_files.peer()
    if p is not None:
        from knurlogic.engine.runtime.model_host import LOAD_STOP
        try:
            out = fill(model, stop=LOAD_STOP.is_set)
            logger.info("weights: %.1f GiB from %s over the link in %.1fs; "
                        "%d arrays, %.2f GiB, read from the files%s",
                        out["fetched"] / GIB, p.source, out["seconds"],
                        out["left"], out["left_bytes"] / GIB,
                        "".join(f"; {k} {n / GIB:.2f} GiB"
                                for n, k in out["largest_left"]))
        except Exception as e:  # only slower: the files are read instead
            logger.warning("weights: the share was not fetched from %s "
                           "(%s: %s); read from the files", p.source,
                           type(e).__name__, e)
        finally:
            _DERIVED.clear()
            weight_files.forget()
        t0 = time.monotonic()
        evaluate(model)
        logger.info("weights: the rest read from the files in %.1fs",
                    time.monotonic() - t0)
        return
    evaluate(model)


def done() -> None:
    """The rank's load is done, its drafting head and vision tower in too:
    stop reading from the peer, and let the ranks waiting their turn read
    (so their reads never compete with this rank's last fetches)."""
    weight_files.use_peer(None)
    _DERIVED.clear()
    plan, link = CURRENT["plan"], CURRENT["link"]
    if plan is not None and plan.ordered and plan.turn == 0 \
            and link is not None:
        link.align()


def plan_from_env(env) -> tuple:
    """(ReadPlan, peer page or "", source name) from KNURLOGIC_WEIGHTS
    (cluster/weights.rank_env); (None, "", "") without one."""
    raw = env.get("KNURLOGIC_WEIGHTS")
    if not raw:
        return None, "", ""
    try:
        d = json.loads(raw)
    except ValueError:
        return None, "", ""
    return (ReadPlan(turn=1 if d.get("turn") == 1 else 0,
                     ordered=bool(d.get("ordered"))),
            str(d.get("from") or ""), str(d.get("source") or ""))
