"""A split rank's read of its share: from a peer's page instead of a network
mount, and in turn (docs/design/weight-streaming.md).

The rank's weight files are opened by engine/model/weight_files.py, which
records what each lazy array is (file, tensor) while this rank has a peer.
After the model is cut to the rank's share, `read_share` waits for the
rank's turn, fetches the bytes of every parameter the model still holds
(`model.parameters()`; a tensor split's cut is redone on the fetched bytes,
`derived`), makes again what the model's sanitize made from the files
(weight_files.remake: its inputs fetched, the same sanitize run on them),
puts them in place with `model.update`, and evaluates what is left from
the files. `done` ends the rank's turn once its drafting head and
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
from knurlogic.machine.units import GIB, gib

from .marker import LOAD_STOP, progress

logger = logging.getLogger(__name__)



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


def _source(a):
    """(source, fn) with a == fn(source) (fn None: a itself), the source
    an array weight_files can give again -- a file's tensor or one the
    model's sanitize made -- or None (read from the files)."""
    fns = []
    for _ in range(8):
        if weight_files.origin(a) is not None or \
                weight_files.made(a) is not None:
            return a, (_chain(fns[::-1]) if fns else None)
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
    one) gives, make again those its sanitize made, and put them in place.
    -> {fetched, made, left, left_bytes, largest_left, seconds}: what came
    over the link, how many were made again, and what (by name and size) is
    read from the files."""
    from mlx.utils import tree_flatten, tree_unflatten
    t0 = time.monotonic()
    names: dict = {}                     # id(array) -> its parameter names
    read: dict = {}                      # id(array) -> (file tensor, fn)
    make: dict = {}                      # id(array) -> (sanitize's, fn)
    left: list = []                      # (bytes, name): read from the files
    for name, a in tree_flatten(model.parameters()):
        if id(a) not in names:
            s = _source(a)
            if s is None:
                left.append((int(a.nbytes), name))
            elif weight_files.origin(s[0]) is not None:
                read[id(a)] = s
            else:
                make[id(a)] = s
        names.setdefault(id(a), []).append(name)
    built: dict = {}
    got = _read(read, built, stop)
    if make:
        got += _make(make, built, stop)
    left += [(int(src.nbytes), names[k][0]) for k, (src, _) in make.items()
             if k not in built]
    model.update(tree_unflatten([(n, built[k]) for k in built
                                 for n in names[k]]))
    progress(fetched_bytes=got, reading_from="")
    left.sort(reverse=True)
    return {"fetched": got, "made": sum(k in built for k in make),
            "left": len(left), "left_bytes": sum(n for n, _ in left),
            "largest_left": left[:4], "seconds": time.monotonic() - t0}


def _read(read: dict, built: dict, stop) -> int:
    """Fetch the file tensors of `read` into `built` (the cut redone).
    -> bytes."""
    import mlx.core as mx
    p = weight_files.peer()
    took = [0]

    def put(k, a) -> None:
        took[0] += a.nbytes
        fn = read[k][1]
        if fn is not None:
            a = fn(a)
            mx.eval(a)
        built[k] = a
        if len(built) % 64 == 0:
            progress(fetched_bytes=took[0], reading_from=p.source)
    _, got = weight_files.fetch(
        {k: weight_files.origin(src) for k, (src, _) in read.items()}, p,
        each=put, stop=stop)
    return got


def _make(make: dict, built: dict, stop) -> int:
    """Make again what the model's sanitize made (weight_files.remake) into
    `built`, the cut redone; one whose name, shape or dtype does not match
    stays out (read from the files). -> bytes fetched."""
    import mlx.core as mx
    try:
        out, got = weight_files.remake(stop=stop)
    except InterruptedError:
        raise
    except Exception as e:  # only slower: the files are read instead
        logger.warning("weights: what the model's sanitize made was not "
                       "made again (%s: %s); read from the files",
                       type(e).__name__, e)
        return 0
    new = []
    for k, (src, fn) in make.items():
        a = out.get(weight_files.made(src))
        if a is None or a.shape != src.shape or a.dtype != src.dtype:
            continue
        built[k] = a if fn is None else fn(a)
        new.append(built[k])
    mx.eval(new)
    return got


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
        try:
            out = fill(model, stop=LOAD_STOP.is_set)
            logger.info("weights: %.1f GiB from %s over the link in %.1fs "
                        "(%d arrays made by the model's sanitize from it); "
                        "%d arrays, %.2f GiB, read from the files%s",
                        out["fetched"] / GIB, p.source, out["seconds"],
                        out["made"], out["left"], out["left_bytes"] / GIB,
                        "".join(f"; {k} {gib(n, 2)}"
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
