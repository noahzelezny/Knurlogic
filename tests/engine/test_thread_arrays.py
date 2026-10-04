"""No array is made lazily where another thread could evaluate it first.

On mlx 0.32.3 a lazy array belongs to the stream of the thread that made
it; a server thread evaluating it first raises "There is no Stream(gpu, 0)
in current thread" -- live, a DeepSeek split's first admission died on
deepseek_v4's module-level _NO_W (an mx.zeros made at import). Arrays made
from Python data (mx.array([...])) hold their values and are safe; so is
anything evaluated before it is kept.

Two guards: a source scan for lazy constructors in the places that outlive
a call (module and class bodies, containers there, functions behind a
cache decorator), and a run that imports every family's modules on the
main thread and evaluates every array they keep from a second thread.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "knurlogic"

#: mx functions that build a lazy array (not from Python data)
LAZY = {"zeros", "ones", "full", "arange", "eye", "linspace", "tri", "tril",
        "triu", "concatenate", "stack", "broadcast_to", "zeros_like",
        "ones_like", "full_like", "identity", "repeat", "tile", "where",
        "exp", "log", "sqrt", "cumsum", "sum", "mean", "matmul", "outer"}
CACHES = {"lru_cache", "cache", "cached_property"}


def _lazy_calls(node) -> list:
    """Line numbers of mx.<lazy>(...) calls inside `node`, not looking into
    nested function bodies (those run per call)."""
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                and n is not node:
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in LAZY \
                and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "mx":
            out.append(n.lineno)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _decorated_cache(fn) -> bool:
    for d in fn.decorator_list:
        name = d.func if isinstance(d, ast.Call) else d
        name = name.attr if isinstance(name, ast.Attribute) else \
            getattr(name, "id", "")
        if name in CACHES:
            return True
    return False


def _kept_bodies(tree):
    """Statements whose values outlive a call: the module body, every class
    body, and the bodies of cached functions."""
    yield from tree.body
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            yield from n.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and _decorated_cache(n):
            yield from n.body


def test_no_lazy_array_is_kept_where_another_thread_can_reach_it():
    found = []
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text())
        for stmt in _kept_bodies(tree):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            for line in _lazy_calls(stmt):
                found.append(f"{f.relative_to(SRC)}:{line}")
    assert not found, found


CODE = r"""
import importlib, pkgutil, sys, threading
import mlx.core as mx
from knurlogic.engine import register, families
from knurlogic.engine.arch import host_for
register.register(override=True)
for name in register.available():
    importlib.import_module(f"{host_for(name)}.models.{name}")
# the families' own packages (vision towers, heads, processors)
import knurlogic.engine.families as F
for m in pkgutil.walk_packages(F.__path__, F.__name__ + "."):
    if ".architecture" in m.name:
        continue                      # vendored: registered above
    try:
        importlib.import_module(m.name)
    except Exception as e:            # an optional dependency: say so
        print("skip", m.name, type(e).__name__)
dirs = [str(p) for p in F.__path__]
mods = [m for m in list(sys.modules.values())
        if any(str(getattr(m, "__file__", "") or "").startswith(d)
               for d in dirs)]

def arrays(v, depth=0):
    if isinstance(v, mx.array):
        yield v
    elif depth < 3 and isinstance(v, (list, tuple, set)):
        for x in v:
            yield from arrays(x, depth + 1)
    elif depth < 3 and isinstance(v, dict):
        for x in v.values():
            yield from arrays(x, depth + 1)

kept = []
for m in mods:
    for k, v in list(vars(m).items()):
        if isinstance(v, type) and v.__module__ == m.__name__:
            for ck, cv in list(vars(v).items()):
                kept += [(f"{m.__name__}.{k}.{ck}", a) for a in arrays(cv)]
        else:
            kept += [(f"{m.__name__}.{k}", a) for a in arrays(v)]
bad = []
def run():
    for name, a in kept:
        try:
            mx.eval(a + 0)
        except Exception as e:
            bad.append(f"{name}: {e}")
t = threading.Thread(target=run); t.start(); t.join()
print(len(mods), "modules", len(kept), "arrays", bad)
raise SystemExit(1 if bad or not mods else 0)
"""


def test_every_kept_array_evaluates_first_in_another_thread():
    pytest.importorskip("mlx.core")
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
