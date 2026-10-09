"""Does the tensor split hold for a module whose layout no split rule knows
(tensor_rules.unverified)? Run it and see, before the ring starts.

One instance per distinct unknown layout, built from the real weights by
the artifact's own loader (lazy: only that module is read), is cut n ways
by the same `split_params` and predicate the ranks use, and a small random
input goes through the whole module and through the n parts reduced as the
ranks reduce them (all-to-sharded: concatenated outputs; sharded-to-all:
the input cut n ways and the partial outputs summed in float32). The two
must agree within 1% of the output's largest value -- the tolerance the
two-rank split test holds the whole model to.

A part whose geometry differs from what the rule promises (an expert axis
cut, the wrong output or input width) fails before any kernel runs: a
kernel indexing experts a part no longer holds reads past its arrays.
"""

from __future__ import annotations

import copy
import logging
import time

logger = logging.getLogger(__name__)

#: tokens and routed experts per token of the probe input
TOKENS, TOP_K = 8, 4
#: |split - whole| must stay under this fraction of max|whole|
TOLERANCE = 0.01

#: (path, n, the layouts) -> refusals, for this process
_SEEN: dict = {}


def _module(layer, path: str):
    m = layer
    for p in path.split("."):
        m = getattr(m, p, None)
        if m is None:
            return None
    return m


def _geometry(m) -> dict:
    return {k: int(getattr(m, k)) for k in
            ("num_experts", "output_dims", "input_dims")
            if isinstance(getattr(m, k, None), int)}


def check_module(m, rule, n: int, key_dim: int = 0, seed: int = 0):
    """None when `m` cut n ways by `rule` computes what it computes whole,
    else why not with the numbers."""
    import mlx.core as mx

    from knurlogic.engine.split.tensor import split_params

    from .tensor_rules import A2S, predicate, segment_points

    g = _geometry(m)
    if "input_dims" not in g:
        return "it says no input_dims, so it cannot be run"
    pred = predicate(rule.kind)
    params = m.parameters()
    parts = []
    for r in range(n):
        try:
            cut = split_params(params, pred, r, n,
                               segment_points(rule, key_dim))
        except ValueError as e:
            return f"its arrays cannot be cut {n} ways ({e})"
        p = copy.copy(m)
        p.update(cut)
        parts.append(p)
    want = dict(g)
    if rule.kind == A2S and "output_dims" in g:
        want["output_dims"] = g["output_dims"] // n
    elif "input_dims" in g:
        want["input_dims"] = g["input_dims"] // n
    got = _geometry(parts[0])
    if got != want:
        bad = ", ".join(f"{k} {g[k]} -> {got.get(k)} (want {want[k]})"
                        for k in want if got.get(k) != want[k])
        return f"a part's geometry is not the rule's: {bad}"

    IN = g["input_dims"]
    k = mx.random.key(seed)
    kx, ki = mx.random.split(k)
    switch = "num_experts" in g
    if switch:
        x = mx.random.normal((TOKENS, 1, 1, IN), key=kx).astype(mx.float16)
        idx = mx.random.randint(0, g["num_experts"], (TOKENS, TOP_K),
                                key=ki).astype(mx.uint32)
        run = lambda mod, xx: mod(xx, idx)              # noqa: E731
    else:
        x = mx.random.normal((TOKENS, IN), key=kx).astype(mx.float16)
        run = lambda mod, xx: mod(xx)                   # noqa: E731
    whole = run(m, x).astype(mx.float32)
    if rule.kind == A2S:
        split = mx.concatenate([run(p, x).astype(mx.float32)
                                for p in parts], axis=-1)
    else:
        xs = mx.split(x, n, axis=-1)
        split = sum(run(p, xs[r]).astype(mx.float32)
                    for r, p in enumerate(parts))
    if split.shape != whole.shape:
        return f"split output {split.shape}, whole {whole.shape}"
    err = float(mx.abs(split - whole).max())
    top = float(mx.abs(whole).max())
    if not err <= TOLERANCE * top:
        return (f"split output differs from whole by {err:.4g} "
                f"(max |whole| {top:.4g}; tolerance {TOLERANCE:.0%})")
    return None


def refusals(path, n: int, unverified: dict, cfg: dict,
             executes_artifact_code: bool = False) -> list:
    """`unverified` (tuning/tensor_split.tensor_unverified) run as above on the
    artifact at `path`; one line per module that fails. Loads the model
    lazily on this machine: only the probed modules' weights are read."""
    key = (str(path), n, tuple(sorted(unverified.items())))
    if key in _SEEN:
        return _SEEN[key]
    from knurlogic.engine.serve.load import load_unlocked

    from .tensor_rules import RULES
    t0 = time.perf_counter()
    tc = cfg.get("text_config", cfg)
    kd = (tc.get("linear_num_key_heads") or 0) * \
        (tc.get("linear_key_head_dim") or 0)
    model, _ = load_unlocked(str(path), executes_artifact_code, lazy=True)
    out = []
    for (rpath, leaves), layer in sorted(unverified.items()):
        where = f"layers.{layer}.{rpath} ({', '.join(leaves)})"
        m = _module(model.layers[layer], rpath)
        if m is None:
            out.append(f"{where}: not in the built model, so the split "
                       f"cannot be checked")
            continue
        try:
            why = check_module(m, RULES[rpath], n, kd)
        except Exception as e:  # a part the module cannot run is a refusal
            why = f"a part does not run: {type(e).__name__}: {e}"
        if why:
            out.append(f"{where}: the tensor split does not hold -- {why}")
    logger.info("tensor viability: %d module(s) of %s run whole and split "
                "%d ways in %.1fs: %s", len(unverified), path, n,
                time.perf_counter() - t0, out or "all hold")
    _SEEN[key] = out
    return out
