"""Where GLM-5.3's prefill in the latent pays: one attention layer, both
paths, at several contexts (glm5_next edit 10, KNURLOGIC_SPARSE_PREFILL_FROM).

    python tools/bench_glm5_sparse_prefill.py <artifact> \\
        [--contexts 8192,32768,131072] [--chunk 2048] [--repeat 3]

Builds ONE sparse-attention layer at the artifact's own shapes (its
config.json; random bf16 weights -- the cost is the shapes', not the
values'), fills its cache to each context, then times one prefill chunk
through the expanded path (off) and the latent path (0) and reads mlx's
peak memory for each. Prints a row per context and the crossover: the
smallest context from which the latent path is the faster, which is what
KNURLOGIC_SPARSE_PREFILL_FROM should be set to on this machine (the latent
path's memory is the smaller everywhere past index_topk). A path that
runs out of memory is reported as such, not as a crash. Loads no model
weights; never run from tests/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENV = "KNURLOGIC_SPARSE_PREFILL_FROM"
GIB = 1 << 30


def _text_config(artifact: Path) -> dict:
    cfg = json.loads((artifact / "config.json").read_text())
    return cfg.get("text_config") or cfg


def _layer(cfg: dict):
    import mlx.core as mx

    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    tc = TextConfig.from_dict(cfg)
    layer = L.Glm5NextSparseAttention(tc)
    layer.set_dtype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer, tc


def _cache():
    from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.cache import (  # noqa: E501
        CacheList,
        KVCache,
    )
    return CacheList(KVCache(), KVCache())


def _mask(x, cache):
    from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models import (  # noqa: E501
        base,
    )
    return base.create_attention_mask(x, cache[0], return_array=True)


def _fill(layer, tc, cache, tokens: int, step: int = 4096) -> None:
    """The cache to `tokens` of context, through the latent path (the
    cache it leaves is the same either way)."""
    import mlx.core as mx
    os.environ[ENV] = "0"
    done = 0
    while done < tokens:
        n = min(step, tokens - done)
        x = mx.random.normal((1, n, tc.hidden_size)).astype(mx.bfloat16)
        mx.eval(layer(x, mask=_mask(x, cache), cache=cache))
        done += n
    mx.clear_cache()


def _time(layer, tc, cache, chunk: int, setting: str, repeat: int):
    """(seconds, peak GiB) for one chunk at `setting`, or (None, why)."""
    import mlx.core as mx
    os.environ[ENV] = setting
    x = mx.random.normal((1, chunk, tc.hidden_size)).astype(mx.bfloat16)
    best, peak = None, 0
    for _ in range(repeat):
        state = [c.state for c in cache.caches]
        offs = [c.offset for c in cache.caches]
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        try:
            mx.eval(layer(x, mask=_mask(x, cache), cache=cache))
        except RuntimeError as e:   # Metal out of memory: report it
            return None, str(e).splitlines()[0][:80]
        dt = time.perf_counter() - t0
        peak = max(peak, mx.get_peak_memory())
        best = dt if best is None else min(best, dt)
        # the chunk again from the same context next time
        for c, s, o in zip(cache.caches, state, offs):
            c.state = s
            c.offset = o
        cache.caches[1]._pool = None
    return best, peak / GIB


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("artifact", type=Path)
    p.add_argument("--contexts", default="8192,32768,131072")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    cfg = _text_config(a.artifact)
    layer, tc = _layer(cfg)
    contexts = sorted(int(c) for c in a.contexts.split(","))
    print(f"{a.artifact.name}: one sparse-attention layer, heads "
          f"{tc.num_attention_heads}, kv_lora_rank {tc.kv_lora_rank}, "
          f"index_topk {tc.index_topk}, chunk {a.chunk}")
    print(f"{'context':>9}  {'expanded s':>10} {'GiB':>6}  "
          f"{'latent s':>9} {'GiB':>6}  faster")
    cache, have, crossover = _cache(), 0, None
    for ctx in contexts:
        _fill(layer, tc, cache, ctx - have)
        have = ctx
        e_s, e_m = _time(layer, tc, cache, a.chunk, "off", a.repeat)
        l_s, l_m = _time(layer, tc, cache, a.chunk, "0", a.repeat)
        win = ("latent" if e_s is None or (l_s is not None and l_s < e_s)
               else "expanded")
        if win == "latent" and crossover is None:
            crossover = ctx

        def cell(s, m):
            return (f"{'OOM':>10} {'':>6}" if s is None
                    else f"{s:>10.3f} {m:>6.2f}")
        print(f"{ctx:>9}  {cell(e_s, e_m)}  {cell(l_s, l_m)}  {win}")
    print(f"\n{ENV}: " + (f"{crossover} (the latent path is the faster from "
                           f"there on this machine)" if crossover is not None
                           else "the expanded path was the faster at every "
                                "context measured; set the largest above, or "
                                "measure longer ones"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
