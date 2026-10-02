"""One rank of tests/test_tensor.py's two-process ring (not a test module).

Builds the tiny text model of a family (qwen3_5_moe unless named: the
second argument), splits it with
engine/runtime/tensor.shard, and prefills + decodes a few tokens; rank 0
also runs the unsplit model and writes both logits for the test to
compare. Run with MLX_RANK and MLX_HOSTFILE set."""
import json
import sys

import numpy as np


def build_deepseek_v4(seed=0):
    """The tiny DeepSeek-V4 of tests/support/goldens (compress ratios
    4 / 8 / 4 / 0, a hash layer, hc_mult 4, 4 heads in 2 o_groups), its
    routed experts 64 wide: they are mxfp4 in groups of 32, so a rank's
    half of down_proj's input is one whole group. Random weights drawn as
    the golden's are; float32 elsewhere."""
    import mlx.core as mx
    from goldens.build_deepseek_v4 import CONFIG
    from mlx.utils import tree_flatten, tree_unflatten

    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    model = M.Model(M.ModelArgs.from_dict(
        dict(CONFIG, moe_intermediate_size=64)))
    rng = np.random.default_rng(seed)
    w = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("tid2eid"):
            a = rng.integers(0, CONFIG["n_routed_experts"],
                             size=v.shape).astype(np.int32)
        elif "switch_mlp" in k and v.dtype == mx.uint8:   # E8M0 scales
            a = rng.integers(118, 124, size=v.shape).astype(np.uint8)
        elif "switch_mlp" in k:                           # packed mxfp4
            a = rng.integers(0, 2 ** 32, size=v.shape,
                             dtype=np.uint64).astype(np.uint32)
        elif k.endswith("norm.weight"):
            a = (1 + 0.1 * rng.standard_normal(v.shape)).astype(np.float32)
        else:
            a = (0.15 * rng.standard_normal(v.shape)).astype(np.float32)
        w.append((k, mx.array(a)))
    model.update(tree_unflatten(w))
    mx.eval(model.parameters())
    return model


def build(family="qwen3_5_moe", seed=0, dtype="float32"):
    import importlib
    if family == "deepseek_v4":
        return build_deepseek_v4(seed)

    import fixtures_vision_qwen as FQ
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from knurlogic.engine import register
    register.register(family, override=True)
    m = importlib.import_module(f"mlx_lm.models.{family}")
    # qwen4_exp's defaults are full size; its tiny config is the fixture's
    cfg = FQ.config(family) if family == "qwen4_exp" else \
        dict(FQ.TEXT[family], model_type=family)
    model = m.Model(m.ModelArgs.from_dict(cfg))
    shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
    w = FQ.init_weights(shapes, seed)
    model.update(tree_unflatten([(k, mx.array(v).astype(getattr(mx, dtype)))
                                 for k, v in w.items()]))
    mx.eval(model.parameters())
    return model


def run(model, ids, then):
    """Prefill `ids`, then decode `then` one token at a time (fixed tokens:
    both models see the same inputs whatever their argmax)."""
    import mlx.core as mx
    cache = model.make_cache()
    out = [model(mx.array([ids]), cache=cache)[:, -1]]
    for t in then:
        out.append(model(mx.array([[t]]), cache=cache)[:, -1])
    y = mx.concatenate(out).astype(mx.float32)
    mx.eval(y)
    return np.array(y)


def main(out_path, family="qwen3_5_moe"):
    from knurlogic.engine.runtime import tensor as T
    link = T.init("ring")
    ids = [5, 17, 3, 99, 42, 7, 64, 11, 23]
    then = [31, 104, 331, 32, 439, 214]
    if family == "deepseek_v4":                 # its vocabulary is 64
        ids, then = [t % 64 for t in ids], [t % 64 for t in then]
    import os

    from knurlogic.engine import kvquant
    bits = kvquant.parse_bits(os.environ.get("KNURLOGIC_KV_BITS"))

    def built():
        m = build(family)
        if bits:
            assert kvquant.install(m, bits) > 0
        return m
    whole = run(built(), ids, then) if link.rank == 0 else None
    model = built()
    T.shard(model, link.group)
    split = run(model, ids, then)
    link.barrier()
    # idle: rank 0 parks rank 1 on the bell, then the next exchange wakes it
    import time

    from knurlogic.engine.runtime import plan as P
    if link.rank == 0:
        T.Ring(link).park()
        time.sleep(1.0)
        link.exchange(0, None)
    else:
        _, data = link.exchange(0, None)
        assert [o["op"] for o in P.decode(data)["ops"]] == ["park"]
        t0 = time.process_time()
        link.sleep()
        slept_cpu = time.process_time() - t0
        link.exchange(0, None)
        assert slept_cpu < 0.2, f"parked rank used {slept_cpu:.2f}s CPU"
    link.barrier()
    if link.rank == 0:
        json.dump({"whole": whole.tolist(), "split": split.tolist()},
                  open(out_path, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
