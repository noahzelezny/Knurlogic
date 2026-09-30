"""One rank of tests/test_tensor.py's two-process ring (not a test module).

Builds the tiny qwen3_5_moe text model, splits it with
engine/runtime/tensor.shard, and prefills + decodes a few tokens; rank 0
also runs the unsplit model and writes both logits for the test to
compare. Run with MLX_RANK and MLX_HOSTFILE set."""
import json
import sys

import numpy as np


def build(seed=0, dtype="float32"):
    import importlib

    import fixtures_vision_qwen as FQ
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from knurlogic.engine import register
    register.register("qwen3_5_moe", override=True)
    m = importlib.import_module("mlx_lm.models.qwen3_5_moe")
    cfg = dict(FQ.TEXT["qwen3_5_moe"], model_type="qwen3_5_moe")
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


def main(out_path):
    from knurlogic.engine.runtime import tensor as T
    link = T.init("ring")
    ids = [5, 17, 3, 99, 42, 7, 64, 11, 23]
    then = [31, 104, 331, 32, 439, 214]
    import os

    from knurlogic.engine import kvquant
    bits = kvquant.parse_bits(os.environ.get("KNURLOGIC_KV_BITS"))

    def built():
        m = build()
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
    sys.exit(main(sys.argv[1]))
