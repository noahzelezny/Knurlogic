"""The deepseek_v4 golden: a tiny random-weight DeepSeek-V4 (4 layers with
compress ratios 4 / 8 / 4 / 0, 4 experts, hc_mult 4, window 8) and its
logits as computed by the FORK the vendored file came from.

    # 1. the tiny artifact (knurlogic's vendored module builds the shapes)
    PYTHONPATH=src python tests/goldens/build_deepseek_v4.py weights
    # 2. the golden, run in an interpreter where the fork (mlx-lm 0.31.9
    #    fork) is installed -- the independent reference
    $FORK_PYTHON tests/goldens/build_deepseek_v4.py \
        golden tests/goldens/deepseek_v4_tiny deepseek_v4_tiny.npz

`logits_of` is what the test runs through knurlogic's load path.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TINY = HERE / "deepseek_v4_tiny"

CONFIG = dict(
    model_type="deepseek_v4", vocab_size=64, hidden_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, head_dim=32,
    qk_rope_head_dim=16, sliding_window=8, compress_ratios=[4, 8, 4, 0],
    index_n_heads=8, index_head_dim=16, index_topk=2,
    moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, hc_mult=4,
    hc_sinkhorn_iters=3,
    rope_scaling={"type": "yarn", "factor": 4,
                  "original_max_position_embeddings": 64,
                  "beta_fast": 32, "beta_slow": 1},
    max_position_embeddings=256, num_nextn_predict_layers=0,
    tie_word_embeddings=False, eos_token_id=1, bos_token_id=0)

#: an 11-token prefill (2 ratio-4 windows + a tail, 1 ratio-8 window, the
#: 8-token window rotated) and 6 decode tokens (two ratio-4 windows and a
#: ratio-8 one complete during decode, so the pools grow on the decode
#: path too)
PROMPT = [3, 17, 42, 5, 9, 60, 33, 2, 11, 48, 27]
DECODE = [7, 55, 21, 36, 4, 12]

#: The golden runs with an index_topk no pool reaches, so the indexer keeps
#: every row: where it must CHOOSE, the vendored file deliberately differs
#: from the fork (PROVENANCE.md: the indexer's query RoPE and its prefill
#: topk), and tests/test_deepseek_v4_arch.py holds that part to the
#: prefill == decode property instead.
GOLDEN_CONFIG = {"index_topk": 64}


def weights(out: Path = TINY) -> None:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    model = M.Model(M.ModelArgs.from_dict(CONFIG))
    w = {}
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
        w[k] = mx.array(a)
    mx.save_safetensors(str(out / "model.safetensors"), w,
                        metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(CONFIG, indent=1))


def logits_of(model, cache=None):
    """[prefill's last position + each decode step] x vocab, float32."""
    import mlx.core as mx
    import numpy as np
    cache = model.make_cache() if cache is None else cache
    rows = [model(mx.array([PROMPT]), cache=cache)[0, -1]]
    for t in DECODE:
        rows.append(model(mx.array([[t]]), cache=cache)[0, -1])
    out = mx.stack(rows).astype(mx.float32)
    mx.eval(out)
    return np.array(out)


def golden(path: str, out: str) -> None:
    import numpy as np
    from mlx_lm.utils import load_model
    import mlx_lm
    model, _ = load_model(Path(path), model_config=dict(GOLDEN_CONFIG))
    np.savez(out, logits=logits_of(model), mlx_lm=mlx_lm.__version__)


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        weights()
    else:
        golden(sys.argv[2], sys.argv[3])
