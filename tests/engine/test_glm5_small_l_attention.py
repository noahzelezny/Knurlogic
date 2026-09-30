"""GLM-5-Next sparse attention: the small-L (MTP verify) path.

A 2-4 token verify forward takes the decode path -- absorbed MLA over the
latent, gathering each row's own top-k -- and the indexer pools
incrementally, instead of expanding the whole latent cache into per-head
K/V and attending densely under a scatter mask. The two must agree: here the
new path is checked against the dense one (SMALL_L=1 routes L>1 back to it) on
a small random GLM-shaped layer, across context lengths, with causal masking
among the L new tokens, and across a speculative rollback.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")


def _layer():
    from knurlogic.engine.families.glm5.architecture.glm5_next import language
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    cfg = TextConfig(
        model_type="glm5_next", vocab_size=64, hidden_size=64,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=None,
        n_routed_experts=None, routed_scaling_factor=1.0, kv_lora_rank=32,
        q_lora_rank=24, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
        num_experts_per_tok=1, first_k_dense_replace=0,
        max_position_embeddings=4096, rms_norm_eps=1e-6, index_topk=16,
        index_head_dim=16, index_n_heads=2, layer_types=["full_attention"],
        mlp_layer_types=["dense"], linear_attn_config={},
    )
    mx.random.seed(0)
    attn = language.Glm5NextSparseAttention(cfg)
    ix = attn.indexer
    ix.index_kpool_compress_ape = mx.random.normal(ix.index_kpool_compress_ape.shape)
    ix.index_kpool_compress_gate = 0.1 * mx.random.normal(
        ix.index_kpool_compress_gate.shape)
    mx.eval(attn.parameters())
    return language, attn


def _cache(language):
    return language.CacheList(language.KVCache(), language.KVCache())


def _mask(language, x, cache):
    # As Glm5NextModel builds it: a bool array from the main cache's offset.
    return language.create_attention_mask(x, cache[0], return_array=True)


def _run(language, attn, small_l, prefix, steps):
    """Prefill `prefix`, then feed each (tokens, rollback) step; return outputs."""
    old = language.SMALL_L
    language.SMALL_L = small_l
    try:
        cache = _cache(language)
        mx.eval(attn(prefix, _mask(language, prefix, cache), cache))
        outs = []
        for x, rollback in steps:
            mask = _mask(language, x, cache)
            y = attn(x, mask, cache)
            mx.eval(y)
            outs.append(y)
            if rollback:
                for c in cache.caches:
                    c.trim(rollback)
        return outs
    finally:
        language.SMALL_L = old


@pytest.mark.parametrize("ctx", [8, 40, 101])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_small_l_matches_the_dense_path(ctx, width):
    language, attn = _layer()
    mx.random.seed(ctx * 10 + width)
    prefix = mx.random.normal((1, ctx, 64))
    steps = [
        (mx.random.normal((1, width, 64)), width - 1),  # verify, reject all drafts
        (mx.random.normal((1, 1, 64)), 0),              # plain decode after rollback
        (mx.random.normal((1, width, 64)), 1),          # verify, reject the last
        (mx.random.normal((1, width, 64)), 0),          # verify, accept all
    ]
    new = _run(language, attn, 4, prefix, steps)
    ref = _run(language, attn, 1, prefix, steps)
    for a, b in zip(new, ref):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4).item(), \
            float(mx.max(mx.abs(a - b)))


def test_rows_see_only_their_own_selection():
    """Causal within the new tokens: an earlier row must not see a later
    token even though that token sits in a later row's block."""
    language, attn = _layer()
    mx.random.seed(7)
    prefix = mx.random.normal((1, 60, 64))
    x = mx.random.normal((1, 4, 64))
    full = _run(language, attn, 4, prefix, [(x, 0)])[0]
    # Row 0 of a 4-wide verify equals a single-token decode of x[:, :1].
    one = _run(language, attn, 4, prefix, [(x[:, :1], 0)])[0]
    assert mx.allclose(full[:, :1], one, atol=1e-4, rtol=1e-4).item()


def test_gather_honours_a_padding_mask():
    language, _ = _layer()
    kv = mx.random.normal((2, 1, 10, 8))
    topk = mx.array([[[0, 1, -1], [2, 3, 4]], [[0, 5, 9], [9, -1, -1]]])
    pad = mx.array([[True] * 10, [False] + [True] * 9])[:, None, None, :]
    g, m = language._gather_selected(kv, topk, pad, 2, 2, 10)
    assert g.shape == (2, 1, 6, 8) and m.shape == (2, 1, 2, 6)
    assert m[0, 0, 0].tolist() == [True, True, False, False, False, False]
    assert m[0, 0, 1].tolist() == [False, False, False, True, True, True]
    assert m[1, 0, 0].tolist() == [False, True, True, False, False, False]
    assert m[1, 0, 1].tolist() == [False, False, False, True, False, False]
