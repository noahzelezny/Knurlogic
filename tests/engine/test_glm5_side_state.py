"""GLM's attention caches hold fields no forward reads -- the MLA latent
cache's zero-width V (`values`, three arrays quantized) and the mx
`offset`. Left lazy, each grew a graph by a link a decode step, every link
holding a scalar buffer alive, until a 40-minute generation hit Metal's
`Resource limit (499000) exceeded`. The batch generator ends them every
step (mtp/caches.settle): after many steps none is deeper than a few
edges, with and without the draft head, unquantized and at 8 bits."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

STEPS = 120
#: a settled field is a few ops deep; the leak was ~10-20 edges a step
MAX_EDGES = 40


def _edges(arr, path) -> int:
    mx.export_to_dot(str(path), arr)
    return path.read_text().count("->")


@pytest.mark.parametrize("bits", [0, 8])
@pytest.mark.parametrize("mtp", [True, False], ids=["mtp", "plain"])
def test_unread_cache_fields_do_not_chain(tmp_path, bits, mtp):
    import build_glm5_next as G

    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.kvquant import install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.mtp.caches import side_state

    mx.random.seed(0)
    cfg = dict(G.CONFIG, kv_lora_rank=64)
    model = L.LanguageModel(TextConfig.from_dict(json.loads(json.dumps(cfg))))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    if bits:
        install(model, bits)
    head = None
    if mtp:
        from knurlogic.engine.families.glm5.heads.glm5 import MTPHeadGlm5
        head = MTPHeadGlm5(model, L)
        for m in head._modules().values():
            m.set_dtype(mx.float32)
        mx.eval([m.parameters() for m in head._modules().values()])
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=16)
    gen.insert([[(7 * i + 3) % 128 for i in range(40)]], max_tokens=[10**6])
    for _ in range(STEPS):
        gen.next()
    b = gen._batch
    arrays = side_state([b.cache, getattr(b, "hcache", None)])
    assert arrays
    worst = max(_edges(a, tmp_path / "g.dot") for a in arrays)
    assert worst <= MAX_EDGES, worst
