"""Build the GLM-5.3-Flash (glm5_next) bf16 reference golden: the tiny model
of build_glm5_next.py run through HF transformers' own Glm5NextTextModel in
bfloat16, as from_pretrained(dtype=bfloat16) would hold it.

    <venv with torch + transformers 5.16.1>/bin/python \\
        tests/support/goldens/build_glm5_next_bf16.py

What "as from_pretrained holds it" means here: every parameter bf16 except
the reference's _keep_in_fp32_modules_strict (e_score_correction_bias,
conv1d, dt_bias, A_log), which stay float32. Every weight is drawn
bf16-representable (the float32 arrays saved are exact bf16 values), so the
storage dtype of a weight cannot be the difference between the two sides.
Experts run through `grouped_mm` (transformers' default experts
implementation; the eager loop is the debug path), which sums the top-k
weighted expert outputs in float32 and casts once, as the shipped engines do.

glm5_next_bf16.npz holds the config, the weights, the prompt and decode
tokens, the bf16 reference's prefill and decode logits, the SAME weights'
float32 reference logits (so the test can say how far bf16 alone moves the
logits, which is what its tolerance is judged against), and `meta`: the
torch / transformers / numpy versions, the experts implementation and the
date it was built.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "glm5_next_bf16.npz"

import build_glm5_next as G  # noqa: E402

KEEP_FP32 = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")


def _run(model, lm_head):
    import torch
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        h = model(input_ids=torch.tensor([G.PROMPT]), past_key_values=cache,
                  use_cache=True)
        out = [lm_head(h.last_hidden_state.to(lm_head.weight.dtype))[0]]
        for t in G.DECODE:
            h = model(input_ids=torch.tensor([[t]]), past_key_values=cache,
                      use_cache=True)
            out.append(lm_head(h.last_hidden_state.to(lm_head.weight.dtype))
                       [0, -1:])
    return (out[0].float().numpy(),
            np.stack([o[-1].float().numpy() for o in out[1:]]))


def main():
    import datetime

    import torch
    import transformers
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    assert Glm5NextTextModel._keep_in_fp32_modules_strict == list(KEEP_FP32) \
        or set(Glm5NextTextModel._keep_in_fp32_modules_strict or []) <= \
        set(KEEP_FP32), Glm5NextTextModel._keep_in_fp32_modules_strict

    torch.manual_seed(0)
    cfg = Glm5NextTextConfig(**{k: v for k, v in G.CONFIG.items()
                               if k != "model_type"})
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "grouped_mm"
    model = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    # build_glm5_next's draw, rounded to bf16 instead of float16
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [
                ("lm_head.weight", lm_head.weight)]:
            r = torch.randn(p.shape, generator=g)
            val = G.draw(name, p, r)
            p.copy_(val.to(torch.bfloat16).float())
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd["lm_head.weight"] = lm_head.weight.detach().clone()

    pre32, dec32 = _run(model, lm_head)

    model = model.to(torch.bfloat16)
    lm_head = lm_head.to(torch.bfloat16)
    for name, p in model.named_parameters():
        if any(k in name for k in KEEP_FP32):
            p.data = p.data.float()
    for name, b in model.named_buffers():
        if any(k in name for k in KEEP_FP32):
            b.data = b.data.float()
    pre16, dec16 = _run(model, lm_head)

    w = G.checkpoint_names(sd, G.CONFIG)
    arrays = {"w/" + k: v.float().numpy() for k, v in w.items()}
    meta = {"torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "experts_implementation": cfg._experts_implementation,
            "attn_implementation": cfg._attn_implementation,
            "keep_in_fp32": list(KEEP_FP32),
            "built": datetime.date.today().isoformat(),
            "device": "cpu"}
    np.savez_compressed(
        OUT, config=np.array(json.dumps(G.CONFIG)),
        meta=np.array(json.dumps(meta)),
        prompt=np.array(G.PROMPT, np.int32),
        decode=np.array(G.DECODE, np.int32),
        prefill_logits=pre16.astype(np.float32),
        decode_logits=dec16.astype(np.float32),
        prefill_logits_fp32=pre32.astype(np.float32),
        decode_logits_fp32=dec32.astype(np.float32),
        **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
