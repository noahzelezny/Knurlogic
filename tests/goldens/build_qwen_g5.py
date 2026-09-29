"""Build the G5 golden: the Qwen trunks' TEXT path as it was BEFORE P1
threaded MRoPE through them -- run once, at the parent of P1's first trunk
edit, in the TEST interpreter (knurlogic's own architectures, no mlx-vlm):

    python3 tests/goldens/build_qwen_g5.py

qwen_g5_text.npz, per family: a fingerprint of the seed-0 weights, a 30-token text
prompt, the logits of a two-chunk prefill (17 + 13, so the second chunk
starts at a non-zero cache offset) and of 12 greedy decode steps.

WHY A SNAPSHOT AND NOT "position_ids=None is the old code". That gate
is a tautology when stated about the source; a
snapshot of main's numbers is not -- any edit that moves the text path by one
ulp turns it red. The comparison is exact (same MLX, pinned, same machine
class); tests/test_vision_qwen.py says what to do if hardware changes that.

Tiny, float32, seed 0; no model files are read.
"""
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
import fixtures_vision as fv  # noqa: E402
import fixtures_vision_qwen as fq  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from knurlogic.engine import register  # noqa: E402

PROMPT = [(7 * i + 3) % 500 for i in range(30)]
SPLIT = 17
STEPS = 12


def fingerprint(model):
    return np.array([float(mx.abs(v).sum().item())
                     for _, v in tree_flatten(model.parameters())])


def run(model, prompt):
    cache = model.make_cache()
    out = []
    a = model(mx.array(prompt[:SPLIT])[None], cache=cache)
    b = model(mx.array(prompt[SPLIT:])[None], cache=cache)
    out.append(b[0, -1])
    tok = int(mx.argmax(b[0, -1]).item())
    toks = [tok]
    for _ in range(STEPS):
        lg = model(mx.array([[tok]]), cache=cache)[0, -1]
        out.append(lg)
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    mx.eval(a)
    return np.stack([np.array(x) for x in out]), np.array(toks)


def main():
    register.register(*fq.FAMILIES, override=True)
    arrays, meta = {}, {"what": "Qwen trunk text path before P1", "seed": 0,
                        "prompt": PROMPT, "split": SPLIT, "steps": STEPS}
    for fam in fq.FAMILIES:
        arch = importlib.import_module(f"mlx_lm.models.{fam}")
        cfg = fq.config(fam)
        mx.random.seed(0)
        model = arch.Model(arch.ModelArgs.from_dict(cfg))
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
        # Weights are re-made from seed 0 by the test (storing them was
        # 20 MB); this fingerprint tells "the init moved" apart from "the
        # text path moved" when the gate goes red.
        arrays[f"{fam}/w_fingerprint"] = fingerprint(model)
        logits, toks = run(model, PROMPT)
        arrays[f"{fam}/logits"] = logits
        arrays[f"{fam}/tokens"] = toks
        meta[f"{fam}/config"] = cfg
    fv.save_golden("qwen_g5_text", arrays, meta)


if __name__ == "__main__":
    main()
