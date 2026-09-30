"""Build the Qwen vision goldens in the reference interpreter (mlx-vlm 0.6.17):

    $KNURLOGIC_VLM_PYTHON tests/goldens/build_qwen.py

One file per family, qwen_<family>.npz, from mlx-vlm's OWN model classes
(`mlx_vlm.models.<family>.Model`, its tower, its `get_rope_index`, its
numpy image processor) on the tiny config of tests/fixtures_vision_qwen.py,
with every weight set by `fixtures_vision_qwen.init_weights` -- numpy only,
so the test rebuilds the same numbers in knurlogic's trees without shipping
weights (shapes ride in the meta).

  img{1,2}           the two tiny images (uint8), so the test preprocesses
                     exactly these pixels
  pv{1,2}, grid{1,2} Qwen3VLImageProcessor output               (G2)
  feats{1,2}         vision_tower(pv, grid)[0]                  (G1)
  ids                [text, vs, img x n1, ve, text, vs, img x n2, ve, text]
  pos, delta         language_model.get_rope_index(ids, grids)  (G3)
  prefill_logits     last-position logits of the one-shot prefill (G4)
  gen, gen_logits    40 greedy tokens, each step's logits        (G4)
  t2_ids             ids + gen + a text turn: turn 2, no new image
  t2_logits, t2_gen  COLD turn 2 (fresh cache, image re-encoded):
                     last-prefill logits and 12 greedy tokens    (G7b)

Tiny, float32, seed 0; no model files are read.
"""
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import fixtures_vision as fv  # noqa: E402
import fixtures_vision_qwen as fq  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    Qwen3VLImageProcessor)

N_GEN = 40
N_T2 = 12
MID = (21, 5, 77, 140)
T2_TEXT = (11, 300, 42, 8, 19, 260, 4, 99)


def processor():
    p = fq.PREPROCESSOR
    return Qwen3VLImageProcessor(
        patch_size=p["patch_size"], temporal_patch_size=p["temporal_patch_size"],
        merge_size=p["merge_size"], min_pixels=p["size"]["shortest_edge"],
        max_pixels=p["size"]["longest_edge"], image_mean=p["image_mean"],
        image_std=p["image_std"])


def greedy(lm, cache, tok, n):
    toks, logits = [], []
    for _ in range(n):
        lg = lm(mx.array([[tok]]), cache=cache).logits[0, -1]
        mx.eval(lg)
        logits.append(np.array(lg))
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    return toks, logits


def build(fam):
    cfg = fq.config(fam)
    m = importlib.import_module(f"mlx_vlm.models.{fam}")
    mx.random.seed(0)
    model = m.Model(m.ModelConfig.from_dict(cfg))
    model.set_dtype(mx.float32)
    params = dict(tree_flatten(model.parameters()))
    shapes = {k: tuple(v.shape) for k, v in params.items()
              if v.dtype == mx.float32}
    w = fq.init_weights(shapes)
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in w.items()]))
    mx.eval(model.parameters())

    t = fq.ids(fam)
    proc = processor()
    out = {}
    imgs = [fv.tiny_image(32, 24, seed=0), fv.tiny_image(40, 20, seed=1)]
    ns = []
    for i, img in enumerate(imgs, 1):
        r = proc([img])
        pv, grid = r["pixel_values"], r["image_grid_thw"]
        feats, _ = model.vision_tower(mx.array(pv), mx.array(grid))
        mx.eval(feats)
        out[f"img{i}"] = np.asarray(img)
        out[f"pv{i}"], out[f"grid{i}"] = pv, grid
        out[f"feats{i}"] = np.array(feats)
        ns.append(int(np.prod(grid[0])) // (fq.PREPROCESSOR["merge_size"] ** 2))

    vs, ve, im = (t["vision_start_token_id"], t["vision_end_token_id"],
                  t["image_token_id"])
    ids = ([3, 17, 45, 9, 101, vs] + [im] * ns[0] + [ve] + list(MID)
           + [vs] + [im] * ns[1] + [ve] + [7, 88, 12, 250, 33, 64])
    grids = np.concatenate([out["grid1"], out["grid2"]])
    pv = np.concatenate([out["pv1"], out["pv2"]])
    pos, delta = model.language_model.get_rope_index(
        mx.array([ids]), mx.array(grids))
    out["ids"], out["pos"], out["delta"] = (np.array(ids), np.array(pos),
                                            np.array(delta))

    lm = model.language_model
    cache = lm.make_cache()
    lg = model(mx.array([ids]), pixel_values=mx.array(pv),
               image_grid_thw=mx.array(grids), cache=cache).logits[0, -1]
    mx.eval(lg)
    out["prefill_logits"] = np.array(lg)
    first = int(mx.argmax(lg).item())
    toks, logits = greedy(lm, cache, first, N_GEN - 1)
    out["gen"] = np.array([first] + toks)
    out["gen_logits"] = np.stack([np.array(lg)] + logits)

    t2 = ids + [first] + toks + list(T2_TEXT)
    cache = lm.make_cache()
    lg = model(mx.array([t2]), pixel_values=mx.array(pv),
               image_grid_thw=mx.array(grids), cache=cache).logits[0, -1]
    mx.eval(lg)
    out["t2_ids"] = np.array(t2)
    out["t2_logits"] = np.array(lg)
    first = int(mx.argmax(lg).item())
    toks, _ = greedy(lm, cache, first, N_T2 - 1)
    out["t2_gen"] = np.array([first] + toks)

    fv.save_golden(f"qwen_{fam}", out, {
        "what": f"mlx_vlm.models.{fam}.Model on the tiny config",
        "seed": 0, "config": cfg, "shapes": shapes,
        "preprocessor": fq.PREPROCESSOR})


def main():
    fams = sys.argv[1:] or list(fq.FAMILIES)
    for fam in fams:
        build(fam)
        print("built", fam)


if __name__ == "__main__":
    main()
