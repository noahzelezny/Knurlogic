"""The deepseek_v4 vision goldens, made by DeepSeek's own reference code
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, `inference/`, MIT) under torch.

    REF=".../deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
    $TORCH_PYTHON tests/support/goldens/build_deepseek_v4_vision.py "$REF"

`$TORCH_PYTHON` is an interpreter with torch, PIL, numpy and safetensors
(torch is not a knurlogic dependency). Nothing in the artifact is written;
only the `vision.*` / `aligner.*` tensors are read, one at a time (~0.9
GB). Writes tests/support/goldens/deepseek_v4_vision.npz:

* routing: the reference `Gate` (bias_vl) on small random inputs, a hash
  layer and a score layer;
* mask: `get_image_visible` + `get_window_topk_idxs_visible` on a prompt
  with two images (window 8, max_image_tokens 12);
* processor: `load_image` + `build_image_block` on synthetic images of
  several aspect ratios: grid sizes, types at four start positions, perm,
  and the sha256 of the patches;
* tower: `ViT` + `Aligner` in float32 on the real weights, on the
  artifact's own examples/images/carrots.jpeg resized to 320x400 (stored
  float16: a 99-token image).
"""
import ast
import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "deepseek_v4_vision.npz"

#: (width, height) of the synthetic processor images: square, landscape,
#: portrait, small (min_pixels upscale), large (safe_resize), and wider
#: and taller than vision_max_wh_ratio
SIZES = [(640, 640), (800, 450), (450, 800), (200, 150), (3000, 2000),
         (2000, 120), (60, 1500), (1000, 333)]
TOWER_SIZE = (320, 400)
V = 32          # the routing / mask tests' vocab


def reference(ref: Path) -> dict:
    """The reference's functions, executed from its own source: model.py
    imports its tilelang kernels at the top, so the four definitions are
    cut out of it with ast, not imported."""
    from typing import Optional

    import torch
    import torch.nn.functional as F
    from torch import nn

    src = (ref / "inference" / "model.py").read_text()
    tree = ast.parse(src)
    want = {"get_window_topk_idxs", "get_image_visible",
            "get_window_topk_idxs_visible", "Gate"}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in want]
    sys.path.insert(0, str(ref / "inference"))
    import image_processor
    import vision
    ns = {"torch": torch, "F": F, "nn": nn, "Optional": Optional,
          "Tuple": tuple, "ModelArgs": object,
          "lru_cache": __import__("functools").lru_cache,
          "linear": lambda x, w, b=None: F.linear(x, w, b),
          "IMAGE_START": image_processor.IMAGE_START,
          "IMAGE_END": image_processor.IMAGE_END,
          "IMAGE": image_processor.IMAGE}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "model.py",
                 "exec"), ns)
    ns["image_processor"] = image_processor
    ns["vision"] = vision
    return ns


def routing(ns, rng) -> dict:
    import torch
    out = {}
    D, N, K, T = 16, 8, 2, 12
    ids = rng.integers(0, V, size=T)
    ids[[2, 3, 7, 11]] = V + np.array([0, 2, 2, 4])   # image tokens
    x = rng.standard_normal((T, D)).astype(np.float32)
    out.update(route_x=x, route_ids=ids)
    for name, layer in (("hash", 0), ("score", 1)):
        args = SimpleNamespace(dim=D, n_activated_experts=K,
                               score_func="sqrtsoftplus", route_scale=1.5,
                               n_hash_layers=1, vocab_size=V,
                               vision_n_layers=2, n_routed_experts=N)
        g = ns["Gate"](layer, args)
        w = rng.standard_normal((N, D)).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32)
        bvl = 3 * rng.standard_normal(N).astype(np.float32)
        with torch.no_grad():
            g.weight.copy_(torch.from_numpy(w))
            g.bias.copy_(torch.from_numpy(b))
            g.bias_vl.copy_(torch.from_numpy(bvl))
            if g.hash:
                t2e = rng.integers(0, N, size=(V, K)).astype(np.int32)
                g.tid2eid.copy_(torch.from_numpy(t2e))
                out[f"route_{name}_tid2eid"] = t2e
            wts, idx = g(torch.from_numpy(x), torch.from_numpy(ids))
        out[f"route_{name}_weight"] = w
        out[f"route_{name}_bias"] = b
        out[f"route_{name}_bias_vl"] = bvl
        out[f"route_{name}_weights"] = wts.numpy()
        out[f"route_{name}_indices"] = idx.numpy().astype(np.int64)
    return out


def mask(ns) -> dict:
    """Two images: one longer than the window, one longer than
    max_image_tokens (the clamps), text between and after."""
    import torch
    win, mx_img = 8, 12
    t = [V + 0] + [V + 2] * 10 + [V + 4]          # 12 tokens
    t2 = [V + 0] + [V + 2] * 16 + [V + 4]         # 18 tokens
    ids = list(range(5)) + [V + 1] * 2 + t + [9, 10, 11] + t2 + [3, 4, 5, 6]
    ids = torch.tensor([ids])
    left, right = ns["get_image_visible"](ids, V, mx_img)
    S = ids.shape[1]
    idx = ns["get_window_topk_idxs_visible"](win, S, left, right, mx_img)
    vis = np.zeros((S, S), dtype=bool)
    for q in range(S):
        for j in idx[0, q].tolist():
            if j >= 0:
                vis[q, j] = True
    plain = ns["get_window_topk_idxs"](win, 1, S, 0)
    pvis = np.zeros((S, S), dtype=bool)
    for q in range(S):
        for j in plain[0, q].tolist():
            if j >= 0:
                pvis[q, j] = True
    return {"mask_ids": ids.numpy()[0], "mask_left": left.numpy()[0],
            "mask_right": right.numpy()[0], "mask_visible": vis,
            "mask_plain": pvis, "mask_window": np.array(win),
            "mask_max_image_tokens": np.array(mx_img)}


def _args(ref: Path):
    import json
    cfg = json.loads((ref / "config.json").read_text())
    a = SimpleNamespace(**{k: v for k, v in cfg.items()
                           if k.startswith("vision_")})
    a.dim = cfg["hidden_size"]
    a.vocab_size = cfg["vocab_size"]
    return a


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def synthetic(w: int, h: int, seed: int):
    from PIL import Image
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                     (xx + yy) % 256], axis=-1)
    noise = rng.integers(0, 32, size=(h, w, 3))
    return Image.fromarray(((base + noise) % 256).astype(np.uint8), "RGB")


def processor(ns, args) -> dict:
    ip = ns["image_processor"]
    out = {}
    for i, (w, h) in enumerate(SIZES):
        rec = {"data": _png(synthetic(w, h, i))}
        patches, nvh, nvw, nlh, nlw = ip.load_image(rec, args)
        p = patches.float().reshape(patches.shape[0], -1).numpy()
        out[f"proc{i}_grid"] = np.array([w, h, nvh, nvw, nlh, nlw])
        out[f"proc{i}_patches_sha"] = np.frombuffer(
            hashlib.sha256(np.ascontiguousarray(p).tobytes()).digest(),
            dtype=np.uint8)
        for s in range(4):
            types, perm = ip.build_image_block(nlh, nlw, 101 + s)
            out[f"proc{i}_types{s}"] = types.numpy()
        out[f"proc{i}_perm"] = perm.numpy()
    return out


def tower(ns, ref: Path, args) -> dict:
    import json

    import torch
    from PIL import Image
    from safetensors import safe_open

    vision = ns["vision"]
    ip = ns["image_processor"]
    img = Image.open(ref / "inference" / "examples" / "images" /
                     "carrots.jpeg").convert("RGB").resize(TOWER_SIZE)
    patches, nvh, nvw, nlh, nlw = ip.load_image({"data": _png(img)}, args)
    vit, al = vision.ViT(args).float(), vision.Aligner(args).float()
    wmap = json.loads((ref / "model.safetensors.index.json").read_text())[
        "weight_map"]
    sd_v, sd_a = {}, {}
    for k, shard in wmap.items():
        if not k.startswith(("vision.", "aligner.")):
            continue
        with safe_open(str(ref / shard), framework="pt") as f:
            t = f.get_tensor(k).float()
        (sd_v if k.startswith("vision.") else sd_a)[k.split(".", 1)[1]] = t
    vit.load_state_dict(sd_v, strict=True)
    al.load_state_dict(sd_a, strict=True)
    with torch.no_grad():
        y = al(vit(patches.float(), nvh, nvw), nvh, nvw)
    # bfloat16, as the reference runs (its RMSNorm weights stay float32):
    # some rows move far from float32 -- the yardstick for knurlogic's own
    # bfloat16 tower is how far the reference's moves
    vit, al = vit.to(torch.bfloat16), al.to(torch.bfloat16)
    for m in vit.modules():
        if isinstance(m, vision.RMSNorm):
            m.weight.data = m.weight.data.float()
    with torch.no_grad():
        y16 = al(vit(patches, nvh, nvw), nvh, nvw).float()
    cos = torch.nn.functional.cosine_similarity(y16, y, dim=-1)
    p = patches.float().reshape(patches.shape[0], -1).numpy()
    if len(sys.argv) > 2:
        np.save(sys.argv[2], y.numpy())       # the float32 output, optional
    return {"tower_grid": np.array([nvh, nvw, nlh, nlw]),
            "tower_bf16_cos": cos.numpy(),
            "tower_patches_sha": np.frombuffer(hashlib.sha256(
                np.ascontiguousarray(p).tobytes()).digest(), dtype=np.uint8),
            "tower_out": y.numpy().astype(np.float16)}


def main(ref: str) -> None:
    import torch
    ref = Path(ref)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ns = reference(ref)
    args = _args(ref)
    arrays = {}
    arrays.update(routing(ns, rng))
    arrays.update(mask(ns))
    arrays.update(processor(ns, args))
    arrays.update(tower(ns, ref, args))
    arrays["__meta__"] = np.array(
        f"torch {torch.__version__}; reference {ref.name}/inference "
        f"(model.py Gate/get_image_visible/get_window_topk_idxs_visible, "
        f"vision.py, image_processor.py); tower image carrots.jpeg "
        f"resized to {TOWER_SIZE}")
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
