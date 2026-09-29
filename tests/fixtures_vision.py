"""The SHARED vision test builder. Per-family fixtures live beside it, one
file per family (tests/fixtures_vision_<family>.py).

Three things here:

1. `tiny_config(family, ...)` -- a real config shrunk to a tiny random model:
   STRUCTURAL keys kept (patch 14 vs 16, merge, temporal patch, model_type,
   rope/mrope sections, pooling, soft-token count), SIZES scaled
   (report-test-plan.md section 1), special token ids remapped into a vocab
   of 512 so embedding tables stay small (option (a) there). `REAL` below
   embeds the structural fields read from each family's released config.json
   (2026-09-23, config.json only -- no weights), so CI needs no artifacts.

2. `StubFamily` -- a complete `Family` with an identity tower and
   `positions() -> (None, 0)`, for P4 to build the serve path against
   without importing any real family. Its features depend on
   the pixels, so two different images give different answers (G9) and the
   same image the same (G6/G7).

3. Goldens -- mlx-vlm reference outputs, made ONCE in the exo interpreter
   (mlx-vlm 0.6.17) and committed as .npz under tests/goldens/, so G1-G4 run
   anywhere without mlx-vlm. `run_reference` runs a builder
   script there; `save_golden` / `load_golden` are the format, numpy only.

This file must import under BOTH interpreters (the test one, and the exo
one a golden builder runs in), so it is stdlib + numpy at module level and
imports mlx / PIL inside functions.

Tiny-fixture rules, from tests/test_batch_drafting.py: float32, seed 0,
vocab >= 512, compare tokens not raw logits (logits get a tolerance).
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

GOLDENS = ROOT / "tests" / "goldens"
#: The interpreter with mlx-vlm 0.6.17 -- the reference. Override with
#: KNURLOGIC_VLM_PYTHON on another box.
REFERENCE_PYTHON = os.environ.get("KNURLOGIC_VLM_PYTHON",
                                  "/opt/anaconda3/envs/exo/bin/python")
REFERENCE_VERSION = "0.6.17"

TINY_VOCAB = 512

# --- 1. real configs, structural fields only ----------------------------------
# Read from config.json of the named artifact on 2026-09-23. Only what the
# vision path or the special ids need; each family's own fixture file adds
# the text_config its trunk needs (P1 already has one in
# tests/test_batch_drafting.py::_tiny).

_QWEN_VISION = dict(deepstack_visual_indexes=[], depth=27,
                    hidden_act="gelu_pytorch_tanh", hidden_size=1152,
                    in_channels=3, intermediate_size=4304, num_heads=16,
                    num_position_embeddings=2304, patch_size=16,
                    spatial_merge_size=2, temporal_patch_size=2)
_QWEN_TOP = dict(image_token_id=248056, video_token_id=248057,
                 vision_start_token_id=248053, vision_end_token_id=248054)
_QWEN_ROPE = dict(mrope_interleaved=True, mrope_section=[11, 11, 10],
                  partial_rotary_factor=0.25, rope_theta=10000000)

REAL: Dict[str, Dict[str, Any]] = {
    "qwen3_5": dict(
        src="Qwen3.8-27B-VQ-3.9bpw", model_type="qwen3_5", **_QWEN_TOP,
        vision_config=dict(_QWEN_VISION, model_type="qwen3_5",
                           out_hidden_size=5120),
        text_config=dict(model_type="qwen3_5_text", hidden_size=5120,
                         vocab_size=248320, rope_parameters=_QWEN_ROPE)),
    "qwen3_5_moe": dict(
        src="Qwen3.6-35B-A3B-VQ-3.4bpw", model_type="qwen3_5_moe",
        **_QWEN_TOP,
        vision_config=dict(_QWEN_VISION, model_type="qwen3_5_moe",
                           out_hidden_size=2048),
        text_config=dict(model_type="qwen3_5_moe_text", hidden_size=2048,
                         vocab_size=248320, num_experts=256,
                         num_experts_per_tok=8, rope_parameters=_QWEN_ROPE)),
    "qwen4_exp": dict(
        src="Qwen3.8-Flash-Next-VQ-2.1bpw", model_type="qwen4_exp",
        **_QWEN_TOP,
        vision_config=dict(_QWEN_VISION, model_type="qwen4_exp",
                           out_hidden_size=2560),
        text_config=dict(model_type="qwen4_exp_text", hidden_size=2560,
                         vocab_size=248320, num_experts=512,
                         num_experts_per_tok=10, rope_parameters=_QWEN_ROPE)),
    "glm5_next": dict(
        src="GLM-5.3-Flash-VQ-2.7bpw", model_type="glm5_next",
        image_token_id=154854, image_start_token_id=154830,
        image_end_token_id=154831, video_token_id=154855,
        video_start_token_id=154832, video_end_token_id=154833,
        vision_config=dict(model_type="glm5_next_vision", attention_bias=True,
                           depth=24, hidden_act="silu", hidden_size=1024,
                           image_size=448, in_channels=3,
                           intermediate_size=4096, num_heads=16,
                           out_hidden_size=4096, patch_size=14,
                           projection_intermediate_size=10240,
                           rms_norm_eps=1e-05, spatial_merge_size=2,
                           swiglu_limit=10.0, temporal_patch_size=2),
        text_config=dict(model_type="glm5_next_text", hidden_size=4096,
                         vocab_size=154880)),
    "gemma4": dict(
        src="gemma-4-e4b-it-VQ-PLE", model_type="gemma4",
        image_token_id=258880, boi_token_id=255999, eoi_token_id=258882,
        audio_token_id=258881, boa_token_id=256000, eoa_token_id=258883,
        video_token_id=258884, vision_soft_tokens_per_image=280,
        vision_config=dict(model_type="gemma4_vision", attention_bias=False,
                           default_output_length=280, global_head_dim=64,
                           head_dim=64, hidden_activation="gelu_pytorch_tanh",
                           hidden_size=768, intermediate_size=3072,
                           num_attention_heads=12, num_hidden_layers=16,
                           num_key_value_heads=12, patch_size=16,
                           pooling_kernel_size=3,
                           position_embedding_size=10240, rms_norm_eps=1e-06,
                           standardize=False, use_clipped_linears=True),
        text_config=dict(model_type="gemma4_text", hidden_size=2560,
                         vocab_size=262144, hidden_size_per_layer_input=256,
                         vocab_size_per_layer_input=262144)),
}
FAMILIES = tuple(REAL)

#: Size keys and their tiny value. Anything not named keeps its real value
#: -- that is how the structural keys survive.
_SCALE_VISION = {"depth": 2, "num_hidden_layers": 2, "hidden_size": 64,
                 "num_heads": 4, "num_attention_heads": 4,
                 "num_key_value_heads": 4, "head_dim": 16,
                 "global_head_dim": 16, "intermediate_size": 128,
                 "projection_intermediate_size": 128}
_SCALE_TEXT = {"hidden_size": 128, "intermediate_size": 256,
               "moe_intermediate_size": 64,
               "shared_expert_intermediate_size": 64,
               "num_hidden_layers": 4, "num_attention_heads": 4,
               "num_key_value_heads": 2, "head_dim": 32,
               "num_experts": 4, "n_routed_experts": 4,
               "num_experts_per_tok": 2, "vocab_size": TINY_VOCAB,
               "vocab_size_per_layer_input": TINY_VOCAB,
               "hidden_size_per_layer_input": 16}


def tiny_ids(family: str) -> Dict[str, int]:
    """The family's special token ids remapped into the tiny vocab: sorted
    by real id and packed at the top (511, 510, ...), so their ORDER is the
    real one and the ordinary ids 0..~500 stay free for text."""
    real = {k: v for k, v in REAL[family].items()
            if k.endswith("_token_id") and isinstance(v, int)}
    ordered = sorted(real.items(), key=lambda kv: kv[1])
    base = TINY_VOCAB - len(ordered)
    return {k: base + i for i, (k, _) in enumerate(ordered)}


def _scale(d: Dict[str, Any], table: Dict[str, int]) -> Dict[str, Any]:
    out = copy.deepcopy(d)
    layers_before = out.get("num_hidden_layers")
    for k, v in table.items():
        if k in out and out[k] is not None:
            out[k] = v
    # per-layer lists (layer_types, ...) follow the layer count
    n = out.get("num_hidden_layers")
    if layers_before and n != layers_before:
        for k, v in list(out.items()):
            if isinstance(v, list) and len(v) == layers_before:
                out[k] = v[:n]
    return out


def tiny_config(family: str, real: Optional[Dict[str, Any]] = None,
                text_config: Optional[Dict[str, Any]] = None,
                **overrides: Any) -> Dict[str, Any]:
    """A tiny config for `family`: `real` (a full config.json, default the
    embedded REAL[family]) with sizes scaled, special ids remapped into
    TINY_VOCAB, vision out_hidden_size set to the tiny text hidden, and
    `text_config` (the family fixture's own, if given) scaled the same way.
    `overrides` are applied last, at the top level."""
    base = copy.deepcopy(real if real is not None else REAL[family])
    base.pop("src", None)
    for k in ("quantization", "quantization_config", "vq_modules",
              "vq_linear", "vq_embed", "vq_other", "vq_ple", "knobs",
              "model_file", "audio_config"):
        base.pop(k, None)
    base.update(tiny_ids(family))
    tc = copy.deepcopy(text_config if text_config is not None
                       else base.get("text_config", {}))
    tc = _scale(tc, _SCALE_TEXT)
    tc.setdefault("vocab_size", TINY_VOCAB)
    tc.setdefault("hidden_size", _SCALE_TEXT["hidden_size"])
    base["text_config"] = tc
    vc = _scale(base.get("vision_config", {}), _SCALE_VISION)
    if "out_hidden_size" in vc:
        vc["out_hidden_size"] = tc["hidden_size"]
    if "default_output_length" in vc:          # gemma: soft tokens per image
        vc["default_output_length"] = 16
    if "vision_soft_tokens_per_image" in base:
        base["vision_soft_tokens_per_image"] = 16
    if "position_embedding_size" in vc:
        vc["position_embedding_size"] = 256
    base["vision_config"] = vc
    base.update(overrides)
    return base


def tiny_image(w: int = 32, h: int = 24, seed: int = 0):
    """A deterministic random RGB PIL image."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                           "RGB")


def png_bytes(img, **save_kw) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", **save_kw)
    return buf.getvalue()


# --- 2. the stub family ---------------------------------------------------------

class StubFamily:
    """A complete Family (engine/vision/__init__.py) with no real tower.

    preprocess: resize so each side is a multiple of `patch`, at most
    `max_side`; one token per patch (grid (1, gh, gw), no merge).
    encode:     identity tower -- each patch's pixels in [0, 1], zero-padded
                or truncated to `hidden`. `tower` is a separate method so a
                test counts calls by wrapping it FROM OUTSIDE (a
                counter the code under test increments proves nothing).
    positions:  (None, 0) -- the trunk's own 1D positions.
    chunk_boundaries: every image span when `bidirectional` (gemma-like, so
                P4 can test the chunk snap), else [].
    embed:      embed_fn(ids) then scatter.merge -- the real merge path.
    """

    def __init__(self, image_token_id: int, hidden: int, *, patch: int = 4,
                 max_side: int = 16, bidirectional: bool = True,
                 placeholder: str = "<image>", embed_fn=None,
                 family: str = "stub"):
        from knurlogic.engine.vision import VisionSpec, proc_hash
        self.hidden, self.patch, self.max_side = hidden, patch, max_side
        self.bidirectional, self.placeholder = bidirectional, placeholder
        self.embed_fn = embed_fn
        self.spec = VisionSpec(
            family=family, image_token_id=image_token_id, patch=patch,
            merge=None, min_pixels=patch * patch,
            max_pixels=max_side * max_side, fixed_tokens=None,
            proc_hash=proc_hash({"stub": 1, "patch": patch,
                                 "max_side": max_side, "hidden": hidden}))

    def load_weights(self, model_path: str) -> int:
        return 0

    def preprocess(self, img, sha: str):
        from PIL import Image
        from knurlogic.engine.vision import ImageRef
        p, m = self.patch, self.max_side
        w, h = img.size
        s = min(1.0, m / max(w, h))
        gw = max(1, int(w * s) // p)
        gh = max(1, int(h * s) // p)
        im = img.convert("RGB").resize((gw * p, gh * p), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0          # [H, W, 3]
        a = a.reshape(gh, p, gw, p, 3).transpose(0, 2, 1, 3, 4)
        a = a.reshape(gh * gw, p * p * 3)
        ref = ImageRef(sha=sha, proc_hash=self.spec.proc_hash,
                       n_tokens=gh * gw, grid_thw=(1, gh, gw))
        return {"pixel_values": a}, ref

    def tower(self, pixel_values):
        import mlx.core as mx
        x = mx.array(pixel_values)
        d = x.shape[-1]
        if d >= self.hidden:
            return x[:, :self.hidden]
        return mx.concatenate(
            [x, mx.zeros((x.shape[0], self.hidden - d), x.dtype)], axis=1)

    def encode(self, pixels, ref):
        import mlx.core as mx
        from knurlogic.engine.vision import EncodedImage
        feats = self.tower(pixels["pixel_values"])
        mx.eval(feats)
        assert feats.shape[0] == ref.n_tokens
        return EncodedImage(ref=ref, feats=feats)

    def placeholder_text(self, ref) -> str:
        return self.placeholder

    def _embed_tokens(self, model):
        if self.embed_fn is not None:
            return self.embed_fn
        for path in ("language_model.model.embed_tokens",
                     "model.embed_tokens", "embed_tokens"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("no embed_tokens on the model; pass embed_fn")

    def embed(self, model, key, start, features):
        import mlx.core as mx
        from knurlogic.engine.vision import key as K
        from knurlogic.engine.vision.scatter import merge
        sl = list(key[start:])
        ids = mx.array(K.to_ids(sl, self.spec.image_token_id))[None]
        text = self._embed_tokens(model)(ids)
        return {"input_embeddings": merge(text, sl, features)}

    def positions(self, key, refs):
        return None, 0

    def chunk_boundaries(self, key):
        from knurlogic.engine.vision.key import image_spans
        if not self.bidirectional:
            return []
        return [(s.start, s.end) for s in image_spans(key)]


def stub_build(model_path: str, text_model, config: Dict[str, Any]):
    """Registry-shaped builder for the stub: point a monkeypatched
    registry.FAMILIES entry at a module exposing this to route a tiny model
    through `registry.build` exactly as a real family is."""
    tc = config.get("text_config", config)
    return StubFamily(config["image_token_id"], tc["hidden_size"])


# --- 3. goldens -----------------------------------------------------------------

def golden_path(name: str) -> Path:
    return GOLDENS / f"{name}.npz"


def save_golden(name: str, arrays: Dict[str, Any],
                meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write tests/goldens/<name>.npz: the arrays as float32/int numpy plus a
    `__meta__` JSON string (reference version, seed, config, command) so a
    golden says how it was made. Called by a builder running in the
    reference interpreter."""
    GOLDENS.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(v) for k, v in arrays.items()}
    m = dict(meta or {})
    m.setdefault("reference_python", sys.executable)
    try:
        import importlib.metadata as md
        m.setdefault("mlx_vlm", md.version("mlx-vlm"))
        m.setdefault("mlx", md.version("mlx"))
    except Exception:
        pass
    out["__meta__"] = np.array(json.dumps(m, sort_keys=True, default=str))
    p = golden_path(name)
    np.savez_compressed(p, **out)
    return p


def load_golden(name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """(arrays, meta). A missing golden is a FAILURE with the command that
    makes it, not a skip: a gate that skips when its reference is missing
    is a gate nobody notices is off."""
    p = golden_path(name)
    if not p.is_file():
        raise FileNotFoundError(
            f"golden {p} missing; build it in the reference interpreter: "
            f"{REFERENCE_PYTHON} tests/goldens/build_<name>.py")
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    return arrays, meta


def run_reference(script: Path, *args: str,
                  timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a golden builder script in the mlx-vlm interpreter, checking the
    reference version first. Builders are tiny-fixture only: float32, seed
    0, no model files -- safe on the shared Mac."""
    ver = subprocess.run(
        [REFERENCE_PYTHON, "-c",
         "import importlib.metadata as m; print(m.version('mlx-vlm'))"],
        capture_output=True, text=True, timeout=120)
    if ver.returncode != 0 or ver.stdout.strip() != REFERENCE_VERSION:
        raise RuntimeError(f"{REFERENCE_PYTHON} does not have mlx-vlm "
                           f"{REFERENCE_VERSION}: {ver.stdout}{ver.stderr}")
    return subprocess.run([REFERENCE_PYTHON, str(script), *args],
                          capture_output=True, text=True, timeout=timeout,
                          check=True, cwd=str(ROOT))
