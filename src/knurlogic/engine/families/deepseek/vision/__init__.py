"""DeepSeek-V4-Flash-Vision-Exp (deepseek_v4 with vision_n_layers > 0).

The Family `registry.build("deepseek_v4", ...)` resolves to
(docs/design/vision-contracts.md, docs/design/deepseek-vision.md). Its
tower (tower.py: ViT + aligner) and processor (processor.py) are ports of
the artifact's own `inference/` code; the tower loads STANDALONE from the
`vision.*` / `aligner.*` keys, which the trunk's sanitize drops.

An image is a block of typed tokens, id = vocab_size + type: IMAGE_START,
then the aligner's rows in an N-layout with an IMAGE_NEW_LINE after each
row, IMAGE_PADs to even it out, IMAGE_END. The block is the image's run
of sentinels in the key, and `feats` holds every row of it: the aligner's
rows where the type is IMAGE, the learned image_start / pad / newline /
end rows elsewhere. Before the block go 0-3 IMAGE_PADs that put its
IMAGE_START at a position == 3 (mod 4); they depend on where the image
sits, so they are not part of the image: `frame_key` writes them into the
key as plain ids (vocab_size + IMAGE_PAD), and the trunk embeds such an id
as `image_pad` itself.

The trunk reads the types (bias_vl routing, image-span attention) from the
`vl_ids` `embed` returns beside the embeddings; the types follow from
(n_llm_h, n_llm_w), which ride in `ref.grid_thw` so a pipeline follower,
which has refs but no store, computes the same.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knurlogic.engine.vision import EncodedImage, ImageRef, VisionSpec, proc_hash

from .processor import IMAGE, IMAGE_PAD, VisionArgs, compress_pad, image_block

#: The image placeholder of DeepSeek's encoder (encoding_dsv4.py)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
#: its id in the Vision-Exp tokenizer.json, when that file cannot be read
IMAGE_TOKEN_ID = 129264
_PREFIXES = ("vision.", "aligner.")
_ROWS = ("image_start", "image_pad", "image_newline", "image_end")


def _header(path: Path) -> dict:
    """A safetensors file's header: name -> {dtype, shape, data_offsets}."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def _read(path: Path, keys: list[str]) -> dict:
    """`keys` out of one safetensors file, each read on its own (a shard
    holding the vision tensors can hold the 1 GB embedding too).
    safetensors' own reader goes through numpy, which has no bfloat16."""
    import struct

    import mlx.core as mx
    import numpy as np

    np_of = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None),
             "F32": (np.float32, None), "U8": (np.uint8, None),
             "U32": (np.uint32, None), "I32": (np.int32, None)}
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        for k in keys:
            m = h[k]
            if m["dtype"] not in np_of:
                raise ValueError(f"{k}: dtype {m['dtype']} not read here")
            dt, view = np_of[m["dtype"]]
            a, b = m["data_offsets"]
            f.seek(8 + n + a)
            arr = mx.array(np.frombuffer(f.read(b - a), dtype=dt)
                           .reshape(m["shape"]))
            out[k] = arr.view(view) if view is not None else arr
    return out


def _image_token_id(model_path: str) -> int:
    p = Path(model_path) / "tokenizer.json"
    if p.is_file():
        for t in json.loads(p.read_text()).get("added_tokens", ()):
            if t.get("content") == IMAGE_PLACEHOLDER:
                return int(t["id"])
    return IMAGE_TOKEN_ID


class DeepseekVisionFamily:
    """`Family` (vision-contracts.md) for DeepSeek-V4-Flash-Vision-Exp."""

    #: DeepSeek's encoder joins a user message's parts with "\n\n"
    #: (encoding_dsv4.py, render_message); engine/vision/request.py puts it
    #: between the parts of a message with an image
    part_separator = "\n\n"

    def __init__(self, config: dict[str, Any], image_token_id: int):
        self.args = VisionArgs.from_config(config)
        a = self.args
        self.vocab_size = a.vocab_size
        self.image_token_id = int(image_token_id)
        step = a.vision_patch_size * a.vision_downsample_ratio
        self.spec = VisionSpec(
            family="deepseek_v4",
            image_token_id=self.image_token_id,
            patch=a.vision_patch_size,
            merge=a.vision_downsample_ratio,
            min_pixels=a.vision_min_pixels,
            max_pixels=a.vision_max_n_token * step * step,
            fixed_tokens=None,
            proc_hash=proc_hash({
                "family": "deepseek_v4", "patch": a.vision_patch_size,
                "downsample": a.vision_downsample_ratio,
                "max_n_token": a.vision_max_n_token,
                "min_pixels": a.vision_min_pixels,
                "max_wh_ratio": a.vision_max_wh_ratio,
            }),
        )
        self.tower = None   # built lazily, mx is engine-only
        self.rows = None    # [5, dim]: image_start, pad, pad, newline, end

    # -- weights ---------------------------------------------------------

    def _build_tower(self):
        if self.tower is None:
            from .tower import Tower
            self.tower = Tower(self.args)
        return self.tower

    def set_rows(self, rows: dict[str, Any]) -> None:
        import mlx.core as mx
        self.rows = mx.stack([rows["image_start"], rows["image_pad"],
                              rows["image_pad"], rows["image_newline"],
                              rows["image_end"]])

    def load_weights(self, model_path: str) -> int:
        """The `vision.*` / `aligner.*` keys and the four image rows, read
        one tensor at a time out of the shards the index names for them
        (the trunk's shards are not opened otherwise). The image rows are
        also the trunk's (sanitize keeps them); this copy is what `encode`
        builds a block from."""
        import mlx.core as mx

        root = Path(model_path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            wmap = json.loads(index.read_text()).get("weight_map", {})
        else:
            wmap = {k: p.name for p in sorted(root.glob("*.safetensors"))
                    for k in _header(p)}
        want: dict[str, list[str]] = {}
        for k, shard in wmap.items():
            name = k[len("model."):] if k.startswith("model.image_") else k
            if name.startswith(_PREFIXES) or name in _ROWS:
                want.setdefault(shard, []).append(k)
        tower, rows = {}, {}
        for shard, keys in sorted(want.items()):
            for k, v in _read(root / shard, keys).items():
                name = k[len("model."):] if k.startswith("model.") else k
                if name in _ROWS:
                    rows[name] = v
                else:
                    tower[name] = v
        if not tower:
            return 0
        missing = [r for r in _ROWS if r not in rows]
        if missing:
            raise ValueError(f"deepseek_v4 vision: no {missing} in "
                             f"{model_path}; an image block needs them")
        model = self._build_tower()
        from knurlogic.engine.vision.quant import artifact_quantization, quantize_like
        quantize_like(model, tower, artifact_quantization(model_path))
        model.load_weights(list(tower.items()), strict=True)
        self.set_rows(rows)
        mx.eval(model.parameters(), self.rows)
        return len(tower) + len(rows)

    # -- preprocess / encode ----------------------------------------------

    def preprocess(self, img, sha: str):
        from .processor import load_image

        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(img, self.args)
        types, _ = image_block(n_llm_h, n_llm_w)
        ref = ImageRef(sha=sha, proc_hash=self.spec.proc_hash,
                       n_tokens=int(types.size),
                       grid_thw=(1, n_llm_h, n_llm_w))
        return {"patches": patches, "n_vit": (n_vit_h, n_vit_w),
                "n_llm": (n_llm_h, n_llm_w)}, ref

    def encode(self, pixels: dict[str, Any], ref) -> EncodedImage:
        """The tower once, then the whole block: `rows[types]`, with the
        aligner's rows (in `perm` order) where the type is IMAGE -- the
        reference's merge_image_embeddings."""
        import mlx.core as mx
        import numpy as np

        tower = self._build_tower()
        dtype = tower.vision.patch_embed.proj.weight.dtype
        if dtype not in (mx.bfloat16, mx.float16, mx.float32):
            dtype = mx.bfloat16
        n_vit_h, n_vit_w = pixels["n_vit"]
        x = mx.array(pixels["patches"]).astype(dtype)
        out = tower(x, n_vit_h, n_vit_w)
        types, perm = image_block(*pixels["n_llm"])
        if types.size != ref.n_tokens:
            raise ValueError(f"deepseek_v4: a {pixels['n_llm']} grid makes "
                             f"{types.size} tokens; the ref says "
                             f"{ref.n_tokens}")
        feats = self.rows.astype(out.dtype)[mx.array(types)]
        feats[mx.array(np.flatnonzero(types == IMAGE))] = out[mx.array(perm)]
        mx.eval(feats)
        return EncodedImage(ref=ref, feats=feats)

    def placeholder_text(self, ref) -> str:
        return IMAGE_PLACEHOLDER

    # -- the key ----------------------------------------------------------

    def frame_key(self, key: list, segments: list[list]):
        """The IMAGE_PADs before each image (processor.compress_pad), by
        the position its block would start at in the key built so far --
        as the reference's prepare_vl_inputs counts it. They go into the
        image's segment, ahead of its first sentinel. -> (key, segments)."""
        from knurlogic.engine.vision.key import is_sentinel

        pad = self.vocab_size + IMAGE_PAD
        out_segs: list[list] = []
        n = 0
        for seg in segments:
            s: list = []
            for x in seg:
                if is_sentinel(x) and x[3] == 0:
                    k = compress_pad(n)
                    s.extend([pad] * k)
                    n += k
                s.append(x)
                n += 1
            out_segs.append(s)
        return [x for s in out_segs for x in s], out_segs

    def types_of(self, ref) -> Any:
        _, n_llm_h, n_llm_w = ref.grid_thw
        return image_block(n_llm_h, n_llm_w)[0]

    # -- embed / positions / chunking -------------------------------------

    @staticmethod
    def _core(model):
        """The trunk's DeepseekV4Model (it owns `embed`)."""
        for path in ("model", "language_model.model", "_model.model"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if hasattr(obj, "embed") and hasattr(obj, "vocab_size"):
                return obj
        raise AttributeError("no deepseek_v4 trunk core with embed() found")

    def embed(self, model, key, start, features) -> dict[str, Any]:
        import mlx.core as mx

        from knurlogic.engine.vision import key as K
        from knurlogic.engine.vision.scatter import merge

        sl = list(key[start:])
        ids = K.to_ids(sl, self.image_token_id)
        vl = list(ids)
        for s in K.image_spans(sl):
            types = self.types_of(features(s.sha, s.proc_hash).ref)
            vl[s.start:s.end] = (self.vocab_size
                                 + types[s.k0:s.k0 + s.end - s.start]).tolist()
        # the trunk's own embedding: an IMAGE_PAD id before a block takes
        # `image_pad` there; every sentinel row is replaced from the store
        text = self._core(model).embed(mx.array(ids)[None])
        return {"input_embeddings": merge(text, sl, features),
                "vl_ids": mx.array(vl)[None]}

    def positions(self, key, refs) -> tuple[Any | None, int]:
        # the trunk's own 1D positions: an image token is one position
        return None, 0

    def chunk_boundaries(self, key):
        """Every image block whole: inside it a token sees forward to
        IMAGE_END, and the reference prefills a block in one call."""
        from knurlogic.engine.vision.key import image_spans
        return [(s.start, s.end) for s in image_spans(key)]


def build(model_path: str, text_model: Any, config: dict[str, Any]):
    if int(config.get("vision_n_layers") or 0) <= 0:
        return None
    # the tower is read by serve/vision.bind (fam.load_weights), not here
    return DeepseekVisionFamily(config, _image_token_id(model_path))
