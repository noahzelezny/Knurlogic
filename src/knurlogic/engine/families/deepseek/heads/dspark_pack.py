"""Pack DeepSeek-V4-Flash-Vision-Exp's DSpark stages into the sidecar
heads/deepseek_v4_dspark.py binds (`mtp-head-dspark-mxfp4.safetensors`).

    python -m knurlogic.engine.families.deepseek.heads.dspark_pack \\
        SOURCE OUT_DIR

SOURCE is the HF checkpoint (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:
`mtp.<stage>.*` in its own names, FP8 linears with 128x128 E8M0 block
scales, FP4 routed experts with per-32 E8M0 scales, bf16 / fp32 rest).
Only its index and the `mtp.*` tensors are read, one at a time; nothing
in SOURCE is written. OUT_DIR gets the sidecar (an existing file of that
name is refused); for a converted artifact that is its own folder.

The conversion is the trunk's (architecture/deepseek_v4.py `sanitize`):
FP8 linears dequantized to bf16 (exact: an e4m3 value times a power of
two fits bf16's mantissa), routed experts reinterpreted as mxfp4 (the
same bits, stacked per projection), names in the trunk's post-sanitize
layout (heads/deepseek_v4_dspark.py lists them). The file is written as
it is read -- a stacked expert projection is the concatenation of its
experts' bytes -- so memory holds one source tensor at a time, not the
~10.5 GiB written (measured on the real checkpoint: 45 s from the HDD,
2.8 GB peak RSS).
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

from .deepseek_v4_dspark import SIDECAR_NAME

#: safetensors dtype -> (numpy dtype of its bytes, bytes per element)
_NP = {"F32": np.float32, "BF16": np.uint16, "F8_E4M3": np.uint8,
       "U8": np.uint8, "I8": np.uint8}
_W = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_HC = re.compile(r"^hc_(attn|ffn)_(fn|base|scale)$")


def _header(f: Path) -> tuple[dict, int]:
    with f.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


class _Source:
    """The checkpoint's mtp.* tensors, by name, read on demand."""

    def __init__(self, root: Path):
        self.root = Path(root)
        idx = json.loads((self.root / "model.safetensors.index.json")
                         .read_text())["weight_map"]
        self.files = {k: v for k, v in idx.items() if k.startswith("mtp.")}
        self._hdr: dict = {}

    def meta(self, name: str) -> tuple[dict, Path, int]:
        f = self.root / self.files[name]
        if f not in self._hdr:
            self._hdr[f] = _header(f)
        hdr, base = self._hdr[f]
        return hdr[name], f, base

    def dtype(self, name: str) -> str:
        return self.meta(name)[0]["dtype"]

    def shape(self, name: str) -> list:
        return list(self.meta(name)[0]["shape"])

    def raw(self, name: str) -> np.ndarray:
        m, f, base = self.meta(name)
        a, b = m["data_offsets"]
        with f.open("rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dtype=_NP[m["dtype"]]).reshape(m["shape"])


def _e4m3() -> np.ndarray:
    """float32 value of every e4m3fn byte."""
    b = np.arange(256)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return (s * v).astype(np.float32)


_E4M3 = _e4m3()


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round to nearest even."""
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def _fp8_to_bf16(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """FP8 [M, N] with E8M0 128x128 block scales -> bf16 bits [M, N]."""
    M, N = w.shape
    scale = np.exp2(s.astype(np.float32) - 127.0)
    full = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)[:M, :N]
    return _bf16_bits(_E4M3[w] * full)


def plan(src: _Source) -> list:
    """[(sidecar name, safetensors dtype, shape, producer)] in file order;
    producer() -> the tensor's bytes (an iterable of chunks)."""
    stages = sorted({int(k.split(".")[1]) for k in src.files})
    out = []
    for s in stages:
        pre = f"mtp.{s}."
        names = sorted(k for k in src.files if k.startswith(pre))
        experts = {}
        for k in names:
            rest = k[len(pre):]
            if rest.endswith(".scale"):
                continue                     # read with its weight
            m = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.weight$", rest)
            if m:
                experts.setdefault(m.group(2), []).append(int(m.group(1)))
                continue
            scale = k[:-len("weight")] + "scale" \
                if k.endswith(".weight") else None
            dt = src.dtype(k)
            if dt == "F8_E4M3":
                if scale not in src.files:
                    raise ValueError(f"{k}: FP8 without its .scale")
                out.append((pre + _rename(rest), "BF16", src.shape(k),
                            lambda k=k, sc=scale: [_fp8_to_bf16(
                                src.raw(k), src.raw(sc)).tobytes()]))
            elif dt in ("BF16", "F32"):
                out.append((pre + _rename(rest), dt, src.shape(k),
                            lambda k=k: [src.raw(k).tobytes()]))
            else:
                raise ValueError(f"{k}: unexpected dtype {dt}")
        for w, ids in sorted(experts.items()):
            ids = sorted(ids)
            if ids != list(range(len(ids))):
                raise ValueError(f"{pre}ffn.experts.*.{w}: experts "
                                 f"{ids[:3]}.. are not 0..{len(ids) - 1}")
            k0 = f"{pre}ffn.experts.0.{w}"
            ws, ss = src.shape(k0 + ".weight"), src.shape(k0 + ".scale")
            if src.dtype(k0 + ".weight") != "I8" or ss[-1] * 16 != ws[-1]:
                raise ValueError(f"{k0}: not FP4 with per-32 scales")
            dst = f"{pre}ffn.switch_mlp.{_W[w]}"
            E = len(ids)

            def chunks(sfx, w=w, E=E, pre=pre):
                for e in range(E):
                    yield src.raw(f"{pre}ffn.experts.{e}.{w}.{sfx}").tobytes()

            # FP4 bytes [out, in/2] read 4 at a time are mxfp4's uint32
            # words [out, in/8]: element 2i in the low nibble either way
            out.append((dst + ".weight", "U32", [E, ws[0], ws[1] // 4],
                        lambda c=chunks: c("weight")))
            out.append((dst + ".scales", "U8", [E] + ss,
                        lambda c=chunks: c("scale")))
    return out


def _rename(rest: str) -> str:
    """An official per-stage name -> the sidecar's (the trunk's layout)."""
    m = _HC.match(rest)
    if m:
        return f"hc_{m.group(1)}.{m.group(2)}"
    if rest == "ffn.gate.bias":
        return "ffn.gate.e_score_correction_bias"
    m = re.match(r"ffn\.shared_experts\.(w[123])\.weight$", rest)
    if m:
        return f"ffn.shared_experts.{_W[m.group(1)]}.weight"
    return rest


_SIZE = {"F32": 4, "BF16": 2, "U32": 4, "U8": 1}


def write(src: _Source, path: Path, meta: dict | None = None) -> int:
    """Stream the plan into a safetensors file; returns its byte size."""
    items = plan(src)
    hdr, off = {}, 0
    for name, dt, shape, _ in items:
        n = int(np.prod(shape)) * _SIZE[dt]
        hdr[name] = {"dtype": dt, "shape": shape,
                     "data_offsets": [off, off + n]}
        off += n
    hdr["__metadata__"] = {"format": "mlx", **(meta or {})}
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name, dt, shape, produce in items:
            want = hdr[name]["data_offsets"][1] - hdr[name]["data_offsets"][0]
            got = 0
            for chunk in produce():
                fh.write(chunk)
                got += len(chunk)
            if got != want:
                raise ValueError(f"{name}: wrote {got} bytes, expected {want}")
    tmp.rename(path)
    return path.stat().st_size


def pack(source, out_dir) -> Path:
    src = _Source(Path(source))
    if not src.files:
        raise ValueError(f"{source}: no mtp.* tensors in its index")
    out = Path(out_dir) / SIDECAR_NAME
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it first")
    recipe = {"family": "deepseek_v4", "kind": "dspark", "expert_bits": 4,
              "source": Path(source).name}
    write(src, out, {"knurlogic_mtp": json.dumps(recipe)})
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    p = pack(sys.argv[1], sys.argv[2])
    print(f"{p}  {p.stat().st_size / (1 << 30):.2f} GiB")
