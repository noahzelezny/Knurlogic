"""KV-cache precision: attention K/V stored at 8, 6 or 4 bits.

WHAT MLX 0.31.2 / MLX-LM 0.31.3 GIVE US, and what they do not:

  mx.quantize / mx.dequantize   affine, per-group scale + bias, bits 2-8
                                including 6 (and 3, 5); groups of 32/64/128
                                along the last axis.
  mlx_lm QuantizedKVCache       ONE sequence. There is no batched quantized
                                cache in mlx-lm 0.31.3 -- BatchKVCache and
                                BatchRotatingKVCache are bf16 only, and
                                `_make_cache` refuses anything else.
  quantized SDPA                mlx_lm.models.base routes to
                                `quantized_scaled_dot_product_attention`
                                (two mx.quantized_matmul) when the cache has
                                a `bits` attribute; mx.fast.scaled_dot_
                                product_attention itself takes no quantized
                                K/V. Only attention written against
                                mlx_lm.models.base takes that path; the
                                vendored families each have their own.

So knurlogic keeps its own pair here: a single-row cache (what a prefill
and the prompt cache hold) and a batched one (what the decode loop holds),
both STORING the quantized triple and handing the attention code back
dequantized arrays of the input dtype. Every family's attention is then
untouched -- including gemma4's KV-shared layers, which reuse the returned
arrays -- and neither class has a `bits` attribute, so mlx-lm's base SDPA
does not mistake the dequantized arrays for quantized ones (the attribute
is `kv_bits`).

The cost of that choice, stated: memory is what shrinks (the stored cache,
the prompt cache's entries). Decode reads the quantized cache and writes a
dequantized copy per layer per step, which is MORE memory traffic than
bf16, not less -- expect decode to be slower, not faster. Nothing here is
measured on a real model yet.

What is quantized: mlx-lm's plain `KVCache` (exact type) in the list
`model.make_cache()` returns, and inside a CacheList. Not: recurrent state
(ArraysCache: deltanet/SSM), sliding windows (RotatingKVCache, bounded by
their window), a family's own cache classes (qwen4_exp's _AttnCache, GLM's
mlx_vlm caches), or the MTP head's draft cache. Rollback and prompt-cache
trims move the write index only, exactly as for bf16: the stale tail is
overwritten by the next update.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.cache import (BatchKVCache, CacheList, KVCache,
                                 dynamic_roll)

#: bits a person can ask for; None (or "bf16") is the unquantized cache
BITS = (8, 6, 4)


def parse_bits(v) -> Optional[int]:
    """'bf16'/''/None/16 -> None; '8'/'6'/'4' -> int. Anything else raises."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "bf16", "16", "none", "off"):
        return None
    b = int(s)
    if b not in BITS:
        raise ValueError(f"KV bits {v!r}: bf16, 8, 6 or 4")
    return b


def group_for(dim: int) -> int:
    for g in (64, 32):
        if dim % g == 0:
            return g
    raise ValueError(f"head dim {dim} is not a multiple of 32: its K/V "
                     f"cannot be quantized in groups")


def _q(x, g, bits):
    return tuple(mx.quantize(x, group_size=g, bits=bits))


def _dq(parts, g, bits, dtype):
    return mx.dequantize(*parts, group_size=g, bits=bits).astype(dtype)


def _group(dk: int, dv: int) -> int:
    """One group size for K and V: 64 when both allow it, else 32."""
    group_for(dk), group_for(dv)
    return 64 if not (dk % 64 or dv % 64) else 32


def _alloc(B, H, L, dim, bits, g, dtype):
    return (mx.zeros((B, H, L, dim * bits // 32), mx.uint32),
            mx.zeros((B, H, L, dim // g), dtype),
            mx.zeros((B, H, L, dim // g), dtype))


def _slice(parts, a, b):
    return tuple(p[..., a:b, :] for p in parts)


def _cat(xs, ys):
    return tuple(mx.concatenate([x, y], axis=2) for x, y in zip(xs, ys))


def _nbytes(parts) -> int:
    return sum(int(p.nbytes) for p in parts) if parts is not None else 0


class QuantKVCache(KVCache):
    """One sequence's attention cache, stored quantized. `keys`/`values`
    are (packed, scales, biases) triples; `update_and_fetch` returns
    dequantized arrays of the dtype it was given."""

    step = 256

    def __init__(self, kv_bits: int = 8):
        super().__init__()
        self.kv_bits = int(kv_bits)
        self.group = None
        self.dims = None               # (k head dim, v head dim)

    def update_and_fetch(self, keys, values):
        B, H, S, Dk = keys.shape
        Dv = values.shape[3]
        if self.group is None:
            self.group = _group(Dk, Dv)
            self.dims = (Dk, Dv)
        g, bits, prev = self.group, self.kv_bits, self.offset
        if self.keys is None or prev + S > self.keys[0].shape[2]:
            n = (self.step + S - 1) // self.step * self.step
            nk = _alloc(B, H, n, Dk, bits, g, keys.dtype)
            nv = _alloc(B, H, n, Dv, bits, g, keys.dtype)
            if self.keys is None:
                self.keys, self.values = nk, nv
            else:
                self.keys = _cat(_slice(self.keys, 0, prev), nk)
                self.values = _cat(_slice(self.values, 0, prev), nv)
        self.offset += S
        for dst, src in ((self.keys, _q(keys, g, bits)),
                         (self.values, _q(values, g, bits))):
            for d, s in zip(dst, src):
                d[..., prev:self.offset, :] = s
        return (_dq(_slice(self.keys, 0, self.offset), g, bits, keys.dtype),
                _dq(_slice(self.values, 0, self.offset), g, bits,
                    values.dtype))

    @property
    def state(self):
        return (_slice(self.keys, 0, self.offset),
                _slice(self.values, 0, self.offset))

    @state.setter
    def state(self, v):
        self.keys, self.values = (tuple(x) for x in v)
        self.offset = self.keys[0].shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.kv_bits, self.group or 0)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.kv_bits, g = map(int, v)
        self.group = g or None

    def to_quantized(self, *a, **k):
        raise TypeError("already quantized")

    @classmethod
    def merge(cls, caches):
        return BatchQuantKVCache.merge(caches)

    @property
    def nbytes(self):
        return _nbytes(self.keys) + _nbytes(self.values)


class BatchQuantKVCache(BatchKVCache):
    """mlx-lm's BatchKVCache (left-padded rows, one offset per row, shared
    write index), stored quantized. Trim, masks and offsets are the
    parent's; everything that touches the stored arrays is here."""

    step = 256

    def __init__(self, left_padding: List[int], kv_bits: int = 8):
        super().__init__(left_padding)
        self.kv_bits = int(kv_bits)
        self.group = None
        self.dims = None

    def _meta_from(self, o) -> None:
        if self.group is None and getattr(o, "group", None) is not None:
            self.group, self.dims = o.group, o.dims

    def update_and_fetch(self, keys, values):
        B, H, S, Dk = keys.shape
        Dv = values.shape[3]
        if self.group is None:
            self.group = _group(Dk, Dv)
            self.dims = (Dk, Dv)
        g, bits, prev = self.group, self.kv_bits, self._idx
        if self.keys is None or prev + S > self.keys[0].shape[2]:
            n = (self.step + S - 1) // self.step * self.step
            nk = _alloc(B, H, n, Dk, bits, g, keys.dtype)
            nv = _alloc(B, H, n, Dv, bits, g, keys.dtype)
            if self.keys is None:
                self.keys, self.values = nk, nv
            else:
                self.keys = _cat(_slice(self.keys, 0, prev), nk)
                self.values = _cat(_slice(self.values, 0, prev), nv)
        self.offset += S
        self._idx += S
        for dst, src in ((self.keys, _q(keys, g, bits)),
                         (self.values, _q(values, g, bits))):
            for d, s in zip(dst, src):
                d[..., prev:self._idx, :] = s
        return (_dq(_slice(self.keys, 0, self._idx), g, bits, keys.dtype),
                _dq(_slice(self.values, 0, self._idx), g, bits,
                    values.dtype))

    def finalize(self):
        if self._right_padding is not None:
            pad = self._right_padding[:, None]
            self.keys = tuple(dynamic_roll(p, pad, axis=2) for p in self.keys)
            self.values = tuple(dynamic_roll(p, pad, axis=2)
                                for p in self.values)
            self.offset -= self._right_padding
            self.left_padding += self._right_padding
            self._right_padding = None

    @property
    def state(self):
        return (_slice(self.keys, 0, self._idx),
                _slice(self.values, 0, self._idx),
                self.offset, self.left_padding)

    @state.setter
    def state(self, v):
        k, vv, self.offset, self.left_padding = v
        self.keys, self.values = tuple(k), tuple(vv)
        self._idx = self.keys[0].shape[2]

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tuple(p[batch_indices] for p in self.keys)
            self.values = tuple(p[batch_indices] for p in self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        m = self.left_padding.min().item()
        if m > 0:
            if self.keys is not None:
                self.keys = _slice(self.keys, m, None)
                self.values = _slice(self.values, m, None)
            self._idx -= m
            self.left_padding -= m

    def _empty_parts(self, B, H, dt):
        Dk, Dv = self.dims
        g, bits = self.group, self.kv_bits
        return (_alloc(B, H, 0, Dk, bits, g, dt),
                _alloc(B, H, 0, Dv, bits, g, dt))

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding,
                                                other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        self._meta_from(other)
        other._meta_from(self)
        src = self if self.keys is not None else other
        H, dt = src.keys[0].shape[1], src.keys[1].dtype
        max_idx = max(self._idx, other._idx)
        size = max(c.keys[0].shape[2] for c in (self, other)
                   if c.keys is not None)

        def pad(c):
            if c.keys is None:
                k, v = c._empty_parts(c.offset.shape[0], H, dt)
            else:
                k, v = c.keys, c.values
            left = max_idx - c._idx
            right = size - k[0].shape[2] - left
            if right < 0:
                k, v = _slice(k, 0, right), _slice(v, 0, right)
                right = 0
            if left or right:
                w = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = tuple(mx.pad(p, w) for p in k)
                v = tuple(mx.pad(p, w) for p in v)
            return k, v, c.offset, c.left_padding + left

        (k1, v1, o1, l1), (k2, v2, o2, l2) = pad(self), pad(other)
        self.keys = tuple(mx.concatenate([a, b]) for a, b in zip(k1, k2))
        self.values = tuple(mx.concatenate([a, b]) for a, b in zip(v1, v2))
        self.offset = mx.concatenate([o1, o2])
        self.left_padding = mx.concatenate([l1, l2])
        self._idx = max_idx

    def extract(self, idx):
        c = QuantKVCache(self.kv_bits)
        c.group, c.dims = self.group, self.dims
        pad = self.left_padding[idx].item()
        if self.keys is not None:
            c.keys = tuple(mx.contiguous(p[idx:idx + 1, :, pad:self._idx])
                           for p in self.keys)
            c.values = tuple(mx.contiguous(p[idx:idx + 1, :, pad:self._idx])
                             for p in self.values)
            c.offset = c.keys[0].shape[2]
        return c

    @classmethod
    def merge(cls, caches):
        bits = next(c.kv_bits for c in caches)
        lengths = [c.size() for c in caches]
        L = max(lengths)
        if L == 0:
            return cls([0] * len(caches), bits)
        first = next(c for c in caches if c.keys is not None)
        pad = [L - n for n in lengths]
        B = len(caches)
        H = first.keys[0].shape[1]
        keys = tuple(mx.zeros((B, H, L, p.shape[3]), p.dtype)
                     for p in first.keys)
        values = tuple(mx.zeros((B, H, L, p.shape[3]), p.dtype)
                       for p in first.values)
        for i, (p, c) in enumerate(zip(pad, caches)):
            if c.keys is None:
                continue
            for dst, src in ((keys, c.keys), (values, c.values)):
                for d, s in zip(dst, src):
                    d[i:i + 1, :, p:p + c.offset] = s[..., :c.offset, :]
        out = cls(pad, bits)
        out.group, out.dims = first.group, first.dims
        out.keys, out.values = keys, values
        out.offset += L
        out._idx = L
        return out

    @property
    def nbytes(self):
        return _nbytes(self.keys) + _nbytes(self.values)


def quantize_cache_list(caches: list, bits: int) -> tuple:
    """(new list, how many were quantized): each plain mlx-lm KVCache --
    exact type, empty -- becomes a QuantKVCache; a CacheList's members the
    same way. Everything else passes through untouched."""
    n = 0

    def one(c):
        nonlocal n
        if type(c) is KVCache and c.keys is None:
            n += 1
            return QuantKVCache(bits)
        if type(c) is CacheList:
            return CacheList(*(one(s) for s in c.caches))
        return c

    out = [one(c) for c in caches]
    return out, n


def install(model, bits: Optional[int]) -> int:
    """Wrap `model.make_cache` so every new cache list stores attention
    K/V at `bits`. Returns how many layers are quantized (0: none of this
    model's caches are the kind this can quantize -- the caller refuses)."""
    if bits is None:
        return 0
    whole = model.make_cache
    probe, n = quantize_cache_list(whole(), bits)
    if n == 0:
        return 0

    def make_cache():
        return quantize_cache_list(whole(), bits)[0]

    model.make_cache = make_cache
    return n

