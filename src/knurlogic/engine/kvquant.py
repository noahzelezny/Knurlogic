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
the prompt cache's entries). Dequantizing per layer per step is MORE
memory traffic than bf16, so at 8 bits a decode step (query length 1)
instead goes through engine/kvattn's Metal kernel, which reads the packed
K/V directly. The dequantized arrays still returned beside it are lazy:
computed only if something reads them -- which gemma4's KV-shared layers
do (they attend over the owner layer's returned keys with cache=None), so
there the copy is still made for those layers. M4, Qwen3.6-35B-A3B, decode tok/s bf16 / dequantize / kernel:
70.0 / 63.9 / 66.3 at 6k context, 65.5 / 54.0 / 61.9 at 16k. Prefill and
6/4 bits stay dequantize + sdpa.

What is quantized: mlx-lm's plain `KVCache` (exact type) in the list
`model.make_cache()` returns, and inside a CacheList; and a family's own
cache class its manifest names (`kv_quant.caches`): qwen4_exp's
attention cache (K/V quantized, the sparse indexer's keys kept exact) and
GLM's MLA latent (quantized; the DSA indexer's cache kept exact). Not:
recurrent state (ArraysCache: deltanet/SSM), sliding windows
(RotatingKVCache, bounded by their window), or the MTP head's draft cache
(one layer; the draft stays bf16). Rollback and prompt-cache
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


#: query lengths up to this remember their packed K/V for the 8-bit
#: decode kernel (engine/kvattn). 1: decode steps only -- an MTP verify
#: (2-4 rows per sequence) measured 1.2-1.7x SLOWER through the kernel than
#: dequantize+sdpa on the M4 (tools/kv8/tune.py, L=2/4), so it stays there.
KERNEL_MAX_QUERY = 1


def _fetch(c, n, S, kdt, vdt):
    """The dequantized K/V `update_and_fetch` returns (lazy: never
    computed if the attention takes the kernel instead), and, for a short
    8-bit query with the kernel on, what engine/kvattn.sdpa needs."""
    g, bits = c.group, c.kv_bits
    K, V = _slice(c.keys, 0, n), _slice(c.values, 0, n)
    k, v = _dq(K, g, bits, kdt), _dq(V, g, bits, vdt)
    if getattr(c, "kv8_fetch", None) is not None:
        # handed out last step and never taken: its attention did not go
        # through kvattn.sdpa (it calls mx.fast directly, or a family's own)
        from knurlogic.engine import kvattn
        kvattn.count(False, "the attention never asked for it")
    c.kv8_fetch = None
    if c.kv8_kernel and bits == 8 and S <= KERNEL_MAX_QUERY:
        c.kv8_fetch = (k, K, V, g)
    return k, v


def _carry(out, caches):
    """`out` takes the kernel setting of the caches it was built from."""
    out.kv8_kernel = any(getattr(c, "kv8_kernel", False) for c in caches)
    return out


def _nbytes(parts) -> int:
    return sum(int(p.nbytes) for p in parts) if parts is not None else 0


class QuantKVCache(KVCache):
    """One sequence's attention cache, stored quantized. `keys`/`values`
    are (packed, scales, biases) triples; `update_and_fetch` returns
    dequantized arrays of the dtype it was given."""

    step = 256
    #: this cache hands its packed K/V to engine/kvattn's decode kernel:
    #: set per cache by `install`'s make_cache, carried by merge / extract /
    #: extend (never process-global: two models in one process each keep
    #: their own setting)
    kv8_kernel = False
    kv8_fetch = None

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
        return _fetch(self, self.offset, S, keys.dtype, values.dtype)

    @property
    def state(self):
        return (_slice(self.keys, 0, self.offset),
                _slice(self.values, 0, self.offset))

    @state.setter
    def state(self, v):
        self.keys, self.values = (tuple(x) for x in v)
        self.offset = self.keys[0].shape[2]

    # the kernel flag rides in meta_state: a cache rebuilt by mlx-lm's
    # from_state (cls.__new__, no make_cache) came back with it off -- the
    # silent dequantize path, not even counted as a kernel miss
    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.kv_bits, self.group or 0,
                               int(bool(self.kv8_kernel)))))

    @meta_state.setter
    def meta_state(self, v):
        v = tuple(map(int, v))
        self.offset, self.kv_bits, g = v[:3]
        self.group = g or None
        self.kv8_kernel = bool(v[3]) if len(v) > 3 else False

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
    #: this cache hands its packed K/V to engine/kvattn's decode kernel:
    #: set per cache by `install`'s make_cache, carried by merge / extract /
    #: extend (never process-global: two models in one process each keep
    #: their own setting)
    kv8_kernel = False
    kv8_fetch = None

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
        return _fetch(self, self._idx, S, keys.dtype, values.dtype)

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

    @property
    def meta_state(self):
        # as QuantKVCache's: what from_state needs beyond the arrays,
        # the kernel flag included
        dk, dv = self.dims or (0, 0)
        return tuple(map(str, (self.kv_bits, self.group or 0, dk, dv,
                               int(bool(self.kv8_kernel)))))

    @meta_state.setter
    def meta_state(self, v):
        bits, g, dk, dv, k = (tuple(map(int, v)) + (0,) * 5)[:5]
        self.kv_bits, self.group = bits or 8, g or None
        self.dims = (dk, dv) if dk else None
        self.kv8_kernel = bool(k)
        if not hasattr(self, "_right_padding"):
            self._right_padding = None     # from_state skips __init__

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
        self.kv8_kernel = self.kv8_kernel or other.kv8_kernel
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
        c.kv8_kernel = self.kv8_kernel
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
            return _carry(cls([0] * len(caches), bits), caches)
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
        out = _carry(cls(pad, bits), caches)
        out.group, out.dims = first.group, first.dims
        out.keys, out.values = keys, values
        out.offset += L
        out._idx = L
        return out

    @property
    def nbytes(self):
        return _nbytes(self.keys) + _nbytes(self.values)


def family_caches() -> dict:
    """{(arch, class name): factory(bits)}: the family cache classes a
    manifest says how to quantize (`kv_quant.caches`, "module:attr")."""
    import importlib

    from knurlogic.engine import families
    out = {}
    for arch, spec in families.build_maps()["kv_quant"].items():
        for name, where in (spec.get("caches") or {}).items():
            mod, attr = where.split(":")
            out[(arch, name)] = (lambda m, a: lambda bits: getattr(
                importlib.import_module(m), a)(bits))(mod, attr)
    return out


def _family_factory(c, table):
    """The factory for `c`'s exact class: its name, and an arch that is a
    component of its module (mlx_lm.models.qwen4_exp, or
    ...glm5_next.language)."""
    t = type(c)
    parts = t.__module__.split(".")
    for (arch, name), make in table.items():
        if name == t.__name__ and arch in parts:
            return make
    return None


def quantize_cache_list(caches: list, bits: int, table=None) -> tuple:
    """(new list, how many were quantized): each plain mlx-lm KVCache --
    exact type, empty -- becomes a QuantKVCache; a family cache class a
    manifest names (qwen4_exp's attention cache, GLM's MLA latent) becomes
    what its factory builds; a CacheList's members the same way (mlx-lm's
    or the vendored one, by its `caches`). Everything else passes through
    untouched."""
    n = 0
    table = family_caches() if table is None else table

    def one(c):
        nonlocal n
        if type(c) is KVCache and c.keys is None:
            n += 1
            return QuantKVCache(bits)
        make = _family_factory(c, table)
        if make is not None and getattr(c, "keys", None) is None:
            n += 1
            return make(bits)
        if isinstance(getattr(c, "caches", None), (tuple, list)):
            return type(c)(*(one(s) for s in c.caches))
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
    table = family_caches()
    probe, n = quantize_cache_list(whole(), bits, table)
    if n == 0:
        return 0

    kernel = False
    if bits == 8:
        from knurlogic.engine import kvattn
        kernel = kvattn.enabled() and kvattn.patch_model(model) > 0
        kvattn.reset(kernel)
    model.kv8_kernel = kernel

    def make_cache():
        out = quantize_cache_list(whole(), bits, table)[0]
        if kernel:
            mark_kernel(out)
        return out

    model.make_cache = make_cache
    return n


def mark_kernel(caches) -> None:
    """Let kvquant's caches (family subclasses included) in `caches` hand
    their packed K/V to the 8-bit decode kernel (engine/kvattn)."""
    for c in caches:
        if isinstance(c, (QuantKVCache, BatchQuantKVCache)):
            c.kv8_kernel = True
        elif isinstance(getattr(c, "caches", None), (tuple, list)):
            mark_kernel(c.caches)
