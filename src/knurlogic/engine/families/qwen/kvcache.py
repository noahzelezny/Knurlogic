"""qwen4_exp (Flash-Next) attention cache, K/V stored quantized.

The vendored `_AttnCache` / `_BatchAttnCache` (architecture/qwen4_exp.py,
pinned: not edited here) carry the QSA indexer's raw keys and rope
positions beside K/V, and move them column for column with K/V. These
subclasses put engine/kvquant's quantized K/V under that: every indexer
operation is the vendored class's, every K/V operation kvquant's. The
indexer stays exact -- it picks which tokens are attended, so a rounding
there would change the sparse set, not just the values read.

Named by the manifest (`kv_quant.caches`); engine/kvquant.install swaps
each empty `_AttnCache` of a fresh cache list for `QuantAttnCache(bits)`.
The arch module is mlx-lm's registered copy, imported on first use.
"""
from __future__ import annotations

import importlib

import mlx.core as mx

from knurlogic.engine.kvquant import BatchQuantKVCache, QuantKVCache

_ARCH = "mlx_lm.models.qwen4_exp"
_CLASSES: dict = {}


def _classes():
    if _CLASSES:
        return _CLASSES["single"], _CLASSES["batch"]
    Q = importlib.import_module(_ARCH)

    class QuantAttnCache(Q._AttnCache, QuantKVCache):
        """`_AttnCache` over kvquant.QuantKVCache."""

        def __init__(self, kv_bits: int = 8):
            super().__init__()
            self.kv_bits = int(kv_bits)

        @property
        def state(self):
            return (*QuantKVCache.state.fget(self), self.indexer.state)

        @state.setter
        def state(self, v):
            *kv, self.indexer.state = v
            QuantKVCache.state.fset(self, tuple(kv))

        @classmethod
        def merge(cls, caches):
            return BatchQuantAttnCache.merge(caches)

    class BatchQuantAttnCache(Q._BatchAttnCache, BatchQuantKVCache):
        """`_BatchAttnCache` over kvquant.BatchQuantKVCache: its finalize,
        trim, filter and extend run the kvquant parent's K/V step (their
        `super()`) and then their own indexer step."""

        def __init__(self, left_padding, kv_bits: int = 8):
            super().__init__(left_padding)
            self.kv_bits = int(kv_bits)

        def extract(self, idx):
            q = BatchQuantKVCache.extract(self, idx)
            c = QuantAttnCache(self.kv_bits)
            c.keys, c.values, c.offset = q.keys, q.values, q.offset
            c.group, c.dims = q.group, q.dims
            c.kv8_kernel = self.kv8_kernel
            pad = self.left_padding[idx].item()
            if self.indexer.keys is not None:
                c.indexer.keys = mx.contiguous(
                    self.indexer.keys[idx:idx + 1, pad:self._idx])
            if self.indexer.pos is not None:
                c.indexer.pos = mx.contiguous(
                    self.indexer.pos[idx:idx + 1, pad:self._idx])
            return c

        @classmethod
        def merge(cls, caches):
            if max(c.size() for c in caches) == 0:
                out = cls([0] * len(caches), caches[0].kv_bits)
                out.kv8_kernel = any(c.kv8_kernel for c in caches)
                return out
            # the vendored merge's super() is BatchQuantKVCache.merge here,
            # which builds `cls(padding, bits)`; the indexer rows follow
            return Q._BatchAttnCache.merge.__func__(cls, caches)

        @property
        def state(self):
            return (*BatchQuantKVCache.state.fget(self), self.indexer.state)

        @state.setter
        def state(self, v):
            *kv, self.indexer.state = v
            BatchQuantKVCache.state.fset(self, tuple(kv))

    _CLASSES.update(single=QuantAttnCache, batch=BatchQuantAttnCache)
    return QuantAttnCache, BatchQuantAttnCache


def QuantAttnCache(bits: int):
    """A fresh qwen4_exp attention cache storing K/V at `bits`."""
    return _classes()[0](bits)

