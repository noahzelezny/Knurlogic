import mlx.core as mx
from mlx_lm.models.cache import KVCache
from knurlogic.engine import kvquant as kq
class Grow(kq.QuantKVCache):
    """write path: grow capacity geometrically, write the 3 parts per K/V"""
    def update_and_fetch(self, keys, values):
        B, H, S, Dk = keys.shape; Dv = values.shape[3]
        if self.group is None: self.group = kq._group(Dk, Dv); self.dims = (Dk, Dv)
        g, bits, prev = self.group, self.kv_bits, self.offset
        if self.keys is None or prev + S > self.keys[0].shape[2]:
            want = max(prev + S, prev * 3 // 2)
            n = (want - prev + self.step - 1) // self.step * self.step
            nk = kq._alloc(B, H, n, Dk, bits, g, keys.dtype); nv = kq._alloc(B, H, n, Dv, bits, g, keys.dtype)
            if self.keys is None: self.keys, self.values = nk, nv
            else:
                self.keys = kq._cat(kq._slice(self.keys, 0, prev), nk); self.values = kq._cat(kq._slice(self.values, 0, prev), nv)
        self.offset += S
        for dst, src in ((self.keys, kq._q(keys, g, bits)), (self.values, kq._q(values, g, bits))):
            for d, s in zip(dst, src): d[..., prev:self.offset, :] = s
        return self._fetch(keys.dtype)
    def _fetch(self, dt):
        g, bits = self.group, self.kv_bits
        return (kq._dq(kq._slice(self.keys, 0, self.offset), g, bits, dt), kq._dq(kq._slice(self.values, 0, self.offset), g, bits, dt))
class GrowQDec(Grow):
    """plus: decode (S==1) hands back the quantized triples for quantized_matmul sdpa"""
    def _fetch(self, dt):
        if self._S == 1:
            return kq._slice(self.keys, 0, self.offset), kq._slice(self.values, 0, self.offset)
        return super()._fetch(dt)
    def update_and_fetch(self, k, v):
        self._S = k.shape[2]; return super().update_and_fetch(k, v)
MAKERS = {"grow": Grow, "growqdec": GrowQDec}
def _bits(self):
    if getattr(self, "_S", 0) != 1: raise AttributeError("bits")
    return self.kv_bits
GrowQDec.bits = property(_bits)
GrowQDec.group_size = property(lambda s: s.group)
