"""Dense VQ modules — VQLinear and VQEmbedding.

WHY THESE EXIST. vq_switch.VQSwitchLinear is EXPERT-shaped: its __call__
takes routing indices and its kernels index a leading expert axis. A dense
model (gemma-4-e4b) has neither, so shipping a VQ e4b needs drop-ins for
plain nn.Linear and nn.Embedding.

THE EMBEDDING IS THE GOOD CASE. A [262144, 10752] table is 35.5% of e4b's
bytes, and an embedding lookup only ever needs the ROWS a batch touches —
so VQEmbedding gathers those code rows and decodes just those, never
materialising the table. Memory saving with no throughput cost.

THE LINEAR IS THE HONEST CASE. A matmul needs the whole weight, so
VQLinear decodes W per call: codebook[codes] -> [OUT, NSUB, D] ->
[OUT, IN], scaled group-wise. That trades compute for resident memory and
will be SLOWER than an 8-bit matmul; measure before believing any speed
claim. Correctness first — a fused dense kernel is a later optimisation,
and per E62 the fused work has to reproduce the scored numbers exactly
before it may replace this path.

Both mirror the fitter's contract exactly: scales are fp16 max-abs per
group of G along `in`, codes index a [K, D] fp16 codebook.
"""
import os

import mlx.core as mx
import mlx.nn as nn


# Rows below which the fused kernel beats decode-the-weight-then-GEMM.
#
# This is TWO numbers, because the decode path's cost depends entirely on
# whether the codes are packed. Measured 2026-09-01 on the 27B mlp shape
# (OUT 17408, IN 5120, d=2), ms per call:
#
#   packed (K512/bits9)   N      1     8    32    68   128   256
#                         fused  0.6   1.4   4.2   8.3  14.9  29.2
#                         decode 9.6   9.5   9.7  10.2  10.2  11.3   -> cross ~100
#
#   unpacked (K256)       N      1     8    16    32    68    96
#                         fused  0.5   1.2   2.2   3.9   7.9  11.1
#                         decode 1.7   1.5   1.7   1.7   2.3   2.3   -> cross ~12
#
# Packed decode is ~9.5 ms flat because _unpack_rows dominates it; unpacked
# decode is 1.65 ms flat. So the fused path stays ahead six times longer when
# codes are packed.
#
# Allocation CHURN is the other reason -- though NOT the emergency it first
# looked like, and the correction is worth keeping. The decode path allocates
# and frees a ~170 MB fp16 weight per VQ linear, so mx.get_peak_memory(),
# which is a cumulative high-water counter of transient allocations, reads
# 37.9 GiB at 2048 context on a 14.6 GiB model. That is not resident memory:
# the same run completes under a 22 GiB mx.set_memory_limit cap, and process
# RSS stays at 15.2 GiB (12.2 / 14.2 / 15.2 for the 3.9 / 4.5 / 4.8 rungs,
# i.e. each runs in its own size).
#
# So the fused path wins here on speed and churn, not on survival. Never
# quote get_peak_memory() as a RAM requirement -- measure RSS. Doing the
# former produced a false alarm that briefly went into a public model card.
_DENSE_FUSED_MAX_N_PACKED = int(os.environ.get("VQ_DENSE_FUSED_MAX_N", 96))
_DENSE_FUSED_MAX_N_PLAIN = int(os.environ.get("VQ_DENSE_FUSED_MAX_N_PLAIN", 12))

# THE PACKED CUTOFF IS GEOMETRY-DEPENDENT, and 96 was a d=2 number.
#
# The table above was measured on the 27B **d=2** mlp shape (K512/bits9,
# i.e. the 4.8bpw rung). The 3.9bpw rung is d=4/K=4096/packed-12 and got a
# dense kernel only later, so it inherited a cutoff nothing had measured for
# it -- and at d=4 the fused kernel does 4 codes per step against a 64 KB
# DEVICE codebook, which is a different cost curve entirely.
#
# Re-measured per rung with BOTH current paths (scripts/bench_dense_crossover.py,
# real L20 gate_proj, ms/call min, fused/decode ratio):
#
#     N            32     64     96    128
#     3.9 (d4)    0.90   1.62   2.27   2.85    crossover ~35
#     4.8 (d2p9)  0.45   0.87   1.20   1.61    crossover ~72
#     4.5 (d2)    1.47   2.77   3.46   4.31    crossover ~20
#
# So at d=4 the shipped 96 ran the fused kernel 2.27x SLOWER than decoding
# the weight over the whole band N=36..96; at d=2 packed it was mildly high;
# and the unpacked default of 12 is conservative but safe (crossover ~20).
#
# Keyed on d, because that is what the measurement says the curve depends
# on. VQ_DENSE_FUSED_MAX_N still overrides, and still wins, for A/B.
#
# NOTE ON NUMERICS. The fused and decode paths are NOT bit-identical to each
# other and never were (the fused kernel keeps an fp32 accumulator per scale
# group; the decode path materialises fp16 weights and hands them to the
# GEMM) -- only fused-vs-fused and decode-vs-decode identity is claimed
# anywhere. Lowering the cutoff therefore moves the N=36..96 band from the
# fused path onto the DECODE path, which is the path every published score
# for these artifacts was produced on. The change moves those shapes toward
# the scored numerics, not away from them.
_DENSE_FUSED_MAX_N_BY_D = {2: 72, 4: 32}


def _fused_max_n(pack_bits, d):
    if not pack_bits:
        return _DENSE_FUSED_MAX_N_PLAIN
    if "VQ_DENSE_FUSED_MAX_N" in os.environ:
        return _DENSE_FUSED_MAX_N_PACKED
    return _DENSE_FUSED_MAX_N_BY_D.get(d, _DENSE_FUSED_MAX_N_PACKED)


def _resolve_kernel(name):
    """Find the fused kernel, most-local copy first.

    Order: (1) this module's own globals — the BUNDLED case, where model.py
    concatenates vq_switch + vq_dense and the kernel is already here;
    (2) a sibling vq_switch module — the PACKAGE / same-directory case, which
    works in a completely stock venv; (3) mlx_lm.models.vq_switch — a
    VQ-PATCHED install only. The old order tried (3) directly after (1) and
    therefore worked on every lab machine and failed on a stock one — found
    the first time the selftest ran in a genuinely fresh venv.
    """
    f = globals().get(name)
    if f is not None:
        return f
    import importlib
    for mod in ("vq_switch", "vqlab.vq_switch", "mlx_lm.models.vq_switch"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(
        f"no source for {name}: not bundled alongside this module, no sibling "
        f"vq_switch importable, and mlx_lm is not VQ-patched")


def _decode(codes, codebook, scales, group_size, out_d, in_d):
    """codes [R, NSUB] -> dense [R, in_d] fp16, scaled group-wise."""
    w = codebook[codes.reshape(-1)].reshape(out_d, in_d)
    w = (w.reshape(out_d, in_d // group_size, group_size)
         * scales[..., None]).reshape(out_d, in_d)
    return w


def _unpack_rows(packed, nsub, bits):
    """mlx-native inverse of vq_pack.pack for a [R, WPR] uint32 slab.

    Runs on-GPU at gather time so packed EMBEDDING rows never round-trip
    through numpy. Same 32-codes-per-block layout as vq_pack.py; verified
    against vq_pack.unpack bit-exactly before first use.

    MEMORY (2026-09-01). This is called on the PREFILL path, once per VQ
    linear, and mlx builds the whole forward before evaluating it -- so every
    layer's temporaries are alive at once and this function sets the peak.
    The original cast the slab to uint64 (8 bytes per element) and returned
    uint32, which cost ~630 MB of transient per 27B mlp layer against the
    unpacked path's 178 MB. Across 64 layers that is the difference between a
    15.9 GB peak and a 30.0 GB one: the 4.8bpw artifact needed twice its own
    weight in RAM, which defeats the entire point of packing and can OOM a
    machine the model card says it fits on.

    Two changes, both bit-preserving:
      - stay in uint32. bits <= 16 always, so a field either lies inside one
        word or straddles two, and both cases are exact in 32-bit as long as
        the shift that would be >= 32 is never evaluated (see below).
      - return the narrowest dtype the codebook needs, not always uint32.

    The `sh + bits > 32` branch is why the uint64 was there: `32 - sh` is a
    32-bit shift by 32 when sh == 0, which is undefined. sh == 0 also means
    the field cannot straddle (bits <= 32), so that branch is unreachable
    then, and it is a Python-level `if` on compile-time constants here rather
    than a data-dependent select -- the bad shift is never emitted at all.
    """
    R, wpr = packed.shape
    nblk = nsub // 32
    src = packed.reshape(R, nblk, bits)
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    outs = []
    for i in range(32):
        off = i * bits
        w, sh = divmod(off, 32)
        v = src[:, :, w] >> mx.array(sh, dtype=mx.uint32) if sh else src[:, :, w]
        if sh + bits > 32:
            # sh > 0 here, so 32 - sh is in [1, 31] and the shift is defined.
            v = v | (src[:, :, w + 1] << mx.array(32 - sh, dtype=mx.uint32))
        outs.append(v & mask)
    ct = mx.uint8 if bits <= 8 else mx.uint16
    # outs[i] is code i of every block -> interleave back to row order
    return mx.stack(outs, axis=-1).reshape(R, nsub).astype(ct)


# PREFILL PEAK: the transient that made a 11.6 GiB model need 18.7 GiB.
#
# THE MEASUREMENT (Noah, 2026-09-03, 27B 3.9bpw, 2048-token prompt, stock
# mlx-lm): peak 18.7 G against 11.6 G active. The ~7 G is this file's
# large-N path -- above _DENSE_FUSED_MAX_N_PACKED every VQLinear leaves the
# fused kernel and materialises a decoded fp16 weight (17408 x 5120 x 2 B =
# 178 MB per mlp linear, and the packed rungs allocate an unpacked code slab
# on the way there too).
#
# WHY IT IS NOT 178 MB. mlx is LAZY: it builds the entire forward graph
# before evaluating any of it, so nothing forces layer L's decoded weight to
# be freed before layer L+1's is allocated. Across 64 layers x 3 linears the
# live set is bounded only by how far ahead the graph is built -- which is
# how 178 MB of legitimate working memory becomes ~7 G of peak.
#
# WHY PEAK IS THE BUDGET, NOT RSS. The earlier note in this file (correctly)
# said not to quote get_peak_memory() as a RAM requirement, because a single
# model in isolation completes under a much smaller cap. That still holds for
# the single-model case. It does NOT hold for Noah's use case: several models
# resident at once, where every model's peak is charged against the same
# machine at the same time. There, "space-saving model" has to mean
# peak ~= resident + KV + a small BOUNDED buffer, so this path has to have a
# fixed transient budget rather than a graph-depth-dependent one.
#
# THE FIX IS THE FORCED EVAL, NOT THE TILING -- and that distinction cost a
# wrong claim, so it is written down. The plan was to decode in OUT-row tiles
# under a byte budget, on the reasoning that splitting along OUT reorders no
# reduction: output element (n, o) is the same length-IN dot product
# whichever tile row o lands in. That reasoning is correct about the MATH and
# wrong about MLX. Measured: at 51-row tiles the output bits DIFFER from the
# untiled path (tests/test_vq_dense_peak.py caught it), because mlx's GEMM
# chooses how to split its own K reduction based on the operand SHAPE -- so
# changing the output width changes the accumulation order inside the matmul,
# even though nothing in this file reordered anything. Row tiling is
# therefore NOT bit-identical in general, and whether it happens to be at any
# given shape is luck (it holds at 64 MB on the real 27B shapes; it does not
# at 51 rows).
#
# So the DEFAULT is the part that IS provably bit-identical: decode the whole
# weight exactly as before -- one tile, one GEMM, same shapes, same bits --
# and simply mx.eval the result before returning. Forcing evaluation cannot
# change a value; it only decides WHEN the graph runs. That alone caps the
# live set at ONE layer's transient instead of the whole forward's, which is
# 13x of the 14x that was available.
#
# Row tiling stays reachable (VQ_DENSE_DECODE_TILE_MB > 0) for anyone who
# needs a harder, width-independent bound and can accept a GEMM-shape
# dependent last bit. It is OFF by default, like every other
# non-bit-identical option in this codebase.
#
# MEASURED (scripts/bench_dense_prefill_peak.py, 48 REAL VQLinears from the
# 3.9bpw artifact, N=2048, transient = peak - resident, every arm checked
# against arm 0 by uint16 BIT PATTERN -- not array_equal, which reports
# NaN != NaN on a deep unnormalised chain):
#
#     tile_MB       0     512     256      64      16
#     transient  6.811G  0.519G  0.519G  0.390G  0.227G
#     wall       1.072s  1.054s  1.058s  1.057s  1.378s
#     bits         ref    same    same    same    same   (at THESE shapes)
#
# and the old path's transient GROWS WITH DEPTH -- 2.758 G at 16 linears,
# 6.811 G at 48 -- exactly as the lazy-graph diagnosis predicts, while every
# forced-eval arm is FLAT in depth. The 6.8 G at 48 linears is the ~7 G Noah
# measured on the real 192-linear forward.
#
# DEFAULTS: eval ON, row tiling OFF. The eval is free (1.054 vs 1.072 s --
# it pays for itself by not building a 48-layer-deep graph) and takes the
# transient from 6.811 G to 0.519 G, i.e. 13.1x of the 30x that row tiling
# down to 16 MB would reach. The remaining factor is not worth a bit that
# depends on which GEMM mlx picked, and the 16 MB arm is also where the sync
# cost finally bites (+30%).
#
# VQ_DENSE_DECODE_EVAL=0 restores the fully lazy pre-arc path for A/B.
# VQ_DENSE_DECODE_TILE_MB=<mb> additionally row-tiles to a hard byte budget:
# width-independent, but NOT bit-identity-guaranteed (see above).
_DENSE_DECODE_EVAL = os.environ.get("VQ_DENSE_DECODE_EVAL", "1") != "0"
_DENSE_DECODE_TILE_MB = float(
    os.environ.get("VQ_DENSE_DECODE_TILE_MB", 0))


def _decode_matmul(xf, codes_in, codebook, scales, group_size, OUT, IN,
                   pack_bits):
    """xf @ decode(codes).T, with a bounded transient. Returns [N, OUT].

    Default arm (eval on, no row tiling) is BIT-IDENTICAL to the pre-arc
    path: identical shapes, identical GEMM, only the evaluation TIME differs.
    """
    cbk = codebook.astype(mx.float16)
    budget = int(_DENSE_DECODE_TILE_MB * (1 << 20))
    row_bytes = IN * 2
    D = int(codebook.shape[1])

    def _tile(r0, r1):
        if pack_bits:
            wdec_fits = _resolve_kernel("wdec_fits")
            if wdec_fits(D, IN // D, group_size, pack_bits):
                # One kernel replaces slab + gather + scale-mul; bit-identity
                # against the graph arm is gated by tests/test_vq_wdec.py.
                # Full codes/scales tensors — the row base rides in dims.
                w = _resolve_kernel("wdec_decode")(
                    codes_in, cbk, scales, r1 - r0, IN, group_size,
                    pack_bits, r0=r0)
                return xf @ w.T.astype(xf.dtype)
        c = codes_in[r0:r1]
        if pack_bits:
            c = _unpack_rows(c, IN // D, pack_bits)
        w = _decode(c, cbk, scales[r0:r1], group_size, r1 - r0, IN)
        return xf @ w.T.astype(xf.dtype)

    if budget <= 0 or row_bytes * OUT <= budget:
        # ONE tile -- the whole weight, exactly the shapes the pre-arc path
        # used, so exactly its bits. The mx.eval is the entire mechanism:
        # it caps the live set at one layer's transient.
        y = _tile(0, OUT)
        if _DENSE_DECODE_EVAL:
            mx.eval(y)
        return y
    # Row-tiled arm: harder bound, and the output width per GEMM changes, so
    # the last bit can move (see the block comment). Opt-in only.
    tile_rows = max(1, budget // row_bytes)
    parts = []
    for r0 in range(0, OUT, tile_rows):
        p = _tile(r0, min(r0 + tile_rows, OUT))
        mx.eval(p)          # frees this tile's decoded weight before the next
        parts.append(p)
    return mx.concatenate(parts, axis=-1)


class VQLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear whose weight is VQ-coded.

    codes may be unpacked ([OUT, NSUB] uint8/uint16) or packed
    ([OUT, WPR] uint32 of pack_bits-wide fields, vq_pack.py layout).
    """

    def __init__(self, codes, codebook, vq_scales, group_size=64,
                 pack_bits=0, in_features=None):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features")
        self._in_features = in_features
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @property
    def input_dims(self):
        if self.pack_bits:
            return self._in_features
        return self.codes.shape[-1] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[0]

    def __call__(self, x):
        # Same sharding guard as VQSwitchLinear: a sliced codebook does not
        # error, it silently decodes garbage (exo PR #2268).
        if int(self.codebook.shape[0]) != self._k_expect:
            raise RuntimeError(
                f"VQ codebook was sharded: K={self.codebook.shape[0]}, "
                f"expected {self._k_expect}. The codebook is a SHARED lookup "
                f"table and must be REPLICATED across tensor-parallel ranks.")
        OUT = self.codes.shape[0]
        IN = (self._in_features if self.pack_bits
              else self.codes.shape[1] * self.codebook.shape[1])
        # A dense linear IS an expert layer with E=1 and every token routed
        # to expert 0 — so the small-N path rides the PRODUCTION-VALIDATED
        # fused kernels from vq_switch (E62: bit-identical, ~3x decode)
        # instead of decoding the whole weight per call, which measured
        # 11.5 tok/s vs the incumbent's 84 (2026-08-19). Prefill (large N)
        # amortises one full decode across the whole batch, which is faster
        # than the fused gather at that shape — same split vq_switch uses.
        if self.pack_bits:
            IN = self._in_features
        orig_shape = x.shape
        xf = x.reshape(-1, IN)
        N = xf.shape[0]
        # d=2 is the only geometry with a dense fused kernel. The expert
        # kernel (_fused, E=1) was tried as the d!=2 small-N path and DIED AT
        # KERNEL LOAD on the first real dense model: at 27B mlp shapes
        # (IN 17408) its threadgroup allocation is 36864 bytes vs Metal's
        # 32768 cap — shapes the 397B's experts never produce. Caught by
        # III.10 smoke-gen (E95). Until a dense kernel exists for d>2,
        # small-N takes the decode path: bit-exact, just slow at decode
        # (~11.5 tok/s measured 08-19) — fine for gates and prefill-shaped
        # scoring, NOT a shipping configuration.
        # THREADGROUP CEILING (2026-08-21, E124). The d=2 dense kernel ALSO
        # dies at kernel load on 27B mlp shapes: NSUB = IN/d = 8704 needs
        # 35840 B of threadgroup memory against Metal's 32768 cap. So neither
        # d=2 NOR d>2 has a usable fused dense path at this model's widths —
        # the 397B's expert shapes simply never produce them. Measured, not
        # inferred: the failure is a RuntimeError naming both numbers.
        # Falling back to decode keeps the artifact CORRECT (bit-exact, the
        # same path that produced its scores) at decode speed. This is not a
        # performance fix and must not be read as one; a dense kernel that
        # fits the cap is the real fix and does not exist yet.
        # THREADGROUP BUDGET — must count the CODEBOOK too (E134 addendum,
        # 2026-08-22). Both dense d2 kernels cache `threadgroup half2
        # cb[MAX_K]` AND `threadgroup half2 xs[MAX_NSUB]`, i.e. 4 bytes per
        # entry each, so the requirement is (K + NSUB) * 4 + overhead. The
        # original guard counted only NSUB and would have reported "fits" for
        # a large-K artifact that then died at KERNEL LOAD — the same failure
        # the guard exists to prevent, and the same defect E134 fixed on the
        # MoE side, where a K>=4096 d4 artifact scored normally and could not
        # generate a token. Does not bite at any rung on disk today
        # (d2/K4096 at NSUB 2560 needs 26,624 B) and would have bitten at
        # d2/K8192.
        # NSUB TILING (2026-09-01). The budget above is the UNTILED
        # kernel's, and it grows with layer width: at 27B mlp shapes it
        # reports "too big" and every forward then decoded the whole weight,
        # which measured 0.43 tok/s against stock's 16.7. The tiled kernels
        # stage one block span instead of the whole x row, so their budget is
        # (K + 32*(G/2)) * 4 and does not depend on width at all. Ask
        # vq_switch whether EITHER kernel can serve the shape; it picks.
        # Bit-exactness against the untiled path is asserted in
        # tests/test_vq_dense_tiled.py. Note this gate is DECODE ONLY
        # (N <= 32): scoring and prefill are large-N and take the
        # decode-to-dense path below, which is untouched, so no published
        # quality number depends on anything here.
        _kk = int(self.codebook.shape[0])
        _G = IN // int(self.vq_scales.shape[-1])
        _fits_tg = _resolve_kernel("dense_fits")(
            _kk, IN, _G, int(self.codebook.shape[1]))
        # d is NOT re-checked here: dense_fits owns which geometries
        # have a kernel (d=2 tiled/untiled, d=4 device-codebook), and
        # duplicating that as `== 2` here is what kept the d4 rung on
        # the decode path after its kernel existed.
        _maxn = _fused_max_n(self.pack_bits, int(self.codebook.shape[1]))
        if N <= _maxn and _fits_tg:
            # DENSE fused kernel (2026-08-19): one simdgroup per output row
            # instead of one thread, no expert axis. Bit-identical to the
            # expert-kernel path below (the kernel replicates its float
            # ordering exactly — verified, and the E62 KL number reproduces).
            # VQ_DENSE_REF=1 keeps the old expert-shaped path callable
            # as the reference for A/B checks.
            if os.environ.get("VQ_DENSE_REF") and self.pack_bits == 0 \
                    and int(self.codebook.shape[1]) == 2:
                _fused = _resolve_kernel("_fused")
                eidx = mx.zeros((N,), dtype=mx.uint32)
                y = _fused(xf, eidx, self.codes[None], self.codebook,
                           self.vq_scales[None], pack_bits=self.pack_bits)
            else:
                _dense_fused = _resolve_kernel("_dense_fused")
                y = _dense_fused(xf, self.codes, self.codebook,
                                 self.vq_scales, pack_bits=self.pack_bits,
                                 in_features=self._in_features)
            return y.astype(x.dtype).reshape(*orig_shape[:-1], OUT)
        y = _decode_matmul(xf, self.codes, self.codebook, self.vq_scales,
                           self.group_size, OUT, IN, self.pack_bits)
        return y.reshape(*orig_shape[:-1], OUT)


class VQEmbedding(nn.Module):
    """Drop-in for nn.Embedding whose table is VQ-coded.

    Decodes ONLY the gathered rows — the whole point of putting VQ on a
    262k-row table. as_linear() is provided because gemma ties the output
    head to the input embedding in some configs; it decodes the full table
    and is deliberately NOT used by the PLE path.
    """

    def __init__(self, codes, codebook, vq_scales, group_size=64,
                 pack_bits=0, in_features=None, out_dtype=mx.bfloat16):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        # The decode runs in fp16 (fp16 codebook gather x fp16 scales) for
        # speed and footprint, but the RESULT has to leave in the model's
        # dtype. nn.Embedding, which this replaces, returns its weight dtype.
        #
        # Returning fp16 into a bf16 stack is not a rounding detail: mlx
        # promotes bf16 (+,*) fp16 to FLOAT32, so one fp16 PLE output turns
        # every downstream activation -- h, q/k/v, and the attention mask --
        # fp32 from that layer on. On gemma4 that lands in
        # mx.fast.scaled_dot_product_attention at head_dim 256/512 in fp32
        # with an fp32 mask, whose steel_attention kernel wants 53760 B of
        # threadgroup memory against Metal's 32768 B limit -- so generation
        # dies with "Unable to load kernel" as soon as the prompt is long
        # enough to leave the vector kernel. VQPLEEmbedding (vq_switch) has
        # always cast on the way out; this is the same rule, and the dense
        # PLE path is the one that lost it.
        self.out_dtype = out_dtype
        self.group_size = group_size
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features")
        self._in_features = in_features
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @property
    def num_embeddings(self):
        return self.codes.shape[0]

    @property
    def dims(self):
        return self.codes.shape[1] * self.codebook.shape[1]

    def __call__(self, ids):
        if int(self.codebook.shape[0]) != self._k_expect:
            raise RuntimeError("VQ codebook was sharded; must be replicated.")
        flat = ids.reshape(-1)
        rows = self.codes[flat]                       # [N, NSUB] or [N, WPR]
        sc = self.vq_scales[flat]                     # [N, in/G]
        D = self.codebook.shape[1]
        if self.pack_bits:
            NSUB = self._in_features // D
            rows = _unpack_rows(rows, NSUB, self.pack_bits)
        N, NSUB = rows.shape
        IN = NSUB * D
        w = self.codebook.astype(mx.float16)[rows.reshape(-1)].reshape(N, IN)
        w = (w.reshape(N, IN // self.group_size, self.group_size)
             * sc[..., None]).reshape(N, IN)
        return w.reshape(*ids.shape, IN).astype(self.out_dtype)

    def as_linear(self, x):
        OUT = self.codes.shape[0]
        IN = self.dims
        w = _decode(self.codes, self.codebook.astype(mx.float16),
                    self.vq_scales, self.group_size, OUT, IN)
        return x @ w.T.astype(x.dtype)
