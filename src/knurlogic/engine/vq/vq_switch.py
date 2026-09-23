# quantlab M1d — VQ expert runtime for mlx_lm.
# Canonical copy lives in quantlab/vq_switch.py; patch_mlx_lm.py installs it
# as mlx_lm/models/vq_switch.py and hooks load_model. Keep both in sync by
# re-running the patcher, never by editing the installed copy.
#
# Format (per VQ'd expert tensor, produced by vq_*_codes fitters):
#   {p}.codes      uint8 (K<=256) / uint16   [E, out, in/d]
#   {p}.codebook   fp16                      [K, d]      (d=2, 4 or 8)
#   {p}.vq_scales  fp16                      [E, out, in/group]
#
# d=8 (E36 mixed geometry, 2026-08-15): each module dispatches on its own
# codebook shape, so a model can mix d4 and d8 tensors freely. d8 codebooks
# above 2K entries (K4096 = 64 KB fp16) cannot live in Apple's 32 KB
# threadgroup memory — the d8 kernels keep the codebook in DEVICE memory and
# rely on L2 residency (measured, not assumed: see m1b_bench d8 rows). A
# threadgroup d8 variant exists for K<=1024 (16 KB codebook).
#
# Two execution regimes (measured on M4, m1b/m1c benches, 2026-08-15):
#   decode (small N):  fused LUT-matmul kernel, threadgroup codebook + x,
#                      uchar4 code loads — 0.66-0.88x gather_qmm
#   prefill (large N): decode experts to dense fp16 chunks + ONE padded
#                      batched GEMM per chunk — 1.21-1.28x gather_qmm
#   The row-batched gather_mm path is a known trap (0.43x): do not "simplify"
#   the prefill path back to it.
#
# READ THIS BEFORE TRUSTING THE 1.21-1.28x ABOVE (VQ-PF1, 2026-08-16): that
# figure was measured with an `rng.integers` router, which pads only 1.20x.
# REAL MoE routing is skewed ~8.7x (max 1574 rows/expert vs mean 180), and
# the padded GEMM pads every expert in a chunk to that chunk's MAX row count
# — so expert-id-ordered chunking did 5.80x the necessary FLOPs and the
# prefill path measured ~9x SLOWER than gather_qmm end-to-end, not faster.
# The kernel was never at fault: at the real workload shape it is at parity
# (61.5 ms both). The fix is in `_prefill` below — chunk experts by SIMILAR
# ROW COUNT, not by expert id. Any future prefill benchmark MUST use a real
# router histogram or it will reproduce the same blind spot.

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# below this many (token, expert) pairs use the fused kernel; above, the
# decode+padded-GEMM path (decode cost amortizes). Tune in M1e if needed.
VQ_FUSED_MAX_N = int(os.environ.get("VQ_FUSED_MAX_N",
                     4096))

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured 2026-08-15 on a 128 GB M4 Max running the
# 110.8 GiB 397B, prefill grew 3.35 MB/token where the KV cache theory is
# only 0.059 MB/token — a 57x gap that is entirely these buffers. The
# transient is chunk * out * in * 2 bytes:
#     chunk=128 -> 1.0 GiB (down_proj) / 2.0 GiB (gate_up)
#     chunk= 16 -> 0.12 GiB            / 0.25 GiB
# On a box where the model nearly fills RAM, they are what caps your context
# length. Auto-sized from free memory at import; override with
# VQ_DECODE_CHUNK.
#
# DEFAULT IS 32, AND 32 IS NOT ARBITRARY (2026-08-17, resident probes on the
# M4 across all three artifacts). Steady-state ms/bucket, real block, weights
# RESIDENT (probe_block_prefill.py, [256,36] bucket):
#
#     chunk |  2.2bpw  2.4bpw  3.1bpw
#       16  |   677.2   651.9   663.6
#       32  |   716.1   683.0   691.9
#       64  |   791.4   756.3   775.7
#      128  |   984.1   943.2   947.3     <- the old default
#
# 128 -> 32 is 1.37x / 1.38x / 1.37x — the knee is the SAME for K128, K256
# and K2048, so codebook size does not move it.
#
# 32 rather than 16, which is marginally faster still, because 32 is the
# smallest chunk that reproduces the published perplexity EXACTLY
# (nll 9452.9414 on 2.2bpw; 16 gives 9465.2217 and 8 gives 9450.3555).
# Those shifts are float ORDERING, not quality: `ne` is the batched-GEMM
# batch dim, a different ne picks a different Metal tiling, and fp16 sums in
# a different order — chunk 8 lands BELOW the published number and chunk 16
# above it. A lower ppl from reordering cannot be banked; it is the same
# model measured differently. So the rule is: take the knee, keep exactness.
#
# CAUTION FOR ANYONE RE-MEASURING THIS. Do NOT time it with
# score_tasks_streaming.py --selftest: that re-reads the whole model from
# disk every pass (~63 s of a ~100 s run at 100 GiB), so it is DISK-bound and
# reports "no difference" for a change worth 1.37x. That mistake was made on
# 08-17 and reversed by measuring a resident block instead.
def _default_decode_chunk():
    env = os.environ.get("VQ_DECODE_CHUNK", None)
    if env:
        return max(1, int(env))
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        headroom = info["max_recommended_working_set_size"] - mx.get_active_memory()
        # keep the largest transient (gate_up, out*in ~ 8M elems fp16 = 16 MB
        # per expert) under ~1/8 of remaining headroom
        per_expert = 2048 * 4096 * 2
        return max(4, min(32, int(headroom / 8 / per_expert)))
    except Exception:
        return 32


_DECODE_CHUNK = None  # resolved lazily on first prefill (after weights load)

_SRC_FUSED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 cb[MAX_K];
    threadgroup float4 xs[MAX_NSUB];
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = float4((float)codebook[i*4], (float)codebook[i*4+1],
                       (float)codebook[i*4+2], (float)codebook[i*4+3]);
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(cb[(uint)crow[j]],   xs[j])
                  + dot(cb[(uint)crow[j+1]], xs[j+1])
                  + dot(cb[(uint)crow[j+2]], xs[j+2])
                  + dot(cb[(uint)crow[j+3]], xs[j+3]);
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=8, codebook in DEVICE memory (L2-resident; K4096 = 64 KB > 32 KB
# threadgroup). x cached in threadgroup as float4; each code costs two half4
# codebook loads. Codes are uint16 (12-bit values for K4096).
_SRC_FUSED_D8 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 xs[MAX_NX4];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb4 = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = (uint)crow[j];
            gacc += dot(float4(cb4[2*c]),   xs[2*j])
                  + dot(float4(cb4[2*c+1]), xs[2*j+1]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=8, DEVICE codebook, ONE SIMDGROUP PER OUTPUT ROW (E141, 2026-09-01).
#
# WHY. _SRC_FUSED_D8 above gives each output row to ONE thread, which walks
# its NSUB codes strictly sequentially. Every code costs two DEPENDENT device
# loads (read the code, then index the 256 KB codebook with it), so the whole
# row is a chain of ~2*NSUB serialised round-trips with one load in flight.
# Measured on the 397B shapes that is 14-27 GB/s of code traffic against a
# 546 GB/s machine: latency-bound, not bandwidth-bound.
#
# FIX. The layout the DENSE kernels already use (_SRC_DENSE_D4_TILED): one
# 32-lane simdgroup per row, lane `l` owning scale group `b*32+l`. The 32
# lanes issue their code and codebook loads independently, so a row has up to
# 32 loads in flight instead of one, and the per-group partials are combined
# with a simd_shuffle reduction.
#
# BIT-EXACTNESS. The shuffle reduction walks i = 0..31 in ASCENDING group
# order and applies srow[b*32+i] in exactly the sequence _SRC_FUSED_D8's
# sequential `for g` loop does, and each lane accumulates its own group in the
# same q order with the same float4 dots — so the float op order is identical
# and the output is bit-for-bit equal (asserted in tests/test_vq_expert_simd.py).
# x is staged as float4, matching _SRC_FUSED_D8's cache, so no extra rounding
# is introduced for a non-half T either.
#
# The row guard is a PREDICATE (`active`), never an early return: the tile
# loop below contains threadgroup barriers, and a subset of a threadgroup
# returning early past a barrier is undefined behaviour.
#
# NGRP >= 32 IS REQUIRED, AND THE DISPATCHER ENFORCES IT. Lane `l` owns scale
# group `l`, so a layer with fewer than 32 groups leaves lanes idle. Measured
# per-call over N=1..10, packed and unpacked:
#
#     NGRP     shape                       M4 Max        M3 Ultra
#       64     397B gate/up (IN=4096)   2.15-2.34x      1.20-1.84x
#       16     397B down    (IN=1024)   1.02-1.62x      1.20-1.83x
#       10     F-Next down  (IN= 640)   0.94-1.03x      1.06-1.37x
#
# NGRP=64 wins on BOTH machines by a wide margin; NGRP<=16 is machine-
# dependent and at NGRP=10 the M4 loses outright. So the gate is NGRP >= 32 --
# every lane has work -- which is never slower on either box.
#
# WHY NOT SPLIT A GROUP ACROSS LANES to fill the idle lanes at NGRP<32? Because
# it cannot be made BIT-EXACT. The thread-per-row kernel sums a group's SPG
# products strictly sequentially; any lane split computes partial sums and
# recombines them, which re-associates the fp32 adds and changes the last bits.
# Every published number for these rungs was produced by the old ordering, so
# a re-association is not available to us here. NGRP<32 keeps the old kernel.
_SRC_FUSED_D8_SIMD = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int TILE = 32 * SPG * 2;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup float4 xs[MAX_TILE];
    const device T* xrow = x + (size_t)t * IN;
    const device half4* cb4 = (const device half4*)codebook;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)rr * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NX4 - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = (uint)crow[j];
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=8, codebook in THREADGROUP memory — K<=1024 only (half4 pairs: K*16 B
# = 16 KB at K1024, + x as half4 = 8 KB at IN=4096 -> 24 KB total).
_SRC_FUSED_D8_TG = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[2 * MAX_K];
    threadgroup half4 xs[MAX_NX4];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)(2 * K); i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = (uint)crow[j];
            gacc += dot(float4(cb[2*c]),   float4(xs[2*j]))
                  + dot(float4(cb[2*c+1]), float4(xs[2*j+1]));
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=2 (gemma d2 rungs, 2026-08-18). The mirror case of d8: where d8 needs
# two half4 loads per code, d2 needs only a half2 — so both codebook and x
# cache as half2. Threadgroup footprint is tiny (K*4 B codebook: 1 KB at
# K256, 8 KB even at K2048) but NSUB doubles vs d4 (IN/2), so the x cache is
# the bigger term (NSUB*4 B: 5.5 KB at IN=2816) — still nowhere near the
# 32 KB ceiling. half2 caching is value-identical, same argument as the
# d4 bigK kernel: codebook is fp16 on disk and x is cast to fp16 before
# dispatch. Codes stay one-per-subvector (uint8/uint16), crow steps by NSUB
# exactly like d4 — only the load width changes.
_SRC_FUSED_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[j]))
                  + dot(float2(cb[(uint)crow[j+1]]), float2(xs[j+1]))
                  + dot(float2(cb[(uint)crow[j+2]]), float2(xs[j+2]))
                  + dot(float2(cb[(uint)crow[j+3]]), float2(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=2, THREADGROUP codebook, ONE SIMDGROUP PER OUTPUT ROW (E14x, 2026-09-02).
#
# MEASURED AND DELIBERATELY NOT DISPATCHED --- kept as the reference twin and
# so the negative result is not re-derived. On real 2.1bpw codes (M3 Ultra,
# 48 dispatches/eval, E=64 slice) it runs 0.80-1.09x the thread-per-row kernel:
#
#     tensor      N   NGRP   uchar us   simd us
#     gate_proj  10     40      57.8      70.7
#     gate_proj  20     40      78.3      93.8
#     up_proj    10     40      69.5      61.9
#     up_proj    20     40      69.9      84.2
#     down_proj  10     10      57.6      58.8
#
# WHY IT DOES NOT WIN, unlike its d8 twin. The layout exists to hide DEPENDENT
# DEVICE round-trips: at d=8/K=16384 the 256 KB codebook lives in device memory,
# so every code costs read-then-index against L2 and only one load is in flight
# per row. At d=2/K=256 the codebook is 1 KB and sits in THREADGROUP memory, so
# there is no device round-trip to hide, and the layout's own costs (two barriers
# per block, a 32-step shuffle reduction, 31 of 32 lanes idle at the write) are
# pure overhead. It is not occupancy either: this launches 204,800 threads
# against the old kernel's 6,400 for the same time. The d2 bottleneck was the
# WIDTH OF THE CODE LOAD, which _SRC_FUSED_D2_U32 below fixes instead (1.47-1.86x).
#
# WHY. This is the geometry the Flash-Next 2.1bpw rung actually dispatches --- its
# expert geometry is d=2 / K=256 / unpacked uint8 (codebook [256,2]), so the d4
# u8-view fast path does not apply and the D==2 branch above is taken. Until now
# d2 had NO simdgroup expert kernel at all: _SRC_FUSED_D2 gives each output row
# to ONE thread walking NSUB codes strictly sequentially. At the gate/up shape
# (OUT=640, IN=2560 -> NSUB=1280) a decode step with N=10 pairs launches only
# 6,400 threads for 8.2M dependent code reads --- latency-bound AND badly
# under-occupied on an 80-core M3 Ultra.
#
# MEASURED COST OF THE OLD LAYOUT (M3 Ultra, 2026-09-02, artifact resident,
# 377-token greedy, stub-ablation of VQSwitchLinear.__call__): the VQ expert
# path costs 9.84 ms/token against stock gather_qmm's 3.09 on the same
# architecture and prompt --- 34% of the whole 19.7 ms/token decode gap.
#
# FIX. The layout _SRC_DENSE_D2_TILED already uses, with the expert stride
# folded into crow/srow: one 32-lane simdgroup per row, lane `l` owning scale
# group `b*32+l`, per-group partials combined by simd_shuffle. 32 independent
# loads in flight per row instead of one, and 32x the threads.
#
# BIT-EXACTNESS. _SRC_FUSED_D2's sequential `for g` loop covers group g with
# subvectors j in [g*SPG, (g+1)*SPG) (its running j advances 4 per q over
# QPG = G/8 iterations, i.e. G/2 = SPG per group), accumulating the same four
# dots per q into one float gacc, then applying srow[g] by fma in ASCENDING g.
# Each lane here computes exactly that expression for its own g, and the
# shuffle loop applies srow[b*32+i] for i = 0..31 ascending --- same operands,
# same order, same rounding. x is staged as half2 exactly as the old kernel
# caches it, and cb is the same threadgroup half2 copy. So the output is
# bit-for-bit equal, not merely close (asserted in tests/test_vq_expert_simd.py).
#
# The row guard is a PREDICATE (`active`), never an early return: the tile loop
# contains threadgroup barriers and a subset of a threadgroup returning early
# past a barrier is undefined behaviour.
#
# NGRP >= 32 IS REQUIRED AND THE DISPATCHER ENFORCES IT, for the same reason as
# the d4/d8 twins: lane `l` owns group `l`, so fewer than 32 groups idles lanes.
# On this rung gate_proj/up_proj have NGRP = 2560/64 = 40 (gated in, 96 of the
# 144 expert calls) and down_proj has NGRP = 640/64 = 10 (gated out, keeps the
# thread-per-row kernel, where its 2560 rows already give decent occupancy).
_SRC_FUSED_D2_SIMD = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int SPG  = G / 2;
    const int TILE = 32 * SPG;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_TILE];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)rr * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize)
            xs[i] = half2((half)xrow[(base + (int)i)*2],
                          (half)xrow[(base + (int)i)*2 + 1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = j - base;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[m]))
                      + dot(float2(cb[(uint)crow[j+1]]), float2(xs[m+1]))
                      + dot(float2(cb[(uint)crow[j+2]]), float2(xs[m+2]))
                      + dot(float2(cb[(uint)crow[j+3]]), float2(xs[m+3]));
                j += 4; m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=2, unpacked uint8 codes read FOUR AT A TIME as uint32 (E14x, 2026-09-02).
#
# WHY. _SRC_FUSED_D2 reads `crow[j]` as a uchar, one code per load. Measured on
# the shipped 2.1bpw gate_proj (real codes, E=64 slice, N=10, 48 dispatches per
# eval, M3 Ultra): 68.4 us/dispatch to move 8.7 MB of codes+scales = 128 GB/s on
# an 819 GB/s machine, i.e. 16% of peak, while stock gather_qmm at 3-bit hits
# 292 GB/s on the same shape in situ. The kernel is neither latency-bound (the
# 1 KB K=256 codebook is in threadgroup memory, so there is no dependent device
# round-trip per code, which is why the simdgroup-per-row twin above buys
# nothing here: 1.01-1.09x) nor occupancy-bound (that twin launches 32x the
# threads for the same time). It is spending its bandwidth on scalar byte loads.
#
# FIX. This is the d=2 analogue of the U8-VIEW dispatch the d4 path already
# uses: an unpacked uint8 code row is byte-for-byte a little-endian uint32 word
# stream, so the host reinterprets `codes` with mx.view (zero copy) and the
# kernel reads ONE word per four codes. The existing loop already steps four
# subvectors per q (QPG = G/8 iterations of four dots), so a q is exactly one
# word and the restructuring is mechanical.
#
# BIT-EXACTNESS. Byte i of the little-endian word is code j+i --- the same
# uint8 values the uchar loads produced --- fed to the same cb[] lookups, the
# same four dots summed in the same a+b+c+d expression, the same per-group
# float accumulator and the same ascending-group fma of srow[g]. Nothing about
# the arithmetic or its order changes; only the width of the code load does.
# Output is bit-for-bit equal (asserted in tests/test_vq_expert_simd.py against
# _SRC_FUSED_D2 on real artifact codes).
#
# REQUIRES NSUB % 4 == 0 so a row is a whole number of words and rows stay
# word-aligned; the dispatcher checks it and otherwise keeps the uchar kernel.
_SRC_FUSED_D2_U32 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NWRD = NSUB / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * NWRD + (size_t)r * NWRD;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            const uint w = crow[j >> 2];
            gacc += dot(float2(cb[ w        & 255u]), float2(xs[j]))
                  + dot(float2(cb[(w >>  8) & 255u]), float2(xs[j+1]))
                  + dot(float2(cb[(w >> 16) & 255u]), float2(xs[j+2]))
                  + dot(float2(cb[ w >> 24         ]), float2(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# --- packed-code variants (d=4) -------------------------------------------
# Codes live as uint32 words, BITS per code, blocks of 32 codes = BITS words
# (see vq_pack.py for the format and its bit-exactness tests). The fetch is
# the ONLY difference from the unpacked kernels above: same threadgroup
# caching, same accumulation, same output. Row math is unchanged because
# packing is row-local — crow just steps by WPR instead of NSUB.
# NOTE: mx.fast.metal_kernel splices this source INSIDE a function body, so a
# nested `inline uint vq_code(...) {}` is a compile error ("function definition
# is not allowed here"). Hence macros. The `(sh + BITS > 32)` guard is a
# ternary, so the `32 - sh` shift is never evaluated when sh == 0 (which would
# be a 32-bit shift on a 32-bit type, i.e. UB).
_PACK_FETCH = r"""
    #define VQ_MASK   ((1u << BITS) - 1u)
    #define VQ_OFF(j) (((j) & 31) * BITS)
    #define VQ_W(j)   ((((j) >> 5) * BITS) + (VQ_OFF(j) >> 5))
    #define VQ_SH(j)  (VQ_OFF(j) & 31)
    #define VQ_CODE(crow, j) ( ( ((crow)[VQ_W(j)] >> VQ_SH(j)) \
        | ((VQ_SH(j) + BITS > 32) ? ((crow)[VQ_W(j) + 1] << (32 - VQ_SH(j))) : 0u) \
        ) & VQ_MASK )
"""

# F58 (2026-09-11): bit-walker code fetch for the packed d4 threadgroup
# kernel, default ON. VQ_D4_WALK=0 restores the VQ_CODE macro path.
_D4_WALK = os.environ.get("VQ_D4_WALK", "1") != "0"

# d=4, LARGE K (>1024) — and every packed artifact.
#
# The original _SRC_FUSED caches the codebook as float4 (16 B/entry). At
# K2048 that alone is 32 KB, so with x cached the threadgroup allocation is
# 36 KB and Metal refuses to load the kernel ("Threadgroup memory size 36864
# exceeds the maximum 32768"). That is a hard ceiling on E's geometry
# REGARDLESS of packing — found 2026-08-15 while gating the packed kernels.
#
# Fix: cache codebook and x as half4 (8 B/entry). This is VALUE-IDENTICAL,
# not an approximation: the codebook is fp16 on disk and x is cast to fp16
# before dispatch, so both round-trips are exact; only the threadgroup
# footprint changes (K2048 + NSUB1024 -> 16 KB + 8 KB = 24 KB).
#
# BITS is the packed code width; BITS == 0 selects plain uint8/uint16 codes,
# so one source serves both formats.
_SRC_FUSED_D4_BIGK = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    #define VQ_AT(j) ((uint)crow[j])
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[VQ_AT(j)]),   float4(xs[j]))
                  + dot(float4(cb[VQ_AT(j+1)]), float4(xs[j+1]))
                  + dot(float4(cb[VQ_AT(j+2)]), float4(xs[j+2]))
                  + dot(float4(cb[VQ_AT(j+3)]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

_SRC_FUSED_PACKED = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(xs[j]))
                  + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(xs[j+1]))
                  + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(xs[j+2]))
                  + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# d=4 PACKED, threadgroup codebook, BIT-WALKER code fetch (2026-09-11, F58,
# decode gather-chain front). The generic VQ_CODE macro re-reads its 32-bit
# word for EVERY code it extracts (~3 codes/word at BITS=11) and redoes the
# shift/or/mask each time; a 3-arm probe showed that work IS the whole 1.30x
# decode deletion bound (threadgroup-gather bank conflicts and index-load
# latency both measured null). This variant walks the row once: each code
# word is loaded exactly once into a 64-bit bit-buffer and codes are shifted
# out sequentially. The pack layout (32-code blocks x BITS words, LSB-first,
# tail block padded) makes the walk seamless across block boundaries:
# 32*BITS bits consumed per block == BITS words loaded, so the buffer is
# word-aligned at every boundary and pad codes are never consumed (the loop
# covers exactly NSUB codes). Code values, dot operands and the
# ((d0+d1)+d2)+d3 summation shape are unchanged -> bit-identical (verified
# 60-step greedy logits-checksum transcript vs the VQ_CODE kernel).
# Measured (35B-3.4, d4-K2048, M3, 200-step decode, interleaved 2 passes):
# 55.2 -> 61.8 tok/s = 1.12x. A 32-bit two-word variant tied it, so the
# ulong is not a cost; the residual to the 1.30x bound is the mandatory
# code-word reads the deletion arm also removed.
_SRC_FUSED_PACKED_D4_WALK = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    int w = 0;
    ulong buf = 0;
    int nb = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            uint cq[4];
            for (int u = 0; u < 4; ++u) {
                if (nb < BITS) { buf |= (ulong)crow[w++] << nb; nb += 32; }
                cq[u] = (uint)(buf & (ulong)VQ_MASK);
                buf >>= BITS; nb -= BITS;
            }
            gacc += dot(float4(cb[cq[0]]), float4(xs[j]))
                  + dot(float4(cb[cq[1]]), float4(xs[j+1]))
                  + dot(float4(cb[cq[2]]), float4(xs[j+2]))
                  + dot(float4(cb[cq[3]]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=4, codebook in DEVICE memory (E134, 2026-08-22).
#
# WHY. _SRC_FUSED_D4_BIGK and _SRC_FUSED_PACKED both cache the codebook in
# THREADGROUP memory as half4 (8 B/entry) alongside the x cache, so the
# allocation is (K + NSUB) * 8 bytes against Apple's hard 32768 cap. Measured
# on the M4 across four sibling 35B artifacts (IN=2048 -> NSUB=512):
#
#     K256   ( 256+512)*8 =  6,144 B   loads, generates
#     K2048  (2048+512)*8 = 20,480 B   loads, generates
#     K4096  (4096+512)*8 = 36,864 B   FAILS to load
#     K8192  (8192+512)*8 = 69,632 B   FAILS to load
#
# Metal reports this as "Unable to load kernel ... Compilation failed due to
# an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED", NOT as a
# threadgroup-size error, which is why it reads like a broken compiler
# service. It is not: a trivial custom kernel compiles on the same box
# seconds later. Do not chase the XPC message.
#
# The consequence was that a K>=4096 d4 artifact SCORED normally (the
# streaming referee is prefill-shaped and never dispatches this kernel) while
# being unable to generate a single token. Both 35B offering candidates
# reached release consideration in that state.
#
# FIX. Mirror _SRC_FUSED_D8, which has kept its codebook in device memory
# since K4096 for exactly this reason: drop the threadgroup codebook cache
# and read cb straight from device memory (L2-resident in practice). x stays
# cached in threadgroup, so the allocation becomes NSUB * 8 bytes and is
# independent of K. Arithmetic is UNCHANGED -- same half4 loads, same
# float4 dot, same fma accumulation order -- so results are bit-identical to
# the threadgroup variants, not merely close.
_SRC_FUSED_D4_DEVCB = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 xs[MAX_NSUB];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[(uint)crow[j]]),   float4(xs[j]))
                  + dot(float4(cb[(uint)crow[j+1]]), float4(xs[j+1]))
                  + dot(float4(cb[(uint)crow[j+2]]), float4(xs[j+2]))
                  + dot(float4(cb[(uint)crow[j+3]]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=4 PACKED, codebook in device memory. _SRC_FUSED_PACKED with the
# threadgroup codebook cache removed; code fetch and accumulation identical.
_SRC_FUSED_PACKED_D4_DEVCB = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 xs[MAX_NSUB];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(xs[j]))
                  + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(xs[j+1]))
                  + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(xs[j+2]))
                  + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# d=4, DEVICE codebook, one simdgroup per output row. The d4 counterpart of
# _SRC_FUSED_D8_SIMD, and the expert-axis counterpart of _SRC_DENSE_D4_TILED
# (identical body once the expert stride is folded into crow/srow). x is
# staged as half4, matching _SRC_FUSED_D4_DEVCB's cache exactly, so the dots
# see the same values and the output is bit-identical.
#
# NGRP >= 32 IS A REQUIREMENT, NOT AN OPTIMISATION. Lane `l` owns scale group
# `l`, so a layer with fewer than 32 groups leaves lanes idle and the layout
# LOSES. Measured on the 35B down_proj shape (IN=512, G=64 -> NGRP=8) it ran
# 0.70-0.89x the thread-per-row kernel across N=1..10.
#
# A PACKED twin of this kernel was written and measured, and is deliberately
# NOT shipped: on the 35B shapes the shipped rungs actually use it scored
# 0.94-1.07x at NGRP=32 and 0.70-0.89x at NGRP=8, i.e. no win to bank. The
# packed thread-per-row kernel is already 1.5x the unpacked one (a packed row
# is 4x fewer code bytes and several codes share a uint32 word), so it is not
# the latency-bound case this layout fixes. d8, which IS, keeps its packed
# simd twin.
_SRC_FUSED_D4_DEVCB_SIMD = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    const int QPG  = SPG / 4;
    const int TILE = 32 * SPG;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half4 xs[MAX_TILE];
    const device half4* cb = (const device half4*)codebook;
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)rr * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = j - base;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float4(cb[(uint)crow[j]]),   float4(xs[m]))
                      + dot(float4(cb[(uint)crow[j+1]]), float4(xs[m+1]))
                      + dot(float4(cb[(uint)crow[j+2]]), float4(xs[m+2]))
                      + dot(float4(cb[(uint)crow[j+3]]), float4(xs[m+3]));
                j += 4; m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# d=2 packed (gemma d2 K512/K1024 rungs, 2026-08-19). Same shape as
# _SRC_FUSED_D2 — half2 codebook + x caches, QPG*4 subvectors per group —
# with the code fetch swapped for the VQ_CODE bit-field read. The packing
# layout (vq_pack.py) is dim-agnostic: blocks of 32 codes = BITS uint32
# words, row-local, so crow steps by WPR just like the d4 packed kernel.
# What is NOT dim-agnostic is NSUB (= IN/2, double the d4 value for the same
# IN) — deriving it inside the kernel from IN keeps the d4/d2 asymmetry that
# caused the original silent d2 bug out of the picture; threadgroup sizing
# uses MAX_NSUB from the host, which also computes NSUB = IN/D explicitly.
_SRC_FUSED_PACKED_D2 = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float2(cb[VQ_CODE(crow, j)]),   float2(xs[j]))
                  + dot(float2(cb[VQ_CODE(crow, j+1)]), float2(xs[j+1]))
                  + dot(float2(cb[VQ_CODE(crow, j+2)]), float2(xs[j+2]))
                  + dot(float2(cb[VQ_CODE(crow, j+3)]), float2(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


_SRC_FUSED_PACKED_D8 = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 xs[MAX_NX4];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb4 = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = VQ_CODE(crow, j);
            gacc += dot(float4(cb4[2*c]),   xs[2*j])
                  + dot(float4(cb4[2*c+1]), xs[2*j+1]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# Packed twin of _SRC_FUSED_D8_SIMD: same simdgroup-per-row layout, with the
# code fetch swapped for the VQ_CODE bit-field read. THIS is the kernel the
# shipped 397B rungs actually dispatch (their codes are packed uint32:
# down_proj is [512, 4096, 56] = ceil(128/32)*14 words/row, d=8, K=16384), so
# the unpacked _SRC_FUSED_D8_SIMD above is the reference twin, not the hot
# path. Bit-identical to _SRC_FUSED_PACKED_D8 -- same dots, same q order,
# same ascending-group scale application.
_SRC_FUSED_PACKED_D8_SIMD = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int TILE = 32 * SPG * 2;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (j < NSUB)
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup float4 xs[MAX_TILE];
    const device T* xrow = x + (size_t)t * IN;
    const device half4* cb4 = (const device half4*)codebook;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)rr * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NX4 - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# SIMD_SUM REDUCTION twin of _SRC_FUSED_PACKED_D8_SIMD (2026-09-02, arc 4).
# OFF BY DEFAULT: it is NOT bit-identical, and this arc's gate is bit-identity.
#
# WHAT IT CHANGES. Only the per-block reduction. The shipped kernel finishes
# each 32-group block with 32 SEQUENTIALLY DEPENDENT steps
#     acc = fma(srow[b*32+i], simd_shuffle(gacc, i), acc)
# so every lane redundantly walks a 32-long fp32 dependency chain and issues 32
# scale loads, against only SPG = G/8 = 8 iterations of actual code work. This
# variant has each lane hold its own group's scale and calls simd_sum once.
#
# WHY IT EXISTS. Arc 4 ablated the kernel instruction by instruction (cost-only
# arms, each WRONG on purpose, each a lower bound), real L20 tensors, E=64,
# rows=8, min/median over 15 interleaved reps x 50 dispatches:
#
#     arm (N=10 / N=20, fraction of base)      gate      up
#       inner code loop deleted             0.57/0.52  0.57/0.52
#       reduction deleted                   0.86/0.82  0.84/0.81
#       reduction's scale loads deleted     0.91/0.88  0.91/0.88
#       reduction chain broken into 4 ILP   0.91/0.90  0.92/0.90
#       simd_sum (this variant)             0.87/0.84  0.85/0.84
#
# i.e. the reduction owns ~19% of the dispatch and simd_sum recovers ~16 of
# those 19 points -- it lands essentially ON the "reduction is free" floor.
# The chain is LATENCY, not shuffle throughput: breaking it into 4 independent
# partials recovers most of the same ground.
#
# This also explains an asymmetry arc 2 recorded but did not attribute: the
# NGRP=10 down_proj, which is DECLINED from this simd layout and runs
# thread-per-row with NO reduction, was the FASTEST shape per code byte
# (95.8 GB/s vs gate's 71.8). The layout it is excluded from is the layout
# carrying the reduction tax.
#
# WHY IT IS OFF. simd_sum re-associates the sum into a tree and un-fuses the
# scale multiply, so the last mantissa bit can move. Measured on real tensors:
# 0.05-0.16% of output elements differ, every one of them by ONE half-ULP.
# Against an independent fp32 mlx reference the two are a TIE -- identical max
# error to every printed digit, identical mean error, and which one is closer
# flips between shapes. So this is a last-place-bit tie, not a loss of
# accuracy. It still is not bit-identity, and bit-identity is the gate, so
# turning it on is a POLICY decision (relax the gate to 1-ULP equivalence plus
# a ppl/KL re-referee and an end-to-end A/B) that this arc does not take
# unilaterally. VQ_D8_SIMDSUM=1 dispatches it so the finding stays reproducible.
#
# Untouched by construction: the d2/K256 layers 0-1 never reach this kernel,
# and the NGRP=10 down_proj is declined from the simd layout by the NGRP>=32
# gate, so neither can regress.
_SRC_FUSED_PACKED_D8_SIMD_SS = _SRC_FUSED_PACKED_D8_SIMD.replace(
    """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
""",
    """        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            acc += simd_sum(sv * gacc);
        }
""")
assert _SRC_FUSED_PACKED_D8_SIMD_SS != _SRC_FUSED_PACKED_D8_SIMD, (
    "the packed-d8 simd reduction text drifted; the simd_sum twin did not "
    "apply and would silently be a duplicate of the base kernel")

_D8_SIMDSUM = os.environ.get("VQ_D8_SIMDSUM", "0") != "0"


# DEVICE-X twin of _SRC_FUSED_PACKED_D8_SIMD (2026-09-02, kernel arc 5).
# ON BY DEFAULT: it is BIT-IDENTICAL and 1.19-1.23x at decode N.
#
# WHAT IT CHANGES. Only where x is read from. The base kernel stages x into a
# threadgroup tile in 32-group blocks (two threadgroup_barriers per block) and
# the inner loop reads the tile. This one deletes the tile, the staging loop
# and every barrier: each lane reads its own x slice straight from device
# memory as half4 and converts with the same float4(half4) widening, so the
# values entering the dots -- and the dot/fma order -- are exactly the base
# kernel's. mx.array_equal on real L20 gate/up tensors at N=1/5/10/20 holds
# (asserted in scripts/bench_d8_stage.py before any timing, and pinned by the
# test suite's simd-vs-base bit-exact parametrisation).
#
# WHY IT WINS (scripts/bench_d8_stage.py, real L20 tensors, E=64, arc-standard
# interleaved min/med): the cost split of the ~36% arc 4 left unprobed is
#     staging deleted  0.52 of base   barriers deleted  0.98   empty  0.15
# i.e. the BARRIERS are ~2% and the STAGING is nearly half the dispatch. The
# tile was built to save device reads of x, but at rows=8 every threadgroup
# re-stages the whole row (OUT/8 = 80 threadgroups re-read 5 KiB per token
# row), and the threadgroup store+load round-trip plus the barrier convoy
# costs far more than L1-served device reads of the same bytes. Reading x
# directly measures 0.81-0.84 of base at N=10/20 on both gate and up
# (1.19-1.23x), 0.95-1.02 at the launch-bound N=1, and composes with nothing
# it invalidates: the reduction, code fetch and scale walk are untouched.
#
# VQ_D8_DEVX=0 restores the staged kernel for A/B without a rebuild. The
# simd_sum twin above still derives from the STAGED base so the arc 4
# negative stays reproducible bit-for-bit; a devx+simd_sum combination is a
# one-line replace if the bit-identity gate is ever relaxed.
_SRC_FUSED_PACKED_D8_SIMD_DEVX = _SRC_FUSED_PACKED_D8_SIMD.replace(
    """    threadgroup float4 xs[MAX_TILE];
""", "").replace(
    """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NX4 - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
""",
    """    const device vec<T,4>* xr4 = (const device vec<T,4>*)xrow;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = 0;
""").replace(
    """                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
""",
    """                gacc += dot(float4(cb4[2*c]),   float4(xr4[m]))
                      + dot(float4(cb4[2*c+1]), float4(xr4[m+1]));
""")
assert ("xs[" not in _SRC_FUSED_PACKED_D8_SIMD_DEVX
        and "barrier" not in _SRC_FUSED_PACKED_D8_SIMD_DEVX), (
    "the packed-d8 simd staging text drifted; the devx twin did not fully "
    "apply")

_D8_DEVX = os.environ.get("VQ_D8_DEVX", "1") != "0"

# DEVX + SIMD_SUM COMPOSITION (2026-09-02, acceptance eval d4855e4). The SS
# reduction rewrite applies verbatim on top of the devx source because devx
# replaced the staging and dot text, not the reduction. This is the shipping
# default: quality delta measured EXACTLY ZERO with the kernel exercised
# (chunk-16 referee: identical nll to every digit over 2048 tokens, KL
# mean/max 0.0 over 459 positions) and +2.4% end-to-end over devx alone
# (18.39 -> 18.84 tok/s A-B-A). Not bit-identical to the pre-arc-4 kernel:
# ~0.1% of elements differ by one half-ULP (rounding ties; equally accurate
# vs fp32). The bit-identity gate is relaxed to 1-ULP equivalence for this
# kernel only, by decision 2026-09-02. VQ_D8_SS=0 restores plain devx
# (bit-identical to the legacy kernel) without a rebuild.
_SRC_FUSED_PACKED_D8_SIMD_DEVX_SS = _SRC_FUSED_PACKED_D8_SIMD_DEVX.replace(
    """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
""",
    """        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            acc += simd_sum(sv * gacc);
        }
""")
assert _SRC_FUSED_PACKED_D8_SIMD_DEVX_SS != _SRC_FUSED_PACKED_D8_SIMD_DEVX, (
    "the packed-d8 reduction text drifted; the devx+ss composition did not "
    "apply and would silently fall back to plain devx")

_D8_SS = os.environ.get("VQ_D8_SS", "1") != "0"


# REGISTER-BUFFERED twin of _SRC_FUSED_PACKED_D8_SIMD (2026-09-02).
#
# The kernel above is LOAD-ISSUE bound, not bandwidth bound: it moves 61-97
# GB/s of the M3 Ultra's 819, and the profiling arc that measured it REFUTED
# both codebook residency (shrinking the hot set 256 KiB -> 4 KiB buys 8%)
# and load width. What is left is issue: ~4 device loads per code against one
# load issued per core per cycle, and 1-2 of those four are the VQ_CODE
# bit-field read, which re-reads the SAME uint32 word for every code that
# lands in it.
#
# _SRC_DENSE_PACKED_D2 already fixes this for d=2 (82 us vs 169 on a
# comparable dependent chain) by copying a lane's code words into registers
# once. It does not port mechanically, and that is the whole difficulty: at
# d=2/G=64 a lane's group is EXACTLY one 32-code block, so its words start at
# bit 0 of the block and every extraction offset is compile-time. At d=8 a
# lane owns only SPG = G/8 codes, so NPH = 32/SPG lanes share a block and
# lane p starts at bit p*SPG*BITS INTO it -- a runtime offset, and indexing a
# register array with a runtime index SPILLS TO MEMORY on Apple GPUs, which
# is strictly worse than the device reads it replaces.
#
# The obvious fix -- an if/else chain over the NPH phases with each body
# compile-time specialised -- was BUILT AND MEASURED FIRST, and it is 1.9x
# SLOWER (see the ledger, 2026-09-02). The reason is structural and is the
# thing that makes d=8 hard: the phase p = g & (NPH-1) varies WITHIN a
# simdgroup (lane L owns group b*32+L, so consecutive lanes have consecutive
# phases), so all NPH bodies are executed serially by the whole simdgroup.
# Compile-time specialisation on a per-lane-varying value buys specialisation
# and pays NPH-way divergence. Regrouping lanes by phase does not help --
# divergence on Apple GPUs is per-simdgroup, and every branch still has an
# active lane.
#
# So the phase is handled BRANCHLESSLY instead. Each lane loads the SAME
# constexpr NW words from a runtime word offset (pointer arithmetic, not
# register indexing, so no spill and no divergence), then FUNNEL-SHIFTS that
# window down by the runtime bit offset SH0:
#
#     v[i] = (wb[i] >> SH0) | (wb[i+1] << (32 - SH0))
#
# after which every code sits at bit q*BITS from v[0] and every v[] index and
# shift in the extraction is a literal, exactly as in the d=2 kernel. The
# funnel is a uniform-cost select, identical work on every lane.
#
# Word footprint per lane, at the shipped SPG=8/BITS=14: a lane's 8 codes are
# 112 bits starting at p*112 within the block's 14 words, so p in {0,1,2,3}
# starts at word {0,3,7,10} bit {0,16,0,16}, and NW = 4 words covers the
# widest case (16 + 112 = 128 bits) -- 4 loads per 8 codes where VQ_CODE
# issued ~12. The window NEVER leaves the block: the largest read is
# blk[10+3] = blk[13] = blk[BITS-1]. _d8_regbuf_ok asserts that in Python
# rather than trusting it.
#
# BIT-EXACTNESS. Extraction order, the two dots per code, the per-lane gacc,
# and the ascending-group simd_shuffle scale reduction are copied verbatim
# from the kernel above; only WHERE the code words are read from changes.
# Output is bit-identical by construction and asserted as such against
# _SRC_FUSED_PACKED_D8_SIMD (tests/test_vq_expert_simd.py).
_SRC_FUSED_PACKED_D8_SIMD_RB = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int TILE = 32 * SPG_C * 2;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (j < NSUB)
    constexpr int NPH  = 32 / SPG_C;           // lanes sharing one 32-code block
    constexpr uint MASK = (1u << BITS) - 1u;
    // widest start bit any phase can land on, and the window that covers it
    constexpr int SH0MAX = ((NPH - 1) * SPG_C * BITS) & 31;
    constexpr int NW = (SH0MAX + SPG_C * BITS + 31) >> 5;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup float4 xs[MAX_TILE];
    const device T* xrow = x + (size_t)t * IN;
    const device half4* cb4 = (const device half4*)codebook;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)rr * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;

    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NX4 - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            const int j0 = g * SPG_C;
            const int SB = (g & (NPH - 1)) * SPG_C * BITS;   // start bit in block
            const device uint* blk = crow + (size_t)(j0 >> 5) * BITS + (SB >> 5);
            const uint SH0 = (uint)(SB & 31);
            // one window, then funnel it down so every extraction below is at
            // a compile-time offset. wb/v are indexed only by unrolled loop
            // counters, so they stay in registers.
            uint wb[NW + 1];
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < NW; ++wi) wb[wi] = blk[wi];
            wb[NW] = 0u;      // bits at/after 32*NW are never needed (see above)
            uint v[NW];
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < NW; ++wi)
                v[wi] = SH0 ? ((wb[wi] >> SH0) | (wb[wi + 1] << (32 - SH0)))
                            : wb[wi];
            int m = 2 * j0 - base;
            #pragma clang loop unroll(full)
            for (int q = 0; q < SPG_C; ++q, m += 2) {
                const int wi = (q * BITS) >> 5;
                const int sh = (q * BITS) & 31;
                uint c = v[wi] >> sh;
                if (sh + BITS > 32) c |= v[wi + 1] << (32 - sh);
                c &= MASK;
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


_SRC_FUSED_PACKED_D8_TG = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[2 * MAX_K];
    threadgroup half4 xs[MAX_NX4];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)(2 * K); i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = VQ_CODE(crow, j);
            gacc += dot(float4(cb[2*c]),   float4(xs[2*j]))
                  + dot(float4(cb[2*c+1]), float4(xs[2*j+1]));
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# --- DENSE d=2 kernels (e4b VQLinear decode path, 2026-08-19) --------------
# The expert kernels above give a dense layer at E=1 correctness but not
# speed: ONE thread walks a whole code row (NSUB = IN/2 = up to 5120 codes)
# sequentially, so decode is LATENCY-bound, not bandwidth-bound (measured
# 43 tok/s on e4b-VQ-d2K2048-packed vs 84 for 8-bit, whose qmv splits a row
# across a simdgroup). These dense variants do the same: a 32-lane simdgroup
# owns each output row, lane l handles scale-groups l, l+32, ..., and the
# per-group partials are stitched back together with a simd_shuffle broadcast
# loop so the group-order fma chain is executed in EXACTLY the sequential
# order of the expert kernel. That makes the output BIT-IDENTICAL to the
# _SRC_FUSED_D2 / _SRC_FUSED_PACKED_D2 path — required, because the shipped
# KL numbers (E62) were scored through those kernels, and a reduction in a
# different float order would move the printed number. The per-group inner
# expression (QPG iterations of four dots summed a+b+c+d) is copied verbatim
# for the same reason: `gacc += a+b+c+d` and four `gacc += d_i` round
# differently.
#
# No expert axis, no eidx gather: codes/scales are [OUT, ...], crow/srow are
# plain row offsets. Threadgroup caching is unchanged (half2 codebook + x:
# K2048 -> 8 KB, IN=10240 -> NSUB 5120 -> 20 KB; 28 KB < 32 KB ceiling).
_SRC_DENSE_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int SPG  = G / 2;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT) return;
    const device CT* crow = codes + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)r * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (g < NGRP) {
            int j = g * SPG;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[j]))
                      + dot(float2(cb[(uint)crow[j+1]]), float2(xs[j+1]))
                      + dot(float2(cb[(uint)crow[j+2]]), float2(xs[j+2]))
                      + dot(float2(cb[(uint)crow[j+3]]), float2(xs[j+3]));
                j += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# Packed twin. G=64 ONLY, and that is load-bearing: at d=2, a scale-group of
# 64 weights is 32 codes, and vq_pack blocks are 32 codes = BITS uint32
# words — so each lane's group is EXACTLY one word-aligned block. The lane
# copies its block's BITS words into registers ONCE and extracts all 32
# codes from registers; going through the generic VQ_CODE device-memory
# macro instead re-reads words per code and measured 169 us/matmul on a
# dependent e4b-shaped chain vs 82 for this version (M3 Ultra, 2026-08-19).
# Bit extraction order and arithmetic are unchanged — output stays
# bit-identical to _SRC_FUSED_PACKED_D2 (verified).
# The LC(i) offsets are compile-time (i and BITS are constants after
# unrolling), so wbuf indexing does not spill. Code 31 ends at bit 32*BITS-1
# exactly, so no extraction ever reads past wbuf[BITS-1].
_SRC_DENSE_PACKED_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT) return;
    const device uint* crow = codes + (size_t)r * WPR;
    const device half* srow = scales + (size_t)r * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (g < NGRP) {
            uint wbuf[BITS];
            const device uint* blk = crow + (size_t)g * BITS;
            for (int wi = 0; wi < BITS; ++wi) wbuf[wi] = blk[wi];
            int j = g * 32;
            for (int q = 0; q < 8; ++q) {
                const int i0 = q * 4;
                #define LC(i) (((wbuf[((i)*BITS)>>5] >> (((i)*BITS)&31)) \
                    | (((((i)*BITS)&31) + BITS > 32) \
                       ? (wbuf[(((i)*BITS)>>5)+1] << (32-(((i)*BITS)&31))) \
                       : 0u)) & ((1u<<BITS)-1u))
                gacc += dot(float2(cb[LC(i0)]),   float2(xs[j]))
                      + dot(float2(cb[LC(i0+1)]), float2(xs[j+1]))
                      + dot(float2(cb[LC(i0+2)]), float2(xs[j+2]))
                      + dot(float2(cb[LC(i0+3)]), float2(xs[j+3]));
                j += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# --- NSUB-TILED dense d2 twins ---------------------------------------------
# The kernels above stage the WHOLE x row in threadgroup memory
# (`xs[MAX_NSUB]`), so their allocation is (K + NSUB) * 4 + overhead and grows
# with LAYER WIDTH. At the 27B mlp shapes (IN 17408 -> NSUB 8704) that is
# 34,816 bytes for xs alone against Metal's 32,768 cap, so the kernel fails at
# LOAD and VQLinear falls back to decoding the entire weight per call --- the
# path that measured 0.43 tok/s against stock's 16.7 (docs/DENSE-VQ-DECODE.md).
#
# The fix is free, because the existing block loop already IS the tile. Lane
# `lane` of block `b` owns scale-group g = b*32 + lane and touches exactly
# xs[g*SPG .. g*SPG+SPG). Across the 32 lanes of one block that is a
# CONTIGUOUS span of 32*SPG sub-vectors, so staging just that span is enough:
#
#     allocation = (K + 32*SPG) * 4 + overhead        <- no NSUB term at all
#
# For G=64 that is 1024 half2 = 4 KB of x regardless of how wide the layer is;
# with K=4096 the whole thing is 20 KB. Total device traffic is unchanged ---
# each x element is still read once per threadgroup, just later.
#
# BIT-EXACTNESS. Every arithmetic operation, its operands and its order are
# untouched: the same half conversion, the same four-way dot per q, the same
# per-lane gacc, the same simd_shuffle reduction applying srow[] in ascending
# group order. Only the storage location of xs changes. So these kernels are
# bit-identical to the untiled twins by construction, and that is asserted
# against them on every shape where both can run.
#
# BARRIER DISCIPLINE. The untiled kernels early-return on `r >= OUT` before
# the loop. That is illegal once barriers live INSIDE the loop --- a
# threadgroup barrier must be reached by every thread --- so the tiled twins
# carry the row guard as a predicate (`active`) and let inactive threads run
# the loop for its barriers only. Inactive rows shuffle garbage among
# themselves (one row == one simdgroup, since threadgroup.x is 32) and never
# write y.
_SRC_DENSE_D2_TILED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int SPG  = G / 2;
    const int TILE = 32 * SPG;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_TILE];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const device CT* crow = codes + (size_t)rr * NSUB;
    const device half* srow = scales + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize)
            xs[i] = half2((half)xrow[(base + (int)i)*2],
                          (half)xrow[(base + (int)i)*2 + 1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = j - base;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[m]))
                      + dot(float2(cb[(uint)crow[j+1]]), float2(xs[m+1]))
                      + dot(float2(cb[(uint)crow[j+2]]), float2(xs[m+2]))
                      + dot(float2(cb[(uint)crow[j+3]]), float2(xs[m+3]));
                j += 4; m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# Packed twin. Same tiling, same guarantees; the ONLY difference from
# _SRC_DENSE_PACKED_D2 is where xs lives, exactly as above.
_SRC_DENSE_PACKED_D2_TILED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int SPG  = G / 2;
    const int TILE = 32 * SPG;
    const int WPR  = (NSUB + 31) / 32 * BITS;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_TILE];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const device uint* crow = codes + (size_t)rr * WPR;
    const device half* srow = scales + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize)
            xs[i] = half2((half)xrow[(base + (int)i)*2],
                          (half)xrow[(base + (int)i)*2 + 1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            uint wbuf[BITS];
            const device uint* blk = crow + (size_t)g * BITS;
            for (int wi = 0; wi < BITS; ++wi) wbuf[wi] = blk[wi];
            int m = g * 32 - base;
            for (int q = 0; q < 8; ++q) {
                const int i0 = q * 4;
                #define LC(i) (((wbuf[((i)*BITS)>>5] >> (((i)*BITS)&31)) \
                    | (((((i)*BITS)&31) + BITS > 32) \
                       ? (wbuf[(((i)*BITS)>>5)+1] << (32-(((i)*BITS)&31))) \
                       : 0u)) & ((1u<<BITS)-1u))
                gacc += dot(float2(cb[LC(i0)]),   float2(xs[m]))
                      + dot(float2(cb[LC(i0+1)]), float2(xs[m+1]))
                      + dot(float2(cb[LC(i0+2)]), float2(xs[m+2]))
                      + dot(float2(cb[LC(i0+3)]), float2(xs[m+3]));
                m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# --- dense d4, tiled x + DEVICE codebook -----------------------------------
# The 3.9bpw 27B rung is d=4/K=4096 and had no dense kernel of any kind, so
# every one of its 192 VQ linears decoded its whole weight per forward:
# 0.426 tok/s against stock's 16.687 (docs/DENSE-VQ-DECODE.md). Tiling alone
# does not rescue it, because at d=4 the CODEBOOK is the thing that does not
# fit --- K=4096 float4 entries is 65,536 bytes, twice the whole 32 KB
# threadgroup budget, before x is considered at all.
#
# So this kernel takes the same route _SRC_FUSED_D8 already takes for the same
# reason: leave the codebook in DEVICE memory and let L2 hold it (64 KB of
# fp16 is comfortably L2-resident, and every lane in the threadgroup reads the
# same table). Only x is staged, and it is tiled, so the threadgroup
# allocation is 32*SPG*8 bytes --- 4 KB at G=64 --- independent of BOTH K and
# layer width.
#
# Arithmetic mirrors _SRC_FUSED (the d4 expert kernel) exactly: float4 dots,
# four codes unrolled per step, one float accumulator per scale group, scales
# applied in ascending group order. The group-parallel simd_shuffle reduction
# is the dense d2 layout, which applies srow[] in the same ascending order as
# the expert kernel's sequential loop, so the results agree.
_SRC_DENSE_D4_TILED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    const int QPG  = SPG / 4;
    const int TILE = 32 * SPG;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half4 xs[MAX_TILE];
    const device half4* cbg = (const device half4*)codebook;
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const device CT* crow = codes + (size_t)rr * NSUB;
    const device half* srow = scales + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = j - base;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float4(cbg[(uint)crow[j]]),   float4(xs[m]))
                      + dot(float4(cbg[(uint)crow[j+1]]), float4(xs[m+1]))
                      + dot(float4(cbg[(uint)crow[j+2]]), float4(xs[m+2]))
                      + dot(float4(cbg[(uint)crow[j+3]]), float4(xs[m+3]));
                j += 4; m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# Packed twin of _SRC_DENSE_D4_TILED, i.e. _SRC_FUSED_PACKED_D4_DEVCB with the
# expert axis removed, the group loop spread across the 32 lanes of a
# simdgroup, and x staged one block span at a time. This is the geometry the
# 3.9bpw 27B rung actually uses (d=4, K=4096, pack_bits=12, G=64), which is
# why that rung had no fused path at all: d4 had no dense kernel, and its
# codebook is 64 KB besides.
_SRC_DENSE_PACKED_D4_TILED = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    const int QPG  = SPG / 4;
    const int TILE = 32 * SPG;
    const int WPR  = (NSUB + 31) / 32 * BITS;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half4 xs[MAX_TILE];
    const device half4* cb = (const device half4*)codebook;
    const device T* xrow = x + (size_t)t * IN;

    const bool active = (r < (uint)OUT);
    const uint rr = active ? r : 0;
    const device uint* crow = codes + (size_t)rr * WPR;
    const device half* srow = scales + (size_t)rr * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (active && g < NGRP) {
            int j = g * SPG;
            int m = j - base;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(xs[m]))
                      + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(xs[m+1]))
                      + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(xs[m+2]))
                      + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(xs[m+3]));
                j += 4; m += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (active && lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# --- DEVICE-X twins of every dense kernel (2026-09-03, dense-kernel arc) ---
# ON BY DEFAULT: bit-identical by construction and 1.66-1.77x per dispatch.
#
# THE TWIN OF ARC 5's devx, AND A BIGGER PRIZE THAN THE EXPERT ONE. Arc 5
# found that the packed-d8 EXPERT kernel spent ~48% of a dispatch staging x
# through a threadgroup tile and ~2% on its barriers, and deleting the tile
# bought 1.17-1.24x. Every dense kernel has the same structure, so the same
# ablation was run on the dense side first (scripts/bench_dense_stage.py, real
# 27B 3.9bpw L20 tensors, cost-only arms each WRONG on purpose = lower
# bounds, fraction of base min/med at N=1 / N=20):
#
#     arm         gate N=1    gate N=20   down N=1    down N=20
#     nostage     0.36/0.36   0.34/0.34   0.36/0.36   0.34/0.33
#     nobar       0.68/0.68   0.66/0.66   0.62/0.63   0.61/0.60
#     nostagebar  0.36/0.37   0.34/0.34   0.36/0.36   0.34/0.34
#     nored       0.68/0.69   0.67/0.66   0.69/0.69   0.68/0.67
#     empty       0.04/0.04   0.006/0.006 0.04/0.04   0.003/0.003
#
# The dense cost map is NOT the expert one, and two of the differences matter:
#   * staging is ~66% of a dense dispatch, not ~48%;
#   * the BARRIERS are ~32-39%, not ~2%. Arc 5 killed the "barrier convoy"
#     theory for the expert kernel; on the dense side the convoy is REAL,
#     because a dense threadgroup is 32 simdgroups (1024 threads) against the
#     expert kernel's 8, so every barrier synchronises 4x the threads and the
#     staging loop they are all waiting on is the same 4 KB.
#   * the launch floor is NOISE here (0.3-4%), not arc 5's ~15%: one dense
#     dispatch covers the whole [17408, 5120] layer and costs 200-4200 us
#     against an expert dispatch's 13-67, so there is nothing to win by
#     dispatching less often. The "fewer dispatches" lever does not exist on
#     this side of the house.
#
# WHAT THE TWINS CHANGE. Only where x is read from: the threadgroup tile, the
# staging loop and (where legal) the barriers are deleted, and each lane reads
# its own x slice from device as vec<T,2> / vec<T,4>.
#
# BIT-IDENTITY. The staging loop narrowed T to half on the way in
# (`xs[i] = half2((half)xrow[..], ..)`), so the twins narrow the same way with
# an explicit `half2(...)` / `half4(...)` cast around the device read rather
# than a bare `float2(xr2[m])`. That distinction is load-bearing: a bare widen
# would be identical only for T=half and would silently skip the narrowing for
# a bf16 or fp32 activation. With the cast the values entering every dot --
# and the dot/fma order, the code fetch, the scale walk and the reduction --
# are exactly the base kernel's, so these are bit-identical for EVERY T by
# construction, asserted on real tensors before any timing was printed
# (scripts/bench_dense_stage.py) and pinned by tests/test_vq_dense_devx.py.
#
# BARRIER DISCIPLINE, and why the d2 twins keep one. The d4 kernels hold their
# codebook in DEVICE memory, so once x staging is gone they need no barrier at
# all. The d2 kernels stage the CODEBOOK into threadgroup memory
# (`cb[MAX_K]`), and today the barrier that publishes it is the first one
# inside the block loop. Deleting every barrier there would leave the codebook
# fill unsynchronised -- a data race that would read garbage codebook entries,
# not a slower kernel. So the d2 twins keep exactly ONE barrier, hoisted out
# of the loop, immediately after the codebook fill.
#
# MEASURED, through the real dispatcher, off -> on, us/dispatch min/med
# (27B 3.9bpw L20, d4/K4096/packed-12, rows=32):
#
#     gate (OUT 17408, IN 5120, NGRP 80)
#       N=1   215.2/216.1 -> 130.7/132.0   1.65/1.64
#       N=10 2091.2/2116.8 -> 1229.9/1231.9 1.70/1.72
#       N=20 4152.0/4183.4 -> 2451.2/2453.2 1.69/1.71
#     down (OUT 5120, IN 17408, NGRP 272)
#       N=1   216.7/219.6 -> 127.5/128.6   1.70/1.71
#       N=20 4114.1/4161.4 -> 2356.4/2361.4 1.75/1.76
#
# VQ_DENSE_DEVX=0 restores the staged kernels for A/B without a rebuild.
def _devx_dots(src, w, idx, name):
    """`float{w}(xs[{idx}+k])` -> `float{w}(half{w}(xr{w}[{idx}+k]))`.

    Explicit per-use-site rewrite with a hard assert, rather than a loose
    substring swap: the narrowing cast has to land on EVERY dot or the twin
    is silently no longer bit-identical at non-half T.
    """
    n = 0
    for k in range(4):
        sub = idx if k == 0 else f"{idx}+{k}"
        old = f"float{w}(xs[{sub}])"
        if old not in src:
            continue
        src = src.replace(old, f"float{w}(half{w}(xr{w}[{sub}]))")
        n += 1
    assert n == 4, f"{name}: rewrote {n}/4 dense dot sites, not all four"
    assert "xs[" not in src, f"{name}: a threadgroup x read survived"
    return src


def _dense_devx(src, w, tiled, staged_cb, name):
    """Build the device-x twin of a dense kernel source."""
    out = src.replace(f"    threadgroup half{w} xs[MAX_TILE];\n", "", 1)
    out = out.replace(f"    threadgroup half{w} xs[MAX_NSUB];\n", "", 1)
    assert out != src, f"{name}: no threadgroup x declaration to delete"
    ptr = f"    const device vec<T,{w}>* xr{w} = (const device vec<T,{w}>*)xrow;\n"
    if tiled:
        # per-block staging + its two barriers -> one device pointer, and
        # `base = 0` so the inner `m = j - base` addresses the whole row.
        # staged_cb keeps ONE barrier, hoisted out of the loop, so the
        # codebook fill above is still published before it is read.
        bar = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n" if staged_cb else ""
        d4_stage = """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""
        d2_stage = """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize)
            xs[i] = half2((half)xrow[(base + (int)i)*2],
                          (half)xrow[(base + (int)i)*2 + 1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""
        stage = d4_stage if w == 4 else d2_stage
        assert stage in out, f"{name}: tiled staging text drifted"
        out = out.replace(stage, ptr + bar + """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = 0;
""")
        idx = "m"
    else:
        # untiled: the whole row was staged once, before the single barrier
        # that also publishes the codebook. Drop the x loop, keep the barrier.
        stage = """    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
"""
        assert stage in out, f"{name}: untiled staging text drifted"
        out = out.replace(stage, ptr)
        idx = "j"
    out = _devx_dots(out, w, idx, name)
    nbar = out.count("threadgroup_barrier")
    assert nbar == (1 if staged_cb else 0), (
        f"{name}: {nbar} barriers left, expected {1 if staged_cb else 0}")
    return out


_SRC_DENSE_D4_TILED_DEVX = _dense_devx(
    _SRC_DENSE_D4_TILED, 4, True, False, "dense_d4_tiled")
_SRC_DENSE_PACKED_D4_TILED_DEVX = _dense_devx(
    _SRC_DENSE_PACKED_D4_TILED, 4, True, False, "dense_packed_d4_tiled")
_SRC_DENSE_D2_TILED_DEVX = _dense_devx(
    _SRC_DENSE_D2_TILED, 2, True, True, "dense_d2_tiled")
_SRC_DENSE_PACKED_D2_TILED_DEVX = _dense_devx(
    _SRC_DENSE_PACKED_D2_TILED, 2, True, True, "dense_packed_d2_tiled")
_SRC_DENSE_D2_DEVX = _dense_devx(
    _SRC_DENSE_D2, 2, False, True, "dense_d2")
_SRC_DENSE_PACKED_D2_DEVX = _dense_devx(
    _SRC_DENSE_PACKED_D2, 2, False, True, "dense_packed_d2")

_DENSE_DEVX = os.environ.get("VQ_DENSE_DEVX", "1") != "0"


# --- dense simd_sum twins: ON BY DEFAULT since 2026-09-19 (F148) ----------
# The dense reduction text is character-identical to the packed-d8 one, so
# arc 4's simd_sum rewrite applies verbatim: instead of a 32-step serial
# `acc = fma(srow[i], simd_shuffle(gacc, i), acc)` chain, each lane scales its
# own group partial and one simd_sum reduces the 32 of them in tree order.
#
# IT IS FAST. Composed on devx, per dispatch: 2.01-2.10x vs the staged base,
# i.e. a further 1.19-1.22x on top of devx alone (gate N=20 2451 -> 2004 us).
# The ablation says why -- the reduction is ~33% of a dense dispatch (nored
# 0.66-0.69), against ~19% in the expert kernel.
#
# IT IS NOT COVERED BY THE 1-ULP DECISION, AND IT IS NOT MINE TO TURN ON.
# The 2026-09-02 gate relaxation (f6aa628) is explicitly "for this reduction
# only", meaning the packed-d8 expert kernel, where the measured divergence
# was ~0.1% of elements at ONE HALF-ULP. The dense divergence is an order of
# magnitude larger: measured max 1.00 / 4.00 / 2.00 / 7.00 ULP (gate, N=1/5/
# 10/20) and up to 8.00 ULP on down_proj. Two structural reasons -- a dense
# row reduces over NGRP=80-272 groups spread across NBLK=3-9 blocks, so the
# tree/serial disagreement compounds per block instead of once; and there is
# no expert axis to average it away.
#
# So this is NOT a 1-ULP change and the 2026-09-02 d8 decision did not reach
# it: turning it on needed its own referee pass on the dense line. That pass
# now exists and the switch is ON by default since 2026-09-19 (Noah's call).
#
# WHAT THE REFEREE PASS FOUND -- read this before flipping it back:
#   SPEED   1.1461x (3.9, d4 packed), 1.1535x (4.5, d2 unpacked), 1.1348x
#           (4.8, d2 packed). Measured AT shipped decode width, 200 tokens
#           best-of-3, one process per arm, idle box. F136 + F142.
#   QUALITY NULL at shipped decode width on both rungs scored there: worst
#           cell +1.101 mnats at t=+1.36. F144, scored at chunk 8 with NO
#           gate overrides -- inside the fused gates, so the arms really
#           ran these kernels.
#   THE ULP READING WAS BACKWARDS. The "up to 8.00 ULP" above is
#           DISAGREEMENT between two roundings, not error. A tree reduction's
#           error grows O(log n) against a serial chain's O(n), so the tree
#           is the better-rounded one; measured, no corpus is worse and
#           prose improved at |t|=4.31 (F138). Nobody had asked which
#           rounding was closer before treating the figure as a cost.
#
# CAVEAT, stated so it is not lost: F138/F139's favourable QUALITY numbers
# were scored at N=512 with the fused gate FORCED OPEN, which is not the
# shipped decode regime (F137). At shipped decode width the effect is null,
# not favourable. The honest claim is FASTER AND NEUTRAL, not "more
# accurate". The speed half is measured where it ships.
#
# SCOPE: the evidence covers the 27B dense rungs (d4 packed, d2 packed, d2
# unpacked) -- which are, as of 2026-09-19, every dense VQ artifact in the
# fleet. A NEW dense geometry inherits this default UNTESTED; re-run
# decode-ladder + a chunk-inside-the-gate kl-ladder before trusting it there.
# VQ_DENSE_SS=0 restores the serial reduction.
def _dense_ss(src, name):
    old = """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
"""
    assert old in src, f"{name}: dense reduction text drifted"
    return src.replace(old, """        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            acc += simd_sum(sv * gacc);
        }
""")


_SRC_DENSE_D4_TILED_DEVX_SS = _dense_ss(
    _SRC_DENSE_D4_TILED_DEVX, "dense_d4_tiled")
_SRC_DENSE_PACKED_D4_TILED_DEVX_SS = _dense_ss(
    _SRC_DENSE_PACKED_D4_TILED_DEVX, "dense_packed_d4_tiled")
_SRC_DENSE_D2_TILED_DEVX_SS = _dense_ss(
    _SRC_DENSE_D2_TILED_DEVX, "dense_d2_tiled")
_SRC_DENSE_PACKED_D2_TILED_DEVX_SS = _dense_ss(
    _SRC_DENSE_PACKED_D2_TILED_DEVX, "dense_packed_d2_tiled")
_SRC_DENSE_D2_DEVX_SS = _dense_ss(_SRC_DENSE_D2_DEVX, "dense_d2")
_SRC_DENSE_PACKED_D2_DEVX_SS = _dense_ss(
    _SRC_DENSE_PACKED_D2_DEVX, "dense_packed_d2")

# ON by default since 2026-09-19: see the block comment above (F136/F138/
# F139/F142/F144). VQ_DENSE_SS=0 restores the serial reduction.
_DENSE_SS = os.environ.get("VQ_DENSE_SS", "1") == "1"


def _dense_src(base_name, base_src, devx, devx_ss):
    """Pick the dense kernel source and NAME for the current flag state.

    The name carries the arm, so _KERNELS never serves a devx kernel to a
    VQ_DENSE_DEVX=0 caller (or the reverse) out of a stale cache entry.
    """
    if _DENSE_DEVX and _DENSE_SS:
        return base_name + "_devx_ss", devx_ss
    if _DENSE_DEVX:
        return base_name + "_devx", devx
    if _DENSE_SS:
        return base_name + "_ss", _dense_ss(base_src, base_name)
    return base_name, base_src


_SRC_DECODE_PACKED = _PACK_FETCH + r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = VQ_CODE(crow, j0 + q);
        for (int u = 0; u < D; ++u)
            wrow[q * D + u] = (half)(s * (float)codebook[c * D + u]);
    }
"""

_SRC_DECODE = r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = (uint)crow[j0 + q];
        for (int u = 0; u < D; ++u)
            wrow[q * D + u] = (half)(s * (float)codebook[c * D + u]);
    }
"""

# Vector-store variant (2026-09-06, r3 compute survivor, conservative
# form). The ARITHMETIC is unchanged — each element is still one
# fp32 multiply + one RTNE round, so bits cannot move — only the store
# pairs into half2 (and the codebook read pairs likewise), halving the
# store instructions of a store-bound kernel. Requires D % 2 == 0 and
# G % 2 == 0 (every shipped artifact). Gated bit-exact against
# _SRC_DECODE in tests/test_vq_prefill_paths.py; VQ_DECODE_VEC=0 (the
# default until benched on a free box) keeps the scalar kernel.
_SRC_DECODE_VEC = r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half2* wrow2 = (device half2*)(
        w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G);
    const int j0 = g * SPG;
    const int D2 = D / 2;
    for (int q = 0; q < SPG; ++q) {
        const uint c = (uint)crow[j0 + q];
        const device half2* cb2 = (const device half2*)(codebook + c * D);
        for (int u = 0; u < D2; ++u) {
            const half2 cv = cb2[u];
            wrow2[q * D2 + u] = half2((half)(s * (float)cv.x),
                                      (half)(s * (float)cv.y));
        }
    }
"""

# Packed twin: identical to _SRC_DECODE_PACKED except the paired
# codebook read + half2 store (same bit-exactness argument).
_SRC_DECODE_PACKED_VEC = _PACK_FETCH + r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    const int WPR  = (NSUB + 31) / 32 * BITS;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half2* wrow2 = (device half2*)(
        w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G);
    const int j0 = g * SPG;
    const int D2 = D / 2;
    for (int q = 0; q < SPG; ++q) {
        const uint c = VQ_CODE(crow, j0 + q);
        const device half2* cb2 = (const device half2*)(codebook + c * D);
        for (int u = 0; u < D2; ++u) {
            const half2 cv = cb2[u];
            wrow2[q * D2 + u] = half2((half)(s * (float)cv.x),
                                      (half)(s * (float)cv.y));
        }
    }
"""

_DECODE_VEC = os.environ.get("VQ_DECODE_VEC", "0") != "0"

_KERNELS = {}

# SPECIALIZED (template-free) kernel cache, arc 5 (2026-09-02). Passing
# `template=` to a mx.fast.metal_kernel call costs ~4.5-7 us of HOST time PER
# CALL (measured: a null kernel goes 0.8 -> 5.3 us the moment any template is
# passed, and the packed-d8 simd call drops 8.9 -> 2.0 us without it), which
# at 144 expert dispatches/token is ~1 ms/token of pure Python-binding
# overhead -- most of the "module glue" arc 3 SS4 measured. The fix: bake the
# template values into the source as #defines (header), build ONE kernel per
# (name, template values) pair, and call it with NO template argument. The
# generated Metal code is the same code the template instantiation produced
# -- outputs are BIT-IDENTICAL (asserted in tests on real tensors) and the
# GPU time is unchanged (measured 36.9 vs 37.0 us/dispatch). VQ_SPEC_KERNELS=0
# restores the template path for A/B without a rebuild.
_SPEC_KERNELS = os.environ.get("VQ_SPEC_KERNELS", "1") != "0"

# mx dtype -> Metal source type name, for template values that are dtypes.
_METAL_TYPE = {}


def _metal_type_name(dt):
    if not _METAL_TYPE:
        _METAL_TYPE.update({
            mx.float16: "half", mx.bfloat16: "bfloat16_t",
            mx.float32: "float", mx.uint8: "uchar", mx.uint16: "ushort",
            mx.uint32: "uint", mx.int32: "int",
        })
    return _METAL_TYPE.get(dt)


def _get_kernel_spec(name, src, template):
    """A kernel with `template` baked in as #defines, callable WITHOUT the
    template argument. Falls back to the generic path (None) when a template
    value has no source spelling."""
    parts = []
    for tname, tval in template:
        if isinstance(tval, bool):  # bool is an int subclass; keep C spelling
            parts.append((tname, "true" if tval else "false"))
        elif isinstance(tval, int):
            parts.append((tname, str(tval)))
        else:
            mt = _metal_type_name(tval)
            if mt is None:
                return None
            parts.append((tname, mt))
    # string key so tests inspecting _KERNELS names ("simd" in n) keep working
    key = name + "|" + "|".join(f"{n}={v}" for n, v in parts)
    k = _KERNELS.get(key)
    if k is None:
        hdr = "".join(f"#define {n} {v}\n" for n, v in parts)
        spec_name = name + "_s" + "_".join(
            v.replace(" ", "") for _, v in parts)
        inp, outn = _kernel_sig(name)
        k = mx.fast.metal_kernel(
            name=spec_name, input_names=inp, output_names=outn,
            source=src, header=hdr)
        _KERNELS[key] = k
    return k


# dims arrays are tiny constant int32 uploads; building one per call costs
# ~0.6 us of host time x 144 calls/token. Cached by value instead.
_DIMS_CACHE = {}


def _dims_array(*vals):
    a = _DIMS_CACHE.get(vals)
    if a is None:
        a = mx.array(vals, dtype=mx.int32)
        mx.eval(a)
        _DIMS_CACHE[vals] = a
    return a


def _kernel_sig(name):
    """Input/output names for a kernel, chosen from its NAME prefix.

    Single source of truth, shared by _get_kernel and _get_kernel_spec. It
    used to be inlined in _get_kernel only, and _get_kernel_spec hardcoded
    the EXPERT signature -- which is why arc 5's specialized-kernel win never
    reached the dense path: a dense kernel has no `eidx` input, so asking for
    one would have bound the wrong buffers. Dense simply never called the
    spec path, and silently kept paying ~5-7 us of template processing per
    dispatch (2026-09-03, dense-kernel arc).
    """
    if name.startswith("vq_gemmseg"):
        # Fused segmented-tile VQ-GEMM (2026-09-07). MUST precede the
        # fallthrough or it binds the expert signature — arc-5 class.
        return (["codes", "codebook", "scales", "xsrc", "srcrows",
                 "tmeta", "dims"], ["y"])
    if name.startswith("vq_wdec"):
        # Weight-decode kernel: no x, no eidx. MUST precede the fallthrough
        # or it binds the expert signature's eidx — the arc-5 defect class.
        return ["codes", "codebook", "scales", "dims"], ["w"]
    if name.startswith("vq_dense"):
        return ["x", "codes", "codebook", "scales", "dims"], ["y"]
    if name.startswith("vq_fused"):
        return ["x", "eidx", "codes", "codebook", "scales", "dims"], ["y"]
    return ["codes", "codebook", "scales", "eidx", "dims"], ["w"]


def _get_kernel(name, src):
    if name not in _KERNELS:
        inp, out = _kernel_sig(name)
        _KERNELS[name] = mx.fast.metal_kernel(
            name=name, input_names=inp, output_names=out, source=src)
    return _KERNELS[name]


# largest d8 codebook the threadgroup variant may cache (K*16 B + x half4)
_D8_TG_MAX_K = 1024


# Apple's hard threadgroup allocation cap. The d4 threadgroup kernels cache
# BOTH the codebook and x as half4, so they need (K + NSUB) * 8 bytes; past
# this the kernel fails to LOAD (E134). Checked before dispatch rather than
# discovered at kernel load, matching the guard vq_dense.py uses for the
# dense d2 path.
_TG_CAP_BYTES = 32768


def _d4_tg_fits(K, NSUB):
    """True if the d4 threadgroup codebook+x cache fits Metal's cap."""
    return (K + NSUB) * 8 <= _TG_CAP_BYTES


def _d2_tg_fits(K, NSUB):
    """True if the d2 threadgroup codebook+x cache fits Metal's cap.

    All THREE fused d2 kernels (_SRC_FUSED_D2, _SRC_FUSED_D2_U32,
    _SRC_FUSED_PACKED_D2) allocate `half2 cb[MAX_K]` + `half2 xs[MAX_NSUB]`
    = (K + NSUB) * 4 B, so one predicate covers packed and unpacked alike.
    Every shipped d2 artifact is K<=2048 and lands far under the cap
    (gate/up at K2048/NSUB1408 = 13,824 B), so this is a LATENT cliff, not
    a live one -- found while auditing the gemma divergence (see
    docs/GEMMA-DIVERGENCE-2026-09-07.md). Without it, a K=4096 d2 artifact
    on a wide layer would fail at kernel LOAD (E134) with no graceful path,
    which is exactly the failure mode _d4_tg_fits was added to prevent on
    the d4 side. There is no device-cb d2 kernel, so the fallback is the
    decode path (see _fused), which caches nothing in threadgroup memory.
    """
    return (K + NSUB) * 4 <= _TG_CAP_BYTES


# Rows per threadgroup for the simdgroup-per-row EXPERT kernels, matching
# _DENSE_ROWS_TG (one 32-lane simdgroup per row, 32 rows per threadgroup).
_EXPERT_ROWS_TG = 32

# Rows (= 32-lane simdgroups) per threadgroup for the PACKED d8 simd kernel.
# The shipped value was 32, inherited from the dense kernels where it was
# swept. Re-swept 2026-09-02 on REAL Flash-Next 2.1bpw expert tensors
# (E=64, M3 Ultra, interleaved A/B, 21 reps x 50 dispatches, min/median):
#
#   gate L2  N=10   r2 69.2  r4 51.9  r8 49.9  r16 54.7  r32 61.9 us
#   gate L2  N=20   r2 111.5 r4 81.7  r8 82.2  r16 93.0  r32 103.4
#   up   L2  N=10   r2 66.3  r4 49.0  r8 52.1  r16 56.8  r32 60.9
#   up   L20 N=20            r4 83.3  r8 84.4              r32 101.3
#
# A clear interior optimum: r2 starves the machine of threadgroups, r32
# starves each thread of registers, and 4/8 are indistinguishable from each
# other while beating r32 by 1.18-1.24x on every shape/N measured. 8 is
# chosen over 4 for the larger x tile amortisation at equal measured cost.
# N=1 is dispatch-bound and flat across the whole sweep (23.4-24.5 us), so
# nothing regresses there. The kernel source is UNCHANGED, so the output is
# bit-identical by construction and asserted as such in the tests.
_EXPERT_ROWS_TG_D8_PACKED = int(os.environ.get("VQ_D8_ROWS_TG", "8"))

# Register-buffered packed-d8 simd kernel (2026-09-02). It is MEASURED SLOWER
# (see the ledger) and is therefore OFF; VQ_D8_REGBUF=1 dispatches it so the
# negative stays reproducible without a rebuild.
_D8_REGBUF = os.environ.get("VQ_D8_REGBUF", "0") != "0"


def _d8_regbuf_ok(G):
    """Can the register-buffered d8 kernel serve this G?

    SPG = G/8 codes per lane must tile a 32-code pack block exactly (so a
    lane's codes never straddle two blocks and NPH = 32/SPG is an integer),
    and only NPH <= 4 phases are emitted in the source. That is SPG in
    {8, 16, 32}, i.e. G in {64, 128, 256} -- G=64 is what every shipped d8
    artifact uses. Anything else keeps the existing kernel.
    """
    if not _D8_REGBUF or G % 8 != 0:
        return False
    spg = G // 8
    if spg not in (8, 16, 32):
        return False
    # the funnel window must never leave the lane's own 32-code pack block,
    # for EVERY bit width the packer can emit (a read past blk[BITS-1] would
    # be an out-of-bounds device read on the last block of a row).
    nph = 32 // spg
    for bits in range(2, 17):
        sh0max = ((nph - 1) * spg * bits) & 31
        nw = (sh0max + spg * bits + 31) >> 5
        w0max = ((nph - 1) * spg * bits) >> 5
        if w0max + nw - 1 > bits - 1:
            return False
    return True

# E141: the simdgroup-per-row d8 kernel is bit-identical to the thread-per-row
# one, so it is on by default; VQ_EXPERT_SIMD=0 restores the old layout for
# A/B measurement without a rebuild.
_EXPERT_SIMD = os.environ.get("VQ_EXPERT_SIMD", "1") != "0"

# The simd layout is for SMALL N only. Measured end-to-end on the 397B 2.2bpw
# (M4 Max, box otherwise idle, ~/seqcost.py, 2x40 reps, reproducible to
# 0.1 ms): seq=1 53.4 vs 54.9 ms and seq=2 80.1 vs 81.3 ms favour simd, but
# seq=3 is 108.4 vs 95.4 ms AGAINST it (-13.6%) and seq=4 137.1 vs 128.8
# (-6.4%). The isolated-kernel microbench says the opposite (1.35x at the
# same N=24) -- so the microbench is NOT predictive above decode-sized N and
# the whole-forward number is the one this gate follows. The cutoff is 20
# pairs because both measured WINS sit at or below it -- decode is
# top_k=8..10 pairs and the 397B's seq=2 (depth-1 MTP) is 2*10=20, measured
# 80.1 vs 81.3 ms -- while the first measured LOSS is seq=3 = 30 pairs.
# N=21..29 is unmeasured territory and deliberately falls to the old kernel.
_EXPERT_SIMD_MAX_N = int(os.environ.get("VQ_EXPERT_SIMD_MAX_N", "20"))

# E14x: read unpacked uint8 d2 codes four-at-a-time as uint32. Bit-identical,
# so on by default; VQ_D2_U32=0 restores the uchar kernel for A/B.
_D2_U32 = os.environ.get("VQ_D2_U32", "1") != "0"


# PLAN MEMO, arc 5 (2026-09-02). The dispatch logic below is pure on
# (shapes, dtypes, flags) -- ~2 us of Python per call, 144 calls/token. The
# resolved plan (which kernel, grid, threadgroup, dims, code view) is cached
# on exactly the values the logic reads, so a repeat call skips straight to
# the kernel invocation. The three module-level env flags in the key make the
# bench scripts' flag flips (V._D8_SIMDSUM etc.) still take effect. Plans
# live INSIDE _KERNELS (under ("plan", ...) keys) so the tests' existing
# _KERNELS.clear() invalidates them too.


def _fused_decode_fallback(x, eidx, codes, codebook, scales, pack_bits):
    """Serve a fused call through the decode path, in fused row order.

    _prefill wants rows grouped by expert, so sort, run, and invert the
    permutation. A gather is a pure row permutation, so the returned rows
    are the same rows _prefill computed -- no arithmetic is redone here.
    """
    idx_np = np.array(eidx, copy=False)
    order = np.argsort(idx_np, kind="stable")
    inv = np.argsort(order, kind="stable")
    y = _prefill(x[mx.array(order.astype(np.uint32))], idx_np[order],
                 codes, codebook, scales, pack_bits=pack_bits,
                 in_features=x.shape[1])
    return y[mx.array(inv.astype(np.uint32))]


def _fused(x, eidx, codes, codebook, scales, pack_bits=0, simd=None,
           d2_u32=None):
    # d2 threadgroup-capacity fallback. Mirrors the d4 _d4_tg_fits routing,
    # except d4 has a device-codebook kernel to fall back TO and d2 does not,
    # so the graceful path is the decode path -- _prefill decodes weights
    # with vq_decode, which caches nothing in threadgroup memory and is
    # capacity-independent at any K. Without this an oversized d2 artifact
    # fails at kernel LOAD (E134), which no try/except downstream can catch
    # into a useful answer. Latent today: max shipped d2 K is 2048.
    if codebook.shape[1] == 2 and not _d2_tg_fits(codebook.shape[0],
                                                  x.shape[1] // 2):
        return _fused_decode_fallback(x, eidx, codes, codebook, scales,
                                      pack_bits)
    key = ("plan", x.shape, x.dtype, codes.shape, codes.dtype, codebook.shape,
           scales.shape, pack_bits, simd, d2_u32,
           _D8_SIMDSUM, _D8_REGBUF, _D8_DEVX, _D8_SS, _SPEC_KERNELS)
    plan = _KERNELS.get(key)
    if plan is not None:
        view_u32, kern, name, src, template, grid, threadgroup, dims, N, OUT \
            = plan
        if view_u32:
            codes = mx.view(codes, dtype=mx.uint32)
        if kern is not None:
            (y,) = kern(inputs=[x, eidx, codes, codebook, scales, dims],
                        grid=grid, threadgroup=threadgroup,
                        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
        else:
            (y,) = _get_kernel(name, src)(
                inputs=[x, eidx, codes, codebook, scales, dims],
                template=list(template), grid=grid, threadgroup=threadgroup,
                output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
        return y
    return _fused_resolve(key, x, eidx, codes, codebook, scales, pack_bits,
                          simd, d2_u32)


def _fused_resolve(_plan_key, x, eidx, codes, codebook, scales, pack_bits=0,
                   simd=None, d2_u32=None):
    _view_u32 = False
    # U8-VIEW DISPATCH (E77/E90, 2026-08-20). Unpacked uint8 d4 rows are
    # byte-for-byte the pack_bits=8 word layout (little-endian; verified
    # against vq_pack.pack), and the packed fused kernel's simdgroup layout
    # measured 1.38-1.45x the one-thread-per-row unpacked kernel at every N
    # on both 35B expert shapes (+33% end-to-end prefill at step 512,
    # 1009-1020 vs 732-769 tok/s, rotlab--35B-vqK256codes). Zero-copy
    # reinterpret; output is BIT-IDENTICAL (mx.array_equal at N=8/512/4096
    # on real tensors, greedy 200-token generation byte-identical, KL gate
    # reproduced to every printed digit).
    if (pack_bits == 0 and codes.dtype == mx.uint8
            and codebook.shape[1] == 4 and codes.shape[2] % 4 == 0):
        codes = mx.view(codes, dtype=mx.uint32)
        pack_bits = 8
        _view_u32 = True
    # d=2 U8-VIEW (E14x, 2026-09-02): the same zero-copy reinterpret for d=2,
    # feeding _SRC_FUSED_D2_U32 (four codes per uint32 load instead of four
    # uchar loads). Guarded on NSUB % 4 == 0 so a row is a whole number of
    # words. Kept as a separate flag from pack_bits because the d2 packed
    # kernels take the generic VQ_CODE path, which re-reads a word per code.
    if d2_u32 is None:
        d2_u32 = _D2_U32
    _d2_view = (d2_u32 and pack_bits == 0 and codes.dtype == mx.uint8
                and codebook.shape[1] == 2 and codes.shape[2] % 4 == 0)
    if _d2_view:
        codes = mx.view(codes, dtype=mx.uint32)
        _view_u32 = True
    N, IN = x.shape
    E, OUT, _ = codes.shape
    K, D = codebook.shape
    NSUB = IN // D
    G = IN // scales.shape[2]
    dims = _dims_array(OUT, IN, D, G, N, K)
    tgx = 256 if OUT >= 256 else OUT
    # set by the simdgroup-per-row branches; None keeps the thread-per-row grid
    simd_rows = None
    if simd is None:
        simd = _EXPERT_SIMD and N <= _EXPERT_SIMD_MAX_N
    # threadgroup budget: float4 codebook cache is 16 B/entry, so K>1024 with
    # x cached overflows Apple's 32 KB. Large-K (and all packed) go through
    # the half4 variant — value-identical, half the footprint.
    # Dispatch is EXPLICIT on D with a hard raise for anything unhandled.
    # This used to fall through to the d8 kernels for any non-d4 codebook —
    # a d=2 artifact then read across codebook entries and generated pure
    # <pad> with NO error (2026-08-18, gemma26b vq-K256-d2). A wrong-memory
    # read must never be the default branch.
    if pack_bits:
        # Explicit per-D dispatch, same rule as unpacked: a packed kernel for
        # one D reads wrong memory at another D (the d4 kernel at d=2 returns
        # NaN), so anything unimplemented must raise, never fall through.
        if D == 4:
            # E134: fall back to the device-memory codebook when the
            # threadgroup cache would exceed the cap. Bit-identical, and the
            # only thing that changes is where cb is read from.
            if _d4_tg_fits(K, NSUB):
                # DEFAULT since 2026-09-11 (F58): bit-identical, 1.12x
                # decode on 35B d4-K2048. VQ_D4_WALK=0 restores VQ_CODE.
                if _D4_WALK:
                    name = f"vq_fused_packed{pack_bits}_d4_walk"
                    src = _SRC_FUSED_PACKED_D4_WALK
                else:
                    name = f"vq_fused_packed{pack_bits}"
                    src = _SRC_FUSED_PACKED
            else:
                name = f"vq_fused_packed{pack_bits}_d4_devcb"
                src = _SRC_FUSED_PACKED_D4_DEVCB
        elif D == 2:
            # NSUB (= IN/2) doubles vs d4 for the same IN; MAX_NSUB below is
            # computed from the actual IN, so the threadgroup x cache is
            # sized for the doubled subvector count automatically.
            name = f"vq_fused_packed{pack_bits}_d2"
            src = _SRC_FUSED_PACKED_D2
        elif D == 8:
            # d=8 packed (E100, 2026-08-21). NSUB = IN/8 -- a QUARTER of the
            # d4 value for the same IN -- so the codes row is WPR = NSUB/32
            # *BITS words; derived inside the kernel from IN, never passed.
            # K16384*d8*fp16 = 256 KB does not fit threadgroup memory, so the
            # large-K variant streams the codebook from DEVICE memory exactly
            # like _SRC_FUSED_D8. Verified bit-identical to the unpacked d8
            # kernels on synthetic codes at K=256/1024/4096/16384.
            # Unaligned NSUB is legal since the padded-tail format: pack()
            # zero-pads the last block and every packed kernel computes
            # ceil-WPR with inner loops bounded n < NSUB, so pad codes are
            # never read. The WPR shape assert below (already ceil) is the
            # remaining guard against a mis-packed tensor.
            if K <= _D8_TG_MAX_K:
                name = f"vq_fused_packed{pack_bits}_d8_tg"
                src = _SRC_FUSED_PACKED_D8_TG
            elif simd and IN // G >= 32:
                if _d8_regbuf_ok(G):
                    name = f"vq_fused_packed{pack_bits}_d8_simd_rb"
                    src = _SRC_FUSED_PACKED_D8_SIMD_RB
                elif _D8_SIMDSUM:
                    # arc 4: 1.15-1.23x, NOT bit-identical (1 half-ULP on
                    # 0.05-0.16% of elements). Off by default; see the source.
                    name = f"vq_fused_packed{pack_bits}_d8_simd_ss"
                    src = _SRC_FUSED_PACKED_D8_SIMD_SS
                elif _D8_DEVX and _D8_SS:
                    # DEFAULT since 2026-09-02: arc 5 devx + arc 4 simd_sum
                    # composed. 1-ULP equivalent to legacy (measured zero
                    # quality delta), +2.4% over devx alone. VQ_D8_SS=0
                    # drops back to plain devx (bit-identical to legacy).
                    name = f"vq_fused_packed{pack_bits}_d8_simd_devx_ss"
                    src = _SRC_FUSED_PACKED_D8_SIMD_DEVX_SS
                elif _D8_DEVX:
                    # arc 5: 1.19-1.23x at decode N, BIT-IDENTICAL (device x
                    # reads instead of the staged tile). VQ_D8_DEVX=0
                    # restores the staged kernel.
                    name = f"vq_fused_packed{pack_bits}_d8_simd_devx"
                    src = _SRC_FUSED_PACKED_D8_SIMD_DEVX
                else:
                    name = f"vq_fused_packed{pack_bits}_d8_simd"
                    src = _SRC_FUSED_PACKED_D8_SIMD
                simd_rows = _EXPERT_ROWS_TG_D8_PACKED
            else:
                name = f"vq_fused_packed{pack_bits}_d8"
                src = _SRC_FUSED_PACKED_D8
        else:
            raise NotImplementedError(
                f"no FUSED packed kernel for d={D}; only d=4, d=2 and d=8 "
                f"are implemented and each is dispatched explicitly.")
        if codes.shape[2] != (NSUB + 31) // 32 * pack_bits:
            raise ValueError(
                f"packed codes are {codes.shape[2]} words/row, expected "
                f"{(NSUB + 31) // 32 * pack_bits} for IN={IN}, d={D}, "
                f"bits={pack_bits}")
        if simd_rows is not None:
            tile = 32 * (G // 8) * 2 if D == 8 else 32 * (G // 4)
            template = [("T", x.dtype), ("MAX_TILE", tile),
                        ("BITS", pack_bits)]
            if name.endswith("_d8_simd_rb"):
                # SPG must be a COMPILE-TIME constant: the phase bodies are
                # specialised on it, and a runtime SPG would put a runtime
                # index on the register array and spill it.
                template.append(("SPG_C", G // 8))
        elif D == 8:
            template = [("T", x.dtype), ("MAX_NX4", IN // 4),
                        ("BITS", pack_bits)]
            if K <= _D8_TG_MAX_K:
                template.insert(1, ("MAX_K", K))
        elif D == 4 and not _d4_tg_fits(K, NSUB):
            template = [("T", x.dtype), ("MAX_NSUB", NSUB),
                        ("BITS", pack_bits)]
        else:
            template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB),
                        ("BITS", pack_bits)]
    elif _d2_view:
        # four uint8 codes per uint32 load; see _SRC_FUSED_D2_U32.
        name, src = "vq_fused_d2_u32", _SRC_FUSED_D2_U32
        template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif D == 2:
        name, src = "vq_fused_d2", _SRC_FUSED_D2
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif (D == 4 and K > 1024 and not _d4_tg_fits(K, NSUB)
          and simd and G % 16 == 0 and IN // G >= 32):
        # E134 shape, E141 layout: device codebook + one simdgroup per row.
        name = "vq_fused_d4_devcb_simd"
        src = _SRC_FUSED_D4_DEVCB_SIMD
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_TILE", 32 * (G // 4))]
        simd_rows = _EXPERT_ROWS_TG
    elif D == 4 and K > 1024 and not _d4_tg_fits(K, NSUB):
        # E134: (K + NSUB) * 8 over the cap -> device-memory codebook.
        name = "vq_fused_d4_devcb"
        src = _SRC_FUSED_D4_DEVCB
        template = [("T", x.dtype), ("CT", codes.dtype), ("MAX_NSUB", NSUB)]
    elif D == 4 and K > 1024:
        name = "vq_fused_d4_bigk"
        src = _SRC_FUSED_D4_BIGK
        template = [("T", x.dtype), ("CT", codes.dtype), ("MAX_K", K),
                    ("MAX_NSUB", NSUB)]
    elif D == 4:
        name, src = "vq_fused", _SRC_FUSED
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif D == 8 and K <= _D8_TG_MAX_K:
        name, src = "vq_fused_d8_tg", _SRC_FUSED_D8_TG
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NX4", IN // 4)]
    elif D == 8 and simd and IN // G >= 32:
        name, src = "vq_fused_d8_simd", _SRC_FUSED_D8_SIMD
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_TILE", 32 * (G // 8) * 2)]
        simd_rows = _EXPERT_ROWS_TG
    elif D == 8:
        name, src = "vq_fused_d8", _SRC_FUSED_D8
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_NX4", IN // 4)]
    else:
        raise NotImplementedError(f"no fused kernel for subvector dim d={D}")
    if simd_rows is not None:
        grid = (32, ((OUT + simd_rows - 1) // simd_rows) * simd_rows, N)
        threadgroup = (32, simd_rows, 1)
    else:
        grid = (((OUT + tgx - 1) // tgx) * tgx, N, 1)
        threadgroup = (tgx, 1, 1)
    kern = _get_kernel_spec(name, src, template) if _SPEC_KERNELS else None
    _KERNELS[_plan_key] = (_view_u32, kern, name, src, tuple(template),
                           grid, threadgroup, dims, N, OUT)
    if kern is not None:
        (y,) = kern(
            inputs=[x, eidx, codes, codebook, scales, dims],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(N, OUT)],
            output_dtypes=[x.dtype],
        )
        return y
    (y,) = _get_kernel(name, src)(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=template,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(N, OUT)],
        output_dtypes=[x.dtype],
    )
    return y


# rows each threadgroup owns in the dense kernels (one 32-lane simdgroup per
# row). Swept 2/4/8/16/32 on the dependent e4b-shaped chain (M3 Ultra,
# 2026-08-19): 211.9 / 118.8 / 79.6 / 73.0 / 68.1 us per matmul — bigger
# threadgroups amortise the codebook+x threadgroup loads over more rows.
_DENSE_ROWS_TG = 32


# Threadgroup budget, in bytes, for the dense d2 kernels. Both stage a half2
# codebook (K entries) plus a half2 staging area for x; the untiled kernels
# size that area by NSUB (layer width), the tiled ones by one block span.
# +1024 is headroom for the compiler's own threadgroup use, matching the
# guard vq_dense.py has always applied.
_TG_CAP = 32768


def _dense_tg_bytes(K, nx, tiled_span=None):
    return (K + (tiled_span if tiled_span is not None else nx)) * 4 + 1024


def _dense_tiled(K, NSUB, G):
    """Use the NSUB-tiled kernel? Yes whenever the untiled one would not fit,
    which is the case the fallback-to-full-decode used to handle at ~40x the
    cost. The untiled kernel is kept for shapes it already serves so that
    nothing currently working changes path, and so the two can be A/B'd for
    bit-exactness wherever both are legal (tests/test_vq_dense_tiled.py)."""
    if os.environ.get("VQ_DENSE_TILED") == "1":
        return True
    if os.environ.get("VQ_DENSE_TILED") == "0":
        return False
    return _dense_tg_bytes(K, NSUB) > _TG_CAP


def dense_fits(K, IN, G, d=2):
    """Can a dense VQ linear of this shape run on a fused kernel at all?
    vq_dense.py asks this before choosing the decode fallback."""
    if d == 4:
        # The d4 kernel keeps the codebook in device memory and stages only a
        # tile of x, so nothing about the shape can exceed the budget:
        # 32*(G/4) half4 = 8*G bytes. Only the unroll constraint can fail.
        return G % 16 == 0
    if d != 2:
        return False
    NSUB = IN // d
    # Must agree with _dense_tiled, including its debug override: reporting
    # "fusable" for a shape the dispatcher will then build untiled is a
    # kernel-load crash, not a fallback.
    if _dense_tiled(K, NSUB, G):
        return _dense_tg_bytes(K, NSUB, 32 * (G // 2)) <= _TG_CAP
    return _dense_tg_bytes(K, NSUB) <= _TG_CAP


def _dense_dispatch(name, src, template, x, codes, codebook, scales,
                    OUT, IN, D, G, N, K):
    """The one place a dense kernel is launched.

    Carries arc 5's HOST-side wins across to the dense path, where they had
    never been applied (2026-09-03): a specialized template-free kernel when
    VQ_SPEC_KERNELS is on, and a cached dims array. Arc 5 measured the
    `template=` argument at ~5-7 us of host time PER CALL; the 27B dense line
    dispatches 192 VQ linears per token, so that argument alone was ~1.0-1.3
    ms of a ~54 ms token. Both changes are BIT-IDENTICAL: the specialized
    kernel is the same generated Metal code with the template values baked in
    as #defines, and the dims array is the same six int32s.
    """
    rows = _DENSE_ROWS_TG
    dims = _dims_array(OUT, IN, D, G, N, K)
    grid = (32, ((OUT + rows - 1) // rows) * rows, N)
    tg = (32, rows, 1)
    kern = _get_kernel_spec(name, src, template) if _SPEC_KERNELS else None
    if kern is not None:
        (y,) = kern(inputs=[x, codes, codebook, scales, dims],
                    grid=grid, threadgroup=tg,
                    output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
        return y
    (y,) = _get_kernel(name, src)(
        inputs=[x, codes, codebook, scales, dims], template=template,
        grid=grid, threadgroup=tg,
        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
    return y


def _dense_fused(x, codes, codebook, scales, pack_bits=0, in_features=None):
    """Dense d=2 fused VQ matmul: y[N, OUT] = x [N, IN] @ decode(codes).T.

    codes [OUT, NSUB] (uint8/uint16) or [OUT, WPR] (uint32, pack_bits-wide
    fields); scales [OUT, IN/G]. Output is BIT-IDENTICAL to _fused with E=1
    and eidx=0 (see the kernel-source comment) — verified before first ship.
    Dispatch is explicit with hard raises, never a silent fallthrough.
    """
    N, IN = x.shape
    OUT = codes.shape[0]
    K, D = codebook.shape
    if D not in (2, 4):
        raise NotImplementedError(
            f"no DENSE fused kernel for d={D}; only d=2 and d=4 are "
            f"implemented and dispatched explicitly (a kernel for one D reads "
            f"wrong memory at another D — see the expert-kernel dispatch).")
    NSUB = IN // D
    NGRP = scales.shape[1]
    G = IN // NGRP
    if D == 4:
        if G % 16 != 0:
            raise NotImplementedError(
                f"dense d4 kernel unrolls 4 codes at a time, so it needs "
                f"G % 16 == 0 (SPG % 4 == 0), got G={G}")
        if pack_bits:
            exp_w = (NSUB + 31) // 32 * pack_bits
            if codes.shape[1] != exp_w:
                raise ValueError(
                    f"packed dense d4: codes are {codes.shape[1]} words/row, "
                    f"expected {exp_w} for IN={IN}, bits={pack_bits}")
            name, src = _dense_src(
                f"vq_dense_packed{pack_bits}_d4_tiled",
                _SRC_DENSE_PACKED_D4_TILED,
                _SRC_DENSE_PACKED_D4_TILED_DEVX,
                _SRC_DENSE_PACKED_D4_TILED_DEVX_SS)
            tmpl = [("T", x.dtype), ("MAX_TILE", 32 * (G // 4)),
                    ("BITS", pack_bits)]
        else:
            if codes.shape[1] != NSUB:
                raise ValueError(f"dense d4: codes are {codes.shape[1]} cols, "
                                 f"expected NSUB={NSUB} for IN={IN}")
            name, src = _dense_src(
                "vq_dense_d4_tiled", _SRC_DENSE_D4_TILED,
                _SRC_DENSE_D4_TILED_DEVX, _SRC_DENSE_D4_TILED_DEVX_SS)
            tmpl = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_TILE", 32 * (G // 4))]
        return _dense_dispatch(name, src, tmpl, x, codes, codebook, scales,
                               OUT, IN, D, G, N, K)
    if G % 8 != 0:
        raise NotImplementedError(f"dense d2 kernel needs G % 8 == 0, got {G}")
    if pack_bits:
        if G != 64:
            # the packed kernel's register block-fetch assumes one scale
            # group == one 32-code pack block (see its source comment).
            raise NotImplementedError(
                f"packed dense d2 kernel requires group_size=64, got {G}")
        exp_in = in_features
        if exp_in is not None and exp_in != IN:
            raise ValueError(f"packed dense: x is IN={IN} but module says "
                             f"in_features={exp_in}")
        if codes.shape[1] != (NSUB + 31) // 32 * pack_bits:
            raise ValueError(
                f"packed dense: codes are {codes.shape[1]} words/row, "
                f"expected {(NSUB + 31) // 32 * pack_bits} for IN={IN}, "
                f"bits={pack_bits}")
        if _dense_tiled(K, NSUB, G):
            name, src = _dense_src(
                f"vq_dense_packed{pack_bits}_d2_tiled",
                _SRC_DENSE_PACKED_D2_TILED,
                _SRC_DENSE_PACKED_D2_TILED_DEVX,
                _SRC_DENSE_PACKED_D2_TILED_DEVX_SS)
            template = [("T", x.dtype), ("MAX_K", K),
                        ("MAX_TILE", 32 * (G // 2)), ("BITS", pack_bits)]
        else:
            name, src = _dense_src(
                f"vq_dense_packed{pack_bits}_d2", _SRC_DENSE_PACKED_D2,
                _SRC_DENSE_PACKED_D2_DEVX, _SRC_DENSE_PACKED_D2_DEVX_SS)
            template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB),
                        ("BITS", pack_bits)]
    else:
        if codes.shape[1] != NSUB:
            raise ValueError(f"dense: codes are {codes.shape[1]} cols, "
                             f"expected NSUB={NSUB} for IN={IN}, d={D}")
        if _dense_tiled(K, NSUB, G):
            name, src = _dense_src(
                "vq_dense_d2_tiled", _SRC_DENSE_D2_TILED,
                _SRC_DENSE_D2_TILED_DEVX, _SRC_DENSE_D2_TILED_DEVX_SS)
            template = [("T", x.dtype), ("CT", codes.dtype),
                        ("MAX_K", K), ("MAX_TILE", 32 * (G // 2))]
        else:
            name, src = _dense_src(
                "vq_dense_d2", _SRC_DENSE_D2,
                _SRC_DENSE_D2_DEVX, _SRC_DENSE_D2_DEVX_SS)
            template = [("T", x.dtype), ("CT", codes.dtype),
                        ("MAX_K", K), ("MAX_NSUB", NSUB)]
    return _dense_dispatch(name, src, template, x, codes, codebook, scales,
                           OUT, IN, D, G, N, K)


# --------------------------------------------------------------------------- #
# vq_wdec — packed codes -> dense fp16 weight tile, one dispatch.
#
# Serves the PACKED arm of vq_dense._decode_matmul's fallback (prefill,
# N > _fused_max_n). Replaces _unpack_rows (the [R, NSUB] slab) + _decode's
# gather + broadcast-scale — three full materialisations — with one kernel
# whose only buffer is the [rows, IN] fp16 output. The GEMM downstream is
# untouched: same w shape/dtype/strides, so mlx picks the same tiling and
# bits cannot move there (the row-tiling lesson).
#
# Bit-exactness is structural: every w[r, e] is an independent
# (half)(float(s) * float(cb)) — no reduction, no ordering. The exact product
# of two fp16 significands fits fp32, so the float path applies exactly one
# RTNE rounding, same as mlx's half*half elementwise op. That one assumption
# is the ship gate: tests/test_vq_wdec.py compares uint16 BIT PATTERNS (not
# array_equal — NaN != NaN) against _decode(_unpack_rows(...)).
#
# Codebook stays in DEVICE memory — zero threadgroup allocation, so the E134
# "Threadgroup memory size exceeds" load-failure class is structurally
# impossible and wdec_fits has no cap term. (A threadgroup-cached variant was
# considered and rejected: a write kernel reads each codebook vector once per
# code with no intra-threadgroup reuse to pay for the barrier, and
# K4096*d4*2 B is exactly the 32 KB cap anyway.)
# --------------------------------------------------------------------------- #

_SRC_WDEC = _PACK_FETCH + r"""
    // dims: [TROWS, IN, NGRP, r0]   (BITS, D_BAKE, GROUP are baked)
    const int TROWS = dims[0];   // rows in THIS tile, not the module's OUT
    const int IN    = dims[1];
    const int NGRP  = dims[2];
    const int r0    = dims[3];

    const int D    = D_BAKE;
    const int G    = GROUP;      // compile-time => SPG folds to mul/shift
    const int NSUB = IN / D;
    const int SPG  = G / D;      // codes per scale group
    const int WPR  = (NSUB / 32) * BITS;

    uint g = thread_position_in_grid.x;   // scale-group index
    uint r = thread_position_in_grid.y;   // row within tile
    if (g >= (uint)NGRP || r >= (uint)TROWS) return;

    const device uint* crow = codes  + (size_t)(r0 + r) * WPR;
    const device half* srow = scales + (size_t)(r0 + r) * NGRP;
    device half* wrow       = w + (size_t)r * IN + (size_t)g * G;

    const float s = (float)srow[g];       // one broadcast scalar per thread

    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = VQ_CODE(crow, j0 + q);
        const device half* cbp = codebook + (size_t)c * D;
        for (int u = 0; u < D; ++u)
            wrow[q * D + u] = (half)(s * (float)cbp[u]);
    }
"""

# rows of w each threadgroup covers. The expert sweep found 4-8 beats 32 for
# its dot-product kernel; this one is store-bound, so the value must be swept
# on real tensors (VQ_WDEC_ROWS_TG for the sweep), not inherited.
_WDEC_ROWS_TG = int(os.environ.get("VQ_WDEC_ROWS_TG", "4"))

# Escape hatch: VQ_DENSE_DECODE_FUSE=0 restores _unpack_rows + _decode for
# A/B without a rebuild. Read at import like every other arm switch.
_WDEC_FUSE = os.environ.get("VQ_DENSE_DECODE_FUSE", "1") != "0"


def wdec_fits(D, NSUB, G, pack_bits):
    """May the packed fallback take the vq_wdec kernel? On False the caller
    falls through to _unpack_rows + _decode unchanged. NSUB % 32 mirrors
    _unpack_rows' own hard precondition, so A/B parity is exact; unpacked
    codes (pack_bits == 0) keep their bit-frozen one-gather path."""
    return (_WDEC_FUSE
            and D in (2, 4)
            and 0 < pack_bits <= 16
            and NSUB % 32 == 0
            and G % D == 0)


def wdec_decode(codes, codebook, scales, rows, IN, G, pack_bits, r0=0):
    """Decode packed rows [r0, r0+rows) into a [rows, IN] fp16 weight tile.

    `codes`/`scales` are the FULL module tensors — the row base travels in
    dims, so no Python-side slice (and no packed-slab copy) is ever made.
    `codebook` must already be fp16 (the caller's astype), never the raw
    tensor: a bf16 source read as device half* would be silent garbage.
    """
    D = int(codebook.shape[1])
    NGRP = IN // G
    name = f"vq_wdec_packed{pack_bits}_d{D}"
    template = [("BITS", pack_bits), ("D_BAKE", D), ("GROUP", G)]
    dims = _dims_array(rows, IN, NGRP, r0)
    rtg = _WDEC_ROWS_TG
    grid = (((NGRP + 31) // 32) * 32, ((rows + rtg - 1) // rtg) * rtg, 1)
    tg = (min(32, max(1, NGRP)), rtg, 1)
    kern = _get_kernel_spec(name, _SRC_WDEC, template) if _SPEC_KERNELS \
        else None
    if kern is not None:
        (w,) = kern(inputs=[codes, codebook, scales, dims],
                    grid=grid, threadgroup=tg,
                    output_shapes=[(rows, IN)], output_dtypes=[mx.float16])
        return w
    (w,) = _get_kernel(name, _SRC_WDEC)(
        inputs=[codes, codebook, scales, dims], template=template,
        grid=grid, threadgroup=tg,
        output_shapes=[(rows, IN)], output_dtypes=[mx.float16])
    return w


# T5, at import: the sig table must bind this kernel's actual buffers. A
# vq_wdec* name reaching the expert fallthrough binds an eidx that does not
# exist — silent garbage, not an error (the documented arc-5 defect).
assert _kernel_sig("vq_wdec_packed9_d2") == (
    ["codes", "codebook", "scales", "dims"], ["w"])


# --------------------------------------------------------------------------- #
# vq_gemmseg — fused segmented-tile VQ-GEMM for MoE prefill (2026-09-07,
# r4 swarm design: survivors 0/2/6 architecture + survivor 5 inner loop).
#
# Dequantizes packed VQ codes INSIDE the matmul tile loop — the fp16 expert
# weight matrix is never materialized. One dispatch per linear covers ALL
# touched experts via per-tile metadata over the count-sorted contiguous
# rows: (expert, row_start, nrows) per 32-row tile, so a tile never
# straddles experts and padding is bounded by <32 rows per expert instead
# of the padded-GEMM's chunk-cap (measured pad 1.567 -> ~1.0x).
#
# Marlin-style amortization (the E141 fix at prefill scale): each scale
# group's [32 out-rows x G] weight tile is decoded ONCE into threadgroup
# memory by 128 cooperating threads (independent VQ_CODE fetches, no
# dependent-load chain) and reused across 32 token rows — a 32x
# amortization of code extraction vs the per-(row,token) fused kernels.
#
# Threadgroup budget @ K512/d2/G64: cb 2KB + w_tile 4KB + x_tile 4KB
# ~= 10KB << 32KB cap (E134-safe; budget asserted in gemmseg_fits).
#
# ACCEPTANCE CONTRACT (mission r4): fp32 accumulation in-tile; this is
# NOT bit-identical to decode+GEMM (reduction order differs by design);
# gates are numeric (max rel < 1e-3 on real routing) + score-identity,
# and the path is DEFAULT OFF: VQ_MOE_FUSED_GEMM=1 opts in (unscored
# serving) until promoted.
# --------------------------------------------------------------------------- #

_SRC_GEMMSEG = _PACK_FETCH + r"""
    // dims: [OUT, IN, NGRP, K, NTILES]
    // tmeta: int32 [NTILES, 3] = (expert, row_start, nrows)
    // baked: BITS, GROUP, OTILE(=32), RTILE(=32)
    const int OUT   = dims[0];
    const int IN    = dims[1];
    const int NGRP  = dims[2];
    const int K     = dims[3];
    const int G     = GROUP;
    const int SPG   = G / 2;                 // d=2 codes per scale group
    const int NSUB  = IN / 2;
    const int WPR   = (NSUB + 31) / 32 * BITS;

    uint lane = thread_position_in_threadgroup.x;   // 0..31
    uint sg   = thread_position_in_threadgroup.y;   // 0..3
    uint tid  = sg * 32 + lane;                     // 0..127
    uint otile = thread_position_in_grid.x / 32;    // OUT/OTILE tiles
    uint rtile = thread_position_in_grid.y / 4;     // row tiles

    const int e    = tmeta[rtile * 3 + 0];
    const int r0   = tmeta[rtile * 3 + 1];
    const int nrow = tmeta[rtile * 3 + 2];
    const int o0   = (int)otile * OTILE;

    threadgroup half2 cb[MAX_K];
    threadgroup half  wt[OTILE][GROUP];
    threadgroup half  xt[RTILE][GROUP];

    for (uint i = tid; i < (uint)K; i += 128u)
        cb[i] = ((const device half2*)codebook)[i];

    // fp32 accumulators: thread owns out-row (o0+lane) x tokens sg*8..+8
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    const int my_r   = o0 + (int)lane;               // output row (col of y)
    const device uint* crow = codes
        + (size_t)e * OUT * WPR + (size_t)my_r * WPR;
    const device half* srow = scales
        + (size_t)e * OUT * NGRP + (size_t)my_r * NGRP;

    // decode assignment: thread tid covers w-tile row wr = tid/4,
    // code slots q = (tid%4)*SPG/4 .. +SPG/4 (SPG=32 -> 8 codes = 16 halfs)
    const int wr  = (int)tid / 4;
    const int q0  = ((int)tid % 4) * (SPG / 4);
    const device uint* wrow_codes = codes
        + (size_t)e * OUT * WPR + (size_t)(o0 + wr) * WPR;
    // x assignment: thread tid covers x-tile row xr = tid/4, same q span
    const int xr = (int)tid / 4;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int g = 0; g < NGRP; ++g) {
        const int j0 = g * SPG;
        // Phase 1: decode this group's [OTILE x G] weight tile, unscaled
        if (o0 + wr < OUT) {
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                const uint c = VQ_CODE(wrow_codes, j0 + q);
                const half2 v = cb[c];
                wt[wr][q * 2]     = v.x;
                wt[wr][q * 2 + 1] = v.y;
            }
        }
        // Phase 2: stage this group's [RTILE x G] x tile (zeros past nrow)
        {
            const device half* xrow = 0;
            if (xr < nrow)
                xrow = xsrc + (size_t)srcrows[r0 + xr] * IN + (size_t)g * G;
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                xt[xr][q * 2]     = (xr < nrow) ? xrow[q * 2]     : (half)0;
                xt[xr][q * 2 + 1] = (xr < nrow) ? xrow[q * 2 + 1] : (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 3: accumulate. Thread: out-row (o0+lane), tokens sg*8..+8
        if (my_r < OUT) {
            const float s = (float)srow[g];
            for (int t = 0; t < 8; ++t) {
                const int tt = (int)sg * 8 + t;
                float gacc = 0.0f;
                for (int k = 0; k < G; ++k)
                    gacc += (float)wt[lane][k] * (float)xt[tt][k];
                acc[t] = fma(s, gacc, acc[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (my_r < OUT) {
        for (int t = 0; t < 8; ++t) {
            const int tt = (int)sg * 8 + t;
            if (tt < nrow)
                y[(size_t)(r0 + tt) * OUT + my_r] = (half)acc[t];
        }
    }
"""

# v2 (2026-09-07, same night): v1's scalar threadgroup MAC measured 34%
# SLOWER than legacy (11.4 vs 8.5s) — scalar fp32 against threadgroup
# memory loses to steel GEMM's simdgroup pipelines, exactly the mission's
# R3-class warning. v2 keeps v1's segmentation + decode-once staging but
# runs phase 3 as simdgroup_half8x8 matmuls: wt is decoded TRANSPOSED and
# PRE-SCALED into threadgroup (scale applied at half precision — numerics
# covered by the same rel<1e-3 gate), C tiles accumulate in
# simdgroup_float8x8 registers across the whole group loop.
# VQ_MOE_FUSED_GEMM=1 selects v1 (scalar, kept for A/B), =2 selects v2.
_SRC_GEMMSEG2 = _PACK_FETCH + r"""
    // dims: [OUT, IN, NGRP, K, NTILES]; tmeta int32 [NTILES,3]
    // baked: BITS, GROUP(=64), MAX_K, D_BAKE(2|4); tiles fixed 32x32
    // ONE source serves d2 and d4 (2026-09-07, r5 swarm): only phase 1
    // changes — codebook entry width (half2 vs half4) and how many halves
    // one code contributes. SPG folds at compile time so every thread
    // still stages exactly 16 halves of wtT and 16 of xt either way
    // (d2: 8 codes x 2 halves; d4: 4 codes x 4). Phase 3's simdgroup
    // matmul is OUT x TOKEN geometry — completely d-independent.
    const int OUT   = dims[0];
    const int IN    = dims[1];
    const int NGRP  = dims[2];
    const int K     = dims[3];
    const int G     = GROUP;
    const int SPG   = G / D_BAKE;
    const int NSUB  = IN / D_BAKE;
#if BITS == 0
    // UNPACKED codes (uint8/uint16): one CT element per code. The packed
    // row stride ((NSUB+31)/32*BITS) DEGENERATES TO 0 at BITS==0, so every
    // row would read row 0 — silent garbage, the exact class the
    // 2026-08-18 prefill guard exists for. Stride is NSUB CT elements.
    const int WPR   = NSUB;
#else
    const int WPR   = (NSUB + 31) / 32 * BITS;
#endif

    uint lane = thread_position_in_threadgroup.x;   // 0..31
    uint sg   = thread_position_in_threadgroup.y;   // 0..3
    uint tid  = sg * 32 + lane;
    uint otile = thread_position_in_grid.x / 32;
    uint rtile = thread_position_in_grid.y / 4;

    const int e    = tmeta[rtile * 3 + 0];
    const int r0   = tmeta[rtile * 3 + 1];
    const int nrow = tmeta[rtile * 3 + 2];
    // OT2 (0 or 1): output-block pairing (arm 1, KERNEL-BODY-CAMPAIGN).
    // Each threadgroup owns 32*(1+OT2) output columns; the gathered xt slab
    // is staged ONCE per threadgroup, so total xt staging traffic halves at
    // OT2=1 (affine stages X once per BM tile; this closes half the gap).
    // wtT is decoded serially per block -- threadgroup bytes UNCHANGED.
    const int o0   = (int)otile * 32 * (1 + OT2);

#if CB_DEV
    // BIG-K arm (2026-09-07): K*2*D exceeds the 32 KB threadgroup cap
    // (d4 K8192 = 64 KB, d8 K16384 = 256 KB), so the codebook stays in
    // DEVICE memory. This is only viable because phase 1 decodes each
    // [32 out-rows x G] tile ONCE and phase 3 reuses it across 32 token
    // rows: the random-gather cost is paid once per tile, not per token,
    // and 128 threads issue their lookups independently (the E141 fix —
    // no dependent-load chain). Whether that amortization actually holds
    // is a MEASUREMENT, not a claim; see VQGEMM-BENCH.
  #if D_BAKE == 2
    const device half2* cb = (const device half2*)codebook;
  #else
    // d4: one half4 per code. d8: TWO (Metal has no half8) — the same
    // cb4[2*c], cb4[2*c+1] pairing _SRC_FUSED_D8 ships.
    const device half4* cb = (const device half4*)codebook;
  #endif
#else
  #if D_BAKE == 2
    threadgroup half2 cb[MAX_K];        // K*4 B
  #else
    threadgroup half4 cb[MAX_K];        // K*8 B
  #endif
#endif
    // RTILE = token rows per threadgroup. 32 is the shipped tiling; 64
    // pairs two token tiles so phase 1 (the weight decode) runs ONCE per
    // 64 rows instead of twice. wtT is unchanged (it is OUT x G, not
    // token-shaped); xt and ybuf double. CB_DEV only: with a threadgroup
    // codebook at d4-K2048 the sum would be 16384 + 20480 = 36864 > cap.
    threadgroup half  wtT[GROUP][32];    // transposed, pre-scaled (4 KB)
    // XPAD (0 or 8): leading-dimension pad on xt. GROUP = 64 halves is an
    // exact 128 B row stride, so all eight rows of a simdgroup A fragment
    // land on one bank column -- the worst-conflict geometry. Affine's
    // steel pads EVERY staged tile by +16 B for exactly this
    // (quantized.h BK_padded/BN_padded); swarm7 survivor #1, F49 arm.
    // +8 halves = +16 B/row = 512 B total @RTILE=32. Bit-exact: same
    // logical elements, only the row stride changes.
    threadgroup half  xt[RTILE][GROUP + XPAD];  // 4/4.5 KB @32, 8/9 KB @64
#if !DSTORE
    // DSTORE=1 deletes this buffer entirely: the epilogue stores straight
    // from the simdgroup accumulators via steel's lane mapping (mma.h:51-53
    // in mlx -- fm/fn per lane, 2 elements each). -4 KB threadgroup.
    threadgroup float ybuf[RTILE][32];   // 4 KB @32, 8 KB @64
#endif

#if !CB_DEV
    for (uint i = tid; i < (uint)K; i += 128u)
  #if D_BAKE == 2
        cb[i] = ((const device half2*)codebook)[i];
  #else
        cb[i] = ((const device half4*)codebook)[i];
  #endif
#endif

    // decode assignment: thread -> w row wr=tid/4, code span q0..q0+SPG/4
    const int wr = (int)tid / 4;
    const int q0 = ((int)tid % 4) * (SPG / 4);
#if BITS == 0
    const device CT* wrow_base = codes
        + (size_t)e * OUT * WPR + (size_t)(o0 + wr) * WPR;
    #define VQ_FETCH(j) ((uint)wrow_codes[j])
#else
    const device uint* wrow_base = codes
        + (size_t)e * OUT * WPR + (size_t)(o0 + wr) * WPR;
    #define VQ_FETCH(j) VQ_CODE(wrow_codes, j)
#endif
    const device half* srow_base = scales
        + (size_t)e * OUT * NGRP + (size_t)(o0 + wr) * NGRP;
    const int xr = (int)tid / 4;

    simdgroup_float8x8 C0 = simdgroup_float8x8(0);
    simdgroup_float8x8 C1 = simdgroup_float8x8(0);
    simdgroup_float8x8 C2 = simdgroup_float8x8(0);
    simdgroup_float8x8 C3 = simdgroup_float8x8(0);
#if RTILE == 64
    simdgroup_float8x8 C4 = simdgroup_float8x8(0);
    simdgroup_float8x8 C5 = simdgroup_float8x8(0);
    simdgroup_float8x8 C6 = simdgroup_float8x8(0);
    simdgroup_float8x8 C7 = simdgroup_float8x8(0);
#endif
#if OT2
    simdgroup_float8x8 Db0 = simdgroup_float8x8(0);
    simdgroup_float8x8 Db1 = simdgroup_float8x8(0);
    simdgroup_float8x8 Db2 = simdgroup_float8x8(0);
    simdgroup_float8x8 Db3 = simdgroup_float8x8(0);
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

#if OT2 && (RTILE == 64)
#error "OT2 pairs output blocks at RTILE=32 only; combine is unimplemented"
#endif
#if PIPE
    // Prefetch pipeline (KERNEL-BODY-CAMPAIGN arm 1.5, salvaged design):
    // g+1's code words + scale are INDEPENDENT device loads issued before
    // the mma, so they fly during the matmul; after the barrier only the
    // dependent codebook gather remains. SPG/4 <= 8 across d2/d4/d8.
    uint  pfc[1 + OT2][8];
    float pfs[1 + OT2];
#endif
    for (int g = 0; g < NGRP; ++g) {
        const int j0 = g * SPG;
      for (int ob = 0; ob < 1 + OT2; ++ob) {
        const int oo = o0 + ob * 32;
#if BITS == 0
        const device CT* wrow_codes = wrow_base + (size_t)ob * 32 * WPR;
#else
        const device uint* wrow_codes = wrow_base + (size_t)ob * 32 * WPR;
#endif
        const device half* srow_w = srow_base + (size_t)ob * 32 * NGRP;
        if (oo + wr < OUT) {
#if PIPE
            uint  cq[8];
            float s;
            if (g == 0) {
                s = (float)srow_w[0];
                for (int q = q0; q < q0 + SPG / 4; ++q)
                    cq[q - q0] = VQ_FETCH(j0 + q);
            } else {
                s = pfs[ob];
                for (int q = q0; q < q0 + SPG / 4; ++q)
                    cq[q - q0] = pfc[ob][q - q0];
            }
#else
            const float s = (float)srow_w[g];
#endif
            for (int q = q0; q < q0 + SPG / 4; ++q) {
#if PIPE
                const uint c = cq[q - q0];
#else
                const uint c = VQ_FETCH(j0 + q);
#endif
#if D_BAKE == 2
                const half2 v = cb[c];
                wtT[q * 2][wr]     = (half)(s * (float)v.x);
                wtT[q * 2 + 1][wr] = (half)(s * (float)v.y);
#elif D_BAKE == 4
                const half4 v = cb[c];
                wtT[q * 4][wr]     = (half)(s * (float)v.x);
                wtT[q * 4 + 1][wr] = (half)(s * (float)v.y);
                wtT[q * 4 + 2][wr] = (half)(s * (float)v.z);
                wtT[q * 4 + 3][wr] = (half)(s * (float)v.w);
#else
                const half4 v0 = cb[2 * c];
                const half4 v1 = cb[2 * c + 1];
                wtT[q * 8][wr]     = (half)(s * (float)v0.x);
                wtT[q * 8 + 1][wr] = (half)(s * (float)v0.y);
                wtT[q * 8 + 2][wr] = (half)(s * (float)v0.z);
                wtT[q * 8 + 3][wr] = (half)(s * (float)v0.w);
                wtT[q * 8 + 4][wr] = (half)(s * (float)v1.x);
                wtT[q * 8 + 5][wr] = (half)(s * (float)v1.y);
                wtT[q * 8 + 6][wr] = (half)(s * (float)v1.z);
                wtT[q * 8 + 7][wr] = (half)(s * (float)v1.w);
#endif
            }
        } else {
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                for (int u = 0; u < D_BAKE; ++u)
                    wtT[q * D_BAKE + u][wr] = (half)0;
            }
        }
#if PIPE
        // issue g+1's independent loads NOW -- they overlap the mma below
        if (g + 1 < NGRP && oo + wr < OUT) {
            pfs[ob] = (float)srow_w[g + 1];
            const int j1 = (g + 1) * SPG;
            for (int q = q0; q < q0 + SPG / 4; ++q)
                pfc[ob][q - q0] = VQ_FETCH(j1 + q);
        }
#endif
        if (ob == 0) {
            // TIO = I/O element type (half or bfloat16_t). The (half)
            // convert at stage-in is round-to-nearest -- bit-identical to
            // the host astype it replaces; MACs stay half8x8 regardless.
            // Staged ONCE per threadgroup-group even at OT2: the whole
            // point of the pairing (KERNEL-BODY-CAMPAIGN arm 1).
#if PH2V
            // Arm 3 (staging shape): predicate hoisted to one outer branch,
            // loads/stores vectorized at D_BAKE width. Same elements, same
            // rounding -- bit-exact. Alignment: row bases are >=8B-aligned
            // for every XPAD in {0,8} (checked: (GROUP+XPAD)*2 % 8 == 0).
            if (xr < nrow) {
                const device TIO* xrow =
                    xsrc + (size_t)srcrows[r0 + xr] * IN + (size_t)g * G;
                for (int q = q0; q < q0 + SPG / 4; ++q) {
  #if D_BAKE == 2
                    const vec<TIO, 2> v = ((const device vec<TIO, 2>*)xrow)[q];
                    *(threadgroup half2*)&xt[xr][q * 2] =
                        half2((half)v.x, (half)v.y);
  #elif D_BAKE == 4
                    const vec<TIO, 4> v = ((const device vec<TIO, 4>*)xrow)[q];
                    *(threadgroup half4*)&xt[xr][q * 4] =
                        half4((half)v.x, (half)v.y, (half)v.z, (half)v.w);
  #else
                    const vec<TIO, 4> v0 = ((const device vec<TIO, 4>*)xrow)[q * 2];
                    const vec<TIO, 4> v1 = ((const device vec<TIO, 4>*)xrow)[q * 2 + 1];
                    *(threadgroup half4*)&xt[xr][q * 8] =
                        half4((half)v0.x, (half)v0.y, (half)v0.z, (half)v0.w);
                    *(threadgroup half4*)&xt[xr][q * 8 + 4] =
                        half4((half)v1.x, (half)v1.y, (half)v1.z, (half)v1.w);
  #endif
                }
            } else {
                for (int q = q0; q < q0 + SPG / 4; ++q)
                    for (int u = 0; u < D_BAKE; ++u)
                        xt[xr][q * D_BAKE + u] = (half)0;
            }
#else
            const device TIO* xrow = 0;
            if (xr < nrow)
                xrow = xsrc + (size_t)srcrows[r0 + xr] * IN + (size_t)g * G;
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                for (int u = 0; u < D_BAKE; ++u)
                    xt[xr][q * D_BAKE + u] =
                        (xr < nrow) ? (half)xrow[q * D_BAKE + u] : (half)0;
            }
#endif
#if RTILE == 64
            // paired half: same thread stages row xr+32
            const int xr2 = xr + 32;
            const device TIO* xrow2 = 0;
            if (xr2 < nrow)
                xrow2 = xsrc + (size_t)srcrows[r0 + xr2] * IN + (size_t)g * G;
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                for (int u = 0; u < D_BAKE; ++u)
                    xt[xr2][q * D_BAKE + u] =
                        (xr2 < nrow) ? (half)xrow2[q * D_BAKE + u] : (half)0;
            }
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // phase 3: C[tokens 32 x outs 32] += X[32 x G] @ WtT[G x 32]
        // simdgroup sg owns out-block col sg*8; ti indexes token blocks.
#if OT2
        if (ob == 0)
#endif
        for (int k8 = 0; k8 < G / 8; ++k8) {
            simdgroup_half8x8 B;
            simdgroup_load(B, &wtT[k8 * 8][(int)sg * 8], 32);
            simdgroup_half8x8 A;
            simdgroup_load(A, &xt[0][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C0, A, B, C0);
            simdgroup_load(A, &xt[8][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C1, A, B, C1);
            simdgroup_load(A, &xt[16][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C2, A, B, C2);
            simdgroup_load(A, &xt[24][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C3, A, B, C3);
#if RTILE == 64
            simdgroup_load(A, &xt[32][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C4, A, B, C4);
            simdgroup_load(A, &xt[40][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C5, A, B, C5);
            simdgroup_load(A, &xt[48][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C6, A, B, C6);
            simdgroup_load(A, &xt[56][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(C7, A, B, C7);
#endif
        }
#if OT2
        else for (int k8 = 0; k8 < G / 8; ++k8) {
            simdgroup_half8x8 B;
            simdgroup_load(B, &wtT[k8 * 8][(int)sg * 8], 32);
            simdgroup_half8x8 A;
            simdgroup_load(A, &xt[0][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(Db0, A, B, Db0);
            simdgroup_load(A, &xt[8][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(Db1, A, B, Db1);
            simdgroup_load(A, &xt[16][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(Db2, A, B, Db2);
            simdgroup_load(A, &xt[24][k8 * 8], GROUP + XPAD);
            simdgroup_multiply_accumulate(Db3, A, B, Db3);
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

#if DSTORE
    // Direct epilogue (KERNEL-BODY-CAMPAIGN arm 2): accumulators -> device,
    // no ybuf, no barrier, no scalar copy pass. Lane mapping is steel's
    // (mlx mma.h): lane owns 2 elements at row fm, cols fn..fn+1 of its
    // simdgroup's 8x8 fragment. (TIO)float is the same single rounding the
    // ybuf path performed -- bit-exact by construction.
    {
        const short fm = ((lane / 4) & 4) + ((lane / 2) % 4);
        const short fn = (((lane / 4) & 2) * 2) + ((lane % 2) * 2);
        const int   col0 = (int)sg * 8 + fn;
        for (int ob = 0; ob < 1 + OT2; ++ob) {
            const int oo = o0 + ob * 32;
            const int oc = oo + col0;
            if (oc < OUT) {
                const bool oc1 = (oc + 1) < OUT;
#define VQ_DSTORE(CK, ROWB) { \
                const int tt = ROWB + fm; \
                if (tt < nrow) { \
                    thread auto el = (CK).thread_elements(); \
                    device TIO* yp = y + (size_t)(r0 + tt) * OUT + oc; \
                    yp[0] = (TIO)el[0]; \
                    if (oc1) yp[1] = (TIO)el[1]; \
                } }
                if (ob == 0) {
                    VQ_DSTORE(C0, 0)  VQ_DSTORE(C1, 8)
                    VQ_DSTORE(C2, 16) VQ_DSTORE(C3, 24)
#if RTILE == 64
                    VQ_DSTORE(C4, 32) VQ_DSTORE(C5, 40)
                    VQ_DSTORE(C6, 48) VQ_DSTORE(C7, 56)
#endif
                }
#if OT2
                else {
                    VQ_DSTORE(Db0, 0)  VQ_DSTORE(Db1, 8)
                    VQ_DSTORE(Db2, 16) VQ_DSTORE(Db3, 24)
                }
#endif
#undef VQ_DSTORE
            }
        }
    }
#else

    for (int ob = 0; ob < 1 + OT2; ++ob) {
        const int oo = o0 + ob * 32;
#if OT2
        if (ob == 0) {
#endif
        simdgroup_store(C0, &ybuf[0][(int)sg * 8], 32);
        simdgroup_store(C1, &ybuf[8][(int)sg * 8], 32);
        simdgroup_store(C2, &ybuf[16][(int)sg * 8], 32);
        simdgroup_store(C3, &ybuf[24][(int)sg * 8], 32);
#if RTILE == 64
        simdgroup_store(C4, &ybuf[32][(int)sg * 8], 32);
        simdgroup_store(C5, &ybuf[40][(int)sg * 8], 32);
        simdgroup_store(C6, &ybuf[48][(int)sg * 8], 32);
        simdgroup_store(C7, &ybuf[56][(int)sg * 8], 32);
#endif
#if OT2
        } else {
            simdgroup_store(Db0, &ybuf[0][(int)sg * 8], 32);
            simdgroup_store(Db1, &ybuf[8][(int)sg * 8], 32);
            simdgroup_store(Db2, &ybuf[16][(int)sg * 8], 32);
            simdgroup_store(Db3, &ybuf[24][(int)sg * 8], 32);
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // guarded copy: thread tid -> token row tt=tid/4, out span lane%4*8..
        {
            const int tt = (int)tid / 4;
            const int c0 = ((int)tid % 4) * 8;
            if (tt < nrow) {
                for (int c = c0; c < c0 + 8; ++c) {
                    const int oc = oo + c;
                    if (oc < OUT)
                        y[(size_t)(r0 + tt) * OUT + oc] = (TIO)ybuf[tt][c];
                }
            }
#if RTILE == 64
            const int tt2 = tt + 32;
            if (tt2 < nrow) {
                for (int c = c0; c < c0 + 8; ++c) {
                    const int oc = oo + c;
                    if (oc < OUT)
                        y[(size_t)(r0 + tt2) * OUT + oc] = (TIO)ybuf[tt2][c];
                }
            }
#endif
        }
        // ybuf is reused by the next block's stores
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
"""

# PROMOTED 2026-09-07 (Noah): default ON at v2 after 1.43x (35B-4.6,
# K512) and 1.35x (gemma-26b, K2048) measured on real 9k prefills, both
# score-gated within the reordering-noise band. VQ_MOE_FUSED_GEMM=0
# restores the legacy decode+padded-GEMM path; =1 selects the v1 scalar
# kernel (kept for A/B; measured 0.74x — do not use for speed).
_FUSED_GEMM = os.environ.get("VQ_MOE_FUSED_GEMM", "2") != "0"
_FUSED_GEMM_V2 = os.environ.get("VQ_MOE_FUSED_GEMM", "2") == "2"
# Big-K device-codebook arm (d4 K>2560, d8 K>1280). PROMOTED 2026-09-07
# after measurement: 35B-3.8bpw (d4 K8192, which had NO fast path at
# all) went 8.9s -> 4.7s on a real 9k prefill, 1.89x — the largest gain
# of any arm, on the geometry that looked worst. The 32x tile reuse does
# amortize the device gather, as hypothesised; ppl 5.5041 -> 5.4966
# (forced-prefill), inside the reordering band.
# VQ_MOE_FUSED_GEMM_BIGK=0 pins it back off.
_FUSED_GEMM_BIGK = os.environ.get("VQ_MOE_FUSED_GEMM_BIGK", "1") != "0"
# d8 PROMOTED 2026-09-07 after the arm was finally measured on both d8
# artifacts (it shipped off only because it was unbenched, never because it
# was suspect — correctness was gated at K16384, the exact flagship
# geometry, from the start):
#   397B-2.2, 2-node ring, 151/180 modules d8: 25.184s -> 19.489s = 1.29x
#   Flash-2.1, M3 single-box, 138/144 d8:      15.190s -> 11.587s = 1.31x
# Two artifacts, two topologies, agreeing within 0.02x, with non-overlapping
# rep ranges in both pairs. Flash is the stronger of the two: single-box, so
# no pipeline to confound it, and a 1.2% spread. It also takes that artifact
# from 6/144 fused to 144/144.
# CAVEAT ON METHOD: flipping this flag needs a process restart, so unlike
# the d2/d4 numbers these arms are separate loads, NOT interleaved in one
# process. Margins are 3-25x the observed spread, so the direction and rough
# size hold, but treat the exact ratio as softer than the interleaved ones.
# VQ_MOE_FUSED_GEMM_D8=0 pins it back off.
_FUSED_GEMM_D8 = os.environ.get("VQ_MOE_FUSED_GEMM_D8", "1") != "0"
# RTILE: token rows per threadgroup in gemmseg v2. 32 is shipped; 64 pairs
# two token tiles so the phase-1 weight decode runs once per 64 rows instead
# of twice — codes (WPR*4 B/row), scales (NGRP*2 B/row) and the CB_DEV
# codebook loads all halve, while x bytes and MAC counts are invariant.
# CB_DEV ONLY by budget: xt and ybuf double (12288 -> 20480 B), which still
# fits the 32768 cap alone but not beside a 16 KB threadgroup codebook.
# DEFAULT 32 (unmeasured). Set VQ_MOE_GEMMSEG_RTILE=64 to arm it. Ships off
# for the same reason d8 did: an unbenched arm does not ship armed.
# CB_DEV arm selection. "auto" = by budget (the shipped rule: device
# codebook only when K*2*D + 3*4096 exceeds the 32 KB cap). "1"/"0" force it
# on/off where BOTH arms are legal, which is what makes the d4-K2048 A/B
# possible at all — there the threadgroup arm fits (16384 + 12288 = 28672)
# and is chosen today, but the two have opposite traffic profiles and nobody
# had measured which is faster. Forcing OFF is ignored when the budget makes
# it illegal; the gate would refuse the geometry anyway.
_GEMMSEG_CBDEV = os.environ.get("VQ_MOE_GEMMSEG_CBDEV", "auto")
_GEMMSEG_RTILE = int(os.environ.get("VQ_MOE_GEMMSEG_RTILE", "32"))
# xt leading-dimension pad (F49 arm). "1" pads by 8 halves (+16 B/row, the
# affine steel pattern); default OFF until the bench promotes it. The budget
# arithmetic below MUST follow this flag or an edge geometry E134s at load.
_GEMMSEG_XT_PAD = os.environ.get("VQ_GEMMSEG_XT_PAD", "0") == "1"
# bf16 I/O for gemmseg (F49/F50, rev-2 DEFAULT). The prefill kernel reads
# bf16 activations and writes bf16 output directly, deleting both full-tensor
# boundary casts (+1.3-1.8% measured). Stage-in converts bf16->half with the
# astype's own round-to-nearest (bit-identical); the store SINGLE-rounds
# float->bf16 -- one convert fewer and strictly more accurate than the old
# float->half->bf16 double round, differing from shipped by <=1 bf16 ULP
# (the class d8/devx shipped under; gated by the comparator, not checksum).
# "0" is the kill switch back to the cast pipeline.
_GEMMSEG_BF16IO = os.environ.get("VQ_GEMMSEG_BF16IO", "0") == "1"
# Output-block pairing (KERNEL-BODY-CAMPAIGN arm 1). "1" halves the gathered
# xt staging traffic and the dispatch count by giving each threadgroup two
# 32-column output blocks (decoded serially -- threadgroup bytes unchanged).
# RTILE=32 only (the shipped default). Bit-exact per element. Default off
# until benched.
# MEASURED 2026-09-10 (F54): +5.1-6.6% prefill on 35B-3.4, bit-exact
# checksums, interleaved reps -- the largest single kernel win since CB_DEV.
# Default ON for the v2 runtime; "0" is the kill switch.
_GEMMSEG_OT2 = os.environ.get("VQ_GEMMSEG_OTILE64", "1") == "1"
# Direct epilogue store (arm 2). "1" stores accumulators straight to y via
# steel's lane mapping -- deletes ybuf (-4 KB threadgroup), the epilogue
# barrier, and the scalar copy pass (priced at ~+4.9% alone). Default off
# until benched.
_GEMMSEG_DSTORE = os.environ.get("VQ_GEMMSEG_DSTORE", "0") == "1"
# Prefetch pipeline (arm 1.5): g+1's code/scale loads issued before the mma
# so they fly during the matmul. Register cost <= (8 uints + 1 float) per
# ob strand per thread. Default off until benched.
_GEMMSEG_PIPE = os.environ.get("VQ_GEMMSEG_PIPE", "0") == "1"
# Phase-2 staging shape (arm 3): hoisted predicate + vector-width staging.
# Default off until benched.
# MEASURED 2026-09-10 (F56): +3.7-3.9% prefill, bit-exact, interleaved.
# The refuter attacked this arm's traffic claim (correctly) but the real
# mechanism is instruction count + hoisted predication. Default ON for v2.
_GEMMSEG_PH2V = os.environ.get("VQ_GEMMSEG_PH2V", "1") == "1"
# threadgroup tile bytes at RTILE=32, tracked so cb_dev/fits arithmetic
# follows the flags (F51's rule: budget follows every byte change).
_TILES_R32 = (2 if _GEMMSEG_DSTORE else 3) * 4096
# Decode-side bf16 I/O (F46: +3.6% decode measured on 35B-3.4). The small-N
# _fused kernels are templated on x.dtype, so bf16 flows straight through and
# both per-call casts disappear -- ~144+ elementwise passes/token. Numerics
# CHANGE (halfN-staged kernels round in-kernel where the host astype did;
# d8 float4-staged skip the fp16 rounding entirely) -- covered by the v2
# release policy: PPL is retested before anything ships. Kill switch = "0".
_DECODE_BF16IO = os.environ.get("VQ_DECODE_BF16IO", "0") == "1"
_XT_PAD_HALVES = 8 if _GEMMSEG_XT_PAD else 0
# Extra threadgroup bytes the pad costs at each RTILE (halves * 2 B * rows).
_XT_PAD_BYTES_R32 = _XT_PAD_HALVES * 2 * 32
_XT_PAD_BYTES_R64 = _XT_PAD_HALVES * 2 * 64
if _GEMMSEG_RTILE not in (32, 64):
    raise ValueError(f"VQ_MOE_GEMMSEG_RTILE must be 32 or 64, got {_GEMMSEG_RTILE}")


def gemmseg_cb_dev(D, K):
    """Does this geometry take the DEVICE codebook arm? Single source of
    truth — `coverage` asks this instead of restating the rule, which is
    how its arm label silently went stale when the 16 KB preference landed
    on 2026-09-08."""
    cb = K * 2 * D
    return cb + _TILES_R32 + _XT_PAD_BYTES_R32 > _TG_CAP_BYTES or cb >= 16384


def gemmseg_fits(D, K, G, pack_bits, IN):
    """May prefill take the fused segmented VQ-GEMM? Conservative first
    ship: exactly the measured MoE geometry class."""
    # pack_bits == 0 is the UNPACKED arm (plain uint8/uint16 codes, K<=64k):
    # legal, and it is how every Flash-Next rung stores its d2 shared
    # modules. Anything above 16 bits is not a format we emit.
    # d8 is CB_DEV-only by construction (K*16 B never fits threadgroup).
    # Numerically gated (incl. K16384, the 397B-2.2 geometry) and, since
    # 2026-09-07, benched on both d8 artifacts and armed by default —
    # see the _FUSED_GEMM_D8 note above. VQ_MOE_FUSED_GEMM_D8=0 pins it off.
    if D == 8 and not _FUSED_GEMM_D8:
        return False
    # NSUB (= IN/D) need NOT be a multiple of 32. That was conservatism from
    # the first ship ("exactly the measured geometry class"), and it cost
    # Flash-2.1 its 46 d8 in=640 modules (NSUB=80) — the only sub-100%-fused
    # artifact in the fleet. Relaxed 2026-09-08 after verifying, not arguing:
    #   * the PACKER already zero-pads the tail block: the shipped artifact
    #     stores codes [512, 2560, 42] = ceil(80/32)*14, and scales [..,10]
    #     = NGRP = 640/64.
    #   * WPR is already ceil((NSUB+31)/32)*BITS in both kernel sources.
    #   * the decode loop covers exactly NGRP*SPG = (IN/G)*(G/D) = NSUB
    #     codes, so a pad code is never read.
    #   * a 32-code block is 32*BITS bits = a whole number of words for any
    #     BITS, so VQ_CODE's straddle read crow[VQ_W(j)+1] never crosses out
    #     of the row.
    # Gated by test_gemmseg_ragged_nsub_numeric at the exact Flash-2.1
    # geometry. What still MUST divide is SPG/4 (below) — that one is a real
    # thread-assignment constraint, not conservatism.
    if not (_FUSED_GEMM and D in (2, 4, 8) and 0 <= pack_bits <= 16
            and G == 64 and IN % G == 0
            and (G // D) % 4 == 0):   # SPG/4 codes per thread must divide
        return False
    # E134 budget, exact: cb K*2*D B + wtT 4096 + xt 4096 + ybuf 4096.
    # Over the cap the codebook moves to DEVICE memory (CB_DEV arm) —
    # the tiles alone are 12 KB, so the kernel always loads. Big-K is
    # therefore covered too, unless the operator pins it off.
    if K * 2 * D + _TILES_R32 + _XT_PAD_BYTES_R32 <= _TG_CAP_BYTES:
        return True
    return _FUSED_GEMM_BIGK


# ROUTING MEMO (F42, 2026-09-09). Within one forward pass the gate/up/down
# projections of an MoE layer are called with the SAME routing, so exactly 1
# in 3 tile builds is fresh (measured: 240 repeats of 720 calls per prefill,
# and the whole numpy prefix is 3.9% of prefill wall -- a memo recovers
# ~2.6%). Keys are the routing BYTES, not a hash digest: dict equality does a
# memcmp on collision, so a hit is exact by construction and the cached
# tensors are bit-identical to what the prefix would rebuild. Values are
# immutable (mx arrays are; the ints are). Bounded: routing changes every
# prefill step, so entries go stale in one step -- keep a few layers' worth
# and evict FIFO.
_ROUTING_MEMO: "dict[tuple, tuple]" = {}
_ROUTING_MEMO_MAX = 8


def _memo_get(key):
    return _ROUTING_MEMO.get(key)


def _memo_put(key, value):
    if len(_ROUTING_MEMO) >= _ROUTING_MEMO_MAX:
        _ROUTING_MEMO.pop(next(iter(_ROUTING_MEMO)))
    _ROUTING_MEMO[key] = value


# F128: `np.array(idx_flat)` forces the whole pending lazy graph, and
# __call__ pays it ONCE PER LINEAR PER LAYER -- n=120 on a 40-layer 35B, i.e.
# 120 pipeline drains per forward. Host TIME there is ~0 (the numpy path is
# 0.3% of a forward, measured), but a drain serializes CPU and GPU and
# forbids overlap, which is a cost no Python profiler can see.
#
# mlx-lm's SwitchGLU routes ONCE and hands the SAME `indices` object to
# up_proj, gate_proj and down_proj, so drains 2 and 3 of every three
# recompute a byte-identical array. This memo keys on that object's identity
# and holds a reference, which keeps the id valid (an id can be reused only
# after the object dies). Values are identical by construction: the hit is
# returned only when the SAME object is presented.
#
# Off by default until an A/B says otherwise -- a performance fix is an
# instrument change until measured (F120).
_IDX_MEMO = os.environ.get("VQ_IDX_MEMO", "0") == "1"
_IDX_CACHE: "dict[int, tuple]" = {}
_IDX_CACHE_MAX = 4


def _idx_np(indices, idx_flat):
    """Host copy of the routing indices, reused across the three linears."""
    if not _IDX_MEMO:
        return np.array(idx_flat, copy=False)
    key = id(indices)
    hit = _IDX_CACHE.get(key)
    if hit is not None and hit[0] is indices:
        return hit[1]
    arr = np.array(idx_flat, copy=False)
    if len(_IDX_CACHE) >= _IDX_CACHE_MAX:
        _IDX_CACHE.pop(next(iter(_IDX_CACHE)))
    _IDX_CACHE[key] = (indices, arr)     # the ref is what makes id() safe
    return arr


def _gemmseg_prefill(xsrc, src_rows, idx_sorted_np, codes, codebook, scales,
                     pack_bits, IN):
    """One fused dispatch per linear: y[sorted_row, OUT] with dequant
    inside the tile loop. Rows must be count-sorted by expert (they are:
    __call__ sorts, _prefill's contract)."""
    E, OUT, _ = codes.shape
    K = codebook.shape[0]
    # Tile granularity MUST equal the kernel's RTILE: tmeta rows are
    # (expert, row_start, nrows) and the kernel guards on nrow, so a
    # mismatch would silently drop rows past 32 of every 64-row tile.
    # cb_dev is recomputed here (cheaply) because the tiling is decided
    # before the kernel branch below.
    _rt = 64 if (_GEMMSEG_RTILE == 64 and _FUSED_GEMM_V2
                 and codebook.shape[0] * 2 * int(codebook.shape[1])
                 + _TILES_R32 + _XT_PAD_BYTES_R32 > _TG_CAP_BYTES) else 32
    _mk = ("tiles", idx_sorted_np.tobytes(), E, _rt)
    _hit = _memo_get(_mk)
    if _hit is not None:
        tmeta, ntiles = _hit
    else:
        counts = np.bincount(idx_sorted_np, minlength=E)
        touched = np.nonzero(counts)[0]
        starts = np.zeros(E + 1, np.int64)
        starts[1:] = np.cumsum(counts)
        # Vectorized tile build (swarm7 compute:7 / F42 follow-up): the
        # Python double loop was the memo's miss-path cost. Same rows in the
        # same order -- touched ascends (np.nonzero) and within-expert tiles
        # ascend by construction -- so tmeta is BIT-IDENTICAL to the loop
        # (pinned by tests/test_routing_memo.py::test_vectorized_tile_build).
        tc = counts[touched]
        ntiles_per = (tc + _rt - 1) // _rt
        eids = np.repeat(touched, ntiles_per)
        cum = np.cumsum(ntiles_per)
        within = (np.arange(int(cum[-1]) if len(cum) else 0)
                  - np.repeat(cum - ntiles_per, ntiles_per)) * _rt
        rows = starts[eids] + within
        nrows = np.minimum(_rt, counts[eids] - within)
        metas_np = np.stack([eids, rows, nrows], axis=1).astype(np.int32)
        tmeta = mx.array(metas_np.reshape(-1))
        ntiles = int(metas_np.shape[0])
        _memo_put(_mk, (tmeta, ntiles))
    cbk = codebook.astype(mx.float16) if codebook.dtype != mx.float16 \
        else codebook
    dims = _dims_array(OUT, IN, IN // 64, K, ntiles)
    D = int(codebook.shape[1])
    if _FUSED_GEMM_V2:
        name = (f"vq_gemmseg2_packed{pack_bits}_d{D}" if pack_bits
                else f"vq_gemmseg2_u{codes.dtype.size * 8}_d{D}")
        src = _SRC_GEMMSEG2
        _cb_bytes = K * 2 * D
        # Arm selection. The budget still FORCES the device codebook when the
        # threadgroup copy cannot fit. On top of that, 2026-09-08: prefer the
        # device arm at cb >= 16 KB even where the threadgroup arm is legal.
        #
        # Measured, 9k prefills, threadgroup -> CB_DEV:
        #   d4-K2048 16KB single-box  1316 -> 1942 tok/s  1.475x  WIN
        #   d4-K512   4KB 2-node       335 ->  334 tok/s  0.997x  tie
        #   d2-K2048  8KB single-box  1209 -> 1194 tok/s  0.988x  tie
        #   d2-K1024  4KB single-box  1787 -> 1872 tok/s  1.047x
        #
        # So it is NOT D (d4-K512 ties) and NOT codebook size monotonically
        # (8 KB ties while 4 KB is marginally positive). The one decisive case
        # is the LARGEST threadgroup-legal codebook: at 16 KB the bulk load is
        # maximal and leaves only 4096 B of the 32768 B budget spare.
        # MECHANISM UNMEASURED — occupancy pressure at the budget edge is a
        # hypothesis, not a finding.
        # ~447 fleet modules are d4-K2048 (397B-3.1, Flash-3.2, 35B-3.4/4.6).
        cb_dev = _cb_bytes + _TILES_R32 + _XT_PAD_BYTES_R32 > _TG_CAP_BYTES or _cb_bytes >= 16384
        if _GEMMSEG_CBDEV == "1":
            cb_dev = True                      # force device codebook
        elif _GEMMSEG_CBDEV == "0" and _cb_bytes + _TILES_R32 + _XT_PAD_BYTES_R32 <= _TG_CAP_BYTES:
            cb_dev = False                     # force threadgroup (if legal)
        if cb_dev:
            name += "_cbdev"
        # RTILE=64 needs the CB_DEV budget; silently fall back otherwise so
        # arming the flag can never produce an E134 kernel-load failure.
        rtile = 64 if (_GEMMSEG_RTILE == 64 and cb_dev) else 32
        if rtile != 32:
            name += f"_r{rtile}"
        if _GEMMSEG_XT_PAD:
            name += "_xp8"
        # OT2 needs RTILE=32 (compile-guarded in the source); silently keep
        # the flag off at RTILE=64 so arming both can never E134.
        ot2 = 1 if (_GEMMSEG_OT2 and rtile == 32) else 0
        if ot2:
            name += "_ot2"
        if _GEMMSEG_DSTORE:
            name += "_ds"
        if _GEMMSEG_PIPE:
            name += "_pf"
        if _GEMMSEG_PH2V:
            name += "_p2v"
        # TIO follows xsrc's dtype so the kernel cache never mixes builds.
        if xsrc.dtype == mx.bfloat16:
            name += "_bf16io"
        template = [("BITS", pack_bits), ("GROUP", 64),
                    ("MAX_K", 1 if cb_dev else K), ("D_BAKE", D),
                    ("CB_DEV", 1 if cb_dev else 0), ("RTILE", rtile),
                    ("XPAD", _XT_PAD_HALVES), ("TIO", xsrc.dtype),
                    ("OT2", ot2), ("DSTORE", 1 if _GEMMSEG_DSTORE else 0),
                    ("PIPE", 1 if _GEMMSEG_PIPE else 0),
                    ("PH2V", 1 if _GEMMSEG_PH2V else 0)]
        if not pack_bits:
            # unpacked arm reads codes as CT (uchar/ushort), not uint words
            template.append(("CT", codes.dtype))
    else:
        # v1 (scalar MAC, measured 0.74x) was only ever written for d2.
        if D != 2:
            raise NotImplementedError(
                "vq_gemmseg v1 is d2-only; use VQ_MOE_FUSED_GEMM=2")
        if xsrc.dtype != mx.float16:
            raise NotImplementedError("gemmseg v1 is fp16-I/O only")
        name, src = f"vq_gemmseg_packed{pack_bits}_d2", _SRC_GEMMSEG
        template = [("BITS", pack_bits), ("GROUP", 64), ("OTILE", 32),
                    ("RTILE", 32), ("MAX_K", K)]
    N = int(idx_sorted_np.shape[0])
    _ospan = 64 if locals().get("ot2") else 32   # EFFECTIVE ot2 (v2+RTILE32)
    grid = (32 * ((OUT + _ospan - 1) // _ospan), 4 * ntiles, 1)
    tg = (32, 4, 1)
    common = dict(
        inputs=[codes, cbk, scales, xsrc, mx.array(src_rows), tmeta, dims],
        grid=grid, threadgroup=tg,
        output_shapes=[(N, OUT)], output_dtypes=[xsrc.dtype],
    )
    kern = _get_kernel_spec(name, src, template) \
        if _SPEC_KERNELS else None
    if kern is not None:
        (y,) = kern(**common)
    else:
        (y,) = _get_kernel(name, src)(template=template, **common)
    return y


def _decode_chunk(codes, codebook, scales, eidx_chunk, pack_bits=0,
                  in_features=None):
    NE = eidx_chunk.shape[0]
    E, OUT, _ = codes.shape
    D = codebook.shape[1]
    NGRP = scales.shape[2]
    # packed rows are WPR words wide, so shape no longer implies IN — the
    # caller passes it (VQSwitchLinear.input_dims knows it from the format).
    IN = in_features if in_features is not None else codes.shape[2] * D
    G = IN // NGRP
    # PREFILL GUARD (2026-08-18). _fused dispatches explicitly on D, but this
    # path only branched on pack_bits — so d=2 with uint16 codes (K>256) read
    # wrong memory and returned SILENT GARBAGE: gemma vq-K512-d2 verified at
    # relerr 0.0589 (verify_artifact PASS, weights provably fine) yet scored
    # 3154 mnats / 47.14%, worse than the d4 artifact it should beat. d=2 with
    # uint8 (K<=256) is correct and is what ships. Raise rather than decode
    # wrongly — a plausible bad number costs more than a crash.
    # (2026-08-18, later) the vq_decode kernel is D-generic: verified
    # numerically for d2-uint16 (K=512) against a numpy reference, max rel
    # diff 2.5e-4. So unpacked d2 needs no dtype restriction here. The
    # garbage that prompted the original guard came from a STALE COPY of
    # this file inside the venv (site-packages/mlx_lm/models/vq_switch.py)
    # that still had the fall-through dispatch — see E47. Packed d2 remains
    # unimplemented and must still raise.
    # packed d=2 through THIS path is verified D-generic: decoded against a
    # numpy vq_pack.unpack reference at max rel 2.6e-4 (K=512, pack_bits=9).
    # The fused path has its own dedicated packed-d2 kernel (2026-08-19).
    # Arc-5 host-overhead caches applied to THIS call site too (2026-09-05,
    # swarm design doc props 3+4): cached dims array + specialized
    # template-free kernel. Measured 7.9 -> 2.4 us/call host issue (3.29x),
    # bit-identical output (asserted on real tensors); ~0.8 ms/token at the
    # 144-dispatch rate. VQ_SPEC_KERNELS=0 restores the template path.
    dims = _dims_array(OUT, IN, D, G, NE)
    vec = _DECODE_VEC and D % 2 == 0 and G % 2 == 0
    if pack_bits:
        if vec:
            name, src = f"vq_decode_vec_packed{pack_bits}", \
                _SRC_DECODE_PACKED_VEC
        else:
            name, src = f"vq_decode_packed{pack_bits}", _SRC_DECODE_PACKED
        template = [("BITS", pack_bits)]
    else:
        name, src = ("vq_decode_vec", _SRC_DECODE_VEC) if vec \
            else ("vq_decode", _SRC_DECODE)
        template = [("CT", codes.dtype)]
    kern = _get_kernel_spec(name, src, template) if _SPEC_KERNELS else None
    common = dict(
        inputs=[codes, codebook, scales, eidx_chunk, dims],
        grid=(NGRP, OUT, NE),
        threadgroup=(min(32, NGRP), 8, 1),
        output_shapes=[(NE, OUT, IN)],
        output_dtypes=[mx.float16],
    )
    if kern is None:
        (w,) = _get_kernel(name, src)(template=template, **common)
    else:
        (w,) = kern(**common)
    return w


# GATHER FUSION (2026-09-06, swarm r3 fusion survivor + measured 24%).
# The legacy prefill input chain materializes the same rows three times:
# broadcast_to(x).reshape (a stride-0 axis forced into a full [N, IN]
# copy), xf[order] (a second full copy), then xf_sorted[gmap] -> xp. All
# three are row-gathers of the ORIGINAL token matrix, so the indices
# compose on the host for free: xp = x_tokens[(order // K_rep)[gmap]] —
# ONE gather, from a source K_rep(=top_k)x smaller. The gathered VALUES
# are the same rows, so GEMM operands are byte-identical and the output
# is bit-exact by construction (gated in tests/test_vq_prefill_paths.py).
# VQ_MOE_FUSE_GATHER=0 restores the legacy three-copy chain.
_FUSE_GATHER = os.environ.get("VQ_MOE_FUSE_GATHER", "1") != "0"

# EXACT-LENGTH PER-EXPERT GEMMs (2026-09-06, r3 compute survivor).
# Kills ALL padding (measured pad ratio 1.567 at chunk=32 on real 35B
# routing = ~57% wasted GEMM FLOPs) by running one GEMM per expert on its
# already-contiguous sorted row slice — no gmap, no vmask, no pad rows.
# NOT bit-identical to the padded batched GEMM: a different GEMM shape
# picks a different Metal tiling and fp16 sums in a different order (the
# published-score law), so this ships DEFAULT OFF, for unscored serving
# where wall time matters more than bit-reproducibility.
# VQ_MOE_EXACT_GEMM=1 opts in.
_EXACT_GEMM = os.environ.get("VQ_MOE_EXACT_GEMM", "0") != "0"


def _prefill(xf, idx_sorted_np, codes, codebook, scales, pack_bits=0,
             in_features=None, xsrc=None, src_rows=None):
    """xf [N, IN] rows sorted by expert; idx_sorted_np = matching np expert
    ids. Decode touched experts in chunks; one padded batched GEMM each.

    Fused-gather form: ``xsrc`` [T, IN] (the original token rows, fp16) +
    ``src_rows`` [N] (sorted-row -> token-row map) replace ``xf``; the xp
    gather then reads xsrc[src_rows[gmap]] directly and xf may be None.
    """
    global _DECODE_CHUNK
    if _DECODE_CHUNK is None:
        _DECODE_CHUNK = _default_decode_chunk()
    # Fused segmented VQ-GEMM (opt-in, see _SRC_GEMMSEG block): dequant
    # inside the tile loop, no w materialization, no chunk loop. Needs
    # the fused-gather form (xsrc+src_rows); falls through untouched
    # otherwise. Output is already in this function's input row order.
    if xsrc is not None and gemmseg_fits(
            int(codebook.shape[1]), int(codebook.shape[0]),
            in_features // (scales.shape[2]) if in_features else 64,
            pack_bits, in_features or 0):
        return _gemmseg_prefill(xsrc, src_rows, idx_sorted_np, codes,
                                codebook, scales, pack_bits, in_features)
    E, OUT, _ = codes.shape
    counts = np.bincount(idx_sorted_np, minlength=E)
    touched = np.nonzero(counts)[0]
    # COUNT-SORTED CHUNKING (VQ-PF1, 2026-08-16). The GEMM below pads every
    # expert in a chunk up to that chunk's MAX row count, so a chunk that
    # mixes a 1574-row expert with a 20-row one pays 1574 rows for both.
    # Real routing is skewed ~8.7x, so chunking by expert ID (the obvious
    # order, and what shipped) did 5.80x the necessary FLOPs. Grouping
    # experts of SIMILAR size makes cap track the chunk mean: pad falls
    # 5.92x -> 1.19x and _prefill goes 263 -> 75.9 ms at chunk=16.
    # This is a pure host-side reordering of WHICH experts share a GEMM —
    # codes/codebook/scales are read identically, nothing is repacked, so
    # the on-disk artifact layout is untouched (the HARD CONSTRAINT: a fix
    # must ship as a bundled model.py update, never a re-upload).
    touched = touched[np.argsort(counts[touched], kind="stable")]
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    ys = []
    # Reordering experts reorders OUTPUT ROWS, and this function's contract
    # is to return rows in its input order (__call__ applies its own `inv`
    # on top). Track the xf row each output row came from and undo it below.
    row_ids = []
    for c0 in range(0, len(touched), _DECODE_CHUNK):
        eids = touched[c0:c0 + _DECODE_CHUNK]
        ne = len(eids)
        w = _decode_chunk(codes, codebook, scales,
                          mx.array(eids.astype(np.uint32)),
                          pack_bits=pack_bits, in_features=in_features)
        if _EXACT_GEMM:
            # One GEMM per expert on its contiguous sorted slice: zero
            # padding, zero gather maps. Rows/outputs stay in sorted
            # order, so row_ids is just the slice ranges. Eval EVERY y of
            # this chunk (not just the last) so `w` can actually be freed
            # before the next chunk decodes — the same lazy-graph trap the
            # per-chunk eval below exists for.
            chunk_ys = []
            for i, e in enumerate(eids):
                s0, c = int(starts[e]), int(counts[e])
                if xsrc is not None:
                    xe = xsrc[mx.array(src_rows[s0:s0 + c])]
                else:
                    xe = xf[s0:s0 + c]
                chunk_ys.append(xe @ w[i].T)
                row_ids.append(np.arange(s0, s0 + c, dtype=np.uint32))
            mx.eval(chunk_ys)
            ys.extend(chunk_ys)
            del w, chunk_ys
            continue
        cap = int(counts[eids].max())
        # gather map rows -> [ne, cap]; pads point at row 0 (discarded)
        gmap = np.zeros((ne, cap), np.uint32)
        vmask = np.zeros((ne, cap), bool)
        for i, e in enumerate(eids):
            c = counts[e]
            gmap[i, :c] = np.arange(starts[e], starts[e] + c, dtype=np.uint32)
            vmask[i, :c] = True
        if xsrc is not None:
            xp = xsrc[mx.array(src_rows[gmap.reshape(-1)])] \
                .reshape(ne, cap, -1)
        else:
            xp = xf[mx.array(gmap.reshape(-1))].reshape(ne, cap, -1)
        yp = xp @ mx.swapaxes(w, 1, 2)                      # [ne, cap, OUT]
        flat_valid = np.nonzero(vmask.reshape(-1))[0].astype(np.uint32)
        ys.append(yp.reshape(ne * cap, OUT)[mx.array(flat_valid)])
        row_ids.append(gmap.reshape(-1)[flat_valid])
        # CRITICAL: MLX is lazy. Without this eval the whole loop builds one
        # graph and EVERY chunk's decoded weights stay live until the final
        # concatenate — 4 chunks x 2 GiB for gate_up, which is what actually
        # capped context length on a 128 GB box (measured 2026-08-15: prefill
        # grew 3.35 MB/token vs 0.059 MB/token of real KV cache). Evaluating
        # per chunk lets each `w` be freed before the next is decoded.
        mx.eval(ys[-1])
        del w, xp, yp
    y = mx.concatenate(ys, axis=0)
    # Undo the count-sort: output row j currently holds input row row_ids[j].
    # inv[row_ids[j]] = j, so y[inv] restores the caller's row order. One
    # [N, OUT] gather (~2 ms at N=92k) against ~190 ms of padding saved.
    rid = np.concatenate(row_ids)
    inv = np.empty(rid.shape[0], np.uint32)
    inv[rid] = np.arange(rid.shape[0], dtype=np.uint32)
    return y[mx.array(inv)]


# DEFAULT BUFFER-CACHE CEILING (2026-09-03). Long prompts through a VQ
# model allocate and free a decoded-weight-sized buffer per layer; MLX
# parks every freed buffer in its reuse cache and returns nothing to the
# OS until someone calls clear_cache. On a 12 GiB model that reads as a
# ~38 GB "Peak memory" and real system pressure -- allocator behavior,
# not a property of the model, and measured to cost NOTHING to cap
# (GLM-5.3 26k-token prefill: identical wall time, cache pinned at
# 0.0-0.2 GiB; ledger 2026-09-03). So the runtime ships with a sane
# ceiling instead of a card footnote. A user- or host-set limit that is
# ALREADY stricter is respected (set_cache_limit returns the previous
# value, so we can peek without clobbering); VQLAB_CACHE_LIMIT_GB
# overrides ours, and =0 disables entirely.
_DEFAULT_CACHE_LIMIT_GB = 4.0


def _apply_default_cache_limit() -> None:
    raw = os.environ.get("VQLAB_CACHE_LIMIT_GB")
    if raw is not None:
        try:
            gb = float(raw)
        except ValueError:
            return
        if gb <= 0:
            return
    else:
        gb = _DEFAULT_CACHE_LIMIT_GB
    want = int(gb * (1 << 30))
    prev = mx.set_cache_limit(want)
    if prev < want and raw is None:  # someone set a stricter limit; keep it
        mx.set_cache_limit(prev)


_apply_default_cache_limit()


class VQSwitchLinear(nn.Module):
    """Drop-in for QuantizedSwitchLinear over VQ codes. No bias support
    (Qwen3.5 experts are bias-free)."""

    def __init__(self, codes, codebook, vq_scales, group_size: int = 64,
                 pack_bits: int = 0, in_features: int | None = None):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        # pack_bits = 0 -> legacy unpacked codes (uint8/uint16), the format
        # every shipped artifact before 08-16 uses. Non-zero -> uint32 words
        # holding pack_bits-wide fields (vq_pack.py).
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features: a "
                             "packed row is WPR words wide, so its shape no "
                             "longer implies the input dimension")
        self._in_features = in_features
        # SHARDING GUARD (2026-08-19). The codebook is a shared [K, d] lookup
        # table indexed by the codes; it must be REPLICATED across tensor-
        # parallel ranks, never sliced. exo's tensor_auto_parallel sliced it
        # by default until PR #2268, and a sliced codebook does not error —
        # every rank decodes against a fraction of the table and the model
        # emits fluent garbage, which reads as "this quant is broken" rather
        # than "my cluster mis-sharded a LUT". Remember the size we were
        # built with so __call__ can say so plainly.
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @classmethod
    def from_weights(cls, codes, codebook, vq_scales):
        if codes.dtype == mx.uint32:
            # packed codes (vq_pack.py). Geometry is fully derivable from the
            # tensors: the packer stores ceil(log2(K))-bit fields (K is a
            # power of two by construction), and rows are NSUB/32*BITS words.
            k, d = codebook.shape
            bits = int(k - 1).bit_length()
            # WPR -> NSUB is lossy for padded-tail packs; the scales axis
            # (IN/group, default group 64) carries the true input width.
            return cls(codes, codebook, vq_scales,
                       pack_bits=bits, in_features=vq_scales.shape[2] * 64)
        return cls(codes, codebook, vq_scales)

    @property
    def input_dims(self):
        # MUST be derived from the CURRENT tensors, never cached: exo's
        # tensor-parallel path shards `codes` IN PLACE after the module is
        # built (auto_parallel._sharded_to_all -> last axis). A cached value
        # then describes the pre-shard tensor, the kernel computes word
        # offsets for twice the data it holds, and the ring desyncs mid-load
        # (observed 2026-08-16: M4 66 GiB loaded, M3 stalled at 16.5 GiB).
        # Single-box never shards, which is why this only bites the cluster.
        if self.pack_bits:
            # Padded-tail packs (NSUB % 32 != 0) make WPR -> NSUB lossy:
            # ceil(80/32)*BITS words decode back to 96 subvectors, not 80.
            # The scales tensor's last axis is IN/group and shards on the
            # same axis as codes, so it carries the true IN through both
            # the unaligned format and exo's in-place sharding.
            return self.vq_scales.shape[2] * self.group_size
        return self.codes.shape[2] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[1]

    @property
    def num_experts(self):
        return self.codes.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        k_now = self.codebook.shape[0]
        if k_now != self._k_expect:
            raise RuntimeError(
                f"VQ codebook was sharded: K={k_now}, expected "
                f"{self._k_expect}. The codebook is a SHARED lookup table and "
                f"must be replicated across tensor-parallel ranks, not sliced "
                f"(codes and scales shard fine on the default axes). On exo, "
                f"apply the codebook guard in "
                f"src/exo/worker/engines/mlx/auto_parallel.py — upstream PR "
                f"https://github.com/exo-explore/exo/pull/2268. "
                f"Pipeline sharding and single-box "
                f"mlx-lm are unaffected.")
        IN = self.input_dims
        OUT = self.output_dims
        idx_flat = indices.flatten()
        N = idx_flat.size
        xf = mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN)
        in_dtype = xf.dtype
        # Decode (N <= VQ_FUSED_MAX_N) may keep bf16: _fused templates on
        # x.dtype and returns it, so the tail astype becomes an identity.
        # Prefill and the legacy xf paths still get the fp16 cast below.
        _keep_bf16 = (_DECODE_BF16IO and in_dtype == mx.bfloat16)
        if in_dtype not in (mx.float16,) and not _keep_bf16:
            xf = xf.astype(mx.float16)
        pb = self.pack_bits
        # Packed d=2 now has its own fused kernel (vq_fused_packed{bits}_d2,
        # 2026-08-19, verified against vq_pack.unpack numpy reference); the
        # old force-to-_prefill workaround is gone. _fused still raises
        # explicitly for any (D, pack_bits) without a dedicated kernel.
        if N <= VQ_FUSED_MAX_N:
            y = _fused(xf, idx_flat.astype(mx.uint32),
                       self["codes"], self["codebook"], self["vq_scales"],
                       pack_bits=pb)
        else:
            idx_np = _idx_np(indices, idx_flat)
            # Fused-gather form: xf above is broadcast_to(x).reshape — a
            # stride-0 axis forced into a FULL [N, IN] copy, and the
            # unsorted branch then copies it AGAIN via xf[order]. Both are
            # row-repeats/permutations of the true token matrix x2 [T, IN]
            # (row j of xf is x2[j // K_rep]), so the gathers compose on
            # the host and _prefill reads x2 directly: one gather, from a
            # K_rep-times-smaller source, same rows -> bit-identical.
            T = x.size // IN
            if _FUSE_GATHER and N % max(T, 1) == 0:
                x2 = x.reshape(T, IN)
                # bf16 flows straight through when the bf16-I/O kernel is
                # armed (the kernel converts at its edges, bit-identically);
                # otherwise the historical host cast.
                if x2.dtype not in (mx.float16,) and not (
                        _GEMMSEG_BF16IO and x2.dtype == mx.bfloat16):
                    x2 = x2.astype(mx.float16)
                k_rep = N // T
                if not sorted_indices:
                    # Same memo, same reason (F42): gate/up/down share
                    # idx_np, so the two argsorts and the idx gather repeat
                    # verbatim for 2 of every 3 calls. idx_sorted is cached
                    # too -- it feeds the tile memo in _gemmseg_prefill, so a
                    # hit there needs the identical bytes object anyway.
                    _sk = ("sort", idx_np.tobytes(), int(k_rep))
                    _sh = _memo_get(_sk)
                    if _sh is not None:
                        idx_sorted, src, inv_mx = _sh
                    else:
                        order = np.argsort(idx_np, kind="stable")
                        inv = np.argsort(order, kind="stable")
                        src = (order // k_rep).astype(np.uint32)
                        idx_sorted = idx_np[order]
                        inv_mx = mx.array(inv.astype(np.uint32))
                        _memo_put(_sk, (idx_sorted, src, inv_mx))
                    y = _prefill(None, idx_sorted,
                                 self["codes"], self["codebook"],
                                 self["vq_scales"], pack_bits=pb,
                                 in_features=IN, xsrc=x2, src_rows=src)
                    y = y[inv_mx]
                else:
                    src = (np.arange(N, dtype=np.uint32) // k_rep) \
                        .astype(np.uint32)
                    y = _prefill(None, idx_np,
                                 self["codes"], self["codebook"],
                                 self["vq_scales"], pack_bits=pb,
                                 in_features=IN, xsrc=x2, src_rows=src)
            elif not sorted_indices:
                if xf.dtype != mx.float16:      # legacy path is fp16-only
                    xf = xf.astype(mx.float16)
                order = np.argsort(idx_np, kind="stable")
                inv = np.argsort(order, kind="stable")
                y = _prefill(xf[mx.array(order.astype(np.uint32))],
                             idx_np[order],
                             self["codes"], self["codebook"], self["vq_scales"],
                             pack_bits=pb, in_features=IN)
                y = y[mx.array(inv.astype(np.uint32))]
            else:
                if xf.dtype != mx.float16:      # legacy path is fp16-only
                    xf = xf.astype(mx.float16)
                y = _prefill(xf, idx_np,
                             self["codes"], self["codebook"], self["vq_scales"],
                             pack_bits=pb, in_features=IN)
        return y.astype(in_dtype).reshape(*indices.shape, 1, OUT)


class VQPLEEmbedding(nn.Module):
    """VQ'd embedding / PLE table shard: decode is a pure gather.

    codes      uint16 [rows, cols/dim]     (unpacked; NSUB=40 defeats the
                                            32-aligned block pack for now)
    codebook   fp16   [K, dim]
    vq_scales  fp16   [rows, cols/group]

    __call__(ids) -> [.., cols] rows, decoded on the fly:
    codebook[codes[ids]] reshaped, times the row's group scales. No matmul
    anywhere in the path — the original module was already a lookup, so VQ
    composes as a second lookup (LUT of a LUT).
    """

    def __init__(self, codes, codebook, vq_scales, group_size: int = 32,
                 packed_nsub: int = 0):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        # packed rows: BITS-wide codes (BITS from the codebook size — a
        # hardcoded 11 here read K256 rows on an 11-bit stride and scored
        # NaN, 2026-08-29), byte-aligned because nsub*BITS % 8 == 0.
        # Constant gather tables map code i -> its 3-byte window + shift.
        self._pn = packed_nsub
        self._bits = max(1, (codebook.shape[0] - 1).bit_length())
        self._mask = (1 << self._bits) - 1
        # BITS == 8 MAKES _unpack AN IDENTITY (E14x, 2026-09-02). At K<=256 the
        # "packed" row is one byte per code, so bit0 = arange(nsub)*8 gives
        # window byte i, shift 0, mask 0xFF -- the extraction reproduces its own
        # input exactly. It was still being executed: a concatenate, three
        # mx.take gathers and six elementwise ops per shard per token, on the
        # Flash-Next 2.1bpw's 16 touched n-gram shards, i.e. ~160 tiny
        # dispatches a token to compute the identity function. Skip it; the
        # result is the same ARRAY, so bit-identity is not merely likely.
        self._byte_codes = bool(packed_nsub) and self._bits == 8
        if packed_nsub and not self._byte_codes:
            import numpy as _np
            bit0 = _np.arange(packed_nsub) * self._bits
            self._b0 = mx.array(bit0 // 8)
            self._sh = mx.array((bit0 % 8).astype(_np.uint32))

    def _unpack(self, rows_u8):
        # rows_u8 [.., row_bytes] uint8 -> [.., nsub] uint32 codes
        b = rows_u8.astype(mx.uint32)
        # pad 2 bytes so the 3-byte window never reads past the row
        pad = mx.zeros((*b.shape[:-1], 2), dtype=mx.uint32)
        b = mx.concatenate([b, pad], axis=-1)
        w = (mx.take(b, self._b0, axis=-1)
             | (mx.take(b, self._b0 + 1, axis=-1) << 8)
             | (mx.take(b, self._b0 + 2, axis=-1) << 16))
        return (w >> self._sh) & self._mask

    def __call__(self, ids):
        if self._pn and not self._byte_codes:
            c = self._unpack(self.codes[ids])            # [.., nsub]
        else:
            c = self.codes[ids]                              # [.., nsub]
        v = self.codebook[c.astype(mx.uint32)]           # [.., nsub, d]
        # Apply the group scales by BROADCAST over a [.., ngrp, G] view rather
        # than materialising mx.repeat(sc, G)'s full-width copy: element
        # flat[g*G+k] is multiplied by sc[g] either way, so the products and
        # their rounding are identical, but the repeat's [.., cols] temporary
        # (and the dispatch that fills it) is gone.
        G = self.group_size
        sc = self.vq_scales[ids]                         # [.., cols/G]
        ngrp = sc.shape[-1]
        prod = v.reshape(*ids.shape, ngrp, G) * sc[..., None]
        return prod.reshape(*ids.shape, ngrp * G).astype(mx.bfloat16)
