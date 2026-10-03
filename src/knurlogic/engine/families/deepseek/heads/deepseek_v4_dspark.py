"""DSpark: DeepSeek-V4-Flash-Vision-Exp's block drafter, bound to a loaded
deepseek_v4 trunk from a packed sidecar.

Per the official inference/model.py (DSparkAttention, DSparkMarkovHead,
DSparkConfidenceHead, DSparkBlock, Transformer.forward / forward_spec;
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, MIT):

    main_h   the HC-mean (mean over the hc streams) of the OUTPUT of each
             trunk layer in dspark_target_layer_ids, concatenated:
             [B, S, n * D] per committed position
    main_x   main_norm(main_proj(main_h))           stage 0's, shared by
                                                    every stage
    cache    per stage, kv_norm(wkv(main_x)) roped at its position: a
             window of the last sliding_window committed positions
    block    [t, noise, ..., noise] (block_size ids; t at p+1 when the
             last committed position is p), embedded with the trunk's
             table and broadcast to hc streams; each stage is a trunk-style
             block whose attention queries the block (roped p+1..p+K)
             against the window AND the whole block (no causal mask
             inside it), with the stage's own sink
    logits   lm_head(norm(hc_head(x)))[k] + markov_w2(markov_w1(id_k)),
             id_0 = t and id_{k+1} the token drafted from logits k: the
             Markov term chains the block's drafts one by one
    conf     confidence_head.proj([hc_head(x), markov_w1(id_k)]): one score
             per drafted position. The reference returns it and its
             generate.py never reads it; nothing here acts on it either.

The reference's FP8 simulation of the kv's non-rope dims (`act_quant`) is
not done, as the trunk's V4Attention does not do it.

THE SIDECAR (`mtp-head-dspark-mxfp4.safetensors` beside the trunk,
outside its index; heads/dspark_pack.py packs it from the HF
checkpoint). Every key is under `mtp.<stage>.`, in the trunk's
post-sanitize layer layout where the stage is a trunk block:

    attn.{attn_sink, q_norm.weight, kv_norm.weight, wq_a.weight,
          wkv.weight, wq_b.weight, wo_a.weight, wo_b.weight}
    attn_norm.weight, ffn_norm.weight
    ffn.gate.{weight, e_score_correction_bias[, bias_vl]}
    ffn.shared_experts.{gate,up,down}_proj.weight
    ffn.switch_mlp.{gate,up,down}_proj.{weight, scales}   (mxfp4)
    hc_attn.{fn, base, scale}, hc_ffn.{fn, base, scale}
  stage 0:     main_proj.weight, main_norm.weight
  last stage:  norm.weight, hc_head_fn, hc_head_base, hc_head_scale,
               markov_head.markov_w1.weight, markov_head.markov_w2.weight,
               confidence_head.proj.weight

wq_a and wkv stay separate (the trunk fuses them as wqkv_a): the stage's
kv projection runs on main_x and on the block, its q projection on the
block only. Any linear may be packed; the recipe is read off the tensors
(heads/deepseek_v4._recipe).
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .deepseek_v4 import _recipe

SIDECAR_NAME = "mtp-head-dspark-mxfp4.safetensors"
#: official name -> this module's parameter path, under each stage
_OWN = {"hc_head_fn": "hc_head.fn", "hc_head_base": "hc_head.base",
        "hc_head_scale": "hc_head.scale"}


def _to_module(k: str) -> str:
    """`mtp.<i>.<name>` -> `stages.<i>.<path>`."""
    _, i, rest = k.split(".", 2)
    return f"stages.{i}.{_OWN.get(rest, rest)}"


def _to_sidecar(k: str) -> str:
    _, i, rest = k.split(".", 2)
    for src, dst in _OWN.items():
        if rest == dst:
            rest = src
    return f"mtp.{i}.{rest}"


def _rope(x, rope, pos, inverse=False):
    """RoPE on the last rope dims of x ([..., S, head_dim]); `pos` is the
    first position, an int or one per row ([B])."""
    rd = rope.dims
    pe = mx.fast.rope(x[..., -rd:], rd, traditional=True, base=None,
                      scale=-1.0 if inverse else 1.0, offset=pos,
                      freqs=rope.freqs)
    return mx.concatenate([x[..., :-rd], pe], axis=-1)


class _Attention(nn.Module):
    """DSparkAttention: a compress-ratio-0 V4 attention whose keys are the
    window of main kvs plus the block's own."""

    def __init__(self, arch, args):
        super().__init__()
        D, H, hd = args.hidden_size, args.num_attention_heads, args.head_dim
        self.n_heads, self.head_dim = H, hd
        self.n_groups, self.o_lora_rank = args.o_groups, args.o_lora_rank
        self.eps = args.rms_norm_eps
        self.scale = hd ** -0.5
        self.wq_a = nn.Linear(D, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, H * hd, bias=False)
        self.wkv = nn.Linear(D, hd, bias=False)
        self.kv_norm = nn.RMSNorm(hd, eps=self.eps)
        self.attn_sink = mx.zeros((H,), dtype=mx.float32)
        self.wo_a = nn.Linear(H * hd // self.n_groups,
                              self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, D,
                              bias=args.attention_bias)
        # compress ratio 0: base rope_theta, no YaRN (the reference's
        # original_seq_len = 0), as the trunk's window-only layers
        self.rope = arch.DeepseekV4RoPE(args.qk_rope_head_dim,
                                        args.rope_theta, None)
        self._project = arch.V4Attention._grouped_output_projection
        self._sdpa = arch.scaled_dot_product_attention

    def main_kv(self, main_x, pos):
        """[B, S, D] at positions pos.. -> the window's kvs [B, S, hd]."""
        return _rope(self.kv_norm(self.wkv(main_x)), self.rope, pos)

    def __call__(self, x, window, mask, pos):
        """x [B, K, D] (the block at pos..pos+K-1), window [B, W, hd]."""
        B, K, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, K, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = _rope(mx.fast.rms_norm(q, None, self.eps), self.rope, pos)
        kv = _rope(self.kv_norm(self.wkv(x)), self.rope, pos)
        keys = mx.concatenate([window.astype(kv.dtype), kv], axis=1)[:, None]
        o = self._sdpa(q, keys, keys, cache=None, scale=self.scale,
                       mask=mask, sinks=self.attn_sink.astype(q.dtype))
        o = _rope(o, self.rope, pos, inverse=True)
        o = o.transpose(0, 2, 1, 3).reshape(B, K, self.n_heads * self.head_dim)
        return self.wo_b(self._project(self, o))


class _Markov(nn.Module):
    def __init__(self, V, R):
        super().__init__()
        self.markov_w1 = nn.Embedding(V, R)
        self.markov_w2 = nn.Linear(R, V, bias=False)


class _Confidence(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.proj = nn.Linear(n, 1, bias=False)


class _Stage(nn.Module):
    """One DSparkBlock: a trunk-style block (hyper-connections, attention,
    MoE) whose attention is _Attention."""

    def __init__(self, arch, args, skel, stage, n_stages):
        super().__init__()
        D, eps = args.hidden_size, args.rms_norm_eps
        hc = (D, args.hc_mult, eps, args.hc_sinkhorn_iters, args.hc_eps)
        self.attn = _Attention(arch, args)
        self.attn_norm = nn.RMSNorm(D, eps=eps)
        self.ffn_norm = nn.RMSNorm(D, eps=eps)
        # the official layer id: never hash-routed, bias_vl with vision
        self.ffn = arch.DeepseekV4MoE(skel, args.num_hidden_layers + stage)
        self.hc_attn = arch.HyperConnection(*hc)
        self.hc_ffn = arch.HyperConnection(*hc)
        if stage == 0:
            n = len(args.dspark_target_layer_ids)
            self.main_proj = nn.Linear(n * D, D, bias=False)
            self.main_norm = nn.RMSNorm(D, eps=eps)
        if stage == n_stages - 1:
            R = args.dspark_markov_rank
            self.norm = nn.RMSNorm(D, eps=eps)
            self.hc_head = arch.HyperHead(D, args.hc_mult, eps, args.hc_eps)
            self.markov_head = _Markov(args.vocab_size, R)
            self.confidence_head = _Confidence(D + R)

    def __call__(self, h, window, mask, pos, ids):
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn(self.attn_norm(y), window, mask, pos)
        h = self.hc_attn.hc_post(y, residual, post, comb)
        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn(self.ffn_norm(y), ids)
        return self.hc_ffn.hc_post(y, residual, post, comb)


class _DSpark(nn.Module):
    def __init__(self, stages):
        super().__init__()
        self.stages = stages


class DSparkCache:
    """The stages' windows of main kvs: per stage [B, W, head_dim], the
    newest position last, W <= sliding_window; a row shorter than W is
    left-padded (its first W - min(length, window) slots are masked).
    Only committed positions ever enter it, so a speculative step never
    rolls it back; it is not trimmable (the trunk's DeepseekV4Cache is
    not either, so a prefix entry is reused whole or not at all)."""

    def __init__(self, n_stages: int, window: int):
        self.n_stages = n_stages
        self.window = window
        self.keys: list | None = None
        #: committed positions per row
        self.lengths: list[int] = [0]

    # ---------------------------------------------------------- position
    @property
    def offset(self):
        if len(self.lengths) == 1:
            return self.lengths[0]
        return mx.array(self.lengths)

    def positions(self):
        """The next position per row: an int when every row agrees."""
        if len(set(self.lengths)) == 1:
            return self.lengths[0]
        return mx.array(self.lengths, dtype=mx.int32)

    def append(self, kvs: list, S: int) -> None:
        if self.keys is None:
            self.keys = [kv[:, -self.window:] for kv in kvs]
        else:
            self.keys = [mx.concatenate([k, kv.astype(k.dtype)], axis=1)
                         [:, -self.window:] for k, kv in zip(self.keys, kvs)]
        self.lengths = [n + S for n in self.lengths]

    def mask(self, K: int):
        """[B, 1, 1, W + K] (True = attend) or None when every slot is a
        row's own. The block's K keys are visible to every block query."""
        W = int(self.keys[0].shape[1])
        valid = [min(n, self.window) for n in self.lengths]
        if all(v >= W for v in valid):
            return None
        j = mx.arange(W)[None, :]
        win = j >= (W - mx.array(valid))[:, None]
        m = mx.concatenate([win, mx.ones((len(valid), K), dtype=mx.bool_)],
                           axis=1)
        return m[:, None, None, :]

    # ------------------------------------------------------------- batch
    @classmethod
    def merge(cls, caches: list) -> "DSparkCache":
        c0 = caches[0]
        out = cls(c0.n_stages, c0.window)
        out.lengths = [n for c in caches for n in c.lengths]
        have = [c for c in caches if c.keys is not None]
        if not have:
            return out
        W = max(int(c.keys[0].shape[1]) for c in have)
        ref = have[0].keys
        stacked = []
        for s in range(c0.n_stages):
            parts = []
            for c in caches:
                B = len(c.lengths)
                if c.keys is None:
                    parts.append(mx.zeros((B, W, ref[s].shape[-1]),
                                          dtype=ref[s].dtype))
                    continue
                k = c.keys[s]
                pad = W - int(k.shape[1])
                if pad:
                    k = mx.concatenate([mx.zeros((B, pad, k.shape[-1]),
                                                 dtype=k.dtype), k], axis=1)
                parts.append(k)
            stacked.append(mx.concatenate(parts, axis=0))
        out.keys = stacked
        return out

    def extend(self, other: "DSparkCache") -> None:
        m = type(self).merge([self, other])
        self.keys, self.lengths = m.keys, m.lengths

    def filter(self, keep: list) -> None:
        self.lengths = [self.lengths[i] for i in keep]
        if self.keys is not None:
            idx = mx.array(keep)
            self.keys = [k[idx] for k in self.keys]

    def extract(self, i: int) -> "DSparkCache":
        out = type(self)(self.n_stages, self.window)
        n = self.lengths[i]
        out.lengths = [n]
        if self.keys is not None and n:
            v = min(n, self.window)
            out.keys = [mx.array(k[i:i + 1, -v:]) for k in self.keys]
        return out

    # ------------------------------------------------- the cache contract
    @property
    def state(self):
        return list(self.keys or [])

    @property
    def nbytes(self) -> int:
        return sum(int(k.nbytes) for k in self.keys or [])

    def is_trimmable(self) -> bool:
        return False

    def trim(self, n: int) -> int:
        return 0

    def empty(self) -> bool:
        return self.keys is None


class DSparkHead:
    """The DSpark stages bound to a loaded deepseek_v4 trunk. Drafts
    `block_size` tokens per pass (engine/mtp/block_loop.BlockBatch)."""

    def __init__(self, model, arch):
        text = getattr(model, "language_model", model)
        core = text.model
        args = core.args
        if int(getattr(args, "dspark_block_size", 0) or 0) <= 0:
            raise ValueError("deepseek_v4 DSpark: the config has no "
                             "dspark_block_size; this trunk has no DSpark")
        self.model = text
        self.core = core
        self.arch = arch
        self.args = args
        self.block_size = int(args.dspark_block_size)
        self.noise_id = int(args.dspark_noise_token_id)
        self.targets = [int(i) for i in args.dspark_target_layer_ids]
        self.n_stages = int(args.num_nextn_predict_layers)
        skel = dataclasses.replace(
            args, quantization=args.quantization or {"skeleton": True})
        self.m = _DSpark([_Stage(arch, args, skel, s, self.n_stages)
                          for s in range(self.n_stages)])

    # ------------------------------------------------------------ capture
    def capture_paths(self) -> list[str]:
        """Where the target layers' OUTPUTS are captured: each is the next
        layer's input, the last layer's is the trunk hc_head's
        (engine/mtp/capture.py)."""
        n = self.args.num_hidden_layers
        return [f"layers.{i + 1}" if i + 1 < n else "hc_head"
                for i in self.targets]

    def main_hidden(self, captured: list) -> mx.array:
        """The captured [B, S, hc, D] streams -> [B, S, n * D]."""
        return mx.concatenate([h.mean(axis=2) for h in captured], axis=-1)

    def make_draft_cache(self) -> DSparkCache:
        return DSparkCache(self.n_stages, self.args.sliding_window)

    # ------------------------------------------------------------ sidecar
    def bind(self, w: dict) -> dict:
        """Shape the stages to sidecar tensors and check they match exactly
        (names and shapes; `w` may hold header shapes). Returns the values
        under this module's parameter paths."""
        bad = [k for k in w if not k.startswith("mtp.")]
        if bad:
            raise ValueError(f"deepseek_v4 DSpark: {len(bad)} tensors "
                             f"outside 'mtp.', e.g. {bad[:3]}")
        mw = {_to_module(k): v for k, v in w.items()}

        def pred(path, mod):
            return _recipe(mw, path, mod) or False

        nn.quantize(self.m, class_predicate=pred)
        slots = dict(tree_flatten(self.m.parameters()))
        missing = sorted(set(slots) - set(mw))
        extra = sorted(set(mw) - set(slots))
        if missing or extra:
            raise ValueError(
                f"deepseek_v4 DSpark does not match the sidecar: missing "
                f"{missing[:4]} ({len(missing)}), unexpected {extra[:4]} "
                f"({len(extra)})")
        wrong = [(k, tuple(v.shape), tuple(slots[k].shape))
                 for k, v in mw.items()
                 if tuple(v.shape) != tuple(slots[k].shape)]
        if wrong:
            raise ValueError(f"deepseek_v4 DSpark: shape mismatch {wrong[:4]}")
        return mw

    def load_weights(self, w: dict) -> "DSparkHead":
        mw = self.bind(w)
        self.m.load_weights(list(mw.items()), strict=True)
        mx.eval(self.m.parameters())
        return self

    def tensors(self) -> dict:
        return {_to_sidecar(k): v
                for k, v in tree_flatten(self.m.parameters())}

    def save(self, path) -> dict:
        flat = self.tensors()
        mx.save_safetensors(str(path), flat, metadata={"format": "mlx"})
        return flat

    @classmethod
    def from_sidecar(cls, model, arch, path):
        return cls(model, arch).load_weights(mx.load(str(path)))

    # -------------------------------------------------------------- draft
    def advance(self, main_h, cache: DSparkCache) -> DSparkCache:
        """Commit positions: main_h [B, S, n * D] at the cache's next
        positions enter every stage's window."""
        s0 = self.m.stages[0]
        main_x = s0.main_norm(s0.main_proj(main_h))
        pos = cache.positions()
        cache.append([st.attn.main_kv(main_x, pos) for st in self.m.stages],
                     int(main_h.shape[1]))
        return cache

    def block(self, t, cache: DSparkCache):
        """The block pass after the cache's last position p, with t ([B])
        the token at p+1 -> (x [B, K, D] the last stage's hc_head output,
        logits [B, K, V] before the Markov term)."""
        K = self.block_size
        B = int(t.shape[0])
        ids = mx.concatenate(
            [t.reshape(B, 1).astype(mx.int32),
             mx.full((B, K - 1), self.noise_id, dtype=mx.int32)], axis=1)
        e = self.core.embed_tokens(ids)
        h = mx.contiguous(mx.broadcast_to(
            e[:, :, None, :], (B, K, self.args.hc_mult, e.shape[-1])))
        mask = cache.mask(K)
        pos = cache.positions()
        for st, win in zip(self.m.stages, cache.keys):
            h = st(h, win, mask, pos, ids)
        last = self.m.stages[-1]
        x = last.hc_head(h)
        return x, self.model.lm_head(last.norm(x))

    def markov(self, prev):
        """prev [B] -> (logit bias [B, V] float32, embedding [B, R])."""
        mk = self.m.stages[-1].markov_head
        e = mk.markov_w1(prev.astype(mx.int32))
        w = mk.markov_w2
        if isinstance(w, nn.QuantizedLinear):
            return w(e).astype(mx.float32), e
        return e.astype(mx.float32) @ w.weight.astype(mx.float32).T, e

    def confidence(self, x, embeds):
        """x [B, K, D], embeds [B, K, R] -> [B, K] float32."""
        proj = self.m.stages[-1].confidence_head.proj
        z = mx.concatenate([x, embeds.astype(x.dtype)], axis=-1)
        return (z.astype(mx.float32)
                @ proj.weight.astype(mx.float32).T).squeeze(-1)

    def draft(self, t, cache: DSparkCache, pick=None):
        """The reference's forward_head: -> (ids [B, K + 1] with t first,
        logits [B, K, V] float32 with the Markov term, confidence [B, K]).
        `pick(k, logits_k) -> [B]` chooses each draft (argmax by default,
        the reference at temperature 0)."""
        x, base = self.block(t, cache)
        prev = t.astype(mx.int32)
        ids, rows, embeds = [prev], [], []
        for k in range(self.block_size):
            bias, e = self.markov(prev)
            row = base[:, k].astype(mx.float32) + bias
            prev = (mx.argmax(row, axis=-1) if pick is None
                    else pick(k, row)).astype(mx.int32)
            ids.append(prev)
            rows.append(row)
            embeds.append(e)
        return (mx.stack(ids, axis=1), mx.stack(rows, axis=1),
                self.confidence(x, mx.stack(embeds, axis=1)))
