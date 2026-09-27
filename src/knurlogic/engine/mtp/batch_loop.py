"""MTP speculative decoding over a BATCH of sequences.

loop.py runs one sequence: draft one token with the head, verify it inside a
2-token trunk forward, roll back and replay on rejection. This module runs
that same step over B rows at once, which is what lets the batch engine keep
drafting when several requests are in flight instead of trading the head
away for batching.

What makes it batchable at all is a property of the 1-token draft: EVERY row
advances by exactly two positions per step, accepted or not.

  accepted -> the verify forward committed [t1, d2]; the caches are right.
  rejected -> the caches hold [t1, d2] with the wrong second token; the
              trunk's own t2 came out of the same forward, so the step still
              emits two tokens, but position offset-1 has to be rewritten.

Because the row count in the KV cache is lockstep, the batched caches
(mlx-lm's BatchKVCache, one offset per row, shared write index) never need
per-row trimming. Rejection is handled as ONE whole-batch trim(2) and ONE
whole-batch replay forward with the tokens that actually committed — for a
row that accepted, the replay writes back exactly what was there. That costs
an extra forward whenever ANY row rejects, i.e. with acceptance a and B rows
the expected forwards per step are 1 + (1 - a^B) for 2B tokens, against B
tokens per forward without drafting. It is a decaying win, not a free one:
measure it (see the numbers in the runner's engine_mode.py) before assuming
it beats plain batching at a given B.

A row that cannot draft (a request carrying images, whose head seeding the
vision embedding patch would corrupt) rides along: it pays the replay every
step and gets the trunk's own tokens, never a drafted one.

Sampling is per row — temperature, top-p, processors, and the acceptance
test are the row's own — so a batch may mix greedy and sampled requests. At
temperature the verdict is the same exact rejection sampling loop.py uses
(sampling.rejection_correct); at temperature 0 it is `draft == argmax`. One
deliberate difference from the old sequential loop: a row's logits
processors (repetition penalty, eos ban) are applied to the TRUNK row that
verifies the draft, not only to the draft and to t1 — otherwise a penalised token could be committed
through the verify path that sampling would have refused.

Across machines (a pipeline split, engine/runtime/pipeline.py) every rank
runs this same loop; `coord` carries rank 0's regime and drafts (B1) and
its verdicts (B2) to the others, one fixed broadcast each per step.
"""
from __future__ import annotations

import logging

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

import mlx.core as mx
from mlx_lm.generate import _extend_cache, _merge_caches

from .caches import position, restore, snapshot
from .seed import seed_head
from .sampling import Distribution, Keys, rejection_correct

__all__ = ["RowParams", "Row", "Emitted", "RowStep", "MTPBatch", "admit",
           "default_draft_max_rows"]

import time

#: exo takes this from its runner bootstrap; a module
#: logger is the stdlib answer and nothing downstream cares.
logger = logging.getLogger(__name__)



#: the ceiling that means "draft at every width": KNURLOGIC_MTP_DYNAMIC=off
ALWAYS_DRAFT = 1 << 30


def default_draft_max_rows() -> int | None:
    """A fixed row ceiling for drafting from KNURLOGIC_MTP_BATCH_MAX_ROWS (0:
    never draft -- it read as 1); with none, KNURLOGIC_MTP_DYNAMIC=off
    drafts every step (ALWAYS_DRAFT); else None for the adaptive rule
    (`MTPBatch.drafting_pays`)."""
    raw = os.environ.get("KNURLOGIC_MTP_BATCH_MAX_ROWS")
    if not raw:
        dyn = os.environ.get("KNURLOGIC_MTP_DYNAMIC", "").strip().lower()
        return ALWAYS_DRAFT if dyn in ("off", "0", "false", "no") else None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


#: steps taken in a regime before its cost estimate counts; and how often
#: the currently-losing regime is re-tried at the same width.
EXPLORE_STEPS = 6
RECHECK_EVERY = 96
#: a recheck that confirms the winner doubles the wait before the next one
#: (per width), up to this many base intervals; a flip resets it
RECHECK_BACKOFF_MAX = 64


@dataclass
class RowParams:
    """Everything about a request the loop needs, fixed at admission."""

    max_tokens: int
    #: None = greedy (temperature 0). Otherwise logits [1, V] -> Distribution.
    dist: Optional[Callable[[mx.array], Distribution]]
    processors: List[Callable[[mx.array, mx.array], mx.array]]
    eos: set
    #: False for a row the head must not draft for; it still decodes correctly.
    drafts: bool = True
    #: the request's own random stream (a seed), or None for the global one
    keys: Optional[Keys] = None


@dataclass
class Row:
    """One admitted sequence: prefilled, first token sampled, head seeded."""

    uid: int
    params: RowParams
    cache: list                    # single-row trunk caches, full prompt inside
    hcache: Any                    # head cache (an EMPTY one when not drafting)
    t1: mx.array                   # [1] int32, the first token to commit
    row_t1: mx.array               # [1, V] the logits t1 was sampled from
    draft_row: Optional[mx.array]  # [1, V] the head's draft of t2, or None
    n_prompt: int
    drafts: bool
    #: design D4: this row's MRoPE offset (Qwen: positions after an image
    #: run ahead of the token count by rope_delta). Recomputed from the key
    #: at every admission -- positions are pure in the key -- so it needs no
    #: home in the prefix cache.
    rope_delta: int = 0
    #: True when the family gave explicit position ids for this row; decode
    #: then passes `position_ids` for the whole batch (see MTPBatch._pos_kw).
    mrope: bool = False


@dataclass
class Emitted:
    token: int
    from_draft: bool
    finish: Optional[str]
    logits: mx.array               # [V] the trunk row that produced it
    #: every value of `logits` finite -- computed inside the step's own
    #: final eval, so the NaN guard adds no sync of its own
    finite: bool = True


@dataclass
class RowStep:
    uid: int
    tokens: List[Emitted]
    #: running per-row acceptance: (accepted drafts, speculative steps)
    accepted: int
    steps: int


def _apply(row: mx.array, procs, emitted) -> mx.array:
    """Logits processors over the history; `emitted` is a list of ids or
    an id array (lazy: the draft step's history ends in t1, which it has
    not read back yet)."""
    if not procs:
        return row
    hist = emitted if isinstance(emitted, mx.array) else mx.array(emitted)
    for proc in procs:
        row = proc(hist, row)
    return row


def _with(emitted: List[int], t1: mx.array) -> mx.array:
    """The history for the position after t1: what was emitted, then t1."""
    return mx.concatenate([mx.array(emitted, dtype=mx.int32),
                           t1.astype(mx.int32)])


def _finite_rows(rows: mx.array) -> mx.array:
    """[B, V] -> [B] bool, lazy: evaluated with the step's last sync."""
    return mx.isfinite(rows).all(axis=-1)


def _mark(out: List["RowStep"], fin: mx.array) -> None:
    """fin: [B, k] already evaluated; token j of row i gets fin[i][j]."""
    for rs, flags in zip(out, fin.tolist()):
        for em, good in zip(rs.tokens, flags):
            em.finite = bool(good)


def _pick(row: mx.array, p: RowParams, emitted: List[int]) -> mx.array:
    """row: [1, V] -> token [1]."""
    row = _apply(row, p.processors, emitted)
    if p.dist is None:
        return mx.argmax(row, axis=-1)
    return p.dist(row).sample(_key(p, len(emitted)))


def _key(p: RowParams, position: int):
    """The seeded row's key for the token at `position`, or None."""
    return p.keys.at(position) if p.keys is not None else None


def admit(
    model,
    head,
    get_h: Callable[[], mx.array],
    ids: mx.array,
    params: RowParams,
    *,
    uid: int,
    make_draft_cache: Callable[[], Any],
    prefill_step_size: int = 2048,
    prefill_ctx=None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cache: Optional[list] = None,
    hcache: Any = None,
    start_pos: int = 0,
    on_chunk: Optional[Callable[[list, Any], None]] = None,
    embeds: Optional[mx.array] = None,
    extras: Optional[dict] = None,
    chunk_boundaries: Optional[List[tuple]] = None,
    rope_delta: int = 0,
    mrope: bool = False,
    checkpoints: Iterable[int] = (),
    on_checkpoint: Optional[Callable[[int, list, Any, Optional[mx.array]],
                                     None]] = None,
) -> Row:
    """Prefill one prompt and seed the head for it: loop.py's prefill, kept
    as a standalone so the batch can admit rows between steps.

    Prefix reuse: `cache` (trunk) and `hcache` (head) may arrive from the KV
    prefix pool already holding positions 0..start_pos-1, both at offset
    start_pos -- the head's cache is stored in the pool beside the trunk's
    and trimmed with it, so it stays one row per committed token. Only
    ids[start_pos:] is prefilled and the head is seeded over
    start_pos..P-2. A drafting row whose head cache is NOT at start_pos
    cannot be admitted from that prefix (the head would draft off the
    wrong history); the caller falls back to a fresh prefill. `on_chunk`
    is called with (cache, hcache) after every prefill chunk so the caller
    can snapshot recurrent state for the pool.

    `prefill_ctx` (a zero-arg context-manager factory) wraps the chunked
    prompt forwards only — a vision request's embedding patch goes there,
    and the final single-token forward that produces the first logits runs
    outside it, as it does on the stock path.

    The head is seeded from position 0 (see loop.py on alignment), which is
    why an admitted row always starts from a FRESH trunk cache: a reused
    prefix would shift every rotary position the head sees.

    IMAGES (design D5; docs/design/vision.md). A vision row arrives with
    `embeds` -- the trunk's `input_embeddings` for ids[start_pos:] only, as
    `Family.embed` builds them, [1, n - start_pos, D] -- and optional
    `extras`, further trunk kwargs over the same span. An extras value is an
    array whose sequence axis is 1, or an `(array, axis)` pair naming it
    (position ids are [3, 1, L]: axis -1). Every chunk forward gets the
    matching slice of each, the ids still ride along (the trunk ignores
    them when embeddings are given, and a drafting head needs them).
    `chunk_boundaries` are [start, end) spans no chunk edge may fall
    strictly inside -- gemma attends bidirectionally within an image, so a
    chunk that cut one would compute the first half without the second
    (critique B5). Edges snap back to the span start, or forward past the
    span when it starts the chunk (a span longer than the step is one
    chunk). The last forward, which yields the first logits, is widened the
    same way if the prompt ends inside a span. `rope_delta` / `mrope` ride
    on the Row so MTPBatch can hand the trunk this row's positions on every
    decode step (design D4).

    CHECKPOINTS. Positions c (the server's segment ends: after the system
    prompt, after the last user message) where prefill stops a chunk and
    calls `on_checkpoint(c, trunk, head_cache, h)` so the caller can store
    the prompt cache for ids[:c]. It exists for models whose caches cannot
    be trimmed (linear attention): when the next turn's template re-renders
    the tail differently, only an entry that ends before the difference is
    reusable at all. The head is seeded only to c-1 there -- its input at
    c-1 is (h_{c-1}, x_c) and x_c is the token the next turn may change --
    so `h` is h_{c-1}, for the restore to replay that one step with the
    new x_c. A non-drafting row passes head_cache and h as None.
    """
    import contextlib

    if ids.ndim == 1:
        ids = ids[None]
    n = int(ids.shape[1])
    start_pos = max(0, int(start_pos))
    if cache is None:
        cache = model.make_cache()
        start_pos = 0
    drafts = params.drafts and head is not None
    dcache = hcache if hcache is not None else make_draft_cache()
    if drafts and start_pos > 0:
        hoff = position(dcache) or 0
        if hoff != start_pos:
            raise ValueError(
                f"head cache at offset {hoff} cannot seed a prefix at {start_pos}"
            )

    spans = [tuple(b) for b in (chunk_boundaries or ())]
    last = _snap_down(n - 1, spans, start_pos)

    def _kw(a: int, b: int) -> dict:
        """Trunk kwargs for ids[:, a:b]: the embeds/extras slices."""
        if embeds is None and not extras:
            return {}
        kw = {}
        if embeds is not None:
            kw["input_embeddings"] = embeds[:, a - start_pos:b - start_pos]
        for name, v in (extras or {}).items():
            arr, axis = v if isinstance(v, tuple) else (v, 1)
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(a - start_pos, b - start_pos)
            kw[name] = arr[tuple(sl)]
        return kw

    h_chunks: List[mx.array] = []
    seeded = start_pos          # the head is seeded over [start_pos, seeded)
    cps = sorted(c for c in set(checkpoints or ())
                 if start_pos < c <= last and _inside(c, spans) is None
                 and (not drafts or c - start_pos >= 2))
    if on_checkpoint is None:
        cps = []
    ctx = prefill_ctx() if prefill_ctx is not None else contextlib.nullcontext()
    with ctx:
        i = start_pos
        while i < last:
            end = _snap_chunk_end(i, min(i + prefill_step_size, last), spans,
                                  last)
            if cps and i < cps[0] < end:
                end = cps[0]
            chunk = ids[:, i:end]
            if not chunk.shape[1]:
                break
            model(chunk, cache=cache, **_kw(i, end))
            i = end
            # Evaluate the cache and the captured hidden state, never the
            # logits: forcing them would materialise the lm_head projection
            # for every prompt position (loop.py says the same).
            want = [c.state for c in cache if hasattr(c, "state")]
            if drafts:
                h = get_h()
                h_chunks.append(h)
                want.append(h)
            mx.eval(want)
            mx.clear_cache()
            if on_chunk is not None:
                on_chunk(cache, dcache)
            if cps and end == cps[0]:
                c = cps.pop(0)
                h_c = None
                if drafts:
                    # Seed [seeded, c-1), keep h from c-1 on for the rest.
                    h_all = (mx.concatenate(h_chunks, axis=1)
                             if len(h_chunks) > 1 else h_chunks[0])
                    seed_head(head, [h_all], ids, c, dcache,
                              prefill_step_size, start=seeded)
                    h_chunks = [h_all[:, c - 1 - seeded:]]
                    h_c = h_all[:, c - 1 - seeded:c - seeded]
                    seeded = c - 1
                    mx.eval(h_chunks[0], h_c)
                on_checkpoint(c, cache, dcache if drafts else None, h_c)
            if on_progress is not None:
                on_progress(end, n)

    logits = model(ids[:, max(last, 0):], cache=cache, **_kw(max(last, 0), n))
    row_t1 = logits[:, -1]
    t1 = _pick(row_t1, params, []).astype(mx.int32)

    draft_row = None
    if drafts:
        h_chunks.append(get_h())
        h_last = h_chunks[-1][:, -1:]
        mx.eval(t1, h_last)
        # Positions 0..P-2, so the head enters decoding with the trunk's
        # history and cache.offset == P-1 == the true position. Chunked like
        # the trunk's prefill: one call over the whole prompt is quadratic
        # on an attention head (see seed.py).
        seed_head(head, h_chunks, ids, n, dcache, prefill_step_size, start=seeded)
        h_chunks.clear()
        mx.clear_cache()
        # Bootstrap draft at position P-1: input (h_{P-1}, x_P).
        draft_row = head.draft_logits(h_last, t1[None], dcache)[:, -1]
        mx.eval(draft_row)
    else:
        mx.eval(t1)

    return Row(
        uid=uid, params=params, cache=cache, hcache=dcache, t1=t1,
        row_t1=row_t1, draft_row=draft_row, n_prompt=n, drafts=drafts,
        rope_delta=int(rope_delta), mrope=bool(mrope),
    )


def _inside(p: int, spans) -> Optional[tuple]:
    """The span p falls strictly inside (s < p < e), if any."""
    for s, e in spans:
        if s < p < e:
            return (s, e)
    return None


def _snap_down(p: int, spans, floor: int) -> int:
    """p moved back to the start of a span it cuts, never below floor."""
    sp = _inside(p, spans)
    return max(sp[0], floor) if sp is not None else p


def _snap_chunk_end(i: int, end: int, spans, last: int) -> int:
    """A chunk [i, end) whose end would cut a span: end at the span's start
    if that leaves the chunk non-empty, otherwise run past the span (capped
    at `last`, itself already snapped)."""
    sp = _inside(end, spans)
    if sp is None:
        return end
    if sp[0] > i:
        return sp[0]
    return min(sp[1], last) if sp[1] <= last else last


class MTPBatch:
    """The batched speculative state: B rows advancing two tokens a step.

    Rows enter through `extend` (between steps) and leave when a token
    finishes them or through `remove`. Row order is `uids`; every per-row
    structure below is indexed the same way and filtered together.
    """

    def __init__(self, model, head, get_h: Callable[[], mx.array], *,
                 copy_caches: bool, draft_max_rows: int | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.model = model
        self.head = head
        self.get_h = get_h
        self.copy_caches = copy_caches
        # A fixed ceiling (explicit or KNURLOGIC_MTP_BATCH_MAX_ROWS) wins; otherwise
        # the regime is chosen by MEASURED cost per committed token, per row
        # count (see drafting_pays). The unknown is not acceptance alone but
        # how much more a 2-wide forward costs than a 1-wide one at each
        # width -- on the M4 with Qwen3.8-Flash-Next-VQ-4.4bpw the ratio is
        # ~1.5 at one row, so only a timing can decide.
        self.draft_max_rows = (
            draft_max_rows if draft_max_rows is not None else default_draft_max_rows()
        )
        self._clock = clock
        self.acc_est = 0.8                   # EMA of the head's hit rate (logging)
        # (rows, drafting) -> (EMA seconds per token, steps measured)
        self._cost: dict = {}
        self._regime: Optional[bool] = None  # last step drafted? (for the log)
        self._since_recheck = 0
        # rows -> (backoff multiplier, winner when the last recheck began)
        self._backoff: dict = {}
        self._explore: Optional[tuple] = None   # (rows, drafting, steps left)
        #: engine/runtime/pipeline.Coord on a pipeline split: rank 0's regime,
        #: drafts and verdicts reach every rank through it (B1, B2)
        self.coord = None

        self.uids: List[int] = []
        self.params: List[RowParams] = []
        self.emitted: List[List[int]] = []
        self.drafts: List[bool] = []
        self.accepted: List[int] = []
        self.steps: List[int] = []
        self.n_prompt: List[int] = []
        self.rope_delta: List[int] = []
        self.mrope: List[bool] = []

        self.cache: list = []               # batched trunk caches
        self.hcache: Any = None             # batched head cache
        self.t1: Optional[mx.array] = None          # [B] int32
        self.row_t1: Optional[mx.array] = None      # [B, V]
        self.draft_row: Optional[mx.array] = None   # [B, V]

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def any_drafting(self) -> bool:
        return self.head is not None and any(self.drafts)

    def drafting_pays(self, rows: int) -> bool:
        """Draft this step, or take a plain one-token step?

        With no fixed ceiling: each regime's cost per committed token is
        timed at this row count; an unmeasured regime is tried for
        EXPLORE_STEPS steps, the cheaper one is taken, and the loser is
        re-tried every RECHECK_EVERY steps so a head whose acceptance shifts
        with the text (or a batch whose width changed) is not stuck.
        """
        if self.head is None:
            return False
        if self.draft_max_rows is not None:
            return rows <= self.draft_max_rows
        ex = self._explore
        if ex is not None and ex[0] == rows and ex[2] > 0:
            return ex[1]
        self._explore = None
        d = self._cost.get((rows, True))
        p = self._cost.get((rows, False))
        if d is None or d[1] < EXPLORE_STEPS:
            self._explore = (rows, True, EXPLORE_STEPS)
            return True
        if p is None or p[1] < EXPLORE_STEPS:
            self._explore = (rows, False, EXPLORE_STEPS)
            return False
        best = d[0] <= p[0]
        mult, last = self._backoff.get(rows, (1, None))
        if last is not None:
            # a recheck just finished: same winner -> wait twice as long
            mult = 1 if best != last else min(mult * 2, RECHECK_BACKOFF_MAX)
            self._backoff[rows] = (mult, None)
        # Re-trying the loser costs EXPLORE_STEPS of it; stretch the wait by
        # how much it loses so that probe stays a small share of the time
        # (a draft 13x dearer than plain is not re-tried every 96 steps).
        ratio = max(d[0], p[0]) / max(min(d[0], p[0]), 1e-9)
        if self._since_recheck >= RECHECK_EVERY * mult * max(1.0, ratio):
            self._since_recheck = 0
            self._backoff[rows] = (mult, best)
            self._explore = (rows, not best, EXPLORE_STEPS)
            return not best
        return best

    def cost_per_token(self, rows: int) -> Optional[float]:
        """The measured seconds per committed token at this width, in the
        cheaper regime; None until a regime has been timed there."""
        got = [c[0] for c in (self._cost.get((rows, True)),
                              self._cost.get((rows, False))) if c]
        return min(got) if got else None

    def _record_cost(self, rows: int, drafting: bool, seconds: float, tokens: int) -> None:
        if tokens <= 0:
            return
        per_tok = seconds / tokens
        ema, n = self._cost.get((rows, drafting), (per_tok, 0))
        self._cost[(rows, drafting)] = (0.8 * ema + 0.2 * per_tok, n + 1)
        self._since_recheck += 1
        ex = self._explore
        if ex is not None and ex[0] == rows and ex[1] == drafting:
            self._explore = (rows, drafting, ex[2] - 1)

    def _note_regime(self, drafting: bool, rows: int) -> None:
        if drafting != self._regime:
            self._regime = drafting
        
            d = self._cost.get((rows, True))
            p = self._cost.get((rows, False))
            fmt = lambda c: f"{c[0] * 1000:.1f}ms/tok" if c else "unmeasured"
            logger.info(
                f"MTP batch {'drafting' if drafting else 'plain steps'} at {rows} rows "
                f"(draft {fmt(d)}, plain {fmt(p)}, acceptance est {self.acc_est:.2f})"
            )

    # ------------------------------------------------------------ membership

    def extend(self, rows: Iterable[Row]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.cache = _extend_cache(self.cache, _merge_caches([r.cache for r in rows]))
        if self.head is not None:
            hcs = [r.hcache for r in rows]
            merged = type(hcs[0]).merge(hcs)
            if self.hcache is None:
                self.hcache = merged
            else:
                self.hcache.extend(merged)

        V = int(rows[0].row_t1.shape[-1])
        t1 = mx.concatenate([r.t1 for r in rows]).astype(mx.int32)
        row_t1 = mx.concatenate([r.row_t1 for r in rows], axis=0)
        drafts = [
            r.draft_row if r.draft_row is not None
            else mx.zeros((1, V), dtype=row_t1.dtype)
            for r in rows
        ]
        draft_row = mx.concatenate(drafts, axis=0) if self.head is not None else None

        if self.t1 is None:
            self.t1, self.row_t1, self.draft_row = t1, row_t1, draft_row
        else:
            self.t1 = mx.concatenate([self.t1, t1])
            self.row_t1 = mx.concatenate([self.row_t1, row_t1], axis=0)
            if draft_row is not None:
                self.draft_row = mx.concatenate([self.draft_row, draft_row], axis=0)

        for r in rows:
            self.uids.append(r.uid)
            self.params.append(r.params)
            self.emitted.append([])
            self.drafts.append(r.drafts)
            self.accepted.append(0)
            self.steps.append(0)
            self.n_prompt.append(r.n_prompt)
            self.rope_delta.append(r.rope_delta)
            self.mrope.append(r.mrope)
        # A row's arrays are consumed by the first step; nothing to eval here
        # that the step will not force anyway.

    def filter(self, keep: List[int]) -> None:
        """Keep only these row indices, in this order."""
        if keep == list(range(len(self.uids))):
            return
        self.uids = [self.uids[i] for i in keep]
        self.params = [self.params[i] for i in keep]
        self.emitted = [self.emitted[i] for i in keep]
        self.drafts = [self.drafts[i] for i in keep]
        self.accepted = [self.accepted[i] for i in keep]
        self.steps = [self.steps[i] for i in keep]
        self.n_prompt = [self.n_prompt[i] for i in keep]
        self.rope_delta = [self.rope_delta[i] for i in keep]
        self.mrope = [self.mrope[i] for i in keep]
        if not keep:
            self.cache = []
            self.hcache = None
            self.t1 = self.row_t1 = self.draft_row = None
            return
        for c in self.cache:
            c.filter(keep)
        if self.hcache is not None:
            self.hcache.filter(keep)
        idx = mx.array(keep)
        self.t1 = self.t1[idx]
        self.row_t1 = self.row_t1[idx]
        if self.draft_row is not None:
            self.draft_row = self.draft_row[idx]
        # Evaluated now, not at the next step: filtering is lazy, so until
        # then the full-width arrays stay referenced and nothing is freed --
        # and the scheduler's memory guard, stopping one row to get back
        # under the limit, read no drop and stopped them all (Fable 5.1).
        mx.eval(self._arrays())

    def _arrays(self) -> list:
        from mlx.utils import tree_flatten
        out = [a for a in (self.t1, self.row_t1, self.draft_row)
               if a is not None]
        for c in list(self.cache) + ([self.hcache] if self.hcache else []):
            try:
                out += [a for _, a in tree_flatten(c.state)
                        if isinstance(a, mx.array)]
            except Exception:
                pass            # a cache with no state to read yet
        return out

    def remove(self, uids: Iterable[int]) -> None:
        drop = set(uids)
        self.filter([i for i, u in enumerate(self.uids) if u not in drop])

    def _pos_kw(self, width: int) -> dict:
        """`position_ids` for a decode forward `width` tokens wide, or {}.

        Design D4: a Qwen row whose key holds an image decodes at position
        (tokens so far + rope_delta) on all three MRoPE axes -- for EVERY
        step, not only while the image is in the new span; missing it
        degrades silently (critique B1). So as soon as one row needs it the
        whole batch gets explicit ids, [3, B, width]; a row without MRoPE
        gets its plain count, which is what the trunk would have used. The
        count is n_prompt + tokens emitted: the row's cache length before
        this step, tracked here rather than read off a left-padded batch
        cache. No MRoPE row -> {} and the trunk call is exactly as before.
        """
        if not any(self.mrope):
            return {}
        base = [self.n_prompt[i] + len(self.emitted[i]) + self.rope_delta[i]
                for i in range(len(self.uids))]
        p = mx.array(base)[:, None] + mx.arange(width)[None, :]
        return {"position_ids": mx.broadcast_to(p[None], (3, *p.shape))}

    # ------------------------------------------------------------------ step

    def step(self) -> List[RowStep]:
        """One speculative step: every live row commits two tokens.

        Returns one RowStep per row in batch order (rows that finish are
        removed from the batch before returning, so `uids` afterwards lists
        only the survivors).
        """
        B = len(self.uids)
        if B == 0:
            return []
        assert self.t1 is not None and self.row_t1 is not None
        drafting = self.drafting_pays(B)
        pre = None
        if self.coord is not None and self.head is not None:
            # B1, every step with a head: rank 0's regime and drafts
            if self.coord.leader and drafting:
                pre = self._drafts(B)
            drafting, d2 = self.coord.b1(drafting, pre[1] if pre else None, B)
            if drafting and pre is None:
                pre = (self._live(B), d2, [None] * B)
        self._note_regime(drafting, B)
        t0 = self._clock()
        out = self._plain_step() if not drafting else self._draft_step(B, pre)
        self._record_cost(B, drafting, self._clock() - t0,
                          sum(len(rs.tokens) for rs in out))
        return out

    def _live(self, B: int) -> List[bool]:
        # A row drafts this step iff it is a drafting row AND the head has
        # produced a draft for it (the batch may hold only non-drafting rows).
        return [self.drafts[i] and self.draft_row is not None
                for i in range(B)]

    def _drafts(self, B: int):
        """(live, d2 [B] int32, the draft distributions) for this step."""
        live = self._live(B)

        # --- draft d2 per row -------------------------------------------
        d2_rows: List[mx.array] = []
        qs: List[Optional[Distribution]] = []
        for i in range(B):
            p = self.params[i]
            if not live[i]:
                # Placeholder that is always "rejected" below; the replay
                # supplies the trunk's own token.
                d2_rows.append(self.t1[i:i + 1])
                qs.append(None)
                continue
            # position len+1: its history includes t1, as in a plain step
            row = _apply(self.draft_row[i:i + 1], p.processors,
                         _with(self.emitted[i], self.t1[i:i + 1])
                         if p.processors else self.emitted[i])
            if p.dist is None:
                d2_rows.append(mx.argmax(row, axis=-1))
                qs.append(None)
            else:
                q = p.dist(row)
                # the draft for position len+1 (t1 is at len)
                d2_rows.append(q.sample(_key(p, len(self.emitted[i]) + 1)))
                qs.append(q)
        d2 = mx.concatenate(d2_rows).astype(mx.int32)
        return live, d2, qs

    def _draft_step(self, B: int, pre=None) -> List[RowStep]:
        """One speculative step: every row commits t1 and a verified t2.
        `pre`: (live, d2, qs) already drawn (a pipeline's B1)."""
        assert self.t1 is not None and self.row_t1 is not None
        live, d2, qs = pre if pre is not None else self._drafts(B)

        # --- verify: one 2-wide forward over the batch -------------------
        csnap = snapshot(self.cache, copy=self.copy_caches)
        pos2 = self._pos_kw(2)
        lg2 = self.model(mx.stack([self.t1, d2], axis=1), cache=self.cache,
                         **pos2)

        # --- verdicts ----------------------------------------------------
        oks: List[Any] = [False] * B
        t2_rows: List[Any] = [self.t1[i:i + 1] for i in range(B)]
        lazy: List[mx.array] = []
        # a pipeline follower's logits are zeros: its verdicts come in B2
        judge = self.coord is None or self.coord.leader
        for i in (range(B) if judge else ()):
            p = self.params[i]
            row = _apply(lg2[i:i + 1, 0], p.processors,
                         _with(self.emitted[i], self.t1[i:i + 1])
                         if p.processors else self.emitted[i])
            if not live[i]:
                # Non-drafting row: the trunk's own token, committed through
                # the replay below.
                t2 = (mx.argmax(row, axis=-1) if p.dist is None
                      else p.dist(row).sample(
                          _key(p, len(self.emitted[i]) + 1)))
                oks[i] = False
                t2_rows[i] = t2
                lazy.append(t2)
            elif p.dist is None:
                true_t2 = mx.argmax(row, axis=-1)
                ok = true_t2 == d2[i:i + 1]
                oks[i] = ok
                t2_rows[i] = mx.where(ok, d2[i:i + 1], true_t2)
                lazy += [ok, t2_rows[i]]
            elif p.keys is not None:
                # seeded: the target's own draw under the draft's key; the
                # draft is accepted iff they agree (sampling.Keys)
                t2 = p.dist(row).sample(_key(p, len(self.emitted[i]) + 1))
                acc = t2 == d2[i:i + 1]
                oks[i] = acc
                t2_rows[i] = t2
                lazy += [acc, t2]
            else:
                pt = p.dist(row)
                acc, t2 = rejection_correct(pt.probs, qs[i].probs,
                                            d2[i:i + 1])
                oks[i] = acc
                t2_rows[i] = t2
                lazy += [acc, t2]
        if lazy:
            mx.eval(*lazy)
        ok_flags = [bool(o.item()) if isinstance(o, mx.array) else bool(o) for o in oks]
        t2 = mx.concatenate(t2_rows).astype(mx.int32)
        if self.coord is not None:
            # B2: rank 0's verdicts drive every rank's rollback and replay
            ok_flags, t2 = self.coord.b2(ok_flags, t2, B)

        n_live = sum(live)
        if n_live:
            frac = sum(int(ok_flags[i]) for i in range(B) if live[i]) / n_live
            self.acc_est = 0.9 * self.acc_est + 0.1 * frac
        for i in range(B):
            if live[i]:
                self.steps[i] += 1
                self.accepted[i] += int(ok_flags[i])

        # --- rollback + replay if anyone rejected ------------------------
        if not all(ok_flags):
            restore(self.cache, csnap)
            lg2 = self.model(mx.stack([self.t1, t2], axis=1), cache=self.cache,
                             **pos2)

        # NaN guard, lazy: joins the eval at the end of the step.
        fin = mx.stack([_finite_rows(self.row_t1), _finite_rows(lg2[:, 0])],
                       axis=1)

        # --- emit --------------------------------------------------------
        t1_list = self.t1.tolist()
        t2_list = t2.tolist()
        out: List[RowStep] = []
        keep: List[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            toks: List[Emitted] = []
            finish: Optional[str] = None
            for tok, from_draft, row in (
                (t1_list[i], False, self.row_t1[i]),
                (t2_list[i], ok_flags[i], lg2[i, 0]),
            ):
                em.append(tok)
                if tok in p.eos:
                    finish = "stop"
                elif len(em) >= p.max_tokens:
                    finish = "length"
                toks.append(Emitted(token=tok, from_draft=from_draft,
                                    finish=finish, logits=row))
                if finish is not None:
                    break
            out.append(RowStep(uid=self.uids[i], tokens=toks,
                               accepted=self.accepted[i], steps=self.steps[i]))
            if finish is None:
                keep.append(i)

        # --- next-step state for the survivors ---------------------------
        row_t1 = lg2[:, 1]
        h_pair = self.get_h() if self.any_drafting else None
        t_next_rows = [
            _pick(row_t1[i:i + 1], self.params[i], self.emitted[i]) for i in keep
        ]
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            _mark(out, fin)
            return out
        idx = mx.array(keep)
        row_t1 = row_t1[idx]
        t2k = t2[idx]
        t_next = mx.concatenate(t_next_rows).astype(mx.int32)

        if self.any_drafting:
            assert h_pair is not None
            # Advance the head over the two committed positions of every
            # surviving row: (h_i, x_{i+1}) and (h_{i+1}, x_{i+2}). The
            # second output drafts x_{i+3}, next step's speculative token.
            pair_ids = mx.stack([t2k, t_next], axis=1)
            self.draft_row = self.head.draft_logits(h_pair[idx], pair_ids, self.hcache)[:, -1]
            mx.eval(t_next, self.draft_row, fin)
        else:
            self.draft_row = None if self.head is None else self.draft_row
            mx.eval(t_next, fin)
        _mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out

    def _plain_step(self) -> List[RowStep]:
        """One stock decode step: every row commits t1 and samples the next.

        Taken when the batch is too wide for drafting to pay (draft_max_rows).
        The head still advances one position per row so its cache stays one
        row per committed token, and the next draft is ready the moment the
        batch shrinks back under the ceiling.
        """
        B = len(self.uids)
        assert self.t1 is not None and self.row_t1 is not None
        lg = self.model(self.t1[:, None], cache=self.cache,
                        **self._pos_kw(1))                      # [B, 1, V]
        fin = _finite_rows(self.row_t1)[:, None]     # lazy NaN guard
        t1_list = self.t1.tolist()
        # The head already drafted the token after t1 (draft_row); the trunk
        # is about to choose it too, so score the head at no cost and keep
        # the acceptance estimate live while not drafting.
        standing = self.draft_row if self.any_drafting else None
        out: List[RowStep] = []
        keep: List[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            tok = t1_list[i]
            em.append(tok)
            finish: Optional[str] = None
            if tok in p.eos:
                finish = "stop"
            elif len(em) >= p.max_tokens:
                finish = "length"
            out.append(RowStep(
                uid=self.uids[i],
                tokens=[Emitted(token=tok, from_draft=False, finish=finish,
                                logits=self.row_t1[i])],
                accepted=self.accepted[i], steps=self.steps[i]))
            if finish is None:
                keep.append(i)

        row_t1 = lg[:, 0]
        h = self.get_h() if self.any_drafting else None
        t_next_rows = [
            _pick(row_t1[i:i + 1], self.params[i], self.emitted[i]) for i in keep
        ]
        if standing is not None and keep:
            hits = mx.argmax(standing, axis=-1)[mx.array(keep)] == mx.concatenate(t_next_rows)
            frac = float(mx.mean(hits.astype(mx.float32)).item())
            self.acc_est = 0.9 * self.acc_est + 0.1 * frac
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            _mark(out, fin)
            return out
        idx = mx.array(keep)
        row_t1 = row_t1[idx]
        t_next = mx.concatenate(t_next_rows).astype(mx.int32)
        if self.any_drafting:
            assert h is not None
            # (h_i, x_{i+1}) for the one position that committed; its output
            # drafts x_{i+2}, which is next step's speculative token.
            self.draft_row = self.head.draft_logits(
                h[idx], t_next[:, None], self.hcache)[:, -1]
            mx.eval(t_next, self.draft_row, fin)
        else:
            mx.eval(t_next, fin)
        _mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out
