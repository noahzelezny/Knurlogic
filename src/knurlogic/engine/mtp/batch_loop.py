"""MTP speculative decoding over a BATCH of sequences.

Draft one token with the head, verify it inside a 2-token trunk forward,
for B rows at once. It is batchable because with a 1-token draft EVERY row
advances by exactly two positions per step, accepted or not, so the
batched caches (one offset per row, shared write index) never need per-row
trimming: rejection is ONE whole-batch rollback to t1 (caches.rollback)
and ONE 1-wide forward of the committed t2 -- or, for caches that cannot
roll forward (recurrent state), a restore and a 2-wide replay of [t1, t2].
Expected forwards per step are
1 + (1 - a^B) for 2B tokens -- a decaying win; measure before assuming it
beats plain batching at a given B.

Sampling is per row (exact rejection sampling, sampling.rejection_correct;
`draft == argmax` at temperature 0); a row's logits processors apply to the
TRUNK row that verifies the draft. Across a pipeline split every rank runs
this loop; only rank 0 holds the head and `coord` broadcasts its drafts and
verdicts. Design: docs/design/drafting.md (batched loop).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.generate import _extend_cache, _merge_caches

from knurlogic.engine.split.marker import chunk_done

from .caches import position, release, restore, rollback, snapshot
from .sampling import Distribution, Keys, rejection_correct
from .seed import seed_head

__all__ = ["RowParams", "Row", "Emitted", "RowStep", "MTPBatch", "admit",
           "default_draft_max_rows"]

import time

#: A module logger; nothing downstream configures it.
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
    dist: Callable[[mx.array], Distribution] | None
    processors: list[Callable[[mx.array, mx.array], mx.array]]
    eos: set
    #: False for a row the head must not draft for; it still decodes correctly.
    drafts: bool = True
    #: the request's own random stream (a seed), or None for the global one
    keys: Keys | None = None
    #: a pipeline follower's drafting row: rank 0 drafts it with the head
    #: this rank does not hold. The row is prefilled and batched as a
    #: drafting row (same checkpoints, same verify steps); nothing here
    #: seeds or runs a head for it.
    mirror: bool = False


@dataclass
class Row:
    """One admitted sequence: prefilled, first token sampled, head seeded."""

    uid: int
    params: RowParams
    cache: list                    # single-row trunk caches, full prompt inside
    hcache: Any                    # head cache (an EMPTY one when not drafting)
    t1: mx.array                   # [1] int32, the first token to commit
    row_t1: mx.array               # [1, V] the logits t1 was sampled from
    draft_row: mx.array | None  # [1, V] the head's draft of t2, or None
    n_prompt: int
    drafts: bool
    #: this row's MRoPE offset (Qwen: positions after an image
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
    finish: str | None
    logits: mx.array               # [V] the trunk row that produced it
    #: every value of `logits` finite -- computed inside the step's own
    #: final eval, so the NaN guard adds no sync of its own
    finite: bool = True


@dataclass
class RowStep:
    uid: int
    tokens: list[Emitted]
    #: running per-row acceptance: (accepted drafts, speculative steps)
    accepted: int
    steps: int


def apply(row: mx.array, procs, emitted) -> mx.array:
    """Logits processors over the history; `emitted` is a list of ids or
    an id array (lazy: the draft step's history ends in t1, which it has
    not read back yet)."""
    if not procs:
        return row
    hist = emitted if isinstance(emitted, mx.array) else mx.array(emitted)
    for proc in procs:
        row = proc(hist, row)
    return row


def _with(emitted: list[int], t1: mx.array) -> mx.array:
    """The history for the position after t1: what was emitted, then t1."""
    return mx.concatenate([mx.array(emitted, dtype=mx.int32),
                           t1.astype(mx.int32)])


def finite_rows(rows: mx.array) -> mx.array:
    """[B, V] -> [B] bool, lazy: evaluated with the step's last sync."""
    return mx.isfinite(rows).all(axis=-1)


def mark(out: list[RowStep], fin: mx.array) -> None:
    """fin: [B, k] already evaluated; token j of row i gets fin[i][j]."""
    for rs, flags in zip(out, fin.tolist()):
        for em, good in zip(rs.tokens, flags):
            em.finite = bool(good)


def pick(row: mx.array, p: RowParams, emitted: list[int]) -> mx.array:
    """row: [1, V] -> token [1]."""
    row = apply(row, p.processors, emitted)
    if p.dist is None:
        return mx.argmax(row, axis=-1)
    return p.dist(row).sample(row_key(p, len(emitted)))


def row_key(p: RowParams, position: int):
    """The seeded row's key for the token at `position`, or None."""
    return p.keys.at(position) if p.keys is not None else None


class ForwardFailed(RuntimeError):
    """A prefill forward raised part-way: on a cluster rank the other ranks
    are inside the same forward, waiting on this one's half."""


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
    on_progress: Callable[[int, int], None] | None = None,
    cache: list | None = None,
    hcache: Any = None,
    start_pos: int = 0,
    on_chunk: Callable[[list, Any], None] | None = None,
    embeds: mx.array | None = None,
    extras: dict | None = None,
    chunk_boundaries: list[tuple] | None = None,
    rope_delta: int = 0,
    mrope: bool = False,
    checkpoints: Iterable[int] = (),
    on_checkpoint: Callable[[int, list, Any, mx.array | None], None] | None = None,
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

    IMAGES. A vision row arrives with
    `embeds` -- the trunk's `input_embeddings` for ids[start_pos:] only, as
    `Family.embed` builds them, [1, n - start_pos, D] -- and optional
    `extras`, further trunk kwargs over the same span. An extras value is an
    array whose sequence axis is 1, or an `(array, axis)` pair naming it
    (position ids are [3, 1, L]: axis -1). Every chunk forward gets the
    matching slice of each, the ids still ride along (the trunk ignores
    them when embeddings are given, and a drafting head needs them).
    `chunk_boundaries` are [start, end) spans no chunk edge may fall
    strictly inside -- gemma attends bidirectionally within an image, so a
    chunk that cut one would compute the first half without the second.
    Edges snap back to the span start, or forward past the
    span when it starts the chunk (a span longer than the step is one
    chunk). The last forward, which yields the first logits, is widened the
    same way if the prompt ends inside a span. `rope_delta` / `mrope` ride
    on the Row so MTPBatch can hand the trunk this row's positions on every
    decode step.

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

    A BLOCK head (`head.block_size`, block_loop.BlockBatch) is advanced
    over every prompt position, the last one included, chunk by chunk as
    the trunk prefills: its cache holds the trunk's own positions (no
    one-step lag), so a checkpoint's head cache is at c and `h` is None.
    The row enters decoding with no draft; the first step drafts.
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
    block = drafts and getattr(head, "block_size", 0) > 0
    # a drafting row on this rank's side of the batch: with a head, or
    # mirrored for rank 0's (a pipeline follower). Its checkpoints move
    # with it, so they must be the same on every rank.
    row_drafts = drafts or (params.drafts and params.mirror)
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

    h_chunks: list[mx.array] = []
    seeded = start_pos          # the head is seeded over [start_pos, seeded)
    cps = sorted(c for c in set(checkpoints or ())
                 if start_pos < c <= last and _inside(c, spans) is None
                 and (not row_drafts or c - start_pos >= 2))
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
            try:
                model(chunk, cache=cache, **_kw(i, end))
                # Evaluate the cache and the captured hidden state, never
                # the logits: forcing them would materialise the lm_head
                # projection for every prompt position (loop.py says the
                # same).
                want = [c.state for c in cache if hasattr(c, "state")]
                if block:
                    head.advance(get_h(), dcache)
                    want += dcache.state
                elif drafts:
                    h = get_h()
                    h_chunks.append(h)
                    want.append(h)
                elif row_drafts and (h := get_h()) is not None:
                    # a tensor follower of a drafting rank 0: the final
                    # hidden state holds the last layer's all_sum, which
                    # rank 0 runs here for its head (mirror_hidden)
                    want.append(h)
                mx.eval(want)
            except Exception as e:
                raise ForwardFailed(str(e)) from e
            i = end
            mx.clear_cache()
            chunk_done()
            if on_chunk is not None:
                on_chunk(cache, dcache)
            if cps and end == cps[0]:
                c = cps.pop(0)
                h_c = None
                if drafts and not block:
                    # Seed [seeded, c-1), keep h from c-1 on for the rest.
                    h_all = (mx.concatenate(h_chunks, axis=1)
                             if len(h_chunks) > 1 else h_chunks[0])
                    seed_head(head, [h_all], ids, c, dcache,
                              prefill_step_size, start=seeded)
                    h_chunks = [h_all[:, c - 1 - seeded:]]
                    h_c = h_all[:, c - 1 - seeded:c - seeded]
                    seeded = c - 1
                    mx.eval(h_chunks[0], h_c)
                assert on_checkpoint is not None    # cps is empty without one
                on_checkpoint(c, cache, dcache if drafts else None, h_c)
            if on_progress is not None:
                on_progress(end, n)

    logits = model(ids[:, max(last, 0):], cache=cache, **_kw(max(last, 0), n))
    row_t1 = logits[:, -1]
    t1 = pick(row_t1, params, []).astype(mx.int32)

    draft_row = None
    if block:
        head.advance(get_h(), dcache)
        mx.eval(t1, dcache.state)
    elif drafts:
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
        row_t1=row_t1, draft_row=draft_row, n_prompt=n, drafts=row_drafts,
        rope_delta=int(rope_delta), mrope=bool(mrope),
    )


def _inside(p: int, spans) -> tuple | None:
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

    #: the head's next draft rides between steps (`draft_row`); a block
    #: head drafts at the start of its step instead (block_loop.BlockBatch)
    carries_draft = True

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
        # width -- on an M4 Max (128 GB) with Qwen3.8-Flash-Next-VQ-4.4bpw the ratio is
        # ~1.5 at one row, so only a timing can decide.
        self.draft_max_rows = (
            draft_max_rows if draft_max_rows is not None else default_draft_max_rows()
        )
        self._clock = clock
        self.acc_est = 0.8                   # EMA of the head's hit rate (logging)
        # (rows, drafting) -> (EMA seconds per token, steps measured)
        self._cost: dict = {}
        self._regime: bool | None = None  # last step drafted? (for the log)
        self._since_recheck = 0
        # rows -> (backoff multiplier, winner when the last recheck began)
        self._backoff: dict = {}
        self._explore: tuple | None = None   # (rows, drafting, steps left)
        #: engine/split/pipeline.Coord on a pipeline split: rank 0's regime,
        #: drafts and verdicts reach every rank through it (B1, B2)
        self.coord = None
        #: finish_at(uid, tokens) -> the index of the token in `tokens` (a
        #: row's next ones, in order) that ends the row, or None: the batch
        #: engine's stop rules, which this loop's own params leave out. A
        #: step commits no position past it (`_ends`)
        self.finish_at: Callable[[int, list[int]], int | None] | None = None

        self.uids: list[int] = []
        self.params: list[RowParams] = []
        self.emitted: list[list[int]] = []
        self.drafts: list[bool] = []
        self.accepted: list[int] = []
        self.steps: list[int] = []
        self.n_prompt: list[int] = []
        self.rope_delta: list[int] = []
        self.mrope: list[bool] = []

        self.cache: list = []               # batched trunk caches
        self.hcache: Any = None             # batched head cache
        self.t1: Any = None          # [B] int32 (mx.array)
        self.row_t1: Any = None      # [B, V]
        self.draft_row: Any = None   # [B, V]

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
        # A seeded row drafts every step: the timing-chosen regime differs
        # run to run, and a 2-wide verify and a plain step round differently,
        # so a near-tie would flip and the seed would not reproduce.
        if any(p.keys is not None and p.keys.pins
               and d for p, d in zip(self.params, self.drafts)):
            return True
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

    def cost_per_token(self, rows: int) -> float | None:
        """The measured seconds per committed token at this width, in the
        cheaper regime; None until a regime has been timed there."""
        got = [c[0] for c in (self._cost.get((rows, True)),
                              self._cost.get((rows, False))) if c]
        return min(got) if got else None

    def _record_cost(self, rows: int, drafting: bool, seconds: float,
                     tokens: int) -> None:
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

            def fmt(c):
                return f"{c[0] * 1000:.1f}ms/tok" if c else "unmeasured"

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
        draft_row = (mx.concatenate(drafts, axis=0)
                     if self.head is not None and self.carries_draft else None)

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

    def filter(self, keep: list[int]) -> None:
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
        # under the limit, read no drop and stopped them all.
        mx.eval(self._arrays())

    def _arrays(self) -> list:
        from mlx.utils import tree_flatten
        out = [a for a in (self.t1, self.row_t1, self.draft_row)
               if a is not None]
        for c in list(self.cache) + ([self.hcache] if self.hcache else []):
            try:
                out += [a for _, a in tree_flatten(c.state)
                        if isinstance(a, mx.array)]
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass            # a cache with no state to read yet
        return out

    def _ends(self, i: int, toks: list[int]) -> int | None:
        """The index in `toks` (row i's tokens this step, in order) of the
        one that ends the row, or None. A step never commits past it: the
        row's entry is stored at its last token, and a trunk cache that
        took a position more could not be trimmed back to it."""
        p = self.params[i]
        n = len(self.emitted[i])
        own = next((j for j, t in enumerate(toks)
                    if t in p.eos or n + j + 1 >= p.max_tokens), None)
        hook = (self.finish_at(self.uids[i], toks)
                if self.finish_at is not None else None)
        got = [j for j in (own, hook) if j is not None]
        return min(got) if got else None

    def remove(self, uids: Iterable[int]) -> None:
        drop = set(uids)
        self.filter([i for i, u in enumerate(self.uids) if u not in drop])

    def _pos_kw(self, width: int, start: int = 0) -> dict:
        """`position_ids` for a decode forward `width` tokens wide, or {}.

        A Qwen row whose key holds an image decodes at position
        (tokens so far + rope_delta) on all three MRoPE axes -- for EVERY
        step, not only while the image is in the new span; missing it
        degrades silently. So as soon as one row needs it the
        whole batch gets explicit ids, [3, B, width]; a row without MRoPE
        gets its plain count, which is what the trunk would have used. The
        count is n_prompt + tokens emitted: the row's cache length before
        this step, tracked here rather than read off a left-padded batch
        cache. No MRoPE row -> {} and the trunk call is exactly as before.
        """
        if not any(self.mrope):
            return {}
        base = [self.n_prompt[i] + len(self.emitted[i]) + self.rope_delta[i]
                + start for i in range(len(self.uids))]
        p = mx.array(base)[:, None] + mx.arange(width)[None, :]
        return {"position_ids": mx.broadcast_to(p[None], (3, *p.shape))}

    # ------------------------------------------------------------------ step

    def step(self) -> list[RowStep]:
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
        # timed from here: on a split, rank 0's drafts and the B1 broadcast
        # are this step's cost too (timed after them, drafting read cheaper
        # than it was)
        t0 = self._clock()
        pre = None
        if self.coord is not None and (self.head is not None
                                       or self.coord.head):
            # B1, every step while rank 0 holds a head: rank 0's regime and
            # drafts (a follower holds none and asks)
            if self.coord.leader and drafting:
                pre = self._drafts(B)
            drafting, d2 = self.coord.b1(drafting, pre[1] if pre else None, B)
            if drafting and pre is None:
                pre = (self._live(B), d2, [None] * B)
        self._note_regime(drafting, B)
        out = self._plain_step() if not drafting else self._draft_step(B, pre)
        self._record_cost(B, drafting, self._clock() - t0,
                          sum(len(rs.tokens) for rs in out))
        return out

    def _live(self, B: int) -> list[bool]:
        # A row drafts this step iff it is a drafting row AND the head has
        # produced a draft for it (the batch may hold only non-drafting rows).
        # A follower has no head and no drafts of its own: its drafting rows
        # are live on rank 0's word (B1 said drafting).
        return [self.drafts[i] and (self.draft_row is not None
                                    or self.head is None)
                for i in range(B)]

    def _drafts(self, B: int):
        """(live, d2 [B] int32, the draft distributions) for this step."""
        live = self._live(B)

        # --- draft d2 per row -------------------------------------------
        d2_rows: list[mx.array] = []
        qs: list[Distribution | None] = []
        for i in range(B):
            p = self.params[i]
            if not live[i]:
                # Placeholder that is always "rejected" below; the replay
                # supplies the trunk's own token.
                d2_rows.append(self.t1[i:i + 1])
                qs.append(None)
                continue
            # position len+1: its history includes t1, as in a plain step
            row = apply(self.draft_row[i:i + 1], p.processors,
                         _with(self.emitted[i], self.t1[i:i + 1])
                         if p.processors else self.emitted[i])
            if p.dist is None:
                d2_rows.append(mx.argmax(row, axis=-1))
                qs.append(None)
            else:
                q = p.dist(row)
                # the draft for position len+1 (t1 is at len)
                d2_rows.append(q.sample(row_key(p, len(self.emitted[i]) + 1)))
                qs.append(q)
        d2 = mx.concatenate(d2_rows).astype(mx.int32)
        return live, d2, qs

    def _draft_step(self, B: int, pre=None) -> list[RowStep]:
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
        oks: list[Any] = [False] * B
        t2_rows: list[Any] = [self.t1[i:i + 1] for i in range(B)]
        lazy: list[mx.array] = []
        # a pipeline follower's logits are zeros: its verdicts come in B2
        judge = self.coord is None or self.coord.leader
        for i in (range(B) if judge else ()):
            p = self.params[i]
            row = apply(lg2[i:i + 1, 0], p.processors,
                         _with(self.emitted[i], self.t1[i:i + 1])
                         if p.processors else self.emitted[i])
            if not live[i]:
                # Non-drafting row: the trunk's own token, committed through
                # the replay below.
                t2 = (mx.argmax(row, axis=-1) if p.dist is None
                      else p.dist(row).sample(
                          row_key(p, len(self.emitted[i]) + 1)))
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
                t2 = p.dist(row).sample(row_key(p, len(self.emitted[i]) + 1))
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
        elif not judge:
            # a tensor follower's verify holds every layer's all_sum, which
            # rank 0 has just run: they go before B2's all_gather, or the
            # ranks meet in different collectives (a pipeline follower's
            # zeros cost nothing)
            mx.eval(lg2)
        ok_flags = [bool(o.item()) if isinstance(o, mx.array) else bool(o) for o in oks]
        t2 = mx.concatenate(t2_rows).astype(mx.int32)
        # a row that t1 ends: no row commits t2 this step (`_ends`)
        cut = judge and any(self._ends(i, [t]) == 0
                            for i, t in enumerate(self.t1.tolist()))
        if self.coord is not None:
            # B2: rank 0's verdicts drive every rank's rollback and replay
            ok_flags, t2, cut = self.coord.b2(ok_flags, t2, B, cut)

        n_live = sum(live)
        if n_live:
            frac = sum(int(ok_flags[i]) for i in range(B) if live[i]) / n_live
            self.acc_est = 0.9 * self.acc_est + 0.1 * frac
        for i in range(B):
            if live[i]:
                self.steps[i] += 1
                self.accepted[i] += int(ok_flags[i])

        if cut:
            # t1 alone commits: back to before the verify, and a plain
            # step whose next tokens are the t2s (each the target's token
            # at that position: accepted, corrected or its own)
            restore(self.cache, csnap)
            return self._plain_step(then=t2)

        # --- rollback if anyone rejected: t1 stays, t2 is fed ------------
        h_pair = None
        if all(ok_flags):
            release(self.cache, csnap)
        else:
            h_v = self.get_h() if self.any_drafting else None
            if rollback(self.cache, csnap, 1):
                # the verify's first position is t1's: only t2 is new
                lg1 = self.model(t2[:, None], cache=self.cache,
                                 **self._pos_kw(1, start=1))
                lg2 = mx.concatenate([lg2[:, :1], lg1], axis=1)
                if h_v is not None:
                    h_pair = mx.concatenate([h_v[:, :1], self.get_h()],
                                            axis=1)
            else:
                lg2 = self.model(mx.stack([self.t1, t2], axis=1),
                                 cache=self.cache, **pos2)

        # NaN guard, lazy: joins the eval at the end of the step.
        fin = mx.stack([finite_rows(self.row_t1), finite_rows(lg2[:, 0])],
                       axis=1)

        # --- emit --------------------------------------------------------
        t1_list = self.t1.tolist()
        t2_list = t2.tolist()
        out: list[RowStep] = []
        keep: list[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            toks: list[Emitted] = []
            finish: str | None = None
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
        if h_pair is None and self.any_drafting:
            h_pair = self.get_h()
        t_next_rows = [
            pick(row_t1[i:i + 1], self.params[i], self.emitted[i]) for i in keep
        ]
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            mark(out, fin)
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
            self.draft_row = self.head.draft_logits(
                h_pair[idx], pair_ids, self.hcache)[:, -1]
            mx.eval(t_next, self.draft_row, fin)
        else:
            self.draft_row = None if self.head is None else self.draft_row
            mx.eval(t_next, fin)
        mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out

    def _plain_step(self, then: mx.array | None = None) -> list[RowStep]:
        """One stock decode step: every row commits t1 and samples the next.

        Taken when the batch is too wide for drafting to pay (draft_max_rows).
        The head still advances one position per row so its cache stays one
        row per committed token, and the next draft is ready the moment the
        batch shrinks back under the ceiling. `then` ([B]): the next tokens,
        already drawn (a drafting step cut to t1, `_draft_step`).
        """
        B = len(self.uids)
        assert self.t1 is not None and self.row_t1 is not None
        lg = self.model(self.t1[:, None], cache=self.cache,
                        **self._pos_kw(1))                      # [B, 1, V]
        fin = finite_rows(self.row_t1)[:, None]     # lazy NaN guard
        t1_list = self.t1.tolist()
        # The head already drafted the token after t1 (draft_row); the trunk
        # is about to choose it too, so score the head at no cost and keep
        # the acceptance estimate live while not drafting.
        standing = (self.draft_row if self.any_drafting and then is None
                    else None)
        out: list[RowStep] = []
        keep: list[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            tok = t1_list[i]
            em.append(tok)
            finish: str | None = None
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
        if then is not None:
            # the next tokens are not this forward's: nothing below would
            # evaluate it on a tensor follower (no head, no draft), and its
            # layers' all_sums would meet rank 0's next exchange (Desync).
            # Every rank runs it here, before anything else crosses.
            mx.eval(lg)
        h = self.get_h() if self.any_drafting else None
        t_next_rows = [
            then[i:i + 1] if then is not None
            else pick(row_t1[i:i + 1], self.params[i], self.emitted[i])
            for i in keep
        ]
        if standing is not None and keep:
            hits = (mx.argmax(standing, axis=-1)[mx.array(keep)]
                    == mx.concatenate(t_next_rows))
            frac = float(mx.mean(hits.astype(mx.float32)).item())
            self.acc_est = 0.9 * self.acc_est + 0.1 * frac
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            mark(out, fin)
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
        mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out
