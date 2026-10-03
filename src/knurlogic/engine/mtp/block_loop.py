"""Block drafting over a batch: a head that drafts K tokens in one pass
(deepseek_v4's DSpark, families/deepseek/heads/deepseek_v4_dspark.py),
verified in one (K + 1)-wide trunk forward.

batch_loop.MTPBatch is built on a 1-token draft, where every row advances
exactly two positions per step. With K drafts a row accepts anywhere from
0 to K, and the batched trunk caches have one shared write index, so this
loop commits the SAME count for every row: m = the fewest any row
accepted. Each row emits t1 and its first m drafts, and its next t1 is the
token its verdict put at position m (its accepted draft there, its
correction, or with m = K the bonus from the last verify row) -- every one
a token of the target's distribution, so nothing is resampled. One row
loses nothing; at B rows the rows that accepted more give back the excess
(the batch's acceptance is its worst row's). When m < k the trunk rolls
back to the m + 1 committed positions (caches.rollback: deepseek_v4's
cache by its own bookkeeping, architecture edit 19), and the head takes
the verify's hidden states of those positions; a cache that cannot roll
forward is restored and [t1, d_1..d_m] replayed, as MTPBatch replays.

The head drafts all K in its one pass, but a step verifies only the first
k (1..K): a wider MoE forward reads more experts, and drafts the head
rarely gets right cost more to verify than they return. k maximizes
(1 + sum of the chances that the batch accepts j drafts, j <= k) over the
measured seconds of a step at that width (_width); every width is timed,
none assumed. The chances are this step's own, from the head's confidence
scores (sigmoid of each position's score, read as the chance that draft
stands given the ones before it; the batch accepts j only if every row
does, so the rows' products multiply) while those stay calibrated against
the verdicts; otherwise the measured per-position acceptance. The emitted
tokens are the same for every k.

Verdicts are MTPBatch's per position: greedy `draft == argmax`, a seeded
row the target's own draw under the draft's key (sampling.Keys; positions
as a plain step numbers them), otherwise exact rejection sampling against
the distribution each draft was drawn from. A row's logits processors see
the history up to each position, on the draft rows and the verify rows
both. The head's cache only ever takes committed positions (the trunk's
captured main hidden states of the forward that committed them), so it is
never rolled back. The confidence head's score only chooses the verify
width (the reference's generate.py does not use it).

Across a split (pipeline or tensor) every rank runs this loop; only rank
0 holds the head, drafts and judges. Coord.bk carries its regime and the
[B, K] drafts (B1), Coord.bm the committed count and every row's next
token (B2); a follower runs the verify and the replay with them. The
target layers' outputs reach rank 0's head on a pipeline through
pipeline.carry. Design: docs/design/deepseek-vision.md (DSpark).
"""
from __future__ import annotations

from typing import Any

import logging
import os

import mlx.core as mx

from .batch_loop import EXPLORE_STEPS, RECHECK_EVERY, Emitted, MTPBatch, \
    RowStep, _apply, _finite_rows, _key, _mark, _pick
from .caches import release, rollback, snapshot
from .sampling import rejection_correct


logger = logging.getLogger(__name__)

#: KNURLOGIC_MTP_PROFILE=1: time each phase of a step (BlockBatch._prof)
PROFILE = os.environ.get("KNURLOGIC_MTP_PROFILE", "").strip().lower() in (
    "1", "on", "true", "yes")
PROFILE_EVERY = 32
#: the confidence scores' calibration check (BlockBatch.calibrated):
#: verdicts at a position before it counts, the largest |mean predicted -
#: hit rate|, and the window it slides over
CAL_MIN = 32
CAL_TOL = 0.1
CAL_WINDOW = 256


def _fixed_width() -> int | None:
    """KNURLOGIC_MTP_VERIFY=k: verify the first k drafts every step
    (clamped to 1..K); unset, k is chosen by measurement (_width)."""
    v = os.environ.get("KNURLOGIC_MTP_VERIFY", "").strip()
    try:
        return int(v) if v else None
    except ValueError:
        return None


def _hist(emitted: list[int], tail: list) -> Any:
    """The history before a position: what was emitted, then `tail` ([1]
    id arrays, lazy)."""
    if not tail:
        return emitted
    return mx.concatenate([mx.array(emitted, dtype=mx.int32)]
                          + [t.astype(mx.int32) for t in tail])


class BlockBatch(MTPBatch):
    """MTPBatch for a block head: rows enter and leave the same way; a
    drafting step commits 1 + m tokens per row (module docstring)."""

    carries_draft = False

    def __init__(self, *a, block_size: int = 0, **kw):
        super().__init__(*a, **kw)
        #: K: the head's, or on a split's follower (which holds none)
        #: rank 0's (tensor.agree_head)
        self.block_size = int(block_size
                              or getattr(self.head, "block_size", 0))
        #: (rows, k) -> (EMA seconds per drafting step verifying k, steps)
        self._vcost: dict = {}
        #: rows -> [EMA chance the batch accepts >= j drafts, j = 1..K]
        #: (None until a step verified j)
        self._vacc: dict = {}
        self._wexplore: tuple | None = None   # (rows, k, steps left)
        self._wsince = 0
        #: per position j: [sum of predicted P(batch accepts >= j), sum of
        #: hits, steps] -- the confidence scores' calibration check
        self._cal = [[0.0, 0.0, 0] for _ in range(self.block_size)]
        self._pred: list | None = None

    def _live(self, B: int) -> list[bool]:
        return [bool(self.drafts[i]) for i in range(B)]

    def step(self) -> list[RowStep]:
        B = len(self.uids)
        if B == 0:
            return []
        live = self._live(B)
        drafting = self.drafting_pays(B) and any(live)
        # timed from here: on a split, rank 0's block and the B1 broadcast
        # are this step's cost too (MTPBatch.step)
        t0 = self._clock()
        self._prof_start()
        pre = None
        c = self.coord
        if drafting and (c is None or c.leader):
            # the head drafts its whole block; the step verifies k of it
            d, qs, conf = self._draft_block(B, live)
            self._prof("draft", d)
            self._pred = self._predicted(conf, live)
            k = self._width(B, self._pred)
            pre = (d[:, :k], [q[:k] for q in qs])
        if c is not None and (self.head is not None or c.head):
            # B1, every step while rank 0 holds a head: its regime, k and
            # drafts (a follower holds none and asks)
            drafting, d = c.bk(drafting, pre[0] if pre else None, B,
                               self.block_size)
            self._prof("b1")
            if drafting and pre is None:
                pre = (d, None)
        self._note_regime(drafting, B)
        out = self._block_step(B, pre) if drafting else self._plain_step()
        n = sum(len(rs.tokens) for rs in out)
        sec = self._clock() - t0
        self._record_cost(B, drafting, sec, n)
        if drafting:
            self._record_width(B, int(pre[0].shape[1]), sec)
        self._prof_end(drafting, n)
        return out

    # ------------------------------------------------------------ width
    def _predicted(self, conf, live) -> list[float] | None:
        """P(the batch accepts >= j drafts), j = 1..K, from the confidence
        scores [B, K]: per row the running product of sigmoid(score), over
        the drafting rows the product of those (None without scores)."""
        rows = [i for i, x in enumerate(live) if x]
        if conf is None or not rows:
            return None
        p = mx.cumprod(mx.sigmoid(conf.astype(mx.float32)), axis=1)
        return mx.prod(p[mx.array(rows)], axis=0).tolist()

    def calibrated(self) -> bool:
        """Do the confidence predictions match the verdicts? Position 1 and
        every position verified CAL_MIN times must have its mean prediction
        within CAL_TOL of its hit rate."""
        if self._cal[0][2] < CAL_MIN:
            return False
        return all(abs(c[0] - c[1]) / c[2] <= CAL_TOL
                   for c in self._cal if c[2] >= CAL_MIN)

    def _width(self, B: int, pred: list[float] | None = None) -> int:
        """How many of the K drafts this step verifies: the k with the most
        expected tokens per second, (1 + sum_{j<=k} P(batch accepts >= j))
        / (seconds of a step verifying k at this row count). P is `pred`
        (this step's confidence scores) while calibrated(), else the
        measured acceptance.
        Every k is timed EXPLORE_STEPS steps first (K first: its steps
        measure every position's acceptance), and K is re-measured every
        RECHECK_EVERY steps, as acceptance moves with the text."""
        K = self.block_size
        fixed = _fixed_width()
        if fixed is not None:
            return min(max(fixed, 1), K)
        # a seeded row: one width always (a verify of another width rounds
        # differently, and a near-tie would not reproduce; drafting_pays)
        if any(p.keys is not None and d
               for p, d in zip(self.params, self.drafts)):
            return K
        ex = self._wexplore
        if ex is not None and ex[0] == B and ex[2] > 0:
            return ex[1]
        self._wexplore = None
        for k in range(K, 0, -1):
            got = self._vcost.get((B, k))
            if got is None or got[1] < EXPLORE_STEPS:
                self._wexplore = (B, k, EXPLORE_STEPS - (got[1] if got else 0))
                return k
        use = pred if pred is not None and self.calibrated() else None
        best = max(range(1, K + 1), key=lambda k: self._rate(B, k, use))
        self._wsince += 1
        if best != K and self._wsince >= RECHECK_EVERY:
            self._wsince = 0
            self._wexplore = (B, K, EXPLORE_STEPS)
            return K
        return best

    def _rate(self, B: int, k: int, pred=None) -> float:
        """Expected committed tokens per second verifying k drafts."""
        acc = pred if pred is not None else (self._vacc.get(B) or [])
        tokens = 1.0 + sum(a for a in acc[:k] if a is not None)
        return tokens / max(self._vcost[(B, k)][0], 1e-9)

    def _record_width(self, B: int, k: int, seconds: float) -> None:
        ema, n = self._vcost.get((B, k), (seconds, 0))
        self._vcost[(B, k)] = (0.8 * ema + 0.2 * seconds, n + 1)
        ex = self._wexplore
        if ex is not None and ex[0] == B and ex[1] == k:
            self._wexplore = (B, k, ex[2] - 1)

    def _record_accept(self, B: int, k: int, m: int) -> None:
        """The batch accepted m of k verified drafts: P(>= j) for j <= k."""
        acc = self._vacc.setdefault(B, [None] * self.block_size)
        pred, self._pred = self._pred, None
        for j in range(k):
            hit = 1.0 if m > j else 0.0
            acc[j] = hit if acc[j] is None else 0.9 * acc[j] + 0.1 * hit
            if pred is not None:
                c = self._cal[j]
                if c[2] >= CAL_WINDOW:
                    # a sliding record: calibration can be lost and regained
                    c[0] -= c[0] / c[2]
                    c[1] -= c[1] / c[2]
                    c[2] -= 1
                c[0] += pred[j]
                c[1] += hit
                c[2] += 1

    # ---------------------------------------------------------- profile
    # KNURLOGIC_MTP_PROFILE=1: each phase of a step timed behind an
    # mx.eval barrier, means logged every PROFILE_EVERY steps per regime.
    # The barriers cost a little; off, nothing here runs.
    def _prof_start(self) -> None:
        if not PROFILE:
            return
        self._pt = self._clock()
        self._pcur: dict = {}

    def _prof(self, phase: str, *arrays) -> None:
        if not PROFILE:
            return
        if arrays:
            mx.eval(*[a for a in arrays if a is not None])
        now = self._clock()
        self._pcur[phase] = self._pcur.get(phase, 0.0) + now - self._pt
        self._pt = now

    def _prof_end(self, drafting: bool, tokens: int) -> None:
        if not PROFILE:
            return
        self._prof("emit")
        acc = self.__dict__.setdefault("_pacc", {})
        a = acc.setdefault(drafting, {"steps": 0, "tokens": 0})
        a["steps"] += 1
        a["tokens"] += tokens
        for k, v in self._pcur.items():
            a[k] = a.get(k, 0.0) + v
        if a["steps"] % PROFILE_EVERY == 0:
            st, tk = a["steps"], max(a["tokens"], 1)
            parts = "  ".join(f"{k} {a[k] / st * 1000:.1f}" for k in a
                              if k not in ("steps", "tokens"))
            total = sum(v for k, v in a.items() if k not in ("steps", "tokens"))
            logger.info(f"MTP profile {'block' if drafting else 'plain'}: "
                        f"{st} steps, {tk / st:.2f} tok/step, "
                        f"{total / tk * 1000:.1f} ms/tok; per step ms: {parts}")
            acc[drafting] = {"steps": 0, "tokens": 0}

    # ------------------------------------------------------------- draft
    def _draft_block(self, B: int, live: list[bool]):
        """(d [B, K] int32, the draft distributions per row and position,
        the confidence scores [B, K] float32 or None)."""
        head = self.head
        K = self.block_size
        x, base = head.block(self.t1, self.hcache)
        prev = self.t1.astype(mx.int32)
        qs: list[list] = [[None] * K for _ in range(B)]
        plain = all(p.dist is None and not p.processors for p in self.params)
        cols, embeds = [], []
        for k in range(K):
            bias, e = head.markov(prev)
            embeds.append(e)
            rows = base[:, k].astype(mx.float32) + bias
            if plain:
                prev = mx.argmax(rows, axis=-1).astype(mx.int32)
                cols.append(prev)
                continue
            picks = []
            for i in range(B):
                p = self.params[i]
                n = len(self.emitted[i])
                row = _apply(rows[i:i + 1], p.processors,
                             _hist(self.emitted[i],
                                   [self.t1[i:i + 1]]
                                   + [c[i:i + 1] for c in cols])
                             if p.processors else self.emitted[i])
                if not live[i] or p.dist is None:
                    picks.append(mx.argmax(row, axis=-1))
                else:
                    q = p.dist(row)
                    picks.append(q.sample(_key(p, n + 1 + k)))
                    qs[i][k] = q
            prev = mx.concatenate(picks).astype(mx.int32)
            cols.append(prev)
        conf = (head.confidence(x, mx.stack(embeds, axis=1))
                if hasattr(head, "confidence") else None)
        return mx.stack(cols, axis=1), qs, conf

    # -------------------------------------------------------------- step
    def _block_step(self, B: int, pre) -> list[RowStep]:
        """`pre`: (d [B, K'], qs), the K' drafts this step verifies (qs
        None on a follower, which never judges)."""
        live = self._live(B)
        d, qs = pre
        K = int(d.shape[1])
        # rank 0 judges; a follower's verdicts come in B2
        judge = self.coord is None or self.coord.leader

        # --- verify: one (K + 1)-wide forward ----------------------------
        csnap = snapshot(self.cache, copy=self.copy_caches)
        inp = mx.concatenate([self.t1[:, None].astype(mx.int32), d], axis=1)
        pos_v = self._pos_kw(K + 1)
        lg = self.model(inp, cache=self.cache, **pos_v)       # [B, K+1, V]
        h_v = self.get_h()
        self._prof("verify", lg)

        # --- verdicts, every position at once: oks[i][k] says whether d_k
        # stands, tt[i][k] is the target's token at that position ---------
        oks, tt, lazy = [], [], []
        for i in (range(B) if judge else ()):
            p = self.params[i]
            em = self.emitted[i]
            n = len(em)
            ok_i, tt_i = [], []
            for k in range(K + 1):
                row = _apply(lg[i:i + 1, k], p.processors,
                             _hist(em, [self.t1[i:i + 1]]
                                   + [d[i:i + 1, j] for j in range(k)])
                             if p.processors else em)
                key = _key(p, n + 1 + k)
                if k == K or not live[i]:
                    # the bonus row, or a row that does not draft: the
                    # trunk's own token
                    t = (mx.argmax(row, axis=-1) if p.dist is None
                         else p.dist(row).sample(key))
                    ok_i.append(False)
                    tt_i.append(t)
                    lazy.append(t)
                    if not live[i]:
                        break
                    continue
                if p.dist is None:
                    t = mx.argmax(row, axis=-1)
                    ok = t == d[i:i + 1, k]
                elif p.keys is not None:
                    t = p.dist(row).sample(key)
                    ok = t == d[i:i + 1, k]
                else:
                    ok, t = rejection_correct(p.dist(row).probs,
                                              qs[i][k].probs, d[i:i + 1, k])
                ok_i.append(ok)
                tt_i.append(t)
                lazy += [ok, t]
            oks.append(ok_i)
            tt.append(tt_i)
        t1_list = self.t1.tolist()
        d_all = d.tolist()
        m, nxt = 0, [0] * B
        if judge:
            mx.eval(*lazy)
            acc = []
            for i in range(B):
                a = 0
                if live[i]:
                    while a < K and bool(oks[i][a].item()):
                        a += 1
                acc.append(a)
            m = min(acc)
            self._record_accept(B, K, m)
            # no row commits past the token that ends one (MTPBatch._ends)
            for i in range(B):
                end = self._ends(i, [t1_list[i]] + d_all[i][:m])
                if end is not None:
                    m = min(m, end)
            # the token each row puts at position m (module docstring)
            nxt = [d_all[i][m] if acc[i] > m else int(tt[i][m].item())
                   for i in range(B)]

            n_live = sum(live)
            if n_live:
                frac = sum(acc[i] for i in range(B) if live[i]) / (n_live * K)
                self.acc_est = 0.9 * self.acc_est + 0.1 * frac
            for i in range(B):
                if live[i]:
                    self.steps[i] += 1
                    self.accepted[i] += acc[i]
        else:
            # a tensor follower's verify holds every layer's all_sum, which
            # rank 0 has just run for its verdicts: they go before B2's
            # all_gather (a pipeline follower's zeros cost nothing)
            mx.eval(lg)
        self._prof("verdict")
        if self.coord is not None:
            # B2: rank 0's count and next tokens drive every rank's replay
            m, nxt = self.coord.bm(m, nxt, B)
            self._prof("b2")

        # --- rollback to the m + 1 committed positions --------------------
        if m == K:
            release(self.cache, csnap)
            h_c = h_v
        elif rollback(self.cache, csnap, m + 1):
            # the head takes the committed positions' hidden states from
            # the verify: a causal forward's first m + 1 are theirs
            h_c = None if h_v is None else h_v[:, :m + 1]
            self._prof("rollback")
        else:
            # restored: replay the committed tokens
            self.model(inp[:, :m + 1], cache=self.cache, **self._pos_kw(m + 1))
            h_c = self.get_h()
            self._prof("replay", h_c)

        fin = mx.stack([_finite_rows(self.row_t1)]
                       + [_finite_rows(lg[:, k]) for k in range(m)], axis=1)

        # --- emit --------------------------------------------------------
        d_list = [r[:m] for r in d_all]
        out: list[RowStep] = []
        keep: list[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            toks: list[Emitted] = []
            finish: str | None = None
            seq = [(t1_list[i], False, self.row_t1[i])] + [
                (d_list[i][k], True, lg[i, k]) for k in range(m)]
            for tok, from_draft, row in seq:
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

        row_next = lg[:, m]
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            _mark(out, fin)
            return out
        idx = mx.array(keep)
        t_next = mx.array([nxt[i] for i in keep], dtype=mx.int32)
        if self.head is not None:
            self.head.advance(h_c[idx], self.hcache)
            mx.eval(t_next, fin, self.hcache.state)
            self._prof("advance")
        else:
            # a tensor follower: the replay's all_sums, as rank 0's head
            # cache runs them (its captured hidden state, mirror_hidden)
            mx.eval(t_next, fin, *([h_c] if h_c is not None else []))
        _mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_next[idx]
        return out

    def _plain_step(self) -> list[RowStep]:
        """One stock decode step; the head's cache still takes the
        committed position, so the next drafting step can draft."""
        B = len(self.uids)
        lg = self.model(self.t1[:, None], cache=self.cache,
                        **self._pos_kw(1))
        fin = _finite_rows(self.row_t1)[:, None]
        t1_list = self.t1.tolist()
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
        h = self.get_h() if self.any_drafting else None
        t_next_rows = [_pick(row_t1[i:i + 1], self.params[i], self.emitted[i])
                       for i in keep]
        self.filter(keep)
        if not keep:
            mx.eval(fin)
            _mark(out, fin)
            return out
        idx = mx.array(keep)
        t_next = mx.concatenate(t_next_rows).astype(mx.int32)
        want = [t_next, fin]
        if h is not None:
            self.head.advance(h[idx], self.hcache)
            want += self.hcache.state
        mx.eval(*want)
        _mark(out, fin)
        self.t1 = t_next
        self.row_t1 = row_t1[idx]
        return out
