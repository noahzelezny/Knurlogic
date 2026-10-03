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
(the batch's acceptance is its worst row's). When m < K the trunk is
restored and replays [t1, d_1..d_m] (caches.py), as MTPBatch replays.

Verdicts are MTPBatch's per position: greedy `draft == argmax`, a seeded
row the target's own draw under the draft's key (sampling.Keys; positions
as a plain step numbers them), otherwise exact rejection sampling against
the distribution each draft was drawn from. A row's logits processors see
the history up to each position, on the draft rows and the verify rows
both. The head's cache only ever takes committed positions (the trunk's
captured main hidden states of the forward that committed them), so it is
never rolled back. The confidence head's score is not used: the
reference's generate.py does not use it either.

Across a split (pipeline or tensor) every rank runs this loop; only rank
0 holds the head, drafts and judges. Coord.bk carries its regime and the
[B, K] drafts (B1), Coord.bm the committed count and every row's next
token (B2); a follower runs the verify and the replay with them. The
target layers' outputs reach rank 0's head on a pipeline through
pipeline.carry. Design: docs/design/deepseek-vision.md (DSpark).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from .batch_loop import Emitted, MTPBatch, RowStep, _apply, _finite_rows, \
    _key, _mark, _pick
from .caches import restore, snapshot
from .sampling import rejection_correct


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

    def _live(self, B: int) -> list[bool]:
        return [bool(self.drafts[i]) for i in range(B)]

    def step(self) -> list[RowStep]:
        B = len(self.uids)
        if B == 0:
            return []
        live = self._live(B)
        drafting = self.drafting_pays(B) and any(live)
        pre = None
        c = self.coord
        if c is not None and (self.head is not None or c.head):
            # B1, every step while rank 0 holds a head: its regime and
            # drafts (a follower holds none and asks)
            if c.leader and drafting:
                pre = self._draft_block(B, live)
            drafting, d = c.bk(drafting, pre[0] if pre else None, B,
                               self.block_size)
            if drafting and pre is None:
                pre = (d, None)
        self._note_regime(drafting, B)
        t0 = self._clock()
        out = self._block_step(B, pre) if drafting else self._plain_step()
        self._record_cost(B, drafting, self._clock() - t0,
                          sum(len(rs.tokens) for rs in out))
        return out

    # ------------------------------------------------------------- draft
    def _draft_block(self, B: int, live: list[bool]):
        """(d [B, K] int32, the draft distributions per row and position)."""
        head = self.head
        K = self.block_size
        x, base = head.block(self.t1, self.hcache)
        prev = self.t1.astype(mx.int32)
        qs: list[list] = [[None] * K for _ in range(B)]
        plain = all(p.dist is None and not p.processors for p in self.params)
        cols = []
        for k in range(K):
            bias, _ = head.markov(prev)
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
        return mx.stack(cols, axis=1), qs

    # -------------------------------------------------------------- step
    def _block_step(self, B: int, pre=None) -> list[RowStep]:
        """`pre`: (d, qs) already drawn (a split's B1; qs None on a
        follower, which never judges)."""
        K = self.block_size
        live = self._live(B)
        d, qs = pre if pre is not None else self._draft_block(B, live)
        # rank 0 judges; a follower's verdicts come in B2
        judge = self.coord is None or self.coord.leader

        # --- verify: one (K + 1)-wide forward ----------------------------
        csnap = snapshot(self.cache, copy=self.copy_caches)
        inp = mx.concatenate([self.t1[:, None].astype(mx.int32), d], axis=1)
        pos_v = self._pos_kw(K + 1)
        lg = self.model(inp, cache=self.cache, **pos_v)       # [B, K+1, V]
        h_v = self.get_h()

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
        if self.coord is not None:
            # B2: rank 0's count and next tokens drive every rank's replay
            m, nxt = self.coord.bm(m, nxt, B)

        # --- rollback + replay to the m + 1 committed positions ----------
        if m < K:
            restore(self.cache, csnap)
            self.model(inp[:, :m + 1], cache=self.cache, **self._pos_kw(m + 1))
            h_c = self.get_h()
        else:
            h_c = h_v

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
