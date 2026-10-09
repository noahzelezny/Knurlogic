"""The scheduler's memory guard (MemoryGuard, a mixin Scheduler
inherits): outgrowing the GPU working set is not an exception -- Metal
aborts the process -- so before each step, past the limit, the prompt
cache gives up entries first and then the newest rows are stopped with
`OutOfMemory` (a 503). Admission estimates too, from what this model's
caches measured: a row is admitted with checkpoints, LEAN without them,
waits, or with nothing running is refused -- never admitted to abort.
The limit is working set - others - margin; the margin is the next
step's predicted transient, read off lines measured on this model.
macOS memory pressure is watched here and reported, never a refusal.

The state these methods read lives on the Scheduler (its __init__).
Design: docs/design/server.md (memory guard), docs/design/memory-pacing.md.
"""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)
GIB = 1 << 30

#: this server's own memory macOS has compressed or swapped out, past which
#: -- while macOS itself reports memory pressure -- the server warns
#: (Scheduler._pressure_short): the model's pages are being paged back in
#: by every step, and a prefill that took 98 s ran 17+ min with both GPUs
#: at 100% (2026-10-05, another process's leak). Compressed pages alone
#: are not pressure: 11.3 GiB sat compressed with 80 GiB unused and no
#: swap (left from an earlier squeeze; a page stays compressed until it is
#: touched), and the card said "slowed" about nothing.
PRESSURE_BYTES = 1 << 30
#: how often that is read (a task_info call; on the scheduler thread)
PRESSURE_EVERY_S = 5.0

#: a prefill step's transient is read off a line in context x chunk, in
#: units of this chunk (so x reads in tokens of context at chunk 512)
UNIT_CHUNK = 512
#: the smallest prefill chunk an admission is shrunk to before it waits
MIN_CHUNK = 128


class OutOfMemory(RuntimeError):
    """This row was stopped so the server and the other rows keep running.
    `memory`: the guard's terms when it refused (Scheduler._memory), for
    the 503's body, or None."""

    def __init__(self, message: str = "", memory: dict | None = None):
        super().__init__(message)
        self.memory = memory


def _terms(mem: dict) -> str:
    """Scheduler._memory on one log line, in GiB."""
    def g(v):
        return "-" if v is None else f"{v / GIB:.1f}"
    ranks = mem.get("ranks")
    pairs = [(r.get("rank"), g(r.get("over_limit_bytes")))
             for r in ranks or ()]
    per = f" per rank {pairs}" if ranks else ""
    return (f"rank {mem['rank']} working_set {g(mem['working_set'])} "
            f"others {g(mem['others'])} (readings "
            f"{[g(x) for x in mem['others_readings']]}) margin "
            f"{g(mem['margin'])} (transient {g(mem['transient'])} at chunk "
            f"{mem['chunk']}, samples {mem['transient_samples']}, context "
            f"{mem['context']}+{mem['tokens']}) local_active "
            f"{g(mem['local_active'])} active {g(mem['active'])} cached "
            f"{g(mem['cached'])} prompt_cache {g(mem['prompt_cache'])} "
            f"peers_over {g(mem['peers_over'])}{per} limit "
            f"{g(mem['limit'])} room {g(mem.get('room'))} need "
            f"{g(mem.get('need'))}")


class _Wait(Exception):
    """Not admitted yet: the request goes back to the front of the queue."""


class _RingWait(_Wait):
    """The prompt cache gave way on a ring; wait one exchange for the
    peers' memory before deciding (Scheduler._make_room)."""


def _cache_nbytes(cache) -> int:
    n = 0
    for c in cache or ():
        try:
            n += int(getattr(c, "nbytes", 0) or 0)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass    # an estimate: a layer that cannot report its size counts 0
    return n


def _fixed_nbytes(cache) -> int:
    """The bytes of a row's cache that do not grow with its length: the
    layers that cannot trim (a hybrid's linear-attention / recurrent state).
    A layer that does not say is counted as growing -- the safe side."""
    n = 0
    for c in cache or ():
        try:
            t = getattr(c, "is_trimmable", None)
            if callable(t) and not t():
                n += int(getattr(c, "nbytes", 0) or 0)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass    # an estimate: a layer that cannot report its size counts 0
    return n


class MemoryGuard:
    """Scheduler's memory methods (engine/runtime/scheduler)."""

    def _memory(self, n_tokens: int = 0, need: int | None = None,
                room: int | None = None, chunk: int | None = None) -> dict:
        """The guard's terms for a prompt of n_tokens prefilled `chunk`
        tokens at a time (None: the configured chunk), in bytes: limit =
        working_set - others - margin; room = limit - active, on a ring the
        tighter of that and the peers' (_room). Which term made a refusal
        is read off this (the log line, the 503's `memory`)."""
        peers = self.tensor.peers_over_now() if self.tensor is not None \
            else None
        if n_tokens and chunk is None:
            chunk = self.prefill_step_size
        out = {"rank": int(getattr(getattr(self.tensor, "link", None),
                                   "rank", 0) or 0),
               "tokens": int(n_tokens),
               "chunk": int(chunk) if chunk else None,
               "working_set": self._working_set(),
               "others": self._others_bytes(),
               "others_readings": list(self._others),
               "margin": self._margin(n_tokens, chunk),
               "transient": self._transient(self._context() + n_tokens,
                                            chunk),
               "transient_samples": {k: sorted(v.values())
                                     for k, v in self._tx.items()},
               "context": self._context(),
               "local_active": self._local_active(),
               "active": self._active(),
               "cached": self._cached(),
               "prompt_cache": (self.cache.nbytes
                                if self.cache is not None else 0),
               "peers_over": peers,
               "limit": self._limit(n_tokens, chunk)}
        if peers is not None:
            from knurlogic.engine.serve import state
            out["ranks"] = [dict(r) for r in
                            state.SERVED.get("ranks") or []]
        if need is not None:
            out["need"] = int(need)
        if room is not None:
            out["room"] = int(room)
        return out

    def memory_short(self, n_tokens: int = 1024) -> str | None:
        """Why a minimal prompt would be refused now, or None: a loaded
        server that cannot admit one is not ready (/v1/models, the page's
        instance card). Only with nothing running -- then a refusal is
        final, not a wait. Read at the smallest chunk, the one a refusal
        is made at, and with no reserve for a prompt nobody sent: the
        margin is the step about to run's (_margin). Side-effect free
        (_fits). Memory pressure is reported apart (requests()'s
        memory_pressure), never as a refusal."""
        try:
            if self._rows or not self._kv or not self._limit() or \
                    self._fits(n_tokens):
                return None
            m = self._memory(n_tokens, chunk=MIN_CHUNK)
        except Exception:   # a status read must never fail the server
            return None
        peers = m["peers_over"]
        if peers is not None and \
                self._peer_room(peers, n_tokens, MIN_CHUNK) < \
                m["limit"] - m["local_active"]:
            over = [r for r in m.get("ranks", [])
                    if r.get("rank") and r.get("over_limit_bytes", 0) > 0]
            who = (f"rank {over[0]['rank']}" if len(over) == 1
                   else "a peer rank")
            why = (f"{peers / GIB:.1f} GiB over its limit" if peers > 0
                   else f"{-peers / GIB:.1f} GiB under its limit, short of "
                   f"a {n_tokens}-token step's "
                   f"{m['transient'] / GIB:.1f} GiB transient")
        else:
            who = f"rank {m['rank']}"
            why = (f"limit {m['limit'] / GIB:.1f} GiB (working set "
                   f"{m['working_set'] / GIB:.1f} - others "
                   f"{m['others'] / GIB:.1f} - margin "
                   f"{m['margin'] / GIB:.1f}), active "
                   f"{m['local_active'] / GIB:.1f}")
        return f"loaded, no memory for requests: {who}, {why}"

    # ----------------------------------------------------------- pressure

    def _sample_pressure(self, now: float | None = None) -> None:
        """Read this process's compressed/swapped bytes and macOS's pressure
        level every PRESSURE_EVERY_S; one WARNING when both say so (past
        PRESSURE_BYTES, level above normal), one INFO when either drops
        back. Compressed pages with the system at normal are not pressure
        (PRESSURE_BYTES)."""
        if self._compressed is None:
            return
        now = time.monotonic() if now is None else now
        if self._pressure_at and now - self._pressure_at < PRESSURE_EVERY_S:
            return
        self._pressure_at = now
        try:
            b = int(self._compressed() or 0)
            level = int(self._system_pressure() or 1) \
                if self._system_pressure is not None else 1
        except Exception:   # a reading must never fail the server
            b, level = 0, 1
        was = self._pressure_short() is not None
        self._pressure, self._pressure_level = b, level
        now_short = self._pressure_short()
        if now_short and not was:
            logger.warning("%s; requests are still admitted", now_short)
        elif was and not now_short:
            logger.info("memory pressure over: %.1f GiB of this server "
                        "compressed, macOS pressure level %d", b / GIB, level)

    def _pressure_short(self) -> str | None:
        """Why the server may run slow for memory pressure, or None."""
        b = self._pressure
        if b <= PRESSURE_BYTES or self._pressure_level < 2:
            return None
        rank = int(getattr(getattr(self.tensor, "link", None), "rank", 0)
                   or 0)
        what = "critical" if self._pressure_level >= 4 else "warn"
        return (f"rank {rank}: macOS reports memory pressure ({what}) and "
                f"has compressed {b / GIB:.1f} GiB of the model")

    def _working_set(self) -> int:
        if self.working_set is None:
            import importlib  # engine.serve exports a load() function
            load = importlib.import_module("knurlogic.engine.serve.load")
            self.working_set = int(load.memory().get("working_set_bytes")
                                   or 0)
        return self.working_set

    def _margin(self, extra: int = 0, chunk: int | None = None) -> int:
        """Room the next step's temporaries need: the transient THIS model
        is predicted to make in the step about to run (_transient: the
        running rows' context plus `extra`, a prompt being admitted and
        prefilled `chunk` tokens at a time -- None with no prompt: a decode
        step), with a quarter again -- but never below 5% of the working
        set (at least 4 GiB). Never the largest transient ever seen: one
        59k-token prefill's 17 GiB (GLM-5.3-Flash, 2026-10-05) stayed the
        margin of every later step, idle included, and every prompt was
        refused with 30-60 GiB unused across the ring."""
        floor = max(4 * GIB, self._working_set() // 20)
        if extra and chunk is None:
            chunk = self.prefill_step_size
        return max(floor, int(self._transient(self._context() + extra,
                                              chunk) * 1.25))

    def _context(self) -> int:
        """Tokens of context the running rows' next step spans."""
        return sum(r.job.prompt_tokens + r.made for r in self._rows.values())

    def _transient(self, ctx: int, chunk: int | None = None) -> int:
        """A step's predicted transient at `ctx` tokens of context, read
        off two measured lines. A step that admits a prompt prefills it
        `chunk` tokens at a time (the whole admission is one step), each
        chunk's temporaries spanning the chunk against the context before
        it: the prefill line is keyed by x = ctx x chunk / UNIT_CHUNK, so
        a smaller chunk reads lower (GLM-5.3-Flash at chunk 2048, its DSA
        indexer scoring chunk x context: 3.2 GiB at 2.7k tokens, 10.4 at
        33k, 17.0 at 59k). A decode step (`chunk` None) is keyed by the
        context alone. A prefill step decodes the running rows too: never
        under the decode line's reading."""
        dec = self._line("decode", ctx)
        if chunk is None:
            return dec
        return max(dec, self._line("prefill", self._prefill_x(ctx, chunk)))

    def _prefill_x(self, ctx: int, chunk: int) -> int:
        """The prefill line's key: the context a chunk's temporaries span
        (the model's `prefill_span`, else all of it) x chunk / UNIT_CHUNK.
        GLM-5.3's latent prefill reads at most index_topk keys a query: keyed
        by the whole context, one 2.7k-token warm-up carried in proportion
        refused a 339k-token prompt even at chunk 128."""
        span = self._span_fn()
        try:
            c = int(span(ctx)) if span is not None else ctx
        except Exception:  # a model's hint must never fail the guard
            c = ctx
        return c * int(chunk) // UNIT_CHUNK

    def _span_fn(self):
        """The served model's prefill_span, wherever the wrappers put it
        (the model, its language_model, its trunk), or None."""
        m = getattr(self.host, "model", None)
        if getattr(self, "_span_for", self) is not m:   # read per model
            self._span_for, self._span = m, None
            for o in (m, getattr(m, "language_model", None),
                      getattr(m, "model", None)):
                f = getattr(o, "prefill_span", None)
                if callable(f):
                    self._span = f
                    break
        return self._span

    def _line(self, kind: str, x: int) -> int:
        """The measured `kind` transients read at x: between two samples,
        the line between them; past the longest, the line through the
        shortest and longest carried on (once they are 8192 apart and
        rising, else in proportion to x); below the shortest, the larger
        of that first segment and the proportion. With one sample, in
        proportion to x -- the safe side; a shrinking chunk is what makes a
        long prompt fit then, not a guessed slope. Never under a sample
        at or below x. 0 until measured."""
        pts = sorted(self._tx.get(kind, {}).values())
        if not pts or x <= 0:
            return 0
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        if len(pts) == 1:
            est = y0 * x / x0
        elif x <= x0:
            xb, yb = pts[1]
            est = max(y0 * x / x0, y0 + (yb - y0) * (x - x0) / (xb - x0))
        elif x >= x1:
            slope = ((y1 - y0) / (x1 - x0) if x1 - x0 >= 8192 and y1 > y0
                     else y1 / x1)
            est = y1 + slope * (x - x1)
        else:
            k = next(i for i, (px, _) in enumerate(pts) if px >= x)
            (xa, ya), (xb, yb) = pts[k - 1], pts[k]
            est = ya + (yb - ya) * (x - xa) / (xb - xa)
        seen = max((py for px, py in pts if px <= x), default=0)
        return max(int(est), seen, 0)

    def _limit(self, extra: int = 0, chunk: int | None = None) -> int:
        """Active bytes a step may start at: the working set less what
        other processes hold of the GPU and the next step's temporaries
        (_margin). 0 = unguarded."""
        ws = self._working_set()
        return ws - self._others_bytes() - self._margin(extra, chunk) \
            if ws else 0

    def _others_bytes(self) -> int:
        """GPU memory the other processes on this machine hold, the most
        of the last ten readings (at most one every two seconds): the
        wired limit caps everyone's total, and a window server or another
        agent's model grows between readings."""
        if self._gpu_in_use is None:
            return 0
        now = time.monotonic()
        if now - self._others_at >= 2.0:
            self._others_at = now
            total = self._gpu_in_use()
            if total is not None:
                mine = self._local_active() + self._cached()
                self._others = (self._others + [max(total - mine, 0)])[-10:]
        return max(self._others, default=0)

    def _measure(self, before: int, ctx: int = 0,
                 chunk: int | None = None) -> None:
        """The step's TRANSIENT: its peak above the larger of where it
        started and where it ended. What it kept (an admitted row's KV,
        checkpoint copies) is growth, not transient -- counted as spike, one
        50k-token admission ratcheted the margin up for the life of the
        load. `ctx`: the context the step spanned; `chunk`: the step
        prefilled a row at that chunk (the prefill line), None: a decode
        step (the decode line)."""
        import mlx.core as mx
        spike = int(mx.get_peak_memory()) - max(before, self._here())
        kind = "decode" if chunk is None else "prefill"
        x = ctx if chunk is None else self._prefill_x(ctx, chunk)
        grew = spike > self._line(kind, x) * 1.25 and spike > GIB // 4
        if ctx >= 1024 and spike > 0 and x > 0:
            # one sample per eighth of a doubling of x, the largest kept
            pts = self._tx.setdefault(kind, {})
            b = int(math.log2(x) * 8)
            if b not in pts or spike > pts[b][1]:
                pts[b] = (x, spike)
        if grew:
            logger.info("a %s step's transient measured at %.2f GiB over "
                        "%d tokens of context%s", kind, spike / GIB, ctx,
                        f" at chunk {chunk}" if chunk else "")

    def _reset_peak(self) -> int:
        import mlx.core as mx
        mx.reset_peak_memory()
        return self._here()

    def _here(self) -> int:
        """This process's active memory (a step's transient is measured
        here, never against the peers')."""
        return self._active() if self.tensor is None else \
            self._local_active()

    def _active(self) -> int:
        """Active memory as the guard counts it. On a ring, the tightest
        rank rules: the peers' over-limit from the last exchange
        (tensor.Ring.peers_over_now) is read as if it were here."""
        a = self._local_active()
        peers = self.tensor.peers_over_now() if self.tensor is not None \
            else None
        if peers is not None:
            limit = self._limit()
            if limit:
                a = max(a, limit + peers)
        return a

    def _local_active(self) -> int:
        import mlx.core as mx
        return int(mx.get_active_memory())

    def _cached(self) -> int:
        import mlx.core as mx
        return int(mx.get_cache_memory())

    def _release(self) -> None:
        import mlx.core as mx
        mx.clear_cache()

    def _room_to_admit(self) -> bool:
        """A request is considered when nothing is running (it could not
        wait for memory anyone else would free) or memory is under the
        limit; whether ITS prompt fits is _make_room's question. The limit
        already holds a step's measured transient back -- holding a margin
        back again here (and in _room_for) left 397B, whose weights leave 8
        GiB, refusing 20k-token prompts with nothing else running."""
        limit = self._limit()
        return (not limit or not self._rows or self._active() < limit)

    def _learn(self, tokens, cache) -> None:
        """A row's cache as fixed + per-token bytes, from the shortest and
        longest caches seen. One ratio on a short cache charged a hybrid
        model's fixed linear-attention state to every token (Flash 4.4:
        long agent turns refused that fit). Until two lengths 2048 apart
        are known, the ratio at the longest -- an overestimate, the safe
        side."""
        n = len(tokens)
        if n < 256:
            return
        b = _cache_nbytes(cache)
        s = self._samples
        s["fixed"] = max(s.get("fixed", 0), min(_fixed_nbytes(cache), b))
        if "lo" not in s or n < s["lo"][0]:
            s["lo"] = (n, b)
        if "hi" not in s or n > s["hi"][0]:
            s["hi"] = (n, b)
        (n0, b0), (n1, b1) = s["lo"], s["hi"]
        if n1 - n0 >= 2048 and b1 > b0:
            slope = (b1 - b0) / (n1 - n0)
            self._kv = (max(b0 - slope * n0, 0.0), slope)
        else:
            # one length: what cannot grow (a hybrid's recurrent state) is
            # fixed, the rest per token. Charging a Flash's 34 deltanet
            # layers to every token of short runs prices a 24k prompt at
            # 3.4 GiB (0.85 measured) and refuses it
            f = float(s.get("fixed", 0))
            self._kv = (f, max(b1 - f, 0) / n1)

    def _cost(self, n_tokens: int, copies: int) -> int:
        assert self._kv is not None     # _cost is asked once one is known
        fixed, per = self._kv
        return int(copies * (fixed + per * n_tokens))

    def _need(self, n_tokens: int, checkpoints) -> int:
        """Bytes admitting a prompt of n_tokens adds, measured on the 27B
        (an M3 Ultra) to within 2%: the row's cache; a copy of it at
        each checkpoint (a segment boundary past the prompt cache's hit --
        a chat's trailing one-token segments are two nearly whole copies);
        and, with rows running, the copy the running batch concatenates
        it into, while the row's own is still held. Priced as one row and
        one checkpoint, a 57k-token prompt beside two running rows was
        charged 7.3 GiB, grew memory 21.3, and the next step aborted
        Metal."""
        def one(n):
            return self._cost(n, 1)
        need = one(n_tokens) + sum(one(c) for c in checkpoints)
        if self._rows:
            need += one(n_tokens)
        return need

    def _fits(self, n_tokens: int) -> bool:
        """Could a prompt of n_tokens fit once, at the smallest chunk,
        counting what the prompt cache would give up? No side effects: a
        request that waits must not empty the shared prompt cache on every
        tick it waits."""
        if not self._limit() or not self._kv:
            return True
        room = self._room(n_tokens, MIN_CHUNK) + (
            self.cache.nbytes if self.cache is not None else 0)
        return self._need(n_tokens, ()) <= room

    def _chunks(self) -> list[int]:
        """The prefill chunks an admission may take, largest first: the
        launch chunk, halved down to MIN_CHUNK (2048 -> 1024 -> 512 -> 256
        -> 128)."""
        c = max(int(self.prefill_step_size), 1)
        out = [c]
        while c // 2 >= MIN_CHUNK:
            c //= 2
            out.append(c)
        return out

    def _peer_room(self, peers: int, n_tokens: int, chunk: int | None) -> int:
        """Room on the tightest peer rank for the step admitting n_tokens
        at `chunk`. A peer reports its active memory less its own limit,
        which already holds its floor margin (at least 4 GiB) free; the
        step's transient past that floor is taken from its room -- the
        same transient as here (equal shards under tensor; a pipeline's
        stages run one layer's temporaries at a time, near enough)."""
        t = self._transient(self._context() + n_tokens, chunk)
        return -peers - max(0, int(t * 1.25) - 4 * GIB)

    def _room(self, n_tokens: int = 0, chunk: int | None = None) -> int:
        """Bytes free for a prompt of n_tokens' cache in the step that
        prefills it at `chunk`: under this rank's limit for that step, and
        on a ring under every peer's (_peer_room)."""
        limit = self._limit(n_tokens, chunk)
        if self.tensor is None:
            return limit - self._active()
        room = limit - self._local_active()
        peers = self.tensor.peers_over_now()
        if peers is not None:
            room = min(room, self._peer_room(peers, n_tokens, chunk))
        return room

    def _room_for(self, n_tokens: int, checkpoints=(),
                  chunk: int | None = None):
        """(fits, need, room) for a prompt of n_tokens with checkpoints at
        these lengths, prefilled `chunk` tokens at a time (None: the
        launch chunk), the prompt cache giving way if that is what it
        takes."""
        chunk = chunk or self.prefill_step_size
        if not self._limit(n_tokens, chunk) or not self._kv:
            return True, 0, 0
        need = self._need(n_tokens, checkpoints)
        room = self._room(n_tokens, chunk)
        held = self.cache.nbytes if self.cache is not None else 0
        if need > room + held:
            # not even an empty prompt cache would make it fit: evicting
            # for it only threw away the entries a lean admission (or the
            # next request) could have hit
            return False, need, room
        if need > room and held:
            before = self.cache.nbytes
            self.cache.trim_to(max(before - (need - room), 0))
            self._ring_trimmed = self.cache.nbytes < before
            self._release()
            room = self._room(n_tokens, chunk)
            logger.info("the prompt cache gave up %.1f GiB for a %d-token "
                        "prompt", (before - self.cache.nbytes) / GIB,
                        n_tokens)
        return need <= room, need, room

    def _make_room(self, n_tokens: int, checkpoints=None) -> str:
        """"full" if a prompt of n_tokens fits with its checkpoints (at
        these lengths; None = one, at its end), "lean" if only without;
        else _Wait (rows are running and will free memory) or OutOfMemory
        (none are). The chunk it fits at is left in _chunk_pick.

        Cheapest loss first: the prompt cache gives way; then the prefill
        chunk shrinks (2048 -> ... -> 128: a chunk's temporaries span it
        against the whole context before it, and a whole admission is ONE
        step, so the chunk is chosen here, for the whole prompt -- a
        little slower, instead of a refusal); then the checkpoints go;
        then it waits, or with nothing running is refused."""
        if checkpoints is None:
            checkpoints = [n_tokens]
        chunks = self._chunks()
        self._chunk_pick = None
        full = 0
        for c in chunks:
            fits, full, room = self._room_for(n_tokens, checkpoints, c)
            if fits:
                self._picked(c, n_tokens, room)
                return "full"
        for c in chunks:
            fits, need, room = self._room_for(n_tokens, (), c)
            if fits:
                self._picked(c, n_tokens, room)
                logger.info("a %d-token prompt admitted without checkpoints: "
                            "%.1f GiB free, with them it would take %.1f; %s",
                            n_tokens, room / GIB, full / GIB,
                            _terms(self._memory(n_tokens, need, room, c)))
                return "lean"
        limit = self._limit(n_tokens, chunks[-1])
        if self._rows:
            raise _Wait()
        # On a ring the peers' number is as of the last exchange: what the
        # prompt cache just gave up here is given up there only when the
        # pops travel (the next exchange; an idle rank 0 sends them in
        # park()). Wait for that rather than refuse on a stale number --
        # once nothing is left to give up, the refusal below stands.
        if self.tensor is not None and self.cache is not None and \
                self._ring_trimmed:
            self._ring_trimmed = False
            raise _RingWait()
        mem = self._memory(n_tokens, need, room, chunks[-1])
        logger.warning("refused a %d-token prompt: %s", n_tokens, _terms(mem))
        if room <= 0 and not (self.cache is not None and self.cache.nbytes):
            # nothing runs and nothing is left to give up, yet no memory is
            # free: whatever holds it is out of this server's reach
            raise OutOfMemory(
                f"no memory is free under the server's limit "
                f"({limit / GIB:.1f} GiB) with nothing running and nothing "
                f"cached; it is holding memory it cannot release, so this "
                f"server needs a restart", mem)
        raise OutOfMemory(
            f"this prompt ({n_tokens} tokens) needs about {need / GIB:.1f} "
            f"GiB for its cache; {max(room, 0) / GIB:.1f} GiB is free under "
            f"the server's limit ({limit / GIB:.1f} GiB) even prefilled "
            f"{chunks[-1]} tokens at a time. Send a shorter conversation, "
            f"or serve a smaller model", mem)

    def _picked(self, chunk: int, n_tokens: int, room: int) -> None:
        self._chunk_pick = chunk
        if chunk < self.prefill_step_size:
            logger.info("a %d-token prompt is prefilled %d tokens at a time "
                        "(not %d) to fit the memory left: a %.1f GiB "
                        "transient predicted, %.1f GiB free for its cache",
                        n_tokens, chunk, self.prefill_step_size,
                        self._transient(self._context() + n_tokens, chunk)
                        / GIB, room / GIB)

    def _guard_memory(self) -> None:
        limit = self._limit()
        if not limit:
            return
        # mlx's freed buffers stay wired until cleared, and the step's
        # transient lands on top of them: past the limit, they go first
        if self._cached() and self._local_active() + self._cached() > limit:
            self._release()
        over = self._active() - limit
        if over <= 0:
            return
        if self.cache is not None and self.cache.nbytes:
            before = self.cache.nbytes
            # the limit already leaves a step's margin; trimming another
            # would evict a margin's worth of prompt cache for nothing
            self.cache.trim_to(max(before - over, 0))
            self._release()
            over = self._active() - limit
            logger.warning("memory past the limit (%.1f GiB): the prompt "
                           "cache gave up %.1f GiB", limit / GIB,
                           (before - self.cache.nbytes) / GIB)
        while over > 0 and self._rows:
            uid = max(self._rows)             # the newest: least work lost
            if len(self._rows) > 1 and uid in self._queued():
                # not prefilled yet: nothing is lost by queueing it again
                self._requeue(uid, "memory past the limit (%.1f GiB)"
                              % (limit / GIB))
                over = self._active() - limit
                continue
            row = self._rows.pop(uid)
            assert self._ex is not None     # there are rows, so an executor
            self._ex.remove([uid])
            self._release()
            logger.warning("memory past the limit (%.1f GiB): stopped the "
                           "newest request (%d prompt tokens)", limit / GIB,
                           row.job.prompt_tokens)
            self._error(row.job, OutOfMemory(
                f"stopped to keep the server within its memory "
                f"({limit / GIB:.1f} GiB usable, {len(self._rows)} other "
                f"request(s) running); retry, or ask for fewer tokens"))
            over = self._active() - limit


    def _fit_next(self) -> None:
        """Before a step that prefills a row: refit its chunk to the room
        now. It was fitted when inserted; rows inserted in the same tick
        were each fitted against the same free memory, and the rows
        running have grown since. The largest chunk whose step fits on
        every rank is set (on a ring, the `chunk` op carries it to the
        other ranks before the step). If not even MIN_CHUNK fits: with
        other rows running it goes back to the queue (they will free
        memory); alone it is stopped with OutOfMemory. A row once
        prefilled has no chunk left to shrink: its admission is one
        step."""
        uid = self._next_admission()
        row = self._rows.get(uid) if uid is not None else None
        if row is None or not self._kv or not self._limit():
            return
        n = row.job.prompt_tokens
        need = self._cost(n, 2 if len(self._rows) > 1 else 1)
        cur = self._chunk_of.get(uid, self.prefill_step_size)
        for c in self._chunks():
            room = self._room(0, c)
            if need <= room:
                if c != cur:
                    self._chunk_of[uid] = row.chunk = c
                    refit = getattr(self._ex, "refit", None)
                    if refit is not None:
                        refit(uid, c)
                    logger.info("a %d-token prompt about to prefill is "
                                "refitted to chunk %d (was %d): a %.1f GiB "
                                "transient predicted, %.1f GiB free after "
                                "its cache", n, c, cur,
                                self._transient(self._context(), c) / GIB,
                                (room - need) / GIB)
                return
        if len(self._rows) > 1:
            self._requeue(uid, f"no room to prefill at chunk {MIN_CHUNK}")
            return
        self._rows.pop(uid)
        self._chunk_of.pop(uid, None)
        assert self._ex is not None
        self._ex.remove([uid])
        self._release()
        mem = self._memory(0, need, self._room(0, MIN_CHUNK), MIN_CHUNK)
        logger.warning("stopped a %d-token prompt before its prefill: %s", n,
                       _terms(mem))
        self._error(row.job, OutOfMemory(
            f"this prompt ({n} tokens) no longer fits: about "
            f"{need / GIB:.1f} GiB for its cache and a step prefilling it "
            f"{MIN_CHUNK} tokens at a time; retry, or send a shorter "
            f"conversation", mem))
