"""Rank 0's side of a split model's ring: the `Ring` it keeps across
executors (the plan sent before each step, the ops `Journal`, the peers'
memory), `TensorExecutor` (the local batch engine with every admission
and removal journaled), and `assign_seed`.
The plan itself is plan.py."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import mlx.core as mx

from knurlogic.engine.runtime.executor import (
    Admission,
    LocalExecutor,
)

from . import plan as P
from .link import Link

logger = logging.getLogger(__name__)


def publish_ranks(rows) -> list:
    """Every rank's memory as of this exchange, for rank 0's /status.json
    (`ranks`): a follower serves no status of its own, so this is the one
    place a pipeline stage's memory is visible from outside."""
    from knurlogic.engine.model import state
    ranks = [{"rank": i, "active_bytes": int(r[P.ACTIVE]),
              "peak_bytes": int(r[P.PEAK]),
              "over_limit_bytes": int(r[P.OVER])}
             for i, r in enumerate(rows) if len(r) >= P.CONTROL_LEN]
    state.SERVED["ranks"] = ranks
    return ranks


class Journal:
    """Rank 0's ops since the last exchange, in the order they happened."""

    def __init__(self):
        self.ops: list[dict] = []

    def add(self, op: str, **fields) -> None:
        self.ops.append({"op": op, **fields})

    def take(self) -> list[dict]:
        ops, self.ops = self.ops, []
        return ops


class Ring:
    """Rank 0's side of the ring, kept across executors and prompt caches
    (both are rebuilt; the ring is not)."""

    def __init__(self, link: Link, split: str = "tensor"):
        self.link = link
        #: "tensor" or "pipeline" (engine/split/pipeline.py): the plan is
        #: the same; a pipeline's batch engine also carries a Coord
        self.split = split
        self.journal = Journal()
        #: max over ranks >= 1 of (active - limit) at the last exchange, and
        #: rank 0's own active memory then
        self.peer_over: int | None = None
        self.local_then = 0
        self.mismatches = 0

    @property
    def world(self) -> int:
        return self.link.size

    def exchange(self, over: int, plan: dict) -> None:
        payload = None if P.empty(plan) else P.encode(plan)
        rows, _ = self.link.exchange(over, payload)
        self.peer_over = max(r[P.OVER] for r in rows[1:])
        self.local_then = int(mx.get_active_memory())
        publish_ranks(rows)

    def peers_over_now(self) -> int | None:
        """The peers' over-limit as of the last exchange. Under tensor the
        ranks hold equal shards and apply the same ops, so what rank 0 has
        TAKEN since is added; what it has freed is not subtracted -- a free
        here is not yet a free there, and an estimate that drops on rank
        0's word alone lets a peer run past its limit. Under pipeline the
        stages are unequal and nothing about rank 0's memory says anything
        about a peer's: the peers' own number, refreshed every step."""
        if self.peer_over is None:
            return None
        if self.split == "pipeline":
            return self.peer_over
        return self.peer_over + max(
            0, int(mx.get_active_memory()) - self.local_then)

    def stop(self) -> None:
        ops = self.journal.take() + [{"op": "stop"}]
        self.link.exchange(0, P.encode({"ops": ops}))
        self.stopped = True

    def park(self) -> None:
        """Rank 0 has nothing to run: the other ranks sleep on the bell
        instead of spinning in the next collective (Link.bell). The next
        exchange rings it first."""
        if not self.link.socks or getattr(self, "stopped", False):
            return
        # parked with nothing new: stay asleep. Ops taken while idle (the
        # prompt cache's pops as it makes room) ring the others to apply
        # them and report their memory, then park them again.
        if self.link.parked and not self.journal.ops:
            return
        ops = self.journal.take() + [{"op": "park"}]
        self.link.exchange(0, P.encode({"ops": ops}))
        self.link.parked = True


#: sampling's mark for a seed assign_seed drew (popped before sampling)
RING_SEED = "ring_seed"


def assign_seed(sampling: dict) -> dict:
    """Every row gets a seed on a ring: rank 0's draw is then the draw any
    rank would make from the same logits."""
    s = dict(sampling or {})
    if s.get("seed") is None:
        s["seed"] = random.SystemRandom().randrange(1 << 31)
        # not the client's: it must not pin the drafting regime (Keys.pins)
        s[RING_SEED] = True
    return s


class TensorExecutor(LocalExecutor):
    """Rank 0's executor on a ring: the local batch engine, with every
    admission and removal journaled and the plan sent before each step."""

    def __init__(self, generator, ring: Ring, over: Callable[[], int]):
        super().__init__(generator)
        self.ring = ring
        self._over = over
        # every rank's batch engine carries a Coord (pipeline.coordinate):
        # B0 after each admission tells every rank when one rank's failed,
        # so all skip that decode step instead of all_summing different row
        # counts (Coord.diverged); with a head on rank 0, BA / B1 / B2 carry
        # its drafts and verdicts. The scheduler installs it first when it
        # knows whether rank 0 drafts.
        if generator._coord is None:
            from .pipeline import coordinate
            coordinate(generator, ring.link.group)

    def insert(self, a: Admission) -> int:
        if a.wire is None:
            raise ValueError("a ring admission carries its wire fields")
        key = list(a.prefix) + [t for s in a.segments for t in s]
        ids, images, refs = P.key_to_wire(key, self._ref_of)
        uid = super().insert(a)
        segs, at = [], len(a.prefix)
        for s in a.segments:
            segs.append(ids[at:at + len(s)])
            at += len(s)
        self.ring.journal.add(
            "admit", uid=int(uid), prompt=ids, segs=segs,
            hit=len(a.prefix), max_tokens=int(a.max_tokens),
            sampling=dict(a.sampling), penalties=dict(a.wire["penalties"]),
            initial=a.wire["initial"], images=images, refs=refs,
            chunk=int(a.chunk or self.gen.prefill_step_size))
        return uid

    def refit(self, uid: int, chunk: int) -> None:
        """Row `uid`, not yet prefilled, prefills `chunk` tokens at a time
        on every rank (the `chunk` op, applied before the step)."""
        self.ring.journal.add("chunk", uid=int(uid), chunk=int(chunk))

    def _ref_of(self, sha: str, ph: str):
        """(n_tokens, grid_thw) of an image of a key being admitted: the
        ref rank 0's store keeps (never evicted)."""
        vis = self.gen._vision
        if vis is None:
            raise ValueError("an image key on a ring with no vision family")
        r = vis.lookup()[1](sha, ph)
        return r.n_tokens, r.grid_thw

    def remove(self, uids: list[int]) -> None:
        super().remove(uids)
        if uids:
            self.ring.journal.add("remove", uids=[int(u) for u in uids])

    def step(self):
        b = self.gen._batch
        plan = {"ops": self.ring.journal.take()}
        if len(b):
            plan["tokens"] = [[int(u), int(t)]
                              for u, t in zip(b.uids, b.t1.tolist())]
        self.ring.exchange(self._over(), plan)
        return super().step()

    def close(self) -> None:
        try:
            ops = self.ring.journal.take() + [{"op": "reset"}]
            self.ring.link.exchange(0, P.encode({"ops": ops}))
        finally:
            super().close()
