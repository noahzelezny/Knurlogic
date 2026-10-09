"""A rank >= 1 of a split model, from start to stop: `serve_follower`
(join, load its shard or stage, follow), `follow` (apply rank 0's plans
and step), its memory limit (`Mark`), its SIGTERM handling, and
`agree_head` (rank 0's drafting head told to every rank)."""

from __future__ import annotations

import logging

import mlx.core as mx

from knurlogic.engine.prompt_cache.ring import apply_cache_op
from knurlogic.engine.runtime.executor import (
    Admission,
    Checkpoint,
    Finished,
    LocalExecutor,
)

from . import plan as P
from .link import Desync, Link, init
from .tensor import load_config, shard

logger = logging.getLogger(__name__)
GIB = 1 << 30


# --------------------------------------------------------------- follower

class Mark:
    """A follower's memory limit: the working set less one DECODE step's
    measured transient (at least 5% of it, at least 4 GiB). A step that
    prefills a row is not counted: its transient grows with that prompt's
    context x chunk, and rank 0 prices it per admission against this
    rank's room (memory_guard._room) -- held here, one 59k-token prefill's
    17 GiB stayed the margin of every later step and the GLM-5.3-Flash
    pipeline's M3 read 1.4 GiB over its limit with nothing running."""

    def __init__(self, working_set: int):
        self.ws = int(working_set)
        self.spike = 0

    def limit(self) -> int:
        if not self.ws:
            return 0
        return self.ws - max(4 * GIB, self.ws // 20, int(self.spike * 1.25))

    def over(self) -> int:
        lim = self.limit()
        return int(mx.get_active_memory()) - lim if lim else 0

    def around(self, fn, prefill: bool = False):
        mx.reset_peak_memory()
        before = int(mx.get_active_memory())
        out = fn()
        if not prefill:
            self.spike = max(self.spike, int(mx.get_peak_memory())
                             - max(before, int(mx.get_active_memory())))
        return out


def apply_set(op: dict, rank: int) -> str:
    """A follower applies a live knob rank 0 applied (the `set` op) to its
    own engine, exactly as rank 0's Settings apply did. -> what happened."""
    from knurlogic.engine.model.load import apply_live
    said = apply_live({op["name"]: op["value"]}).get(op["name"], "")
    logger.info("rank %d: %s=%s: %s", rank, op["name"], op["value"], said)
    return said


def _stop_load_on_sigterm(rank: int) -> None:
    """Until it follows: a SIGTERM stops this rank's weight read at its next
    batch boundary (host.LOAD_STOP; the load raises LoadCancelled and the
    rank exits) instead of killing it inside one, which left the GPU's
    utilization counter stuck at 100% until a reboot. follow() replaces it
    with _defer_sigterm."""
    import signal

    from knurlogic.engine.runtime.model_host import LOAD_STOP

    def stop(_sig, _frame):
        logger.info("SIGTERM while loading: stopping at the next batch")
        LOAD_STOP.set()
    try:
        signal.signal(signal.SIGTERM, stop)
    except ValueError:              # not the main thread (a test's ring)
        pass


def _defer_sigterm(rank: int) -> None:
    """A following rank leaves on rank 0's `stop`, between steps, not on the
    page's SIGTERM: killed mid-step, it would leave rank 0's GPU waiting on
    a collective that never completes (pinned at 100%, holding the job's
    memory past every exit, until a reboot). Rank 0 stops the ring within
    seconds (http.watch_ring); if it is gone, the page's SIGKILL follows."""
    import signal

    def deferred(_sig, _frame):
        logger.info("rank %d: SIGTERM -- leaving on rank 0's stop, between "
                    "steps", rank)
    try:
        signal.signal(signal.SIGTERM, deferred)
    except ValueError:              # not the main thread (a test's ring)
        pass


def follow(model, tokenizer, model_key, link: Link, *, prompt_cache_size: int,
           completion_batch_size: int, prefill_step_size: int,
           working_set: int, split: str = "tensor", drafting: bool = False,
           why: str = "", vision=None, block: int = 0,
           outputs: tuple = (), disk=None) -> int:
    """Rank >= 1: apply rank 0's plans and step until told to stop. The
    return value is the number of steps taken.

    `split="pipeline"`: this rank holds a run of layers, not a slice of
    every layer. Its trunk returns zeros for logits (pipeline.Silent: its
    samples are never used, so they are not compared with rank 0's).
    `drafting`: rank 0 drafts with an MTP head (agree_head). This rank
    holds none; it runs rank 0's drafting steps -- the verify forward with
    the drafted tokens, the rollback and replay -- as the Coord broadcasts
    say. `block`: rank 0's head drafts that many tokens a pass (DSpark),
    reading the outputs of the layers `outputs`: this rank runs the block
    loop's steps (MTPBatchGenerator.follow_block). `vision`:
    engine.vision.request.MirrorVision for a model with vision (rank 0
    encodes; the admit op carries each image's ref and the admission its
    rows), else None. `disk`: (key, entries read and agreed at load) of the
    prompt cache on disk (engine/prompt_cache/disk): inserted first, as
    rank 0 inserts its own; saved again at `save_cache` (a client's ask)."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.prompt_cache.memory import PromptCache
    from knurlogic.engine.runtime.request import control_machine

    stream = mx.default_stream(mx.default_device())
    _defer_sigterm(link.rank)
    cache = PromptCache(prompt_cache_size)
    from knurlogic.engine.prompt_cache import disk as prompt_disk
    disk_key, restored = disk if disk is not None else (None, [])
    if restored:
        prompt_disk.adopt(cache.owners, cache.pinned, prompt_disk.insert(
            cache.lru, model_key, restored), shared=cache.shared)

    def save_disk(only_new: bool = False, select=None):
        if disk_key is None:
            return
        try:
            cache.live()                    # prunes the side map
            # no sweep of its own: rank 0's saves sweep and name what they
            # delete (a drop_files op), so every rank keeps the same files
            prompt_disk.save(cache.lru, disk_key, owners=cache.owners,
                             shared=cache.shared,
                             only_new=only_new, select=select,
                             sweep_after=False)
        except Exception:  # never stops a rank (logged)
            logger.exception("rank %d: saving the prompt cache failed",
                             link.rank)
    mark = Mark(working_set)
    #: the prefill chunk rank 0 fitted each row not yet prefilled at (the
    #: admit and chunk ops): a rank prefilling in a different number of
    #: chunks deadlocks
    chunks: dict = {}
    ex: LocalExecutor | None = None
    last: dict = {}
    events: list = []
    steps = mismatches = 0

    def executor() -> LocalExecutor:
        nonlocal ex
        if ex is None:
            gen = MTPBatchGenerator(
                model, None, stats={}, vision=vision, why=why,
                completion_batch_size=completion_batch_size,
                prefill_step_size=prefill_step_size, stream=stream)
            from . import pipeline as PL
            if drafting and block:
                gen.follow_block(block, outputs)
            if split == "pipeline":
                PL.silence(gen)
            elif drafting:
                gen.mirror_hidden()
            PL.coordinate(gen, link.group, drafting=drafting)
            ex = LocalExecutor(gen)
        return ex

    while True:
        _, data = link.exchange(mark.over(), None)
        plan = P.decode(data) if data else {"ops": []}
        halt = park = False
        for op in plan.get("ops", []):
            kind = op["op"]
            if kind == "park":
                park = halt = True
            elif kind == "admit":
                prompt = op["prompt"]
                if op["images"]:
                    if vision is None:
                        raise Desync("rank 0 admitted a prompt with images; "
                                     "this rank bound no vision family")
                    vision.add_refs(op["refs"])
                    prompt = P.key_from_wire(prompt, op["images"],
                                             op["refs"])
                    at, segs = op["hit"], []
                    for sg in op["segs"]:
                        segs.append(prompt[at:at + len(sg)])
                        at += len(sg)
                    op = dict(op, segs=segs)
                c, rest = cache.fetch(model_key, prompt)
                if len(prompt) - len(rest) != op["hit"]:
                    raise Desync(f"prompt cache hit {len(prompt) - len(rest)} "
                                 f"here, {op['hit']} on rank 0")
                procs = []
                if op["penalties"]:
                    from mlx_lm.sample_utils import make_logits_processors
                    procs = make_logits_processors(**op["penalties"])
                sm, _ = control_machine(tokenizer, op["initial"])
                uid = executor().insert(Admission(
                    segments=op["segs"], max_tokens=op["max_tokens"],
                    cache=c, prefix=prompt[:op["hit"]],
                    sampling=op["sampling"], processors=procs,
                    state_machine=sm))
                if uid != op["uid"]:
                    raise Desync(f"admitted as {uid}, rank 0 has {op['uid']}")
                chunks[uid] = int(op["chunk"])
            elif kind == "chunk":
                chunks[op["uid"]] = int(op["chunk"])
            elif kind == "remove":
                if ex is not None:
                    ex.remove(op["uids"])
                for u in op["uids"]:
                    chunks.pop(u, None)
            elif apply_cache_op(op, cache, model_key, last, save_disk,
                                prompt_disk.root() / prompt_disk.key_id(
                                    disk_key) if disk_key else None,
                                disk_key):
                pass
            elif kind == "set":
                apply_set(op, link.rank)
            elif kind == "reset":
                if ex is not None:
                    ex.close()
                ex = None
                chunks.clear()
                if vision is not None:
                    vision.clear()
                halt = True
            elif kind == "stop":
                if ex is not None:
                    ex.close()
                logger.info("rank %d: stopped by rank 0 after %d steps "
                            "(%d token mismatches)", link.rank, steps,
                            mismatches)
                return steps
        # the last step's events hold its checkpoint and finished caches:
        # not kept through a removal or a park, which nothing steps past
        last, events = {}, []
        if park:
            # nothing runs: a transient measured under load is not the
            # margin an idle rank holds back (it pinned a pipeline's
            # over-limit above zero after one long prefill, refusing every
            # request, 16 tokens or 40k)
            mark.spike = 0
            link.sleep()
        if halt:
            continue
        e = executor()
        b = e.gen._batch
        toks = plan.get("tokens") or []
        if [int(u) for u in b.uids] != [u for u, _ in toks]:
            raise Desync(f"batch rows {list(b.uids)} here, "
                         f"{[u for u, _ in toks]} on rank 0")
        if toks and split == "pipeline":
            b.t1 = mx.array([t for _, t in toks], dtype=mx.int32)
        elif toks:
            mine = b.t1.tolist()
            theirs = [t for _, t in toks]
            if mine != theirs:
                mismatches += sum(a != c for a, c in zip(mine, theirs))
                logger.warning("rank %d: %d token(s) differ from rank 0's; "
                               "rank 0's are used", link.rank,
                               sum(a != c for a, c in zip(mine, theirs)))
            b.t1 = mx.array(theirs, dtype=mx.int32)
        nxt = e.next_admission()
        if nxt is not None:
            e.set_chunk(chunks.pop(nxt, prefill_step_size))
        events = mark.around(e.step, prefill=nxt is not None)
        steps += 1
        for ev in events:
            if isinstance(ev, Checkpoint):
                last[("checkpoint", ev.uid)] = (ev.tokens, ev.cache)
            elif isinstance(ev, Finished):
                last[("finished", ev.uid)] = (ev.tokens, ev.cache)


def serve_follower(path: str, *, link_kind: str, working_set: int,
                   prompt_cache_size: int, completion_batch_size: int,
                   prefill_step_size: int,
                   executes_artifact_code: bool = False,
                   split: str = "tensor", pipeline: dict | None = None,
                   draft: bool = True, kv_bits: int | None = None,
                   cross_chip: dict | None = None) -> int:
    """A rank >= 1 from start to stop: join, load its shard, follow.
    `pipeline`: agree()'s keyword arguments for a pipeline split.
    `cross_chip`: engine/crosschip.resolve(...) for this job."""
    from knurlogic.engine.runtime.model_host import ModelHost
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _stop_load_on_sigterm(0)
    link = init(link_kind)
    if split == "pipeline":
        from . import pipeline as PL
        shares = PL.agree(link.group, **(pipeline or {}))
        logger.info("pipeline  rank %s: %s", link.rank, shares["reason"])
        def cut(m):
            return PL.split(m, link.group, shares["bounds"])
        cut_config = None
    else:
        def cut(m):
            return shard(m, link.group)

        def cut_config(p):
            return load_config(p, link.group)
    # a follower never loads the MTP head: rank 0 drafts, and tells this
    # rank whether it does (agree_head)
    heads = agree_head(link)
    layout = {"split": split, "world": link.size, "rank": link.rank}
    if split == "pipeline":
        layout["bounds"] = [list(b) for b in shares["bounds"]]
    # nor the vision tower: rank 0 encodes; this rank embeds its rows with
    # the family's own code (engine.vision.request.MirrorVision)
    host = ModelHost(draft=False,
                     executes_artifact_code=executes_artifact_code,
                     shard=cut, shard_config=cut_config, vision=True,
                     tower=False, load_wait_s=3600.0,
                     head_agree=heads, kv_bits=kv_bits,
                     cross_chip=cross_chip)
    host.cache_layout = layout
    disk: list = [None, []]

    def restore_disk():
        # the same point as rank 0's Scheduler._restore_disk: a collective
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        try:
            key = prompt_disk.host_key(host)
            got = prompt_disk.read(key, prompt_disk.candidates(
                key, prompt_cache_size, sweep_first=False)) \
                if key is not None else []
        except Exception:  # a restore that fails is a miss (logged)
            logger.exception("rank %d: reading the saved prompt cache "
                             "failed", link.rank)
            key, got = None, []
        if not prompt_disk.agree(link, [g[0] for g in got]):
            got = []
        disk[:] = [key, got]
    host.after_bind = restore_disk
    host.load(path)
    if host.state != "ready":
        raise RuntimeError(f"rank {link.rank} could not load {path}: "
                           f"{host.error}")
    logger.info("rank %d: %s loaded, %.1f GiB active", link.rank, path,
                mx.get_active_memory() / GIB)
    from .marker import after_load
    after_load()
    drafting = bool(heads and heads.leader)
    from knurlogic.engine.model import state
    return follow(host.model, host.tokenizer, host.model_key, link,
                  vision=state.VISION.get("serve"),
                  prompt_cache_size=prompt_cache_size,
                  completion_batch_size=completion_batch_size,
                  prefill_step_size=prefill_step_size,
                  working_set=working_set, split=split, drafting=drafting,
                  block=heads.block if drafting else 0,
                  outputs=tuple(heads.outputs) if drafting else (),
                  disk=tuple(disk),
                  why=("rank 0 drafts; this rank runs its verify steps"
                       if drafting else "rank 0 does not draft"))


class agree_head:
    """ModelHost's `head_agree` on a split model, on every rank after its
    load: rank 0's answer (it bound a head, or not) told to every rank.
    Only rank 0 holds one; `leader` is what a follower's Coord needs
    (every rank makes B1 / BA, or none does). A block head (DSpark) also
    tells its `block` size and the layers whose outputs it reads
    (`outputs`): a follower runs the block loop's steps, and on a
    pipeline carries those outputs on to rank 0 (pipeline.carry)."""

    #: the most layer outputs a block head may read (a fixed broadcast)
    MAX_OUTPUTS = 16

    def __init__(self, link: Link):
        self.link = link
        self.leader = False
        self.block = 0
        self.outputs: list[int] = []

    def __call__(self, has: bool, head=None) -> bool:
        k = int(getattr(head, "block_size", 0) or 0) if has else 0
        outs = [int(i) for i in getattr(head, "targets", ())] if k else []
        if len(outs) > self.MAX_OUTPUTS:
            # told as no head (raising here would leave the other ranks
            # in the all_gather)
            logger.warning("a block head reading %d layer outputs; the "
                           "broadcast holds %d: not drafting on this split",
                           len(outs), self.MAX_OUTPUTS)
            has, k, outs = False, 0, []
        row = [int(bool(has)), k, len(outs)] + outs + \
            [0] * (self.MAX_OUTPUTS - len(outs))
        # each rank gets here when ITS load is done: line them up first
        self.link.align()
        got = mx.distributed.all_gather(mx.array(row), group=self.link.group,
                                        stream=mx.cpu).tolist()
        self.leader = bool(got[0])
        self.block = int(got[1])
        self.outputs = [int(i) for i in got[3:3 + int(got[2])]]
        return self.leader
