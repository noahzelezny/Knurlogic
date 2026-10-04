"""The batch engine: mlx-lm's BatchGenerator (as a library class), backed
by batch_loop.MTPBatch instead of GenerationBatch -- drafting with an MTP
head when there is one, and the same engine without. knurlogic's server
drives it through engine/runtime/executor.LocalExecutor.

It SUBCLASSES mlx-lm's BatchGenerator: queueing, uid allocation and the
wired-limit handling are inherited; only where a row is prefilled and where
tokens come from are replaced. Capture is installed for the generator's
lifetime; the head's cache rides in the prefix-cache entry beside the
trunk's; the row count moves in lockstep. Every request with an image comes
here (head or not), keyed by the vision cache KEY; segment checkpoints are
handed back one per `next()` call as end-of-segment responses.
Ported from the maintainer's own code in github.com/noahzelezny/exo
(Apache-2.0). Design: docs/design/drafting.md (batch engine).
"""
from __future__ import annotations

import contextlib
import copy
import logging
import os
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, GenerationBatch, PromptProcessingBatch
from mlx_lm.models.cache import make_prompt_cache

from knurlogic.engine.runtime.control import stop_machine
from knurlogic.engine.serve import cache_report as cachereport

from ..vision import key as K
from .batch_loop import ForwardFailed, MTPBatch, RowParams, admit
from .block_loop import BlockBatch
from .caches import position
from .capture import capture_input
from .registry import resolve
from .sampling import Keys, NonFiniteLogits, make_distribution, nonfinite_message

logger = logging.getLogger(__name__)

#: A row this long is never ended by the loop itself. Finishing is decided
#: here, where the row's cache can still be extracted before it is dropped --
#: batch_loop removes a row it finishes before returning.
_NEVER = 1 << 62


@dataclass
class TokenResponse(GenerationBatch.Response):
    """mlx-lm's per-token response plus the control machine's reading of
    the token (runtime/control.py): the state it left the row in and the
    marker sequence it completed. mlx-lm 0.32's Response dropped both when
    its generator stopped tracking state; request.py needs them."""
    current_state: str | None = None
    match_sequence: tuple | None = None


class HeadCarry:
    """The hidden state h_{c-1} a checkpoint entry carries beside its head
    cache (seeded to c-1), so a restore can replay the head's step at c-1
    with the NEW token at c. Rides in the pool entry list, so it answers
    the prompt cache's questions about size and trimming."""

    def __init__(self, h):
        self.h = h

    @property
    def nbytes(self) -> int:
        return int(self.h.nbytes)

    @property
    def state(self):
        return [self.h]

    def is_trimmable(self) -> bool:
        return False


def split_pool_entry(entry: list, n_trunk: int, *, drafts: bool,
                     hit_len: int, replay=None) -> tuple:
    """(trunk caches, head cache or None, usable prefix length) from a pool
    entry restored at `hit_len`.

    A drafting row needs a head cache sitting at exactly hit_len; an entry
    without one (stored by a non-drafting row, or before drafting was on) or
    with one elsewhere is unusable for drafting -> prefix length 0, fresh
    caches. A non-drafting row uses the trunk part of any entry."""
    trunk, extra = list(entry[:n_trunk]), list(entry[n_trunk:])
    if hit_len <= 0:
        return trunk, None, 0
    if not drafts:
        return trunk, None, hit_len
    off = position(extra[0]) if extra else None
    off = -1 if off is None else off
    if off == hit_len:
        return trunk, extra[0], hit_len
    # A checkpoint entry: head one step behind, with the h to replay it.
    if (off == hit_len - 1 and replay is not None and len(extra) > 1
            and isinstance(extra[1], HeadCarry)):
        replay(extra[0], extra[1].h)
        return trunk, extra[0], hit_len
    return [], None, 0


def trunk_offset(trunk: list) -> int | None:
    """The position a trunk's caches sit at: the first one with a position.
    Recurrent layers carry state, not a position, so they cannot answer."""
    for c in trunk:
        off = position(c)
        if off is not None:
            return off
    return None


def sampling_of(sampler) -> dict | None:
    """A row's sampling parameters: the executor passes them as a dict
    (make_distribution's kwargs, plus `seed`). Verification is rejection
    sampling against the target distribution, so the drafting loop needs
    the parameters, not a callable that has already collapsed them. None
    means greedy."""
    return sampler if isinstance(sampler, dict) else None



class LogitsTrunk:
    """A trunk seen the way the batch engine calls one: `input_embeddings`
    in, a logits ARRAY out.

    mlx-lm's text models already are that. glm5_next's LanguageModel (an
    mlx-vlm class, vendored) is not: it names the keyword `inputs_embeds`
    and returns a LanguageModelOutput. Found by running the tiny GLM
    through the serve path (tests/test_vision_e2e.py): admit's
    `logits[:, -1]` raised on the output object, and the image embeddings
    would have gone in under a name the trunk silently drops into
    **kwargs. Everything else (make_cache, layers, ...) passes through.

    `call`: what runs the forward, when not `model` itself. The artifact
    loads as glm5_next's multimodal `Model`, whose `__call__` takes no
    embeddings at all (it embeds `input_ids` itself, and an `inputs_embeds`
    falls into its **kwargs): the forward goes to its `language_model`,
    which takes them -- the same forward for text, and the image rows
    reach the trunk (found on GLM-5.3-Flash: images silently ignored, one
    Mac and split alike)."""

    def __init__(self, model, rename: bool, call=None):
        self.__dict__["_model"] = model
        self.__dict__["_rename"] = rename
        self.__dict__["_call"] = model if call is None else call

    def __call__(self, inputs, cache=None, **kw):
        if self._rename and "input_embeddings" in kw:
            kw["inputs_embeds"] = kw.pop("input_embeddings")
        out = self._call(inputs, cache=cache, **kw)
        return out if isinstance(out, mx.array) else out.logits

    def __getattr__(self, name):
        return getattr(self._model, name)


def logits_trunk(model):
    """`model` itself when it already speaks the batch engine's convention
    (every mlx-lm text model: the text path is untouched), else a
    LogitsTrunk around it."""
    import inspect
    try:
        sig = inspect.signature(model.__call__)
    except (TypeError, ValueError):
        return model
    names = sig.parameters
    call = model
    lm = getattr(model, "language_model", None)
    if "input_embeddings" not in names and "inputs_embeds" not in names \
            and lm is not None:
        try:
            lsig = inspect.signature(lm.__call__)
        except (TypeError, ValueError):
            lsig = None
        if lsig is not None and "inputs_embeds" in lsig.parameters:
            call, sig, names = lm, lsig, lsig.parameters
    rename = "input_embeddings" not in names and "inputs_embeds" in names
    ret = sig.return_annotation
    wraps_output = ret is not inspect.Signature.empty and ret is not mx.array \
        and "array" not in str(ret)
    if rename or wraps_output:
        return LogitsTrunk(model, rename, call)
    return model


class MTPBatchGenerator(BatchGenerator):
    """mlx-lm's BatchGenerator, drafting every row with an MTP head -- or,
    with `head=None`, the same engine without drafting, which is how a
    vision model with no head serves images.

    `vision` is the served model's `engine.vision.request.VisionServe` (or
    None): the family, the image store and the pins taken at tokenize."""

    def __init__(self, model, head, *, stats: dict | None = None,
                 vision=None, why: str = "", **kw):
        super().__init__(model, **kw)
        #: the control machine of a row admitted without one: end on the
        #: generator's stop tokens, as mlx-lm's own default did
        self._default_control = stop_machine(kw.get("stop_tokens"))
        self._stack = contextlib.ExitStack()
        self._head = head
        self._vision = vision
        if head is not None:
            spec = resolve(model)
            arch = spec.arch_module(model)
            core = getattr(getattr(model, "language_model", model), "model",
                           None)
            if core is None:
                raise RuntimeError(
                    f"{type(model).__name__} exposes neither `.model` nor "
                    f"`.language_model.model`; no capture point for the MTP "
                    f"head")
            if getattr(head, "block_size", 0):
                # a block head (DSpark) drafts from several layers' outputs
                # and says which; get_h hands it what it reads. On a
                # pipeline stage some may be another rank's (carried here)
                from knurlogic.engine.runtime import pipeline as PL
                if PL.run_of(core) is not None:
                    gets = self._stack.enter_context(PL.carry(
                        core, head.targets, head.capture_paths()))
                else:
                    gets = [self._stack.enter_context(capture_input(core, p))
                            for p in head.capture_paths()]

                def get_h():
                    return head.main_hidden([g() for g in gets])
            else:
                get_h = self._stack.enter_context(capture_input(core,
                                                                spec.capture))
            self._make_draft_cache = (
                head.make_draft_cache if hasattr(head, "make_draft_cache")
                else (lambda: spec.make_draft_cache(arch)))
            copy = spec.cache_semantics != "reassign"
            name = spec.name
        else:
            # No head: nothing to capture, nothing to seed. A snapshot is
            # only taken by a drafting step, which a headless batch never
            # takes, so the copy flag is moot; True is the safe reading.
            def get_h():
                return None
            self._make_draft_cache = lambda: None
            copy = True
            name = None
        #: what admit and the decode steps call (self.model stays the
        #: server's object: families embed with it, the server compares it)
        self._trunk = logits_trunk(model)
        batch = BlockBatch if getattr(head, "block_size", 0) else MTPBatch
        self._batch = batch(self._trunk, head, get_h, copy_caches=copy)
        self._batch.finish_at = self._finish_at
        self._n_trunk = len(make_prompt_cache(self.model))
        #: engine/runtime/pipeline.Coord on a pipeline split, else None
        self._coord = None
        #: wraps a row's prefill chunks (admit's prefill_ctx): a pipeline
        #: follower's overlapped sends (pipeline.silence), else None
        self._prefill_ctx = None
        # uid -> what the server sent for that row, and what it has seen.
        self._rows: dict = {}
        # uid -> [(key, entry)] checkpoints not yet reported to the server;
        # uid -> (entry, key) for the one reported in this next() call.
        self._ckpt_pending: dict = {}
        self._ckpt_ready: dict = {}
        # uid -> exception, for rows whose admission raised (see _failed).
        self._failed: dict = {}
        # uid -> the server's request object, for its cache report.
        self._requests: dict = {}
        self._stats = stats if stats is not None else {}
        if name:
            logger.info("batch engine drafting with the %s MTP head", name)
        else:
            # `why`: state.DRAFT's reason (no sidecar, MTP off, a pipeline
            # rank that could not bind one). "vision" only when this engine
            # serves images: the line once said it on a text-only ring.
            logger.info("batch engine without a drafting head%s%s",
                        " (vision model)" if vision is not None else "",
                        f": {why}" if why else "")

    def _finish_at(self, uid: int, toks: list[int]) -> int | None:
        """The index in `toks` (row `uid`'s next tokens) of the one that
        would finish it here (`_next`'s rules, its state left as it is),
        or None. The drafting loops commit nothing past it, so the entry
        stored for a finished row holds exactly its `fed` tokens."""
        st = self._rows.get(uid)
        if st is None:
            return None
        n, state = st["n"], st["state"]
        for j, t in enumerate(toks):
            n += 1
            if n >= st["max"]:
                return j
            state, match, cur = st["sm"].match(state, t)
            if match is not None and cur is None:
                return j
        return None

    def follow_block(self, block_size: int, outputs) -> None:
        """A split's follower of a rank 0 drafting with a block head
        (tensor.agree_head: its K and the layers whose outputs it reads):
        the block loop's steps with no head here, and on a pipeline stage
        those outputs carried on to rank 0 (pipeline.carry). Before
        silence / mirror_hidden / coordinate, which set the batch's parts."""
        from knurlogic.engine.runtime import pipeline as PL
        b = self._batch
        self._batch = BlockBatch(self._trunk, None, b.get_h,
                                 copy_caches=b.copy_caches,
                                 block_size=block_size)
        self._batch.finish_at = self._finish_at
        core = PL.core_of(self.model)
        if PL.run_of(core) is not None and outputs:
            self._stack.enter_context(PL.carry(core, outputs))

    def mirror_hidden(self) -> None:
        """A tensor follower of a drafting rank 0, holding no head: capture
        the trunk's final hidden state too. Rank 0 evaluates it after every
        prefill chunk of a drafting row, and under tensor it holds the last
        layer's all_sum; left lazy here, the ranks' collectives part."""
        spec = resolve(self.model)
        core = getattr(self.model, "language_model", self.model).model
        self._batch.get_h = self._stack.enter_context(
            capture_input(core, spec.capture))

    # ------------------------------------------------------------ admission

    def _admit_one(self) -> PromptProcessingBatch.Response | None:
        (uid, segments, max_tokens, cache, all_tokens, sampler, procs,
         sm) = self._unprocessed_sequences.popleft()
        prefix = list(all_tokens or [])
        rest = [t for seg in segments for t in seg]
        prompt = prefix + rest
        n = len(prompt)
        vis = None
        if K.has_image(prompt):
            vis = self._vision
            if vis is None:
                raise RuntimeError("an image key reached a batch generator "
                                   "with no vision family; the serve path "
                                   "is wired wrong")

        via = {"checkpoint": False}

        def replay(hc, h):
            via["checkpoint"] = True
            # The checkpoint's head stopped at c-1; its input there is
            # (h_{c-1}, x_c), and x_c is this prompt's token at c.
            x = K.to_ids(prompt[len(prefix):len(prefix) + 1],
                         vis.family.spec.image_token_id if vis else -1)
            with mx.stream(self._stream):
                self._head.advance(h, mx.array(x)[None], hc)
                mx.eval([a for a in (getattr(hc, "state", None) or [])
                         if isinstance(a, mx.array)])

        replay_fn = replay if self._head is not None and len(prefix) < n \
            else None
        cache, hcache, hit, drafts = self._split_entry(
            list(cache or []), prompt, len(prefix), vis, replay_fn)
        if len(prefix) > 0 and hit == 0:
            logger.info("prompt cache entry at %d/%d tokens has no aligned "
                        "head cache; prefilling from scratch", len(prefix), n)
        sampling = dict(sampling_of(sampler) or {})
        seed = sampling.pop("seed", None)
        # a ring's own seed does not pin drafting (Keys.pins)
        pins = not sampling.pop("ring_seed", False)
        params = RowParams(max_tokens=_NEVER,
                           dist=make_distribution(**sampling),
                           processors=list(procs or []), eos=set(),
                           drafts=drafts, mirror=self._head is None,
                           keys=Keys(seed, pins) if seed is not None else None)
        try:
            with mx.stream(self._stream):
                if vis is None:
                    ids, kw = mx.array(prompt), {}
                else:
                    ids, kw = self._vision_inputs(vis, prompt, hit)
                # Segment ends, as positions in the whole prompt. The last
                # segment's end is the prompt's end: stored when the row
                # finishes, not here.
                bounds, at = [], len(prefix)
                for seg in segments[:-1]:
                    at += len(seg)
                    bounds.append(at)
                stash = []

                def on_checkpoint(c, trunk, hc, h):
                    entry = copy.deepcopy(list(trunk))
                    if hc is not None:
                        entry += [copy.deepcopy(hc)]
                        # a block head's cache is at c already (admit)
                        if h is not None:
                            entry += [HeadCarry(mx.array(h))]
                    stash.append((list(prompt[:c]), entry))

                row = admit(self._trunk, self._head, self._batch.get_h,
                            ids, params, uid=uid,
                            make_draft_cache=self._make_draft_cache,
                            prefill_step_size=self.prefill_step_size,
                            prefill_ctx=self._prefill_ctx,
                            cache=cache or None, hcache=hcache, start_pos=hit,
                            checkpoints=bounds, on_checkpoint=on_checkpoint,
                            **kw)
                if stash:
                    self._ckpt_pending[uid] = stash
                self._report(uid, prompt, len(prefix), hit, n,
                             "checkpoint" if via["checkpoint"] else None,
                             len(stash), vis)
                self._batch.extend([row])
        finally:
            if vis is not None:
                # The features are in the KV now (admit evaluated every
                # chunk); the pin taken at tokenize has done its job.
                vis.release(K.images_in(prompt))
        self._rows[uid] = {"sm": sm, "state": sm.make_state(),
                           "max": max_tokens, "n": 0, "fed": list(prompt)}
        self._counters.prompt_tokens += n - hit
        self._stats["requests"] = self._stats.get("requests", 0) + 1
        return PromptProcessingBatch.Response(uid, (n, n), True, True)

    def _report(self, uid, prompt, offered, hit, n, via, n_ckpt, vis):
        """The cache report for this row's request (engine/cachereport)."""
        req = self._requests.pop(uid, None)
        if req is None:
            return
        spans = K.image_spans(prompt) if vis is not None else []
        imgs = spans
        # whole images only: one the cache boundary cuts was prefilled
        cached_imgs = [sp for sp in spans if sp.end <= hit]
        cachereport.attach(req, {
            "offered": offered, "used": hit, "discarded": offered - hit,
            "prefilled": n - hit,
            "via": via or ("prefix" if hit else "none"),
            "images": {"total": len(imgs), "in_cached_span": len(cached_imgs),
                       "prefilled": len(imgs) - len(cached_imgs),
                       "encoded": int(getattr(req, "_knurlogic_encoded", 0))},
            "checkpoints_stored": n_ckpt,
        })

    def _split_entry(self, entry: list, prompt: list, hit_len: int, vis,
                     replay):
        """(trunk, head cache, hit, drafts) for a row offered a prompt-cache
        entry at `hit_len`. On a pipeline whose rank 0 drafts, rank 0's
        (hit, drafts) are every rank's (Coord.ba): only rank 0 holds a head
        cache to align, so only it can say whether the entry is usable for
        a drafting row, and whether the row drafts decides its
        checkpoints. A follower's own answer is the trunk hit, which rank
        0's is either equal to or 0."""
        coord = self._coord if (self._coord is not None
                                and self._coord.head) else None
        try:
            if vis is None:
                # A model with no head has nothing to align: asking for a
                # head cache discards EVERY prefix hit on gemma (a
                # 509-token shared system prompt re-prefilled on each
                # request).
                drafts = self._head is not None
                trunk, hcache, hit = split_pool_entry(
                    entry, self._n_trunk, drafts=drafts, hit_len=hit_len,
                    replay=replay)
            else:
                trunk, hcache, hit, drafts = self._vision_entry(
                    entry, prompt, hit_len, replay=replay)
        # any admission failure is first reported to the other ranks, then re-raised
        except Exception:
            if coord is not None:
                try:
                    coord.ba(False, 0, False)
                except RuntimeError:
                    pass
            raise
        if coord is None:
            return trunk, hcache, hit, drafts
        hit0, drafts0 = coord.ba(True, hit, drafts)
        if not coord.leader:
            drafts = drafts0
            if hit0 != hit:
                if hit0:
                    raise RuntimeError(
                        f"rank 0 reuses {hit0} cached tokens; this rank's "
                        f"entry gives {hit}")
                trunk, hit = [], 0
        return trunk, hcache, hit, drafts

    def _vision_entry(self, entry: list, key: list, hit_len: int,
                      replay=None):
        """(trunk, head cache, hit, drafts) for a row whose key holds an
        image. No drafting if the uncached span holds an
        image. Otherwise it drafts only off an entry with an aligned head --
        and where there is none (every entry an image row stored, since
        image rows never seed the head) the row keeps the TRUNK hit and does
        not draft, rather than throwing the image's KV away to draft: the
        point is that the image is never prefilled twice (G8)."""
        fam = self._vision.family
        spans = fam.chunk_boundaries(key)
        if hit_len and any(s < hit_len < e for s, e in spans):
            # A hit cut inside a bidirectional block computed its first half
            # blind to its second; that KV is not reusable. The trie matches
            # the same image whole, so this is not expected -- kept loud.
            logger.warning("prefix hit at %d cuts an image block; "
                           "prefilling from scratch", hit_len)
            return [], None, 0, False
        img_new = K.has_image(key[hit_len:]) if hit_len else True
        if self._head is not None and not img_new:
            trunk, hc, hit = split_pool_entry(entry, self._n_trunk,
                                              drafts=True, hit_len=hit_len,
                                              replay=replay)
            if hit == hit_len and hc is not None:
                return trunk, hc, hit, True
        trunk, _, hit = split_pool_entry(entry, self._n_trunk, drafts=False,
                                         hit_len=hit_len)
        return trunk, None, hit, False

    def _vision_inputs(self, vis, key: list, hit: int):
        """(ids, admit kwargs) for a vision row: embeddings for the uncached
        span, position ids over the whole key (even when the new span
        is text only), the family's chunk boundaries."""
        fam = vis.family
        feats, refs = vis.lookup()
        ids = mx.array(K.to_ids(key, fam.spec.image_token_id))
        kw: dict = {"chunk_boundaries": fam.chunk_boundaries(key)}
        extras: dict = {}
        if K.has_image(key[hit:]):
            if self._coord is not None:
                # a split model: rank 0's rows reach every rank (only rank
                # 0 has a tower); every rank decides this the same way --
                # the key and the hit are rank 0's
                feats = self._coord.images(key[hit:], feats, refs)
            got = dict(fam.embed(self.model, key, hit, feats))
            kw["embeds"] = got.pop("input_embeddings")
            extras.update(got)
        pos, delta = fam.positions(key, refs)
        if pos is not None:
            extras["position_ids"] = (pos[..., hit:], -1)
            kw["mrope"] = True
        kw["rope_delta"] = int(delta)
        if extras:
            kw["extras"] = extras
        return ids, kw

    # ----------------------------------------------------------------- step

    def _entry(self, i: int) -> list:
        """Row i's caches for the prompt cache: trunk, plus the head's when
        it sits at the same offset (otherwise a restore could not draft)."""
        trunk = [c.extract(i) for c in self._batch.cache]
        h = self._batch.hcache
        # A row that never drafted (an image row) never advanced
        # its head cache, so it has no head to store -- and in a batch where
        # NO row drafted the batched head cache holds no keys at all, which
        # `extract` does not survive (found by G10).
        if (h is not None and hasattr(h, "extract")
                and self._batch.drafts[i]):
            head = h.extract(i)
            # Compare against a cache that HAS a position. On a hybrid model
            # the first layers are recurrent (ArraysCache, no offset), and
            # comparing against one of those never matches -- which stored
            # every entry without its head and made every restore prefill
            # from scratch, silently.
            pos = trunk_offset(trunk)
            if pos is not None and position(head) == pos:
                return trunk + [head]
        return trunk

    def insert_segments(self, segments, max_tokens=None, caches=None,
                        all_tokens=None, samplers=None,
                        logits_processors=None, stop_sequences=None, *,
                        reports=None, control=None):
        """`reports`: one object per row to receive its cache report (the
        executor passes them; engine/serve/cache_report.attach).
        `control`: one runtime/control.ControlMachine per row (None: the
        generator's stop tokens). It rides in the queue entry's last slot,
        which mlx-lm names stop_sequences: this generator never hands rows
        to mlx-lm's GenerationBatch, so the slot is only read back here."""
        # mlx-lm's own insert() passes stop_sequences (None) positionally
        control = control or stop_sequences or (
            [self._default_control] * len(segments))
        # mlx-lm 0.32 refuses max_tokens 0; here a row ends on its first
        # token at 0 and at 1 alike (_next's `n >= max`), and the API
        # accepts 0, so 0 is sent as 1
        if max_tokens is not None:
            max_tokens = [max(1, m) for m in max_tokens]
        uids = super().insert_segments(segments, max_tokens, caches,
                                       all_tokens, samplers,
                                       logits_processors, control)
        for u, r in zip(uids, reports or []):
            if r is not None:
                self._requests[u] = r
        return uids

    def _report_checkpoint(self) -> list[PromptProcessingBatch.Response]:
        """One stored checkpoint per row per call, oldest first: the server
        collects end-of-segment caches with ONE extract_cache per call,
        keyed by uid, and labels each with the next of that row's segment
        types -- two in one call would lose one and mislabel the other."""
        self._ckpt_ready.clear()
        out = []
        for uid in list(self._ckpt_pending):
            if uid not in self._rows:          # finished or removed
                del self._ckpt_pending[uid]
                continue
            key, entry = self._ckpt_pending[uid].pop(0)
            if not self._ckpt_pending[uid]:
                del self._ckpt_pending[uid]
            self._ckpt_ready[uid] = (entry, key)
            n = len(self._rows[uid]["fed"])
            out.append(PromptProcessingBatch.Response(uid, (n, n), True,
                                                      False))
        return out

    def _failed_responses(self) -> list[PromptProcessingBatch.Response]:
        """A row whose admission raised must fail ITS request, not the
        engine: the exception goes out once as that row's progress (the
        executor turns it into a RowFailure and removes the row); a plain
        progress tuple follows on later calls until the row is removed."""
        out = []
        for uid, err in list(self._failed.items()):
            out.append(PromptProcessingBatch.Response(
                uid, err if err is not None else (0, 0), False, False))
            self._failed[uid] = None
        return out

    def _next(self):
        prompt_responses = self._failed_responses() + self._report_checkpoint()
        if (self._unprocessed_sequences
                and len(self._batch) < self.completion_batch_size):
            # One admission per call, so rows already decoding are not held
            # for a queue of prefills.
            uid = self._unprocessed_sequences[0][0]
            try:
                admitted = self._admit_one()
            # one request's failure goes to that request; the generation thread lives
            # (logged)
            except Exception as e:
                if self._coord is not None and isinstance(e, ForwardFailed):
                    # A rank that fails mid-forward never sends its half: the
                    # other ranks wait on it forever (a pipeline load sat at
                    # 100% while rank 1 had raised). The rank dies instead, so
                    # the job stops and its card says why (recovery). A
                    # failure after the forward keeps the ranks in step.
                    logger.exception("admission of request %s failed on a "
                                     "cluster rank; stopping this rank", uid)
                    logging.shutdown()
                    os._exit(1)
                logger.exception("admission of request %s failed; failing "
                                 "that request only", uid)
                self._rows.pop(uid, None)
                self._ckpt_pending.pop(uid, None)
                self._failed[uid] = e
                self._requests.pop(uid, None)
                prompt_responses += self._failed_responses()
                return prompt_responses, []
            finally:
                if self._coord is not None:
                    # B0, after every admission attempt (a failed one too:
                    # the ranks make the same broadcasts; rank 0's plan
                    # removes the row the follower still holds): the
                    # admitted row's first token is rank 0's (the plan for
                    # this step went out before the admission sampled it)
                    self._batch.t1 = self._coord.b0(self._batch.t1)
            prompt_responses.append(admitted)
            # This row's first checkpoint goes out with its admission.
            uid = prompt_responses[-1].uid
            if uid in self._ckpt_pending and uid not in self._ckpt_ready:
                key, entry = self._ckpt_pending[uid].pop(0)
                if not self._ckpt_pending[uid]:
                    del self._ckpt_pending[uid]
                self._ckpt_ready[uid] = (entry, key)
                n = len(self._rows[uid]["fed"])
                prompt_responses.insert(-1, PromptProcessingBatch.Response(
                    uid, (n, n), True, False))
            if self._coord is not None and self._coord.diverged:
                # another rank failed this admission: no decode this call
                return prompt_responses, []
        if not len(self._batch):
            return prompt_responses, []

        tic = time.perf_counter()
        try:
            with mx.stream(self._stream):
                row_steps = self._batch.step()
        # a failed decode step fails its rows; the generation thread lives (logged)
        except Exception as e:
            # Same rule as a failed admission: the rows in this step fail
            # their own requests; the generation thread lives on.
            logger.exception("a decode step failed; failing its %d rows",
                             len(self._batch))
            uids = list(self._batch.uids)
            self._batch.remove(uids)
            for u in uids:
                self._rows.pop(u, None)
                self._ckpt_pending.pop(u, None)
                self._failed[u] = e
            if self._coord is not None:
                # the same broadcast count as a failed admission's
                self._batch.t1 = self._coord.b0(self._batch.t1)
            return prompt_responses + self._failed_responses(), []

        # NaN guard: a row whose logits went non-finite this step is failed
        # before any of this step's tokens leave (sampled, they are token 0
        # forever -- "!!!!!"). The flags were computed inside the step's own
        # eval (batch_loop), so this costs no sync.
        bad = {}
        for rs in row_steps:
            for em in rs.tokens:
                uid = rs.uid
                if em.finite or uid in bad:
                    continue
                st = self._rows.get(uid)
                bad[uid] = NonFiniteLogits(nonfinite_message(
                    (st["n"] if st else 0) + 1, "batch decode"))
        if bad:
            logger.error("non-finite logits in %d row(s); failing only those "
                         "requests", len(bad))
            self._batch.remove(list(bad))
            for u, err in bad.items():
                self._rows.pop(u, None)
                self._ckpt_pending.pop(u, None)
                self._failed[u] = err
            prompt_responses += self._failed_responses()
            row_steps = [rs for rs in row_steps if rs.uid not in bad]

        self._counters.decode_time += time.perf_counter() - tic
        out: list[TokenResponse] = []
        finished = []
        for rs in row_steps:
            st = self._rows.get(rs.uid)
            if st is None:
                continue
            for em in rs.tokens:
                st["fed"].append(em.token)
            for em in rs.tokens:
                st["n"] += 1
                finish = None
                if st["n"] >= st["max"]:
                    finish = "length"
                st["state"], match, cur = st["sm"].match(st["state"], em.token)
                if match is not None and cur is None:
                    finish = "stop"
                r32 = em.logits.astype(mx.float32)
                lp = r32 - mx.logsumexp(r32, axis=-1, keepdims=True)
                out.append(TokenResponse(
                    uid=rs.uid, token=em.token, logprobs=lp,
                    finish_reason=finish, prompt_cache=None, all_tokens=None,
                    current_state=cur, match_sequence=match))
                if finish is not None:
                    finished.append((rs.uid, len(out) - 1, rs))
                    break                  # nothing after a finish, ever

        if finished:
            idx = {u: i for i, u in enumerate(self._batch.uids)}
            for uid, k, rs in finished:
                st = self._rows.pop(uid)
                out[k].prompt_cache = self._entry(idx[uid])
                out[k].all_tokens = st["fed"]
                # RowStep carries RUNNING totals for the row, so they are
                # counted once, when the row is done.
                self._stats["steps"] = self._stats.get("steps", 0) + rs.steps
                self._stats["accepted"] = (self._stats.get("accepted", 0)
                                           + rs.accepted)
            self._batch.remove([u for u, _, _ in finished])

        self._counters.generation_tokens += len(out)
        self._counters.generation_steps += 1
        if self._counters.generation_steps % 512 == 0:
            mx.clear_cache()
        return prompt_responses, out

    # ------------------------------------------------------ the rest of the
    # contract: the inherited versions look in _generation_batch, which this
    # never fills, so rows in the drafting batch are answered here.

    def extract_cache(self, uids):
        # An end-of-segment request for a row this call reported a
        # checkpoint for gets that checkpoint, not the row's current state.
        ready = {u: self._ckpt_ready.pop(u) for u in list(uids)
                 if u in self._ckpt_ready}
        uids = [u for u in uids if u not in ready]
        mine = {u: i for i, u in enumerate(self._batch.uids) if u in set(uids)}
        out = super().extract_cache([u for u in uids if u not in mine])
        for u, i in mine.items():
            out[u] = (self._entry(i), list(self._rows[u]["fed"]))
        out.update(ready)
        return out

    def remove(self, uids, return_prompt_caches=False):
        for u in uids:
            self._failed.pop(u, None)
            self._requests.pop(u, None)
        caches = self.extract_cache(uids) if return_prompt_caches else {}
        if self._vision is not None:
            # A row dropped before admission still holds its tokenize pin.
            drop = set(uids)
            for seq in self._unprocessed_sequences:
                if seq[0] in drop:
                    key = list(seq[4] or []) + [t for g in seq[1] for t in g]
                    if K.has_image(key):
                        self._vision.release(K.images_in(key))
        super().remove([u for u in uids if u not in set(self._batch.uids)])
        self._batch.remove(uids)
        for u in uids:
            self._rows.pop(u, None)
            # a removed row's checkpoint copies (deep copies of its cache)
            # go with it, not at the next checkpoint report
            self._ckpt_pending.pop(u, None)
            self._ckpt_ready.pop(u, None)
        return caches

    def cost_per_token(self, rows: int):
        """Measured seconds per token at a batch width, or None."""
        return self._batch.cost_per_token(rows)

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        total += sum(c.nbytes for c in self._batch.cache)
        if self._batch.hcache is not None:
            total += getattr(self._batch.hcache, "nbytes", 0)
        return total

    def close(self):
        # rows still queued hold their tokenize pins; remove() drops them
        # (an executor closed after a failed step would leak them into a
        # store that outlives it)
        queued = [s[0] for s in self._unprocessed_sequences]
        if queued:
            self.remove(queued)
        self._batch.filter([])
        self._stack.close()
        super().close()
