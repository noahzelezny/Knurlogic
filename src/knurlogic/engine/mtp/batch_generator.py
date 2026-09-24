"""Drafting when mlx-lm's server batches: its BatchGenerator, backed by
batch_loop.MTPBatch instead of GenerationBatch.

Ported from the exo fork's `generator/mtp_batch_generate.py`. The engine
underneath is the same one -- `admit` prefills a row and seeds the head,
`MTPBatch.step` advances every row with a verified draft -- and so are the
three decisions that fork paid for:

  - capture is installed for the generator's LIFETIME, not per request;
    `close` removes it.
  - the head's cache rides in the prefix-cache entry BESIDE the trunk's
    (entry = trunk caches + [head cache]). mlx-lm's prompt cache trims every
    cache in an entry by the same amount, so a restored prefix hands back a
    head exactly as far along as the trunk; `split_pool_entry` refuses one
    that is not, and the row prefills from scratch instead of drafting off
    the wrong history.
  - the row count moves in lockstep, so rejection is one whole-batch trim and
    replay (see batch_loop's docstring for what that costs and when).

What differs is only the contract on the outside. exo's runner calls
`submit / step / cancel`; mlx-lm's server calls `insert_segments / next /
remove / extract_cache / close / prompt_cache_nbytes`. So this SUBCLASSES
mlx-lm's BatchGenerator: queueing, uid allocation and the wired-limit
handling are inherited unchanged, and only where a row is prefilled and
where tokens come from are replaced.

TWO THINGS THE CONSUMER ALLOWS THAT ARE WORTH SAYING OUT LOUD, because they
were checked in mlx-lm's server loop rather than assumed:

  - `next()` may return more than one generation response for a uid. The
    server feeds each to that request's detokenizer in order, and only a
    response AFTER a finish would be an error. A drafting step commits two
    tokens, so both go out on the same call rather than one being held back.
  - `all_tokens` on a finished response must be exactly the tokens whose KV
    is in the returned cache -- the prompt cache is keyed by it. Every token
    a step commits was fed through the trunk, including one a stop sequence
    then hid, so the list is prompt + everything committed.

IMAGES (design D5, D7 Phase A; docs/design/vision.md). Every request with
an image comes here, head or no head (`head=None` is a plain batch engine
with the same admission), because only `admit` snaps prefill chunks to the
family's image spans. The prompt the server hands over is the cache KEY
(engine/vision/key.py): ids with a sentinel per image token. `_admit_one`
turns it back into ids for the trunk, asks the family for the embeddings of
the uncached span and for positions over the whole key (D4), and keeps the
key -- not the ids -- as the row's `all_tokens`, so the prefix cache is
keyed by it. A row whose uncached span holds an image does not draft
(Phase A): the head would be seeded from embed_tokens(pad) where the trunk
saw image features, the defect exo has. Text-only keys take exactly the
path they took before this existed.

Costs, inherited and stated: a row is prefilled whole inside one `next()`
call (other rows wait for it, as they did in the fork).

Segment checkpoints: the server splits a chat prompt into segments (system,
user, the assistant header's thinking tail) and stores a prompt cache at
each segment end. Prefill stops at those ends (`admit`'s checkpoints) and
the snapshots are handed to the server one per `next()` call as
end-of-segment responses, which it stores through `extract_cache`. For a
model whose caches cannot be trimmed this is the only reuse a new turn gets
when the template re-renders the previous assistant turn (Qwen3.6 drops the
empty think block it generated with).
"""
from __future__ import annotations

import contextlib
import copy
import logging
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.generate import (BatchGenerator, GenerationBatch,
                             PromptProcessingBatch)

from .batch_loop import MTPBatch, RowParams, admit
from knurlogic.engine.serve import cache_report as cachereport
from .capture import capture_input
from .registry import resolve
from .sampling import make_distribution
from ..vision import key as K

logger = logging.getLogger(__name__)

#: A row this long is never ended by the loop itself. Finishing is decided
#: here, where the row's cache can still be extracted before it is dropped --
#: batch_loop removes a row it finishes before returning.
_NEVER = 1 << 62


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
    entry restored at `hit_len`. From the fork, unchanged.

    A drafting row needs a head cache sitting at exactly hit_len; an entry
    without one (stored by a non-drafting row, or before drafting was on) or
    with one elsewhere is unusable for drafting -> prefix length 0, fresh
    caches. A non-drafting row uses the trunk part of any entry."""
    trunk, extra = list(entry[:n_trunk]), list(entry[n_trunk:])
    if hit_len <= 0:
        return trunk, None, 0
    if not drafts:
        return trunk, None, hit_len
    off = int(getattr(extra[0], "offset", -1) or 0) if extra else -1
    if off == hit_len:
        return trunk, extra[0], hit_len
    # A checkpoint entry: head one step behind, with the h to replay it.
    if (off == hit_len - 1 and replay is not None and len(extra) > 1
            and isinstance(extra[1], HeadCarry)):
        replay(extra[0], extra[1].h)
        return trunk, extra[0], hit_len
    return [], None, 0


def trunk_offset(trunk: list) -> Optional[int]:
    """The position a trunk's caches sit at: the first one with an offset.
    Recurrent layers carry state, not a position, so they cannot answer."""
    for c in trunk:
        off = getattr(c, "offset", None)
        if off is not None:
            return int(off)
    return None


def sampling_of(sampler) -> Optional[dict]:
    """The parameters a sampler was BUILT from, if `tag_samplers` saw it.

    Verification is rejection sampling against the target distribution, so
    the drafting loop needs temperature and friends, not a callable that has
    already collapsed them. Untagged means greedy is the only safe reading
    of it -- and that is only right if the tag is always there, which
    `tag_samplers` guarantees for everything the server builds."""
    return getattr(sampler, "_knurlogic_sampling", None)


def tag_samplers(srv) -> None:
    """Wrap the server's `_make_sampler` so every sampler carries its args.

    The sequential path gets these one frame up, in `_serve_single`; the
    batch path never sees `args` at all -- `insert_segments` receives only
    the built sampler. So the sampler is where they have to travel."""
    if getattr(srv._make_sampler, "_knurlogic", False):
        return
    real = srv._make_sampler

    def _make_sampler(args, tokenizer):
        fn = real(args, tokenizer)
        s = args.sampling
        try:
            fn._knurlogic_sampling = dict(
                temp=s.temperature, top_p=s.top_p, top_k=s.top_k,
                min_p=s.min_p, xtc_probability=s.xtc_probability,
                xtc_threshold=s.xtc_threshold,
                xtc_special_tokens=[tokenizer.eos_token_id,
                                    *tokenizer.encode("\n")])
        except AttributeError:
            pass        # a builtin callable cannot carry it: greedy, see above
        return fn

    _make_sampler._knurlogic = True
    srv._make_sampler = _make_sampler


class LogitsTrunk:
    """A trunk seen the way the batch engine calls one: `input_embeddings`
    in, a logits ARRAY out.

    mlx-lm's text models already are that. glm5_next's LanguageModel (an
    mlx-vlm class, vendored) is not: it names the keyword `inputs_embeds`
    and returns a LanguageModelOutput. Found by running the tiny GLM
    through the serve path (tests/test_vision_e2e.py): admit's
    `logits[:, -1]` raised on the output object, and the image embeddings
    would have gone in under a name the trunk silently drops into
    **kwargs. Everything else (make_cache, layers, ...) passes through."""

    def __init__(self, model, rename: bool):
        self.__dict__["_model"] = model
        self.__dict__["_rename"] = rename

    def __call__(self, inputs, cache=None, **kw):
        if self._rename and "input_embeddings" in kw:
            kw["inputs_embeds"] = kw.pop("input_embeddings")
        out = self._model(inputs, cache=cache, **kw)
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
    rename = "input_embeddings" not in names and "inputs_embeds" in names
    ret = sig.return_annotation
    wraps_output = ret is not inspect.Signature.empty and ret is not mx.array \
        and "array" not in str(ret)
    return LogitsTrunk(model, rename) if (rename or wraps_output) else model


class MTPBatchGenerator(BatchGenerator):
    """mlx-lm's BatchGenerator, drafting every row with an MTP head -- or,
    with `head=None`, the same engine without drafting, which is how a
    vision model with no head serves images (design D5).

    `vision` is the served model's `engine.vision.request.VisionServe` (or
    None): the family, the image store and the pins taken at tokenize."""

    def __init__(self, model, head, *, stats: dict | None = None,
                 vision=None, **kw):
        super().__init__(model, **kw)
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
            get_h = lambda: None                          # noqa: E731
            self._make_draft_cache = lambda: None         # noqa: E731
            copy = True
            name = None
        #: what admit and the decode steps call (self.model stays the
        #: server's object: families embed with it, the server compares it)
        self._trunk = logits_trunk(model)
        self._batch = MTPBatch(self._trunk, head, get_h, copy_caches=copy)
        self._n_trunk = len(self._make_new_cache())
        # uid -> what the server gave us for that row, and what it has seen.
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
            logger.info("batch engine without a drafting head (vision)")

    # ------------------------------------------------------------ admission

    def _admit_one(self) -> Optional[PromptProcessingBatch.Response]:
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
        if vis is None:
            cache, hcache, hit = split_pool_entry(
                list(cache or []), self._n_trunk, drafts=True,
                hit_len=len(prefix), replay=replay_fn)
            drafts = True
        else:
            cache, hcache, hit, drafts = self._vision_entry(
                list(cache or []), prompt, len(prefix), replay=replay_fn)
        if len(prefix) > 0 and hit == 0:
            logger.info("prompt cache entry at %d/%d tokens has no aligned "
                        "head cache; prefilling from scratch", len(prefix), n)
        params = RowParams(max_tokens=_NEVER,
                           dist=make_distribution(**(sampling_of(sampler) or {})),
                           processors=list(procs or []), eos=set(),
                           drafts=drafts)
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
                        entry += [copy.deepcopy(hc), HeadCarry(mx.array(h))]
                    stash.append((list(prompt[:c]), entry))

                row = admit(self._trunk, self._head, self._batch.get_h,
                            ids, params, uid=uid,
                            make_draft_cache=self._make_draft_cache,
                            prefill_step_size=self.prefill_step_size,
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
        self._prompt_tokens_counter += n - hit
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

    def _vision_entry(self, entry: list, key: list, hit_len: int,
                      replay=None):
        """(trunk, head cache, hit, drafts) for a row whose key holds an
        image. Phase A (D7): no drafting if the uncached span holds an
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
        span, position ids over the whole key (D4 -- even when the new span
        is text only), the family's chunk boundaries (D5)."""
        fam = vis.family
        feats, refs = vis.lookup()
        ids = mx.array(K.to_ids(key, fam.spec.image_token_id))
        kw: dict = {"chunk_boundaries": fam.chunk_boundaries(key)}
        extras: dict = {}
        if K.has_image(key[hit:]):
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
        # A row that never drafted (an image row, Phase A) never advanced
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
            if pos is not None and getattr(head, "offset", None) == pos:
                return trunk + [head]
        return trunk

    def insert_segments(self, *a, **kw):
        uids = super().insert_segments(*a, **kw)
        req = cachereport.claim()
        if req is not None and len(uids) == 1:
            self._requests[uids[0]] = req
        return uids

    def _report_checkpoint(self) -> List[PromptProcessingBatch.Response]:
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

    def _failed_responses(self) -> List[PromptProcessingBatch.Response]:
        """A row whose admission raised must fail ITS request, not the
        server: an exception out of next() ends mlx-lm's generation thread,
        and that request and every later one then hang with no error. The
        server copies a prompt response's `progress` into the request's
        queue as is, and the reader raises any Exception it finds there --
        so the exception goes out once as progress. After that a plain
        progress tuple goes out on every call until the server, seeing the
        handler's ctx.stop(), removes the uid."""
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
            except Exception as e:
                logger.exception("admission of request %s failed; failing "
                                 "that request only", uid)
                self._rows.pop(uid, None)
                self._ckpt_pending.pop(uid, None)
                self._failed[uid] = e
                self._requests.pop(uid, None)
                prompt_responses += self._failed_responses()
                return prompt_responses, []
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
        if not len(self._batch):
            return prompt_responses, []

        try:
            with mx.stream(self._stream):
                row_steps = self._batch.step()
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
            return prompt_responses + self._failed_responses(), []

        out: List[GenerationBatch.Response] = []
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
                out.append(GenerationBatch.Response(
                    uid=rs.uid, token=em.token, logprobs=lp,
                    finish_reason=finish, current_state=cur,
                    match_sequence=match, prompt_cache=None, all_tokens=None))
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

        self._gen_tokens_counter += len(out)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
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
        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        total += sum(c.nbytes for c in self._batch.cache)
        if self._batch.hcache is not None:
            total += getattr(self._batch.hcache, "nbytes", 0)
        return total

    def close(self):
        self._batch.filter([])
        self._stack.close()
        super().close()
