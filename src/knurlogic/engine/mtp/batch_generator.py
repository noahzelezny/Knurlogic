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

Costs, inherited and stated: a row is prefilled whole inside one `next()`
call (other rows wait for it, as they did in the fork), and there is no
end-of-segment cache insert mid-prompt -- the finished row's entry is what
goes to the prompt cache.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.generate import (BatchGenerator, GenerationBatch,
                             PromptProcessingBatch)

from .batch_loop import MTPBatch, RowParams, admit
from .capture import capture_input
from .registry import resolve
from .sampling import make_distribution

logger = logging.getLogger(__name__)

#: A row this long is never ended by the loop itself. Finishing is decided
#: here, where the row's cache can still be extracted before it is dropped --
#: batch_loop removes a row it finishes before returning.
_NEVER = 1 << 62


def split_pool_entry(entry: list, n_trunk: int, *, drafts: bool,
                     hit_len: int) -> tuple:
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
    if extra and int(getattr(extra[0], "offset", -1) or 0) == hit_len:
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


class MTPBatchGenerator(BatchGenerator):
    """mlx-lm's BatchGenerator, drafting every row with an MTP head."""

    def __init__(self, model, head, *, stats: dict | None = None, **kw):
        super().__init__(model, **kw)
        spec = resolve(model)
        arch = spec.arch_module(model)
        core = getattr(getattr(model, "language_model", model), "model", None)
        if core is None:
            raise RuntimeError(
                f"{type(model).__name__} exposes neither `.model` nor "
                f"`.language_model.model`; no capture point for the MTP head")
        self._stack = contextlib.ExitStack()
        get_h = self._stack.enter_context(capture_input(core, spec.capture))
        self._head = head
        self._make_draft_cache = (head.make_draft_cache
                                  if hasattr(head, "make_draft_cache")
                                  else (lambda: spec.make_draft_cache(arch)))
        self._batch = MTPBatch(model, head, get_h,
                               copy_caches=spec.cache_semantics != "reassign")
        self._n_trunk = len(self._make_new_cache())
        # uid -> what the server gave us for that row, and what it has seen.
        self._rows: dict = {}
        self._stats = stats if stats is not None else {}
        logger.info("batch engine drafting with the %s MTP head", spec.name)

    # ------------------------------------------------------------ admission

    def _admit_one(self) -> Optional[PromptProcessingBatch.Response]:
        (uid, segments, max_tokens, cache, all_tokens, sampler, procs,
         sm) = self._unprocessed_sequences.popleft()
        prefix = list(all_tokens or [])
        rest = [t for seg in segments for t in seg]
        prompt = prefix + rest
        n = len(prompt)

        cache, hcache, hit = split_pool_entry(
            list(cache or []), self._n_trunk, drafts=True, hit_len=len(prefix))
        if len(prefix) > 0 and hit == 0:
            logger.info("prompt cache entry at %d/%d tokens has no aligned "
                        "head cache; prefilling from scratch", len(prefix), n)
        params = RowParams(max_tokens=_NEVER,
                           dist=make_distribution(**(sampling_of(sampler) or {})),
                           processors=list(procs or []), eos=set(), drafts=True)
        with mx.stream(self._stream):
            row = admit(self.model, self._head, self._batch.get_h,
                        mx.array(prompt), params, uid=uid,
                        make_draft_cache=self._make_draft_cache,
                        prefill_step_size=self.prefill_step_size,
                        cache=cache or None, hcache=hcache, start_pos=hit)
            self._batch.extend([row])
        self._rows[uid] = {"sm": sm, "state": sm.make_state(),
                           "max": max_tokens, "n": 0, "fed": list(prompt)}
        self._prompt_tokens_counter += n - hit
        self._stats["requests"] = self._stats.get("requests", 0) + 1
        return PromptProcessingBatch.Response(uid, (n, n), True, True)

    # ----------------------------------------------------------------- step

    def _entry(self, i: int) -> list:
        """Row i's caches for the prompt cache: trunk, plus the head's when
        it sits at the same offset (otherwise a restore could not draft)."""
        trunk = [c.extract(i) for c in self._batch.cache]
        h = self._batch.hcache
        if h is not None and hasattr(h, "extract"):
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

    def _next(self):
        prompt_responses = []
        if (self._unprocessed_sequences
                and len(self._batch) < self.completion_batch_size):
            # One admission per call, so rows already decoding are not held
            # for a queue of prefills.
            prompt_responses.append(self._admit_one())
        if not len(self._batch):
            return prompt_responses, []

        with mx.stream(self._stream):
            row_steps = self._batch.step()

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
        mine = {u: i for i, u in enumerate(self._batch.uids) if u in set(uids)}
        out = super().extract_cache([u for u in uids if u not in mine])
        for u, i in mine.items():
            out[u] = (self._entry(i), list(self._rows[u]["fed"]))
        return out

    def remove(self, uids, return_prompt_caches=False):
        caches = self.extract_cache(uids) if return_prompt_caches else {}
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
