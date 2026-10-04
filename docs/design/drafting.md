# Multi-token-prediction drafting

## src/knurlogic/engine/families/glm5/heads/glm5.py

Upstream GLM-5.3-Flash ships its MTP head as a PLAIN layer one past the
trunk (`layers.45` on Flash: eh_proj/enorm/hnorm glue, a NoPE-MLA
sparse-attention block with the DSA indexer, a 288-expert MoE, and its own
`shared_head.norm`). It is extracted with `vqlab mtp-extract`
(--key-regex '\.layers\.45\.'); the module only LOADS the packed q6 sidecar
(`mtp-head-q6.safetensors`, 889 tensors).

Two structural facts, both read off the graft's key set, make this head
NOT a trunk `Glm5NextDecoderLayer`:

  - NO hyper-connection weights. The trunk's layers are hc layers; the MTP
    layer is a plain-residual DeepSeek-style block. Instantiating the trunk
    layer class would leave randomly-initialized hc modules in the path, so
    the module assembles the block from the attention and MoE sub-modules
    and runs the residual wiring itself, at (B, T, D) -- no hc broadcast,
    no mean-collapse.
  - Its OWN final norm (`shared_head.norm`), applied before the trunk's
    shared lm_head -- the DeepSeek MTP convention.

The trunk is NoPE (qk_rope_head_dim=0): there are no rotary positions to
align, so the head-cache offset question the qwen4_exp head solves does not
exist here. Cache offsets still gate the attention mask, so the
committed-alignment scheme (one head row per COMMITTED token) is kept for
the mask math and for parity with the other heads.

The `model` bound is the mlx_vlm glm5_next LanguageModel (the object with
`.model` = Glm5NextModel and `.args.model_type == "glm5_next"`); when
holding the full VLM wrapper, pass its `.language_model`. `arch` is the
module those classes live in (mlx_vlm.models.glm5_next.language), per the
registry contract.

Measured in vqlab on a single machine (an M4 Max), not on a cluster:
acceptance 0.8516 pooled over 12 prompts x 128 tokens (q6 head, 2.7bpw
trunk) and 1.05x end-to-end WITHOUT the absorbed-MLA shim. Neither number
has been measured on a cluster.

## src/knurlogic/engine/families/qwen/heads/qwen35.py

Structurally far simpler than the qwen4_exp head: this architecture has a
single residual stream, so there are no hyper-connections and no per-stream
norm statistics, and the head's transformer block is a stock `DecoderLayer`
that owns its own rope.

Wiring, from the checkpoint's own key set:

    h     -> RMSNorm(D)  (pre_fc_norm_hidden)     trunk hidden at position t
    e     -> RMSNorm(D)  (pre_fc_norm_embedding)  embedding of token t+1
    concat -> fc  [D, 2D]  -> D
          -> ONE full-attention DecoderLayer (its own MoE, if the trunk is MoE)
          -> RMSNorm(D)  (the head's own final `norm`)
          -> the shared lm_head

THREE details are not determined by the key set, and each one is a silent
zero-acceptance failure if guessed wrong. All three are therefore flags,
and `vqlab mtp-probe35` sweeps them rather than trusting a guess:

  norm_shift  This family stores RMSNorm gains as DELTAS: mlx-lm's
              `TextModel.sanitize` adds 1.0 to every norm it recognizes, but
              only when the checkpoint still carries unsanitized conv1d
              weights, and it drops every `mtp.*` key before it gets there.
              So a hand-loaded head owns the convention. On the source 397B
              checkpoint conv1d.weight is [12288, 1, 4], i.e. unsanitized,
              so the trunk norms ARE shifted and the head's must be too.
              This agrees with the dense 27B measurement (0.0000 -> 0.7285
              with the shift): with the shift the 27B head drafts at 0.6562,
              without it at exactly 0.0000 in all four other wirings.
              Default 1.0.

  fc_order    The checkpoint fuses the two input projections into ONE
              `fc.weight` of [D, 2D], so unlike qwen4_exp (separate
              fc_embedding / fc_hidden tensors) the concat order is not
              recoverable from the file. Measured on the dense 27B (512
              positions): "eh" = [embedding | hidden] gives 0.6562
              acceptance, "he" gives 0.0020 -- i.e. chance. Default "eh".

  h_source    Whether the head consumes the trunk's hidden state before or
              after the trunk's own final norm. Measured: pre_norm 0.6562 vs
              post_norm 0.6582 over 512 positions -- a ONE-token difference,
              so the experiment does not resolve this flag and probably
              cannot: `pre_fc_norm_hidden` is applied immediately after, and
              an RMSNorm of an already-normed vector is close to idempotent.
              Default "pre_norm", which is what the head carrying its own
              hidden norm argues for.

Head precision cannot affect output quality -- the trunk verifies every
drafted token, so a coarser head costs a rejection, never a wrong token.

## src/knurlogic/engine/mtp/__init__.py

Package layout:

  _artifacts.py       what an artifact HAS: a head, graft weights (stdlib)
  registry.py         model_type -> head spec, built from the family
                      manifests (engine/families/); load_head binds one
  batch_loop.py       requests in one batch (MTPBatch, admit): prefill,
                      seed the head, draft and verify
  batch_generator.py  mlx-lm's BatchGenerator contract over batch_loop,
                      incl. segment checkpoints and the cache report
  caches.py           snapshot and rollback for a speculative step
  capture.py seed.py sampling.py   the pieces those share

Drafting across machines comes with the cluster executor
(engine/runtime/executor.py, docs/design/server.md "Cluster readiness").

WHOSE CODE THIS IS. mlx-lm has no MTP path at all, so an artifact's drafting head is weight
nobody else runs. Keeping one copy here, rather than several that drift
apart, is the point of the package.

## src/knurlogic/engine/mtp/_artifacts.py

A scan of 54 local artifacts, over two independent channels -- config.json
for the declaration, the safetensors HEADERS for the tensors, neither
reading the other:

    declares mtp in config .................... 40
    ships upstream `mtp.*` graft weights ....... 1   (the 806 GB bf16 397B)
    ships a BUILT head beside the weights ..... 11   (mtp-head-q6.safetensors)
    declares one and has NOTHING ............... 31

(40 = 31 + 8 declaring rungs with a built head + the 1 graftable one. The
other 3 built heads sit beside GLM rungs whose config never declared one,
so the declaration is not even a reliable signal of the head's absence.)

The reading "40 artifacts of downloaded MTP weights that mlx-lm's
`sanitize()` throws away" is therefore wrong, and wrong in the direction
that matters. The weights are not downloaded: the MLX converters drop
`mtp.*` at CONVERSION, so a rung arrives already without a head while still
carrying the upstream config that declares one. `sanitize()` discarding
those keys is very nearly a no-op on real artifacts.

So MTP here is not "stop discarding what you have". It is "the head is a
separate artifact" -- grafted once from the one checkpoint that carries it,
quantized, and written beside each rung as a sidecar.

THE SIDECAR IS DELIBERATELY OUTSIDE `model*.safetensors`. A loader discovers
weights by that glob, so a file named `mtp-head-q6.safetensors` costs
nothing until something asks for it -- and an index-only scan does not see
it at all. Read the FILES, not the index.

## src/knurlogic/engine/mtp/batch_generator.py

Ported from the maintainer's own code in github.com/noahzelezny/exo
(`generator/mtp_batch_generate.py`, Apache-2.0); it contains no code from
upstream exo.

The engine underneath -- `admit` prefills a row and seeds the head,
`MTPBatch.step` advances every row with a verified draft -- rests on three
decisions:

  - capture is installed for the generator's LIFETIME, not per request;
    `close` removes it.
  - the head's cache rides in the prefix-cache entry BESIDE the trunk's
    (entry = trunk caches + [head cache]). mlx-lm's prompt cache trims every
    cache in an entry by the same amount, so a restored prefix hands back a
    head exactly as far along as the trunk; `split_pool_entry` refuses one
    that is not, and the row prefills from scratch instead of drafting off
    the wrong history.
  - the row count moves in lockstep, so rejection is one whole-batch trim
    and replay (see batch_loop's docstring for what that costs and when).

The outside contract is the executor's: `insert_segments / next / remove /
extract_cache / close / prompt_cache_nbytes`. So the class SUBCLASSES
mlx-lm's BatchGenerator: queueing, uid allocation and the wired-limit
handling are inherited unchanged, and only where a row is prefilled and
where tokens come from are replaced.

Two things the consumer allows, checked in the consumer's loop rather than
assumed:

  - `next()` may return more than one generation response for a uid. The
    server feeds each to that request's detokenizer in order, and only a
    response AFTER a finish would be an error. A drafting step commits two
    tokens, so both go out on the same call rather than one being held back.
  - `all_tokens` on a finished response must be exactly the tokens whose KV
    is in the returned cache -- the prompt cache is keyed by it. Every token
    a step commits was fed through the trunk, including one a stop sequence
    then hid, so the list is prompt + everything committed.

IMAGES (docs/design/vision.md; on a split model, engine/runtime/tensor.py:
rank 0 encodes and ships the rows, every rank embeds with its own family).
Every request with an image comes here, head or no head (`head=None` is a
plain batch engine with the same admission), because only `admit` snaps
prefill chunks to the family's image spans. The prompt the server hands
over is the cache KEY (engine/vision/key.py): ids with a sentinel per image
token. `_admit_one` turns it back into ids for the trunk, asks the family
for the embeddings of the uncached span and for positions over the whole
key, and keeps the key -- not the ids -- as the row's `all_tokens`, so the
prefix cache is keyed by it. A row whose uncached span holds an image does
not draft: the head would be seeded from embed_tokens(pad) where the trunk
saw image features. Text-only keys take the plain path.

Cost, stated: a row is prefilled whole inside one `next()` call (other rows
wait for it).

Segment checkpoints: the server splits a chat prompt into segments (system,
user, the assistant header's thinking tail) and stores a prompt cache at
each segment end. Prefill stops at those ends (`admit`'s checkpoints) and
the snapshots are handed to the server one per `next()` call as
end-of-segment responses, which it stores through `extract_cache`. For a
model whose caches cannot be trimmed this is the only reuse a new turn gets
when the template re-renders the previous assistant turn (Qwen3.6 drops the
empty think block it generated with).

## src/knurlogic/engine/mtp/batch_loop.py

The drafting step -- draft one token with the head, verify it inside a
2-token trunk forward, roll back and replay on rejection -- runs over B rows
at once, which lets the batch engine keep drafting when several requests are
in flight instead of trading the head away for batching.

What makes it batchable at all is a property of the 1-token draft: EVERY
row advances by exactly two positions per step, accepted or not.

  accepted -> the verify forward committed [t1, d2]; the caches are right.
  rejected -> the caches hold [t1, d2] with the wrong second token; the
              trunk's own t2 came out of the same forward, so the step still
              emits two tokens, but position offset-1 has to be rewritten.

Because the row count in the KV cache is lockstep, the batched caches
(mlx-lm's BatchKVCache, one offset per row, shared write index) never need
per-row trimming. Rejection is handled as ONE whole-batch trim(2) and ONE
whole-batch replay forward with the tokens that actually committed -- for a
row that accepted, the replay writes back exactly what was there. That
costs an extra forward whenever ANY row rejects, i.e. with acceptance a and
B rows the expected forwards per step are 1 + (1 - a^B) for 2B tokens,
against B tokens per forward without drafting. It is a decaying win, not a
free one: measure it before assuming it beats plain batching at a given B.

A row that cannot draft (a request carrying images, whose head seeding the
vision embedding would corrupt) rides along: it pays the replay every step
and gets the trunk's own tokens, never a drafted one.

Sampling is per row -- temperature, top-p, processors, and the acceptance
test are the row's own -- so a batch may mix greedy and sampled requests.
At temperature the verdict is exact rejection sampling
(sampling.rejection_correct); at temperature 0 it is `draft == argmax`. A
row's logits processors (repetition penalty, eos ban) are applied to the
TRUNK row that verifies the draft, not only to the draft and to t1 --
otherwise a penalised token could be committed through the verify path that
sampling would have refused.

Across machines (a pipeline split, engine/runtime/pipeline.py) every rank
runs this same loop; `coord` carries rank 0's regime and drafts (B1) and its
verdicts (B2) to the others, one fixed broadcast each per step. Only rank 0
holds the head: a follower's batch has `head=None` and mirrors rank 0's
drafting rows (RowParams.mirror), running the verify and replay forwards
with the tokens B1 and B2 hand it.

A row that finishes mid-step (its t1 is an end token, or its last under
max_tokens) must not leave a position past it in its caches: the entry
stored for it is keyed by the tokens it streamed, and neither the
DeepSeek-V4 cache nor a linear-attention one can be trimmed back. So a
step commits nothing past the token that ends a row (`MTPBatch._ends`:
the loop's own rules and the batch engine's `finish_at`, its stop
sequences and max_tokens, read without moving its state). In this loop
that is t1 alone: the trunk is restored and steps t1 as a plain step
does, every row's t2 becoming its next token. Rank 0 decides it and B2
carries it.

## src/knurlogic/engine/mtp/block_loop.py

A head that drafts K tokens in one pass (deepseek_v4's DSpark) does not
fit the 1-token loop's invariant that every row advances two positions.
BlockBatch keeps the invariant that matters -- every row of the batch
advances by the SAME count -- by committing m + 1 tokens per row, m the
fewest drafts any row accepted. A row that accepted more keeps its token
at position m (its accepted draft: already a sample of the target), so
nothing is resampled and the verdicts stay exact; one row loses nothing.
When m < K the trunk rolls back to the m + 1 committed tokens by its own
cache bookkeeping (DeepSeek-V4 edit 19, `caches.rollback`), with no second
forward; a replay is only the fallback for a cache that cannot roll back.
A step verifies only the first k of its K drafts, k chosen from the head's
confidence (or measured per-position acceptance) over each width's timed
cost (`block_loop._width`; KNURLOGIC_MTP_VERIFY=k fixes it).
Two diagnostic switches, read at serve start: KNURLOGIC_MTP_VERIFY=k
verifies the first k drafts every step (a whole number, else the server
refuses to start); KNURLOGIC_MTP_PROFILE=1 times a DSpark step's phases
behind eval barriers and logs their means and the width's picks every 32
steps (slower: for measurement only).
The head's cache takes only committed positions, from the forward that
committed them, so it never rolls back. Plain steps (when drafting does
not pay) advance it one position. m is also capped at the token that
ends any row (`_ends`, as above), so a finished row's caches end at its
last streamed token.

On a split every rank runs BlockBatch; only rank 0 holds the head. B1
(`Coord.bk`) carries the regime and the [B, K] drafts, B2 (`Coord.bm`)
the committed count m and every row's token at position m; a follower
never judges (a tensor follower evaluates its verify's logits before B2,
so the ranks' all_sums line up). On a pipeline the target layers'
outputs reach rank 0 through `pipeline.carry`
(docs/design/deepseek-vision.md, DSpark).

## src/knurlogic/engine/mtp/registry.py

A `FamilySpec` says four things, and every one is a place where
architectures genuinely differ:

  head             where the drafting head lives ("module:Class"). The class
                   must expose `from_sidecar(model, arch, path)` and
                   `draft_logits(h_row, next_ids, cache)`.
  capture          dotted path, relative to the trunk core, of the submodule
                   whose INPUT is the pre-lm_head activation the head drafts
                   from. There is no public mlx-lm hook for this, so that one
                   module is wrapped for the duration of the generation (see
                   capture.py) rather than monkeypatching the class.
  draft_cache      the attribute on the architecture module that builds the
                   head's own KV cache.
  cache_semantics  "reassign" or "copy" -- see caches.py. qwen4_exp reassigns
                   its recurrent cache slots rather than mutating them, which
                   makes snapshots free; that is an implementation accident
                   of that arch, NOT a contract, so new families start at
                   "copy" and only move to "reassign" once
                   `caches.check_snapshot_semantics` has been run on them.

Families that ship an MTP head upstream but are NOT registered, because
nothing in this repo can test them:

  deepseek_v3 DeepSeek's MTP module is a different shape again (its own
              embed/norm/head rather than a shared lm_head).

glm5_next is registered: mlx-lm has no glm5_next class, but the trunk runs
under mlx_vlm's class, so the head binds to THAT arch module.

Registering another family means writing its head module and running
`caches.check_snapshot_semantics` plus the acceptance probe first. A table
entry without a measured acceptance number is not evidence of anything.
