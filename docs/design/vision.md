# Vision, owned: build design (v2, the one to build from)

*2026-09-23. v1 was produced by a read-only swarm (five mappers, a designer)
and attacked by an adversarial reviewer, who ruled it not ready: five
blockers. v1, the critique and the five source reports are kept in
`vision-evidence/` as evidence. This document supersedes v1 wherever they
differ; every blocker is resolved here, and two decisions the maintainer made after the
critique are folded in: knurlogic owns the VQ runtime, and the stack is
pinned.*

## Goal

The five released families -- qwen3_5, qwen3_5_moe, qwen4_exp (Flash-Next),
glm5_next (GLM-5.3), gemma4 -- serve text AND images from
`pip install knurlogic`, with code knurlogic owns rather than mlx-vlm, and
with **images as real context**: an image is encoded once per conversation,
the prefix cache survives it, and the image is never prefilled twice. (Turn
5 costing only turn 5's new tokens is the aim, and is measured; for
thinking-model templates it waits on the follow-up below.) The page gets a chat modelled on exo's.

Checked before designing: all 20 released rungs carry `vision_config` and
their vision weights (a 333-tensor `model-vision-graft.safetensors` sidecar
for the Qwen families and gemma 26b; inside the main shards for GLM (347
keys, `vision_model.*`) and gemma e4b (661)). mlx-vlm 0.6.17 implements all
five families (MIT) -- the source to vendor from.

## Decisions (each closes a v1 blocker or records a choice)

### D1. knurlogic owns the VQ runtime (the maintainer, 2026-09-23; closes B3)

Today every released rung ships its own `model.py`: 10 distinct runtimes
across 20 rungs, 4,229-5,194 lines each, and each deliberately builds a
TEXT-ONLY model when mlx-lm loads it. v1 tried to graft a vision tower onto
that and the critique showed it contradicts the bundles. Instead:

* **One VQ runtime in knurlogic** (`engine/vq/`), vendored from vqlab's
  canonical `src/vqlab/vq_switch.py` (4,453 lines, pinned by commit and
  digest), which vqlab also builds against. The runtime is family-aware and
  vision-aware; the bundled `model.py` stays in the HF uploads for plain
  mlx-lm users and knurlogic ignores it for any rung on its verified list.
* **Per-rung differences become settings, not code.** Measured: within a
  family most runtimes differ only by the DEFAULT of two numerics flags
  (Flash-Next 2.1 vs 4.4: 4 lines; GLM 2.7 vs 3.1: 4 lines). Those move to
  the `knobs` block of each rung's `config.json`, which
  `Artifact.declared_knobs()` already reads and ranks above everything
  else. Where runtimes differ by version (397B 2.2 vs 2.6: 713 lines, the
  newer carries the D4-walk kernel), the newest is the candidate superset --
  to be proven, not assumed.
* **Gate per released rung (G-VQ):** knurlogic's runtime and that rung's
  bundled runtime give the same logits on the same short prompt (atol
  1e-5; exact greedy tokens over 40), for all 10 distinct runtimes. A rung
  that fails stays on its bundled runtime and is listed, not silently served.
* **Where each rung's knobs come from (answered by the vqlab session,
  2026-09-23, checked against the PUBLISHED Hub model.py):** vendor vqlab
  HEAD (`42df84f`; last functional change `ef4e8dc`), and set every rung's
  knobs from that rung's PUBLISHED `model.py` (`hf download <repo>
  model.py`), never from `~/.exo` copies -- local copies have drifted from
  the Hub on several repos. The shipped artifact is the record of what IS
  shipped: knurlogic reproduces its flags as they are, not a plan's intent.
  Published reality, three generations:
    - v2 (both bf16-I/O on): Flash-Next 2.1 (per plan); **Qwen3.6-35B-A3B
      3.8 / 4.6 / 5.4 (drift: rebundled while v2 was the repo default,
      before `99ef3a1`; not a decision. Leave on v2 or rebundle to v1.5 is
      the maintainer's call -- knurlogic reproduces whatever is published.)**
    - v1.5 (both off): the rest of the flagged rungs.
    - arc6-era, no flags at all: GLM 3.1 / 3.6 on the Hub (the local v2
      copies were never published), GLM 2.7 to be re-read, 397B 2.4 / 2.6 /
      3.1 (republished 2026-09-22 with corrected weights, F168, model.py
      unchanged). 397B 2.2 was republished with a new 4684-line bundle --
      re-read its flags from the Hub.
  **Running an arc6-era rung on HEAD with knobs set to reproduce arc6 is a
  runtime change, not a no-op**: G-VQ (identity against that rung's
  published bundle) must pass before knurlogic serves it on HEAD.
* **Bug this closes, found while deciding:** `tuning/settings.py`
  `RUNTIME_PROFILES["v1.5"]` forces `VQ_GEMMSEG_BF16IO=0` and
  `VQ_DECODE_BF16IO=0` for every VQ artifact, overriding rungs whose shipped
  default is `1` -- published: Flash-Next 2.1 and 35B-A3B 3.8/4.6/5.4.
  These flags are numerics-active (F103/F105, up to +0.97% ppl). The fix: a
  rung's numerics come from the rung (declared knobs set from its PUBLISHED
  model.py, above); the profile applies only when a person asks for it.

### D2. Pin the stack (the maintainer; closes B4)

Exact versions of mlx, mlx-lm and the vendored VQ runtime, recorded in
`pyproject.toml` and in a test that fails on drift (versions plus a digest of
mlx-lm's `server.py`, because the seam wraps its methods). Every wrap
resolves methods by name with an assertion, never by line number. mlx-vlm
reference outputs for the identity gates are generated once as `.npz`
goldens in the exo interpreter (which has mlx-vlm 0.6.17) and committed, so
G1-G4 run anywhere without mlx-vlm installed.

### D3. All image work happens on the generator thread (closes B2)

v1 encoded images on the HTTP thread while generation runs on mlx-lm's
generator thread -- two threads on one GPU, and two uncoordinated
allocations on a shared host. Instead, wrap `ResponseGenerator._tokenize`
(generator thread; it has `request.messages`): pull image parts out, decode
and hash, look up the image store, encode on a miss, replace each part with
the family's single placeholder, call the real `_tokenize`, expand
placeholders to the image's token count, and return the cache KEY as the
prompt. No side table carrying images between threads. `_post` only refuses
images sent to a model without vision (HTTP 400).

### D4. Positions come from the whole key, every time (closes B1)

For Qwen (MRoPE), a text-only turn AFTER an image still needs positions
shifted by that image's `rope_delta`; v1 computed positions only when the
new suffix contained an image, which degrades silently -- fluent text,
subtly wrong grounding. So `positions(key)` is a pure function of the full
key and runs on every prefill and decode of any row whose key contains an
image. Text suffix prefill: `arange(hit, L) + rope_delta` on all three axes.
New gate G7b catches exactly this.

### D5. Vision requests go through the batch engine only (closes B5)

gemma4 attends bidirectionally within an image block, so a prefill chunk
must never split one; mlx-lm's single-request path chunks without a hook.
Every request with an image, seeded or not, goes through
`MTPBatchGenerator` (with `head=None` when there is no head), whose
`admit()` snaps chunk edges to `Family.chunk_boundaries(key)`.

### D6. The cache key

`key.expand()` turns token ids into a key the same length as the KV:
each image token becomes a sentinel `("img", sha, proc_hash, k)` --
`proc_hash` so a processor change can never hit a stale entry. mlx-lm's
prompt trie accepts any hashable. Two different images of the same size
diverge at the image's first token; the same image hits all the way through.
The image store is keyed `(model_key, sha, proc_hash)`, per image (exo's
whole-list key re-encoded everything when a second image was added),
byte-bounded LRU, and COUNTED in the memory budget (`tuning/resolve.py`) --
a GLM image can be ~65 MB of features. Decoded pixels are clamped (max
pixels, and PIL's decompression-bomb limit) before hashing. The segment
split mlx-lm does on prompts is expanded too.

### D7. Drafting and images

Phase A ships: a row whose uncached span contains an image does not draft;
text-only conversations are unchanged. Phase B (drafting on the text that
follows an image -- seeding the head from merged embeddings instead of
`embed_tokens(placeholder)`, the defect exo has) is its own package AFTER
integration, behind a flag until measured: G10 token identity and acceptance
at least 80% of the text-only rate on the 27B and 35B.

## Work packages

Separate git worktrees; file ownership is exclusive. Model split per the maintainer:
Opus for the packages where subtle correctness lives, Sonnet 5 for
well-specified vendoring and UI against frozen contracts, a review for the
final adversarial review.

| | Package | Model | Depends on |
|---|---|---|---|
| P0 | Contracts, key codec, image store, loadlock, pins, goldens, per-family tiny fixtures | Opus | -- |
| P-VQ | knurlogic's VQ runtime (D1) + the numerics-default fix + G-VQ per rung | Opus | P0 |
| P1 | Qwen vision (qwen3_5, qwen3_5_moe, qwen4_exp) + MRoPE threaded through the trunks (D4) | Opus | P0 |
| P2 | gemma4 vision (e4b, 26b) + image-block mask + chunk_boundaries | Sonnet 5 | P0 |
| P3 | glm5_next vision + the seven mlx-vlm siblings (mlx-vlm stops being a dependency) | Sonnet 5 | P0 |
| P4 | Serve path: `_tokenize` wrap (D3), batch-only vision (D5), embeds in admit, drafts off on image rows | Opus | P0 (builds against a stub family) |
| P5 | Interfaces: chat panel, vision in models/state/fit, Anthropic image blocks, gate tool | Sonnet 5 | P0 |
| -- | Integration: registry resolves real families; real-model gates, one model at a time | orchestrator | all |
| P6 | Phase B drafting on image conversations | Opus | integration |
| -- | Adversarial review of the whole | a review | all |

Ownership notes from the critique: only P4 touches `engine/seam.py` and
`engine/mtp/*`; only P1 touches the Qwen architecture files; P-VQ owns
`engine/vq/` and `tuning/settings.py`'s numerics section; each family
package writes its provenance to `engine/vision/<family>/PROVENANCE.md`
(no shared-file appends); tiny fixtures are one file per family, owned by
that family's package, with P0 providing the shared builder.

## Gates

Tiny random fixtures (safe in parallel, no model, no lock):

* **G1** vendored tower == mlx-vlm golden (atol 1e-5); **G2** grid and token
  count exact; **G3** positions exact; **G4** 40 greedy tokens identical to
  golden, including gemma with `prefill_step_size=16` so a chunk would split
  an image block; **G5** text path unchanged when no image is present --
  byte-identical HTTP output against main.
* **G6** tower called once across a two-turn conversation; **G7** warm turn 2
  == cold turn 2 (tokens, last-prefill logits atol 1e-4); **G7b** image in
  turn 1, text-only turn 2 warm == cold through the golden (fails if D4 is
  missing); **G9** a different image of the same size misses and answers
  differently; **G10** image prompt with drafting machinery == without;
  **G11** three mixed rows each == solo.
* Every gate is mutated once: break the thing it guards and confirm it goes
  red (the practice from the placement work, where one test only proved
  itself after it stopped reading the constant it tested).

Real models, serialized by the orchestrator, one at a time behind the
loadlock and `ready()`, unloaded in `finally`: gemma e4b VQ -> Qwen3.8-27B
3.9 -> Qwen3.6-35B-A3B 3.4 -> gemma 26b -> Flash-Next 2.1 (only if it fits).
GLM and the 397B get a tower-only local check; their full gate waits for
cluster serving. Per rung: vision tensor count bound; a text answer; an
image answer ("red"; OCR of "42"); a five-turn conversation where the tower
runs once, the cache covers at least through the end of the last image span
on turns 2-5, and turn 5 recalls the image; memory back to baseline after
unload.

**Reuse through turn 5 is a gate for knurlogic's own chat.** The v1 critique
said Qwen-style templates drop earlier thinking; the released templates say
otherwise (read 2026-09-23). Qwen3.8 / Flash-Next / 35B-A3B
`chat_template.jinja:116` keeps earlier reasoning unless
`preserve_thinking` is explicitly false; GLM-5.3 `chat_template.jinja:149`
keeps it unless `clear_thinking` is explicitly true (default false). So the
prefix is stable by default -- PROVIDED the client sends each earlier
assistant turn's `reasoning_content` back. Most OpenAI-style clients drop
it, the template then renders those turns without their thinking, and the
prefix diverges there. Therefore: knurlogic's chat always echoes
`reasoning_content`, and for it the gate is `prompt - cached <= new + 32` on
turns 2-5; for other clients the metric is reported and the behaviour
documented ("send reasoning_content back to keep reuse").

**gemma4 is the exception** (checked 2026-09-23 after the 397B's review
flagged it): its template renders earlier reasoning only for the latest
turn and only with tool calls (`chat_template.jinja:239`) and strips
thinking from earlier content (`:319`, `:327`), with no switch. So on gemma
the image is still never re-prefilled (it sits in an earlier user turn,
before the divergence), but each new turn re-prefills the previous answer,
whose thinking the model generated into the KV and the template then drops.
Echoing reasoning_content cannot help. For gemma the turn-N gate is "the
image is never re-prefilled"; full reuse needs the checkpoint below. The
checkpoint-per-user-turn is therefore the planned fix for gemma and the
fallback for clients that do not echo reasoning on Qwen and GLM.

## Review by Flash-Next 4.4 (local, 2026-09-23) -- folded in

The design was reviewed by Qwen3.8-Flash-Next-VQ-4.4bpw through exo's
endpoint. It ran its whole 6,000-token budget as reasoning and never wrote an
answer; the reasoning held these, checked against the design and critique
and new to both:

1. **Store eviction breaks "encode once".** A byte-bounded LRU can evict an
   image a live conversation still references, and turn 5 re-encodes it.
   The store PINS an image while any prompt-cache entry references its sha
   (refcount on insert/evict of cache entries); only unreferenced images
   are LRU-evictable. Image METADATA (grid, token count) is never evicted.
2. **Normalise before hashing.** EXIF orientation applied, mode converted
   (alpha composited on white, greyscale to RGB), first frame of animated
   formats; the hash is of the normalised pixels. Otherwise one photo can
   hash two ways, or two different renderings one way.
3. **The placeholder is a real special-token id.** Each family's placeholder
   is its tokenizer's own image token, verified present in the vocabulary
   at load; a string the tokenizer does not know is spelled out as text and
   the image silently never reaches the sequence. A test asserts it.
4. **Budget the KV, not only the features.** An image's KV across all layers
   exceeds its features; the prompt-cache bytes attributable to image spans
   are counted in `tuning/resolve.py` alongside the store.
5. **The goal overstated the gate** (fixed above).

### Second pass, through the harness with tools (same day)

Checked against the source before folding in:
* CONFIRMED: `fetch_nearest_cache` returns a slice of the key it was given
  (`mlx_lm/models/cache.py:1688,1692`), and the single path hands that slice
  to `stream_generate` (`server.py:965-980`). D5 already routes vision to the
  batch engine; mlx-lm sends SEEDED requests down the single path regardless
  (`_is_batchable` is false when a seed is set), so P4 must force the route,
  and a gate covers an exact cache hit with a remainder containing image
  tokens, through the real ResponseGenerator.
* CONFIRMED and it CORRECTS the v1 critique: the released Qwen templates
  keep earlier thinking by default (see the reuse gate above).
* NOT CONFIRMED at the pinned mlx-lm (0.31.3): a prompt-length quota that
  image tokens would eat -- `server.py` has no `prompt_len` or
  `max_total_tokens`; `max_tokens` bounds output only. Re-check if the pin
  moves.
* REFUTED: "GLM rewrites history every turn". `chat_template.jinja:149` keeps
  reasoning unless `clear_thinking` is true, and it defaults false.

The rest of the second pass (findings 3-9), executed against the files:
* REFUTED (#4) "GLM has no image token in its vocabulary": tokenizer.json
  registers `<|image|>` = 154854 (and begin/end_of_image 154830/154831);
  config.json `image_token_id` = 154854; engine/families/glm5/vision reads it. Flash
  had marked this inferred from absence.
* REFUTED (#3, again) "GLM drops earlier thinking": chat_template.jinja:149
  keeps reasoning unless clear_thinking; `<think></think>` is the else.
* REFUTED by execution (#1): P4 ran tuple sentinels through mlx-lm's real
  server objects without failure; vision is forced to the batch path and
  the seeded case is gated (E6).
* REFUTED (#6): the key carries one sentinel per image TOKEN, so key length
  always equals KV length; there is no single-position placeholder.
* REFUTED (#7) as corruption; kept as a cheap invariant: encoding the same
  image twice yields identical features (tower determinism) -- a gate.
* ADOPTED (#8): a per-request cache report {prompt_tokens, cached,
  expanded image spans, tower encodes} in the response's usage, so the cache
  gates can be checked from outside; the chat's token meter reads it.
* Already addressed: #5 (processor config is fixed at load; a change is a
  key miss by construction), #9 (goal restated).

### Third review: Qwen3.5-397B-A17B-VQ-2.2bpw through the harness (same day)

Placed on the Laptop B through knurlogic's own MCP. It confirmed the
model-level claims (image tokens: Qwen 248056, GLM 154854; templates keep
thinking on Qwen and GLM; vision weights on all released rungs), made no
false claims, and said plainly what it could not read (the harness's read roots
exclude site-packages, so it could not open mlx-lm's server). Its critical
item -- tuple sentinels in mlx-lm -- was already settled by execution (P4 and
the end-to-end tests ran them through the real server objects on all five
families). `proc_hash` and the pins exist in code and contracts it was not
pointed at. Its one new finding, the unchecked gemma template, was real and
is above. Calibration across the three local passes: Flash found more and
claimed more, including wrong claims from absence; the 397B found less and
claimed nothing it had not read.

Process note for the code review: Flash reasons at length -- give it a
larger budget or disable thinking for review passes.

## Follow-up (after the release)

* **Thinking-model reuse.** Default: follow the model's template (earlier
  thinking dropped) and store a cache checkpoint at the end of each USER
  message, so a new turn reuses everything up to there and re-prefills only
  the previous answer without its thinking -- behaviour unchanged. Opt-in
  "keep reasoning" setting (the maintainer's "thinking max"): earlier thinking stays in
  context, more context used, behaviour changes; measure before claiming
  it helps long agent tasks.
* **Local reviewers.** Flash-Next 2.1 (fits the M3) and the 397B (needs both
  nodes, so exo's instance comes off the Laptop B for it) review the
  build through knurlogic's own Anthropic endpoint, beside a review. A
  different model lineage, and dogfooding the endpoint at long context;
  their findings are leads to verify, not verdicts. After the real-model
  gates, so they never compete for memory.

## Risks, ranked

1. MRoPE threaded into mlx-lm-derived trunks (P1), and per-row `rope_delta`
   in batched decode -- new code with no reference. G3, G4, G7b guard it.
2. Sentinels reaching mlx-lm code that assumes ints (stats, detokenizer,
   cache-key append). Guarded by running G6-G9 through the real server
   objects; fallback is negative-int sentinels.
3. The VQ runtime superset claim (D1) -- proven per rung by G-VQ or not
   claimed.
4. Recurrent-state families take only exact or shorter prefix hits; an
   edited turn re-prefills from the nearest snapshot. Acceptable; visible.
5. GLM's reference: 0.7.1 is only knurlogic's vendored copy and differs from
   0.6.17 by design (`_limited_swiglu`); G1 for GLM documents the delta.
