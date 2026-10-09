# Supporting a new model: the definition of done

A model is supported when everything its maker ships works in knurlogic,
not when its text runs. Each line below is either done (with its proof) or
refused with a reason the page and the MCP show. Nothing is silently
missing. Where a family lives in code: families.md.

Start from the maker's own reference (their inference code, encoder,
config, model card), not a third-party conversion: a conversion may have
dropped parts (DeepSeek-V4-Flash-Vision-Exp's teacher copy had no tower,
no bias_vl, no MTP layers).

## 1. Inventory (before any code)
- Every weight group in the maker's checkpoint, and which code reads it
  (trunk, tower, MTP/draft head, extra routing biases, learned special
  rows). An unread group is a feature missing, not noise.
- Every config field the reference reads (e.g. `rms_norm_eps`, window
  sizes, `vision_*`, draft block size), and what reads it here.
- The maker's prompt encoder and its options: thinking modes and effort
  levels, tool format, special roles, image/audio placeholders.
- A design note in docs/design/<model>.md listing all of the above and the
  phases.

## 2. Weights and loading
- Loads from the maker's own release and from the usual MLX conversions
  of it, including one that ships no chat template.
- `sanitize` keeps every key a feature needs (no rename that breaks a
  later reader); a strict load proves nothing is left over.
- Fits are computed with every part counted: trunk, draft head, tower and
  image store, KV at the default context. Each optional part (MTP,
  vision) can be turned off at launch and says what it costs.

## 3. Chat
- The maker's template, or knurlogic's port of their encoder, renders
  byte-equal to the maker's encoder on their own examples (golden test).
- Tool calls parse in the maker's format, streamed and not, on OpenAI,
  Anthropic and Ollama endpoints; a reply that is only a tool call has no
  stray text.
- Special roles and system reminders the maker defines (date, language)
  are supported.

## 4. Thinking
- Every thinking mode and effort level the maker defines is offered, in
  the maker's names, with the maker's default. Not a subset, not a
  remapping that hides one (Vision-Exp has four; Flash has three).
- The dialect is detected from the template text, and a sibling model's
  template is never detected as this one (test both directions).
- The maker's recommended sampling per mode, when they publish it.

## 5. Images (and other modalities) when the maker ships them
- The tower, projector/aligner and processor ported to MLX, held to the
  maker's reference on the real weights (max abs diff stated).
- The processor's resize, token counts and layout equal the reference on
  several aspect ratios, including the limits.
- Whatever the trunk does differently for image tokens (routing biases,
  attention inside an image, positions) is implemented and tested against
  the reference functions.
- Prefill never splits what the reference processes whole.
- The VISION tag shows only for an artifact whose config has a tower.

## 6. Drafting
- The maker's MTP / draft head runs, exact and quantized, with its
  acceptance rate and speedup measured on one Mac.
- A seeded request reproduces with drafting on.

## 7. Clusters
- Pipeline and tensor splits, each with drafting on and off, and images
  on both; or the split is refused with the reason. A family the tensor
  split does not know yet is a missing feature, listed, not a scope cut.
- Stopping a cluster job mid-load, mid-warm-up and mid-generation leaves
  every GPU at baseline.

## 7b. Cache saving
- Every cache kind the family makes saves and restores bit for bit
  (tests/engine/test_prompt_disk.py): the next tokens' logits from a
  restored cache equal the original's.
- The family's spec names each cache kind's tensor-split axis (KV heads,
  value heads, replicated), so a saved entry restores under any pipeline
  or tensor split (prompt-cache-shards.md). A kind without one is listed
  here as missing, not dropped.
- Live: save a session, reload under a different split, and it hits.

## 8. Computing what the maker computes

Every item here was found wrong in a family that "worked" (0.1.3 audit,
2026-10-03). A live answer that looks right proves nothing about these:
GLM read images correctly with its image frame missing, and DeepSeek
answered well while ~3% of tokens per layer went to other experts.

The audit: before release, a table per family of every item below, the
maker's value and ours, with file:line on both sides. Every mismatch is
fixed before the tag; "harmless for today's configs" is not a reason.

Config and constants
- An absent config key means what it means in the maker's config class,
  never our released-model value. (Flash-Next hashed with seed 0 against
  the reference's 1234: every token read wrong embedding rows. Qwen's
  output_gate_type, norm_topk_prob; Gemma's layer pattern, softcap,
  double-wide MLP, KV sharing defaults.)
- Read structure from the config (layer_types), not from a rule that
  happens to agree with it (full_attention_interval).
- A buffer the checkpoint stores (hash multipliers, tables) is loaded,
  not rebuilt from a formula.
- Every eps: its value, and what it is added to (sum vs mean of squares:
  Qwen's l2norm eps was 128x too large; inside vs outside a softmax
  denominator: DeepSeek's fused sinkhorn).
- Every clamp, on every path that has the layer: routed experts, shared
  expert, dense MLP, the draft head, the vision MLP (DeepSeek's shared
  expert and GLM's MLPs and MTP head were unclamped).

Precision: hold it in bf16, not only float32
- List every place the reference leaves the model dtype: router logits,
  final logits, pooling / compression, norms (where the weight multiplies
  and where it rounds), MoE combine, quantization rounding of activations
  (FP8 / FP4 act_quant sites and block sizes). DeepSeek's router scores in
  bf16 changed the chosen experts for ~3% of tokens per layer; its logits
  are float32 in the reference.
- Goldens run the maker's code in bf16 too, with a tolerance taken from
  the reference's own bf16 spread. Float32-only goldens hid every
  DeepSeek dtype mismatch.
- The fused kernels the live path takes are in the goldens (a float32
  test takes the fallback and never runs them).

Prompt
- Image placeholders reproduce exactly what the maker's template and
  processor put in the prompt, framing tokens included (GLM's
  <|begin_of_image|> / <|end_of_image|> were missing: our placeholder
  reaches the template as text and skips its image macro).
- Message parts are joined per role as the maker's encoder joins them
  (DeepSeek: "\n\n" for tool results on Flash too).
- The template variant is chosen from the artifact's config, never from
  its folder name.
- The maker's sampling per thinking mode, from the model card when
  generation_config carries only one set (Flash-Next non-thinking).

Images
- The processor's resize policy is the maker's (aspect kept, token
  budget, no upscaling) and its patch order matches the maker's patchify
  (GLM: wrong patch order read one circle as two ovals; stretched resize).
- Tested through the class the real artifact loads: GLM's multimodal
  Model.__call__ dropped the image embeddings into **kwargs, while the
  test rig built the bare language model.
- Image-specific trunk behaviour per layer type and per model (Gemma's
  bidirectional image mask: sliding layers only, only "vision" models).

Heads, memory, threads
- A draft head builds its layers from the trunk's own classes, so every
  edit reaches it.
- The KV estimate equals what the cache stores per token, measured on a
  tiny model (GLM stored its MLA latent twice).
- No lazy array is made at import or kept in a cache: the server
  evaluates on another thread (mlx 0.32: "There is no Stream(gpu, 0) in
  current thread"). tests/engine/test_thread_arrays.py checks it.

Goldens
- Built from the maker's code (their inference/ or transformers), not a
  third party's port (mlx-vlm), with the script and versions recorded in
  the file. Not run through our own sanitize or packing first.

Where it lives
- Every family fact (template variant, tool parser, part separator,
  sampling defaults, vision signature, split rules) is in
  families/<family>/, so the audit reads one folder.

## 9. Live proof before release
- One capped run on the real weights per feature: text, a tool call,
  each thinking level, an image question with a known answer, drafting
  speed, a split.
- The changelog says what works and what is refused.
