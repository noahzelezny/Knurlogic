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
  on pipeline; or the split is refused with the reason.
- Stopping a cluster job mid-load, mid-warm-up and mid-generation leaves
  every GPU at baseline.

## 8. Live proof before release
- One capped run on the real weights per feature: text, a tool call,
  each thinking level, an image question with a known answer, drafting
  speed, a split.
- The changelog says what works and what is refused.
