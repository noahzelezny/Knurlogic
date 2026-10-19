# Auto effort: the model sets its own thinking level

Status: design, not built (2026-10-18). Noah's direction: an `auto`
thinking level, with the model judging how hard the request is. knurlogic
builds the mechanism; harnesses (Chuck) own the outcome data a classifier
would later be trained on.

## Why now

A thinking level is text in the prompt. Until 2026-10-17 four of the five
dialects (DeepSeek-V4, Qwen3.8, GLM-5.3, Gemma 4) rendered it before the
system prompt, so a changed level re-prefilled the whole conversation (a
42k-token DeepSeek conversation: 0 cached, about 5 minutes). Choosing a
level per turn was therefore impractical. Since ab68ba09 every served
template renders the level after the conversation (families'
templates/thinking_moves.json, test_thinking's cache test), so a changed
level costs only the level's own few dozen tokens. Choosing the level per
turn is now cheap.

## The shape

`reasoning_effort: "auto"` is one more value on the request's ladder
(engine/model/thinking). With it, the server answers in two passes over
the same cached conversation:

1. **Rating.** The conversation plus a short instruction asking for one
   word -- the levels the served dialect offers (`levels(template)`), e.g.
   `low | high | max` -- generated at the lowest level, at most a few
   tokens, never shown to the client.
2. **Answer.** The conversation at the level the rating named, which is
   the ordinary request.

Both passes share the conversation's prefix, and the conversation
checkpoint at the messages' end (engine/runtime/prompt._segment) is where
both continue, so the second pass prefills only the level's text and the
reply header.

The response reports the level actually applied (as `thinking.applied`
does today) and the rating's raw answer, so a client sees what was chosen.
A rating that names no offered level falls back to the dialect's default
and says so.

Where it lives:
- the rating prompt per dialect: beside the dialect in the family
  manifest (`thinking`), so the words match the template's own levels;
- the two-pass flow: the HTTP layer that already translates the level
  (interfaces/http/openai.build_job and thinking.translate), so every API
  (/v1/chat/completions, /v1/messages, Ollama) gets it;
- the record: the request ledger (machine/ledger) gains the applied level,
  how it was chosen (asked, default, auto) and the rating's answer, beside
  the thinking length it already implies (output tokens).

The rating pass must not disturb the session's cache: it is not saved, and
under keep-latest it must not replace the session's entries -- it is the
same step as the answer, whose entries are what the session keeps.

## Latency

The extra cost of `auto` is the rating pass plus the level's text in the
answer pass. Measured on Qwen3.6-35B-A3B-VQ-4.6bpw on one M3 Ultra
(2026-10-18, the request ledger), with ~5k tokens of conversation cached:

| piece | tokens | measured |
|---|---|---|
| a step with 1 new prompt token | 1 | ~85 ms (the step's fixed cost) |
| a prefill of ~100 new tokens | 91-113 | 237-260 ms |
| decode | per token | ~15 ms (69 tok/s, one row) |

So, estimated, not yet measured as a whole:
- rating pass: ~50 tokens of instruction (~0.15-0.2 s) and ~3 tokens out
  (~0.05 s) plus admission (~0.05 s): **about 0.25-0.3 s**;
- answer pass: the level's text, 0-100 tokens, **up to ~0.25 s** more than
  the same request sent at that level directly.

About **0.3-0.5 s per turn**, against what a level changes: on this
model a `max`-like answer of ~5k tokens takes ~70 s to decode and a
`low` one of ~600 tokens ~9 s. One rating that saves one unneeded high
level pays for a hundred ratings. On a model that is slow to prefill
(a 397B, a pipeline split over two Macs) the fixed costs grow; the
rating's own tokens stay few.

What would make it slower: a rating pass that is not served from the
cache (a conversation its session never sent before pays the prefill
once, as the answer would have), and a rating answer that rambles (cap
it: a few tokens, stop at the first newline).

## What the A/B runs say about levels (2026-10-17/18)

`auto` only pays where the levels change the answer. Measured with
8 problems x 2 samples per family (each family's
templates/PROVENANCE.md):
- GLM-5.3-Flash: levels scale thinking cleanly (tokens per answer
  ~460 / ~3000-3400 / ~4500-5000 for low / high / max).
- Gemma 4: on or off, about 2x the tokens on.
- Qwen3.8-27B: weaker separation, noisy at this size.
- DeepSeek-V4-Flash-Vision-Exp VQ-3.2bpw: "high" thought *less* than
  no effort text at DeepSeek's own position; to be checked on the
  non-vision Flash before `auto` is offered there.

So `auto` should be offered per dialect, where a family's levels were
measured to differ.

## Data, and a classifier later

Whether models judge their own effort well is open; it is measured, not
assumed. knurlogic's ledger records each auto request's rating and applied
level; only the client knows the outcome (tests passed, a step redone, a
user correction), so the outcome data lives with the harness (Chuck), keyed
by the request id knurlogic returns. A classifier -- a small encoder or a
logistic model on embeddings, milliseconds on the CPU -- is trained on that
once there is enough of it across models, and plugs in at the same place:
it names the level instead of the rating pass, `auto` unchanged for the
client.

A first measurement before building: the same mixed-difficulty set (some
problems a low level gets right, some only a high one), the model's own
rating for each, scored against the cheapest level that gets it right.

## Measurement plan: can a model rate its own difficulty?

Started 2026-10-18 on Qwen3.8-27B; reviewed before the first run. The
level A/B runs (2026-10-17/18) set each level by hand and scored the
answers; no model was asked to choose. A level is prompt text, not a
budget: at `low` a model may still think as long as it likes (DeepSeek
Vision-Exp got all 8 of those problems at `low`, one at 21k tokens), so
whether the levels separate on a problem set is itself the first result.

The goal is the right answer in the fewest tokens for that model (one
model at a time, so tokens stand in for time).

1. **Problem set** (28, integer answers computed by script): easy
   arithmetic and facts; mid counting with famous problems' constants
   changed so the answer is not recall; hard multi-step counting that is
   still workable by reasoning; and code/ops tasks (a 40-line function run
   exactly as written, a short trace, a dependency config, a log where the
   culprit is not named).
2. **Reference, per model:** each problem at every level the dialect
   offers, 4-5 samples, through /v1/chat/completions at the model's own
   sampling. Rows record right/wrong, thinking and completion tokens,
   truncation (reported apart, not silently wrong). The cost compared is
   completion tokens, not thinking tokens: with thinking off the 27B wrote
   its working into the answer, 2-5x the tokens of `xhigh` on easy
   problems (first 34 rows, 2026-10-18).
   The **cheapest sufficient level** is the lowest level at which this and
   every higher level is right in most samples; "none" otherwise. First
   result: the distribution of cheapest levels. If too few problems land
   strictly between the lowest level and "none" (fewer than ~8), problems
   are replaced before any rating is run.
3. **Rating, per model:** the production path -- the problem's
   conversation with the rating instruction appended, thinking off where
   the dialect allows (else its lowest level), stop at a newline, a few
   tokens. How often it names no level is a result, not a retry. Two
   wordings, each also with the level list reversed; and a control with
   the problem replaced by a neutral sentence (the base rate). One
   problem placed after a long unrelated history, to see whether the
   rating follows the last turn. 3 samples; the first is scored as the
   deployed behaviour, agreement reported apart.
4. **Scores**, from the reference rows (no extra runs):
   - primary: accuracy and total thinking tokens for `auto` against
     always-lowest, always-default, a prompt-length rule, and the oracle
     (each problem at its cheapest sufficient level), with confidence
     intervals;
   - under-rating (below the cheapest: costs correctness) and over-rating
     (above it: costs tokens) counted apart; exact match secondary;
   - problems no level solves scored apart (does the model rate them
     highest?);
   - rating latency and tokens, against the estimate above.
5. **Order:** what one Mac holds first. Qwen3.8-27B (graded: off, low,
   medium, xhigh) to build and validate the experiment; Qwen3.6-35B-A3B
   and Gemma 4 as the on/off case (the rating is think or not);
   GLM-5.3-Flash when the cluster is free; the large models last, one
   capped confirmation each, DeepSeek after the non-vision Flash effort
   A/B shows its levels move at all.

Speed: Qwen3.8-27B-VQ-3.9bpw on one M3 Ultra decoded ~24 tok/s alone
and ~32 tok/s across 8 rows (2026-10-18), so the full reference is hours
there. Before scaling it: a faster engine for the reference (an MoE
model, or fewer samples where the easy end is already settled), and the
samples spent where the levels disagree.

`auto` is worth offering on a model where it beats the always-default
level on tokens without losing accuracy outside the intervals; a fixed
loss threshold waits until there are enough samples to resolve one.

## Open

- Mid-answer escalation (start low, restart higher when the thinking
  hedges): only the reply is redone, the conversation stays cached; not in
  the first version.
- Whether the rating sees the tools and the whole history or a summary:
  the whole cached conversation is free, so the whole.
