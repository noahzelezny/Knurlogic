# Auto effort: the model thinks until it has its answer

Status: design, not built. **Thinking: auto** is a probe on the model's
hidden state that closes the think block when the model has its answer;
its threshold is the effort knob (the probe sections below, 2026-10-19).
It learns per model from served traffic, then switches itself on.

How it got here: started 2026-10-18 as the model rating the request's
difficulty before answering; the runs turned that into "default low,
escalate on demand" (the model asks for more thinking after trying), and
level testing paused when no level improved what a model can do enough to
chase. The sections below keep that record in order.

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

## The shape (first design: superseded by escalation, below)

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

## What the runs found (2026-10-18/19)

A level is prompt text: the template adds a line asking for more or less
reasoning, and the model was trained to reason longer when it says so.
Nothing else changes -- the same compute per token, the same model. More
effort means longer reasoning by habit, which a model can also spend
talking itself out of a right answer.

- **Qwen3.8-Flash-Next VQ-2.1bpw**, 28 mixed problems, 2 samples, every
  level: `low` 56/56 at the fewest tokens (978 mean); `medium` 50/56,
  `xhigh` 54/56 at 2081. Off writes its working into the answer (1417).
- Same model, 13 harder problems, `low` vs `xhigh`, 2 samples: at a 16k
  cap `low` 20/26, `xhigh` 16/26 -- all 10 of `xhigh`'s misses ran out of
  room. Rerun at 64k: `low` 22/26, `xhigh` 23/26, `xhigh` at 3-5x the
  tokens. No verbatim loops (Chuck's `ReasoningLoopMonitor` over every
  long run); the runs still open at 64k drifted, re-deriving the same
  partial results ("Wait" 98-143 times).
- **GLM-5.3-Flash VQ-2.7bpw** (both Macs), asked before answering, with
  thinking off, how hard each of 20 problems is: "high" for 19, "low" for
  one (a subtle code bug), "max" never. A rating that is nearly always the
  same word says nothing; and it cost 5-11 s on the split, not the
  0.3-0.5 s estimated above for one Mac. The model judges difficulty
  poorly before it has tried.

So: no fixed cap (it cuts long productive thinking; a loop guard in the
harness is the stop), the lowest level that still lets the model think
as long as it needs, and more effort only where it was measured to fix
answers -- on these models about one problem in 26.

## Escalation: `think_harder`, the model's call after trying

The model answers at `low`. When its own attempt leaves it unsure, it
calls a tool -- `think_harder(reason)` -- and the harness reruns the turn
at the dialect's top level with the first attempt and the reason in the
context. A model that has its answer does not call it. The decision is
made after the attempt, when the difficulty is known, not before it, when
GLM rated nearly everything "high".

Cost: the level renders after the conversation (ab68ba09), so the rerun
reuses the cached conversation and redoes only the reply.

Where it lives: the tool is the harness's (Chuck offers it like any other
and reruns the step); knurlogic needs nothing new -- `reasoning_effort`
per request already picks the level, and the prompt cache already serves
the rerun.

**Test** (the problems and reference rows above, one model at a time):
1. Every problem at `low` with `think_harder` in the tools, 2 samples.
2. Score where it fires against the reference: calls on problems `low`
   gets wrong (good), on problems `low` gets right (wasted), and no call
   on a problem `low` gets wrong (missed).
3. The result is accuracy and total tokens for low+escalation against
   always-low, always-top and the oracle (each problem at its cheapest
   sufficient level).
Offered where it beats always-low on accuracy at a modest token cost;
always-low is the default it has to beat.

## A probe on the hidden state: stop when the model has its answer (2026-10-19)

A level asks the model to think longer; nothing tells it to stop. The
hidden state carries "the working answer is right" before the thinking
ends, and a linear probe reads it with one dot product.

**Experiment** (Qwen3.8-Flash-Next VQ-2.1bpw, the 41 problems above with
known answers; the scripts were one-off and are not in the repo -- the
method below is enough to redo it):
1. traces: every problem at `low` (2 samples, 82) and at `xhigh` (2
   samples, 82), full reasoning kept;
2. one pass per trace in-process: at up to 12 paragraph breaks, keep the
   hidden state of layers 12/24/32/40/47 at the break, and -- on a copy
   of the cache -- close the think block and let the model write only its
   answer, greedily. Right or wrong is the label ("would it be right if it
   stopped here?"). Matches the server's forced answers 26/26;
3. logistic regression per layer, trained on half the problems, scored on
   the other half (split by problem), 10-20 repeats.

**Results.**
- The signal is real and not position: on non-final points at `low`,
  probe AUC 0.878 (layer 32) against 0.517 for what a live server knows
  (thinking so far) and 0.832 for fraction-through (needs the end);
  0.72-0.74 inside the 50-80% band, where position gives 0.55. Layers
  24-40 read best.
- It transfers: trained on `low`, it scores `xhigh` points at AUC 0.90-
  0.91. But calibrated for `low` it stops `xhigh` late; a probe is trained
  (or calibrated) per level -- mixing levels made it worse.
- As a stop, at `xhigh` (80/82 right thinking to the end, 312k thinking
  tokens), with the first, undertrained probe: trained on `xhigh`, layer 40, stop at the first point
  over 0.8 -> **77/82 right at 36% of the thinking**. The probe-free rule
  "force an answer at each break, stop when it repeats 3 times" -> 77/82 at
  56%, and costs a forced answer per check.
- At `low` there is little to cut (the model stops soon after it has the
  answer): 76/82 kept at 96%.
- Where the thinking goes: by the median problem the forced answer is
  first right at ~790 chars at `low`, ~1390 at `xhigh` (1.7x later), and
  64% of `xhigh`'s thinking comes after the answer was already there (24%
  at `low`).
- The 3 answers the stop lost were the model confidently wrong, later
  caught by its own re-checking (1363 for 1365 at ~1.2k tokens, fixed by
  2.7k; 381 for 781 at 13k, fixed at 19k). The probe reads confidence, not
  truth; the long tail's value is these rare corrections.
- Against `low`: `xhigh` + probe stop is 77/82 at 113k tokens; `low` to
  the end is 76/82 at 65k. On this model `low` stays the better default;
  the probe pays where a model's own setting will not stop (GLM-5.3 at
  max).
- Properly fitted (scikit-learn logistic regression, standardized,
  C=0.01, 5 folds by problem; the hand-written trainer above was
  undertrained), the `xhigh` probe reads AUC 0.98-0.99 and stops better.
  First point over the threshold, layer 40 (to the end: 80/82 at 312k):
  0.99 -> 78/82 at 41%; **0.9 -> 77/82 at 30%** (94k tokens); 0.8 ->
  74/82 at 24%; 0.5 -> 72/82 at 21% (64k, about `low`'s tokens). The
  threshold is a smooth knob from "answer as soon as it believes it" to
  "keep checking".
- No label had to be guessed: the first principal component of the
  hidden state, found with no labels, separates right from wrong at AUC
  0.95-0.96. It moves with position (correlation 0.76) but beats it where
  the stop decides: 0.89-0.90 vs 0.82 in the first 30% of the thinking,
  0.87-0.89 vs 0.79 in 30-60%. "The model has settled on an answer" is
  the main thing the hidden state tracks while it thinks.
- Nonlinear readers add nothing: held-out AUC, linear 0.987 (layer 32),
  boosted trees 0.978, a small network 0.974; all five layers together
  0.986. One direction in one layer carries it.

**The effort knob this suggests** (Noah): not a level of text the model
was trained to think longer on, but the probe's threshold -- how sure the
model must be before knurlogic closes the think block. Low threshold:
answer as soon as it believes it has it; high: keep checking. The model
thinks as long as it needs and no longer, with no token cap; the knob is
a confidence, which a user can reason about, and its effect (tokens and
answers lost per threshold) is measured per model.

**A second probe: doubt** (Noah) -- "the answer will still change", where
confidence reads "it is right". Label: the forced answer here differs from
the trace's final one (24% of non-final points at `xhigh`). It reads very
well (AUC 0.98), but it is the confidence probe's mirror: their scores
correlate at -0.99, and 1 - confidence detects the same points (AUC 0.98).
Requiring low doubt on top of high confidence buys at most one answer at
9 more points of thinking (layer 40: 78/82 at 45%). On these traces the
model's hidden state carries one direction for "settled", not two. The
lost answers stay lost because the model itself was settled on a wrong
answer at that point -- no probe of its state can see past that.

**Learning it from use, not from a test set** (Noah, 2026-10-19). The
label need not be the true answer: "does the forced answer here match the
model's final answer" read as well as correctness (AUC 0.98, the same
direction), and the server sees every request's final answer. It is also
the right target for a stop -- one that fires when the answer already
matches what the model would end up saying gives the thinking-to-the-end
output minus the tail; it saves time, it cannot make a model more right.
- **An opt-in setting, "collect thinking traces"** (off by default; local
  only): while serving, keep the hidden state of one layer at paragraph
  breaks in the thinking (8-40 KB a point) and the request's final answer.
- **Idle-time labelling:** replay a sample of kept reasoning (prefill
  only, as the walker does) and force the answer at the breaks; the match
  with the final answer is the label.
- **Fit and calibrate per model:** a logistic regression, seconds; the
  threshold from how often a stop changes the answer; served only once a
  held-out match rate clears a bar. Before any labels, the no-label first
  component (0.95 above) may serve as a starting stop -- untested.
- **Agent turns** mostly end in a tool call, and tool calls compare
  exactly (the same function and arguments): "would the forced call here
  be the final one" is a clean label for most of Chuck's traffic. Prose
  answers stay hard to compare.
- **By task type:** a request says what it is -- the client's role
  (X-Client-Role), whether tools were sent, the kind of answer (a number,
  a tool call, prose) -- and the hidden states may cluster by task on
  their own. Probes and thresholds can be kept per task type, so a math
  calibration does not decide when an agent stops.
- **Volume:** both Macs served about 21,000 requests (4.7M output tokens)
  in the week to 2026-10-19, mostly Chuck; a small sample of a week or two
  is far more than the 164 traces that showed the signal here.
- **What a user sees** (Noah): one setting, **Thinking: auto** -- the
  model thinks until it has its answer, then stops. While a model is new,
  auto serves the model's own default and collects traces in the
  background; once it has enough it switches itself on. The page shows it
  per model: "auto: learning (120/300 traces)", then "auto: ready". A
  probe belongs to a model, and serves all its quants (measured below on
  the 35B: VQ-3.4 to 8-bit, drop <= 0.002); another size is another model.
  The learning is once per model.
- **Enough, then stop:** collection ends by itself once a model and task
  type has a few hundred traces (82 per level gave AUC 0.98 here), and
  resumes only when the model changes (a new quant, a template change) to
  check the probe still fits. Chuck's agent turns are the task type worth
  their own probe. No traffic is kept beyond that.
- **The work in the engine:** reading one layer's state at paragraph
  breaks inside batched, drafted decoding without slowing it, and
  storing it with the request.

**More models, and every quant of one** (2026-10-19 evening).
- Qwen3.6-35B-A3B VQ-4.6, thinking on, the same 41 problems x 2 (82
  traces, 940 points): probe AUC 0.95 at layer 26 of 40 (~65% depth;
  position 0.909); the stop at 0.9 keeps 80/82 right at 41% of the
  thinking (81/82 to the end). The signal is not one model's quirk.
- Every other quant of that model walked the same traces: VQ-3.4, 3.8,
  5.4 and mlx-community's 8-bit (affine, another method). The 4.6's probe
  reads each as well as that quant's own probe -- AUC drop at most 0.002
  (+- 0.002) at every layer, 3 seeds, held out by problem -- and its stop
  at 0.95 keeps 80-81/82 at 48-49% of the thinking on each. **One probe
  per model serves all its quants**: onboarding learns once, and a new
  quant (a 4-bit DWQ) needs no learning. Untested: each quant reading its
  own generated thinking (here all read the 4.6's text); with drops this
  small it is not expected to change the answer.
- (The walker hooks the decoder layer class: the 35B iterates
  `pipeline_layers`, a list rebuilt on every access.)
- A third family, Gemma 4 26B-A4B VQ-6.2 (thinking on; 41 traces, one
  sample each, all right): probe AUC 0.92 at layer 19 of 30 (position
  0.89); the stop at 0.95 keeps 39/41 at 33% of the thinking, 0.99 keeps
  40/41 at 51%.
- **The layer to read sits at about two-thirds of the depth** in all three
  families (Gemma 19/30, the 35B 26/40, the Flash 32-40/48): a default for
  onboarding a new model without a layer search.

**A state monitor from the same signal** (Noah, 2026-10-19): streamed with
the thinking, the confidence gives a harness a glance at each worker --
with four workers running, which is stuck. A face per state, named for
what is measured, not for a feeling:
- 🤔 working: confidence low, climbing;
- 😕 stuck: confidence low and swinging (the traces' dips and recoveries);
- 💡 got it: a jump to high (most traces had one clear jump);
- 😌 settled: high and steady -- where Thinking: auto stops;
- 😬 drifting: high, still going (the re-checking tail at `xhigh`);
- 😤 struggling: from the hidden state, not from words (Noah: "wait,
  wait" can be excitement). Labelled by behaviour -- will the forced
  answer change by the next break -- a probe reads it at AUC 0.97 on the
  Flash's `xhigh` traces, but it is mostly the confidence direction again
  (scores correlate -0.94; low confidence alone predicts it at 0.97). So
  struggling and drifting are patterns of confidence over time. One
  sliver may differ: among confident points the struggle probe still
  picked the rare answers about to change (AUC 0.93-0.95, but only ~8
  such points) -- "confident but about to change its mind", the stop's
  one weakness; worth a larger test.
The monitor is the harness's (Chuck); knurlogic streams the number.

**Next:**
- the same on GLM-5.3 at max (both Macs, when the cluster is free): its
  own traces, does the stop end the thinking that never finishes;
- the remaining single-Mac model (Qwen3.8-27B; the 35B is done, below):
  does the direction exist there too, and at what layer depth;
- probe training as a step of onboarding a new model
  (docs/design/new-model.md), once the stop is a feature;
- finer checks (every paragraph, not 12 per trace), and a probe that does
  not need answers with one checkable value (agent turns, prose) -- what
  it is trained on there is the open question;
- in the engine: the state reading above, and closing the think block
  when the probe fires (the GLM "off" path already closes one).

## Open

- Escalation from inside the thinking (the harness sees hedging and
  restarts higher) rather than a tool call: needs a signal for hedging;
  drift, not verbatim loops, is what the long runs showed.
- Whether the rating sees the tools and the whole history or a summary:
  the whole cached conversation is free, so the whole.
