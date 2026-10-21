"""Thinking: auto -- the model thinks at its top level and stops when a
probe on its hidden state says the answer is ready.

A probe is a logistic reader of one decoder layer's residual stream at a
paragraph break in the thinking: "would an answer forced here be right".
One probe serves every quant of a base model (measured: AUC within 0.002
across VQ-3.4/3.8/5.4 and 8-bit); its threshold is the effort knob.
Design and measurements: docs/design/auto-effort.md.

  store.py     a probe file: weights, normalisation, layer, threshold; where
               it lives (one per base model) and the model's auto state
  tap.py       read one layer's output during decoding; score a batch
  stop.py      the logits processor that closes the think block
  fit.py       train a probe from states and labels; pick the threshold
  problems.py  the onboarding set: problems whose answers are computed
  binding.py   bind the probe, or onboarding's tap, to the loaded model
  shipped/     probes fitted before release, one per base model measured

ACTIVE is the process's bound probe and tap (`store.ACTIVE`), reached by
attribute at call time as engine/mtp/binding.DRAFT is. Onboarding drives
the server as a client would, so it lives with the server
(interfaces/http/onboard.py). Importing this package imports no mlx; tap,
stop and binding do.
"""
