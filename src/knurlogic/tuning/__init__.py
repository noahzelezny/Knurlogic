"""tuning/ -- every setting a model runs with: what it is, what it should
be, and why.

  measured.py        every measured constant beside the run that set it:
                     decode and prefill chunks, the fit reserve, cache
                     limits, low headroom, KV element size and which KV
                     precisions a family takes, the vision allowance
  numerics.py        a VQ model's numerics flags, profiles and sources
  presets.py         the launch presets (the tune axis) and their rows
  knobs.py           the knob registry: doc, help, titles, aliases, tiers,
                     ranges, bounds, and the readers of a set value
  groups.py          knurlogic-wide groups read per request: compaction and
                     the prompt cache on disk
  context_window.py  a model's trained window, YaRN past it, the KV room
  checks.py          every refusal: one knob, a request's knobs, a set on
                     an artifact, a whole launch (and its fit)
  live.py            which knobs a running server can change
  fit.py             the memory arithmetic: KV per token, margins, room,
                     vision and MTP bytes, the single-Mac fit check
  resolve.py         an artifact + a memory budget -> the environment and
                     argv to run it with, and a note for every decision
  tensor_split.py    the tensor split's arithmetic and refusals
  pipeline_split.py  the pipeline split's arithmetic and refusals
  rank_order.py      which machine leads a ring, and the ring's order
  preferences.py     the saved knurlogic-wide settings (settings.json)
  strategy.py        the machine's default preset (strategy.json)

A value here without its evidence is a bug. Depends on machine/ (what an
artifact is, the load budget) and engine/ (how configs spell a family);
imports no mlx and never imports interfaces/ or cluster/.
"""
