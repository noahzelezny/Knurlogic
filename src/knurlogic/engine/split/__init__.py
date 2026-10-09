"""One model served by N ranks in one ring (docs/builders/splits.md). The
ranks are processes of this engine; the machines they run on are
cluster/'s (no mlx there).

  tensor.py        the tensor split: every layer's weights cut N ways (shard)
  tensor_rules.py  which arrays are cut, on which axis (no mlx: tuning reads it)
  viability.py     an unknown layout run whole and split before the ring starts
  pipeline.py      the pipeline split: each rank a run of layers, Coord
  plan.py          the step plan rank 0 sends every rank (pure Python)
  link.py          joining the ring, the per-step exchange, the bell (Link)
  ring.py          rank 0's side: Ring, Journal, TensorExecutor, assign_seed
  follower.py      a rank >= 1: serve_follower, follow, Mark, agree_head
  marker.py        a rank's progress marker, as the engine reports it
"""
