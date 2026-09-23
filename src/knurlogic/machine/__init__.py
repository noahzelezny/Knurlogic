"""machine/ -- what is true about this box, read rather than assumed.

  artifact.py   what a model directory IS: config, size, VQ, bundled runtime
  discover.py   every model on this machine, in every tool's store
  loaded.py     what is in memory now, in every runtime, and where memory went
  wired.py      the GPU wired limit, and `load_budget()`: the one number every
                fit / settings / load answer is computed against
  status.py     the snapshot /status.json serves
  deps.py       which build of mlx, mlx-lm, mlx-vlm and exo each interpreter
                has, read off the fix itself rather than a version string

Every answer here says how it was measured, because each one was wrong at
least once while it was only assumed (see the docstrings).

Depends on engine/ only for the working set and whether a head exists.
"""
