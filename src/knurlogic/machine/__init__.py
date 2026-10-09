"""machine/ -- what is true about this box, read rather than assumed.

  artifact.py   what a model directory IS: config, size, VQ, bundled runtime
  discover.py   every model on this machine, in every tool's store
  loaded.py     what is in memory now, in every runtime
  memory/       this box's memory: the wired limit and `load_budget()` (the
                one number every fit / settings / load answer is computed
                against), the allowance, where memory went, pressure
  status.py     the snapshot /status.json serves
  servers.py    the record of knurlogic servers running on this box
  loadlock.py   the model-load lock: one real load at a time
  identity.py   which machine this is (an id that survives renames)
  folders.py    the model folders this Mac remembers
  metrics.py    how hard this machine is working (GPU, CPU, swap)
  ledger.py     the request ledger: one row per served request
  disk_cache.py a small JSON cache of work whose inputs have not changed
  deps.py       which build of mlx, mlx-lm and mlx-vlm is installed, read off
                the fix itself rather than a version string

Every answer here says how it was measured, because each one was wrong at
least once while it was only assumed (see the docstrings).

Depends on engine/ only for the working set and whether a head exists.
"""
