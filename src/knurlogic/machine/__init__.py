"""machine/ -- what is true about this box, read rather than assumed.

  artifact.py   what a model directory IS: config, size, VQ, bundled runtime
  discover.py   every model on this machine, in every tool's store
  loaded.py     what is in memory now, in every runtime
  memory/       this box's memory: the wired limit and `load_budget()` (the
                one number every fit / settings / load answer is computed
                against), the allowance, where memory went, pressure
  status.py     the snapshot /status.json serves
  servers.py    the record of knurlogic servers running on this box,
                and the default ports (MODEL_PORT, PAGE_PORT)
  loadlock.py   the model-load lock: one real load at a time
  identity.py   which machine this is (an id that survives renames) and its
                hardware: chip(), machine(), kind_from()
  folders.py    the model folders this Mac remembers
  metrics.py    how hard this machine is working (GPU, CPU, swap)
  ledger.py     the request ledger: one row per served request
  disk_cache.py a small JSON cache of work whose inputs have not changed
  deps.py       which build of mlx, mlx-lm and mlx-vlm is installed, read off
                the fix itself rather than a version string

The shared low-level layer every package builds on:

  units.py       GIB, MIB, and gib(n) -> "12.3 GiB"
  paths.py       cache_dir(), config_dir(), state_dir(): where files live
  files.py       write_atomic / write_json / read_json: whole-file writes
  processes.py   running(pid), PROC_ERRORS, command lines, ports
  http_client.py get_json / send_json / request / open_url: HTTP, as a client
  nvidia.py      this machine's Nvidia GPU, as nvidia-smi reports it

Every answer here says how it was measured, because each one was wrong at
least once while it was only assumed (see the docstrings).

Never imports engine/ (tests/integration/test_layers.py): what it needs
from the engine -- a process's memory, the families' sampling tables, the
framework's working set off macOS -- its callers hand in.
"""
