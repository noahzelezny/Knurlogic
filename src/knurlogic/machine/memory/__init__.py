"""machine/memory/ -- this box's memory, read rather than assumed.

  wired.py      the GPU wired limit, and `load_budget()`: the one number every
                fit / settings / load answer is computed against
  allowance.py  the most memory knurlogic may use here, a person's setting
  footprint.py  where memory went: vm_stat's used/available, and every
                process's footprint by runtime (memory_map)
  pressure.py   macOS's pressure level and this process's compressed bytes

What a model needs is tuning/fit.py's; the serving process's guard is
engine/runtime/memory_guard.py.
"""
