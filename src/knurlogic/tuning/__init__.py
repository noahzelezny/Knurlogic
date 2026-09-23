"""tuning/ -- what the settings should be, and why.

  settings.py   every constant, beside the measurement that set it: prefill
                chunks per family, cache limits, VQ kernel flags, the tune
                profiles, what each knob is and who reaches for it
  resolve.py    an artifact + a memory budget -> the environment and argv to
                run it with, and a note for every decision it made

A value here without its evidence is a bug. Depends on machine/ (what an
artifact is) and engine/ (how configs spell a family).
"""
