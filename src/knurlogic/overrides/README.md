# Overrides

A module that replaces one inside an installed package — `mlx_lm.*`,
`mlx_vlm.*` or `exo.*` — without forking it, writing to site-packages, or
carrying a permanent diff.

This directory ships **empty of overrides on purpose**. The rule is the same
one the architecture set follows, and it is the only thing that keeps an
override from becoming a fork by accident:

* Mirror the target's import path on disk: `exo/master/placement_utils.py`,
  `mlx_lm/models/qwen4_exp.py`.
* Every override records what it is **against** (the exact upstream version it
  was taken from and tested on), **why** it exists, and the **measurement**
  that justifies it. No measurement, no override.
* Digest-pinned in `MANIFEST.json`. A file whose digest does not match is
  refused at import, not warned about.
* Dropped the moment upstream ships the equivalent. `arch.py` already checks
  whether upstream has moved ahead; last time it had.

An override is still a diff. It is just an unmerged one that lives in a
versioned package and can be turned off with an environment variable.
