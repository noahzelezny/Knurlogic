"""engine/mtp/cluster/ -- drafting across a pipeline-sharded model. NOT WIRED
into knurlogic yet: nothing imports these modules today.

  speculative.py   exo's glue for MTP on a pipeline ring (its "stage 1"):
                   the head loads on the LAST rank, which is the only rank
                   holding the trunk's true final activation during prefill,
                   so ALL sampling moves there and two broadcasts per step
                   carry tokens and the verdict to the others
  runtime.py       trunk loading for the MTP tools across mlx-lm and mlx-vlm

Kept because knurlogic will build its own cluster pipeline (docs/PLAN.md,
"Next: replace exo") and these hold hard-won detail about which rank drafts.
Copied from exo's worker/engines/mlx/mtp; exo keeps its own live copies.
"""
