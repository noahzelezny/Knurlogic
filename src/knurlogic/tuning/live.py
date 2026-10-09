"""Live or restart: which knobs a running server can change, and for one
knob on one artifact whether a change applies now, needs a restart, or
does nothing at all (its bundled runtime never reads it).

Applying a live knob to the running process is the engine's
(engine/serve/load.apply_live); the page and `serve` ask here which ones
it can.
"""

from __future__ import annotations

from knurlogic.tuning import knobs

#: Knobs that can be changed on a RUNNING process, and how.
#:
#: Measured by reading a real bundled runtime (4523 lines) rather than
#: assuming. Of eleven knobs the resolver emits:
#:
#:   * VQ_DECODE_CHUNK is captured into a module global on FIRST PREFILL
#:     (`_DECODE_CHUNK = _default_decode_chunk()`) and then read inside the
#:     expert loop as a global. Rebinding that global takes effect on the next
#:     prefill -- no reload.
#:   * VQ_CACHE_LIMIT_GB (and its old names) is applied through the
#:     framework's own live API.
#:   * the eight GEMM/numerics flags are read into module globals AT IMPORT and
#:     baked into Metal kernel source that is compiled once. Those genuinely
#:     need a restart, or an override module that reads them per dispatch.
LIVE_KNOBS = ("VQ_DECODE_CHUNK", "VQ_CACHE_LIMIT_GB", "VQLAB_CACHE_LIMIT_GB",
              "KNURLOGIC_CACHE_LIMIT_GB", "KNURLOGIC_CONTEXT_LENGTH",
              "KNURLOGIC_THINKING_DEFAULT")


RESTART_WHY = ("read at import and compiled into the kernel, so it takes a "
               "restart")


def knob_reach(artifact, name: str, live_knobs, restart_why=RESTART_WHY):
    """(reach, why) for one knob on THIS artifact.

    Three outcomes, and keeping them apart is the point: it applies now, it
    needs a restart, or -- the one nobody checks -- the bundled runtime does
    not read it at all, so it will never do anything however it is set.
    """
    if name in knobs.ENGINE_KNOB_NAMES:
        # Read by the engine -- server argv or a process-global mlx call --
        # so whether the artifact's runtime also reads it is beside the point.
        if name in live_knobs:
            return "live", "the engine applies this on the running server"
        return "restart", ("engine server argv, read once at startup: set it "
                           "before loading, or restart to change it")
    reads = artifact.reads_knob(name)
    if reads is False:
        return "no-effect", ("this artifact's bundled runtime never reads "
                             "this variable, so setting it does nothing")
    if name in live_knobs:
        return "live", "can be changed on the running server"
    return "restart", restart_why
