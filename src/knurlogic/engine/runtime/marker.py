"""The rank side of a cluster job's progress marker, as the engine sees it:
calls the engine makes from inside a rank (a step taken, a prefill chunk
done, the model loaded). Stdlib only, a no-op outside a job.

The marker itself (cluster/jobs.Marker) is set here by whoever starts the
rank (interfaces/serve.py); engine never imports cluster.
"""
from __future__ import annotations

import os

#: this process's marker, when it is a rank of a job
CURRENT: dict = {"marker": None}


def progress(**fields) -> None:
    """From anywhere in a rank (engine included): update its marker; a
    no-op outside a job."""
    m = CURRENT["marker"]
    if m is not None:
        m.set(**fields)


def chunk_done() -> None:
    """A prefill chunk finished (engine/mtp/batch_loop.admit). One step
    admits a whole prompt, so a 30k-token prefill of the 397B over two Macs
    is one step of ~2 minutes: the chunk count is what shows it moving."""
    m = CURRENT["marker"]
    if m is not None:
        m.bump("chunk")


def after_load() -> None:
    """The model is in memory: arm the jaccl self-heal deadline, if the
    page asked for one (it does only when the fork is installed and the
    link is jaccl). Read live by libjaccl on every collective."""
    ms = os.environ.get("KNURLOGIC_JACCL_TIMEOUT_MS")
    if ms and ms.isdigit():
        os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] = ms
    progress(phase="ready")
