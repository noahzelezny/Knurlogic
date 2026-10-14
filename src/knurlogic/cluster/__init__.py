"""The other machines: finding them, knowing them, and running one model
across them.

  peers.py      named, remembered and introduced peers, each with a
                reachability state.
  discovery/    `_knurlogic._tcp`, advertised and browsed: Bonjour, Avahi.
  launch.py     a cluster job, page to page: prepare, start, watch, stop.
  jobs.py       a job's files, rank progress markers and health verdict.
  recovery.py   a model that died unasked is relaunched, bounded.
  links.py      which link a peer is reached over and a page answers on.
  protocol.py   the control-plane contract: the message envelope and kinds.
  transport.py  the one client of the control plane, page to page.
  checks.py     `knurlogic doctor --cluster`: what stops Macs finding each
                other, each with the fix.

launch.py and recovery.py never import interfaces/: the page injects its
status, peers, children and load at startup (interfaces/page/server.py
wire, called by make_handler). Nothing here imports mlx.
"""

import http.client  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

#: `knurlogic serve`'s exit status when it refuses to start (bad settings,
#: a model it cannot load as asked): deterministic, so the page fails the
#: load with the REFUSING line and never relaunches it (cluster/recovery).
#: EX_CONFIG, distinct from a crash (1) and from argparse's usage error (2).
REFUSED_EXIT = 78

# What a call to another machine's page can raise.
# Callers catch these (plus whatever their own parsing adds), not Exception.
NET_ERRORS = (OSError, ValueError, http.client.HTTPException)


def every(seconds: float, fn, name: str, *, first: bool = False,
          level: int = logging.WARNING) -> threading.Thread:
    """Run `fn()` every `seconds` on a daemon thread called `name`, for as
    long as the process lives: sleep, call, repeat (`first`: call, then
    sleep). A call that raises is logged at `level` and the loop goes on --
    one bad pass must never end a watcher, a heartbeat or a poll."""
    log = logging.getLogger(__name__)

    def loop():
        if not first:
            time.sleep(seconds)
        while True:
            try:
                fn()
            except Exception:  # the loop outlives any one failed call (logged)
                log.log(level, "%s: a pass failed", name, exc_info=True)
            time.sleep(seconds)
    t = threading.Thread(target=loop, daemon=True, name=name)
    t.start()
    return t

