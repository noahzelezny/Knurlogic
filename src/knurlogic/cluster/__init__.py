"""The other machines: finding them, knowing them, and running one model
across them.

  peers.py      named, remembered and introduced peers, each with a
                reachability state.
  discovery/    `_knurlogic._tcp`, advertised and browsed: Bonjour, Avahi.
  launch.py     the coordinator: place a job, pick its cable, run the
                Prepare and Start rounds, follow it; peer_step answers.
  preparing.py  one page checks a rank of a job against itself.
  starting.py   one page starts the ranks it prepared.
  watching.py   whether each job this page runs is still healthy.
  stopping.py   a job's ranks end here and the other pages are told.
  here.py       this machine for a job: what it offers and holds.
  tables.py     the page's jobs in memory, their locks and ports.
  jobs.py       a job's files, rank progress markers and health verdict.
  recovery.py   a model that died unasked is relaunched, bounded.
  links.py      which link a peer is reached over and a page answers on,
                and the cable a job runs on.
  protocol.py   the control-plane contract: the message envelope and kinds,
                and the link names.
  transport.py  the one client of the control plane, page to page.
  checks.py     `knurlogic doctor --cluster`: what stops Macs finding each
                other, each with the fix.

cluster/ never imports interfaces/: the page injects its
status, peers, children and load at startup (interfaces/page/server.py
wire, called by make_handler). Nothing here imports mlx. Below: what a
call to another page can raise, serve's refusal exit status, and `every`,
the one background loop.
"""

import logging
import threading
import time

#: `knurlogic serve`'s exit status when it refuses to start (bad settings,
#: a model it cannot load as asked): deterministic, so the page fails the
#: load with the REFUSING line and never relaunches it (cluster/recovery).
#: EX_CONFIG, distinct from a crash (1) and from argparse's usage error (2).
REFUSED_EXIT = 78

from knurlogic.machine.http_client import NET_ERRORS  # noqa: E402,F401


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

