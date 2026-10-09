"""The other machines: finding them, knowing them, and running one model
across them.

  peers.py      named, remembered and introduced peers, each with a
                reachability state.
  discovery.py  Bonjour: `_knurlogic._tcp`, advertised and browsed.
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
_wire). Nothing here imports mlx.
"""

import http.client  # noqa: E402
import subprocess  # noqa: E402

# What a call to another machine's page, or to a system tool, can raise.
# Callers catch these (plus whatever their own parsing adds), not Exception.
NET_ERRORS = (OSError, ValueError, http.client.HTTPException)
PROC_ERRORS = (OSError, subprocess.SubprocessError)
