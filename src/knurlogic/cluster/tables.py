"""A page's cluster jobs in memory: the tables every step shares.

What this page prepared, runs, ended and is stopping, the locks over
them, the hooks the page injects (status_fn, peers_fn), and the ring's
ports and slots. The steps -- preparing, starting, watching, stopping --
and the coordinator (launch) read and change these; nothing else here
does work.

Design: docs/design/cluster.md (launch).
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable

from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import LINK_NAMES


def _no_status():
    raise LookupError("no page has set cluster_jobs.status_fn")


# What the cluster needs of the page that runs it, injected by that page at
# startup (interfaces/page/server.wire, from make_handler) so cluster/ never
# imports it. Unset, this machine is alone: no status snapshot (node_info()
# answers) and no peers.
#: () -> (status snapshot, _): the page's own status document
status_fn: Callable = _no_status
#: () -> [peer record]: the page's PEERS store
peers_fn: Callable[[], list] = list

# ------------------------------------------------------------ the tables

#: job -> prepared spec (+ resolved path), awaiting start
PREPARED: dict = {}
#: job -> {"reason", "t", "machines"} for jobs that stopped, shown a while
ENDED: dict = {}
#: job -> spec, for the jobs this page has ranks in (to stop the others)
SPECS: dict = {}
#: job -> Popen for ranks this page process started (exit codes, reaping)
PROCS: dict = {}
#: over every table here
LOCK = threading.Lock()
WATCH = J.Watch()
#: set once the watcher runs
WATCHER = threading.Event()
#: one start at a time on this page: the loading check and the spawn are
#: one step, so two jobs' ranks never begin loading together
START_LOCK = threading.Lock()
#: jobs a stop() in this process is tearing down right now
STOPPING: set = set()
#: the boot_id a peer had when it held its rank of a job: {(job, id): id}
PEER_BOOT: dict = {}
#: (job, peer id) pages that did not answer, warned once per episode
PEER_SLOW: set = set()
#: (job, peer id) -> when that machine's kernel was first found silent
#: while the job was still joining (watching._silent_while_joining)
PEER_SILENT: dict = {}
#: jobs a coordinator's _follow is watching now (recovery leaves a link-init
#: failure of one of these to the cable failover)
FOLLOWING: set = set()

# ------------------------------------------------------------ a job's terms

SPLITS = ("tensor", "pipeline")
#: mlx's distributed backends a spec names (protocol.LINK_NAMES)
LINKS = tuple(LINK_NAMES.values())
#: the prompt chunk a ring runs when a rank cannot work out its own room's:
#: the floor. Otherwise it is the smallest of the ranks' room-based chunks
#: (each rank answers Prepare with its own; see preparing.ring_chunk)
PREFILL_CHUNK = 512
#: the launch setting that names the ring's chunk outright
PREFILL_KNOB = "KNURLOGIC_PREFILL_CHUNK"
RING_CHUNK_WHY = "ring: the smallest rank's room"
#: a prepared job not started within this long is forgotten
PREPARED_S = 120.0
#: how long a prepare waits for a stopping rank of an earlier job to be gone
EXIT_WAIT_S = 30.0
#: mlx's words when a running peer of the ring died (ring.cpp): not a link
#: that failed to come up, so never a reason to move the job to another cable
RING_LOST = "connection to a peer was lost"
#: a rank's link init failing: its log line (mlx's jaccl / ring errors)
LINK_INIT_RX = re.compile(
    rf"[^\n]*\[(?:jaccl|ring)\](?![^\n]*{re.escape(RING_LOST)})[^\n]*"
    r"(?:RTR|RTS|queue pair|connect|Connection)[^\n]*", re.I)


def leader_url(spec: dict) -> str:
    """The job's API as the other machines reach it: rank 0's link
    address (the leader binds loopback and that), or '' when unknown."""
    hs = [h for h in (spec.get("serve_hosts") or [])
          if h not in ("127.0.0.1", "::1", "localhost")]
    port = spec.get("port")
    return f"http://{hs[0]}:{int(port)}/v1" if hs and port else ""


def link_init_failure(text: str) -> str:
    """The line of a rank's output saying its link init failed ("" if
    none): jaccl's queue pair not reaching RTR, a ring that could not
    connect."""
    m = LINK_INIT_RX.search(text or "")
    return m.group(0).strip()[:240] if m else ""


# ------------------------------------------------------------ ring ports

#: first ring port; a job's ranks take ring_port(slot, rank) (+0..+17),
#: rank 0's bell +starting.BELL_OFFSET and the jaccl coordinator
#: +COORDINATOR_OFFSET
RING_PORT = 47200
#: ports per slot, and slots
SLOT_PORTS = 20
SLOTS = 100
COORDINATOR_OFFSET = 19


def slot(job: str, used=None) -> int:
    """The job's ring-port slot: its nonce's, or the next one after it that
    no live job on this machine holds (`used`: default, the registry's)."""
    if used is None:
        used = _used_slots()
    start = _nonce_slot(job)
    for i in range(SLOTS):
        s = (start + i) % SLOTS
        if s not in used:
            return s
    raise ValueError("every ring-port slot is held by a live job here")


def _used_slots() -> set:
    """The ring-port slots of the live jobs in this machine's registry."""
    out = set()
    for job, recs in J.by_job().items():
        for r in recs:
            rp = r.get("ring_port")
            if isinstance(rp, int) and rp >= RING_PORT:
                out.add(slot_of_port(rp))
            else:
                out.add(_nonce_slot(job))
    return out


def _nonce_slot(job: str) -> int:
    """The ring-port slot a job's nonce names (`slot` moves off it when a
    live job holds it)."""
    return int(job[:4], 16) % SLOTS


def ring_port(slot: int, offset: int) -> int:
    """Port `offset` of ring-port slot `slot`."""
    return RING_PORT + slot * SLOT_PORTS + offset


def slot_of_port(port: int) -> int:
    """The ring-port slot a port of it (ring_port) belongs to."""
    return (port - RING_PORT) // SLOT_PORTS

