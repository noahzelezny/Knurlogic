# Liveness: a job stops on evidence, never on a timer

Status: built (2026-10-15). It replaces `PEER_GONE_S` in `cluster/launch.py`
`peer_verdict`, and it ends the hang where a rank whose peer machine
vanished blocks forever.

## The problem

- **A page's silence stops a healthy job.** A page decides that another
  machine's rank is gone when that machine's page cannot be reached, or
  answers without the rank, for `PEER_GONE_S` (20 s). But the page and the
  rank are separate processes, and the rank outlives a page restart. So a
  page redeploy that stays down for more than 20 s stops a job whose ranks
  are all fine.
- **A dead peer machine hangs the job.** On 2026-10-15 the M3 rebooted
  mid-job. The M4's rank 0 sat in mlx's ring `recv` forever. mlx's ring
  sockets (mlx 0.32.3, `distributed/ring/ring.cpp`) set only `TCP_NODELAY`,
  so a peer that vanishes without a FIN reads as `EAGAIN` forever. Because
  the rank was blocked rather than computing, it counted as idle, never as
  stalled, and it held its memory until Noah stopped it.

No single timeout value fixes both.

## The principle

A rank is gone only when something reports it gone:

1. **Its own page says so.** The page has reaped the process, or holds no
   record of the rank.
2. **Its connection closed.** The kernel closes every socket of a dead
   process, SIGKILL included. mlx then raises `[ring] connection to a peer
   was lost` on the thread waiting in the collective.
3. **Its machine's kernel stopped acknowledging.** This is TCP keepalive
   on the bell, an idle connection beside the ring. A peer's kernel answers
   the probes even while its rank is busy, stuck, SIGSTOPped or not reading.
   So they measure whether the machine and the link are there, never how
   long the work takes.

A page that is down or slow is never evidence.

## Part B: the bell watches the peer, and closes the ring when it dies

This part is all knurlogic code, with no mlx change and no system setting.

**What mlx does on its own** (mlx 0.32.3, `distributed/ring/ring.cpp`).
Each ring socket has a thread that calls `recv`/`send` in a non-blocking
loop while a transfer is pending. A closed or failed socket counts as an
error, and after 10 errors mlx fails every pending transfer with `[ring]
connection to a peer was lost`, raised on the thread waiting on the
collective. That thread then takes the existing paths (`scheduler.ring_error`
/ `_ring_fatal` on rank 0 with its mid-step and "Stuck" guards;
`ring_error` -> `SystemExit(1)` on a follower, unguarded, exactly as when a
peer process dies today). So a peer *process* that dies is
already handled: the kernel closes its sockets, SIGKILL included (shown on
the rig). What mlx cannot see is a peer *machine* that goes silent. Its ring
sockets set only `TCP_NODELAY`, so silence reads as `EAGAIN` forever.

**The design.** Every job already holds the bell: a TCP connection from
rank 0 to each other rank, opened before the ring (`bell_early`, else
`Link.bell`) and kept open for the life of the job.

1. **Keepalive on every bell socket, both ends:** `SO_KEEPALIVE`, the idle
   time (`TCP_KEEPALIVE` on macOS, `TCP_KEEPIDLE` on Linux),
   `TCP_KEEPINTVL`, `TCP_KEEPCNT`. Today `bell_dial` sets `SO_KEEPALIVE`
   with the macOS default idle of 2 h, and `bell_answer` sets nothing. The
   bell is idle almost all the time; its few bytes (`align`, `park`, the
   load order) are acknowledged at once. So the probes always run, unlike
   on a ring socket full of unacked data.
2. **A watcher thread per rank** checks each bell socket's kernel TCP
   state once a second: `TCP_CONNECTION_INFO` (0x106) on macOS, `TCP_INFO`
   on Linux; the first byte is the state. Reading it consumes nothing, does
   not race the main thread's reads, and cannot spin on unread data. A
   dropped connection stays closed, so a missed second is never a missed
   event. When the state shows closed (macOS `TCPS_CLOSED` 0, Linux
   `TCP_CLOSE` 7), the watcher reads `SO_ERROR` once to learn why (item 5).
   Polling once a second is a sampling interval, not a deadline.
   **Not `poll`.** On macOS, `poll` with `events=0` reports nothing when
   keepalive drops a connection. It reports only `POLLIN|POLLHUP`, which
   readable data also triggers. Measured: below.
3. **On a kernel-silence error** (item 5), the watcher:
   - logs which rank and why ("the machine holding rank N stopped
     answering");
   - calls `shutdown(SHUT_RDWR)` on each of mlx's ring sockets. To mlx's
     threads this is a peer that closed, so a rank blocked in a collective
     raises on its own thread through the normal path;
   - sets `link.dead` and wakes the scheduler (`link.on_dead`, set by the
     scheduler to its `_wake.set`). The scheduler's loop checks `link.dead`
     when it wakes and calls `_ring_fatal` on its own thread. This is for
     an idle rank 0. A rank in no collective gets no error from mlx, since
     mlx's threads only count errors while a transfer is pending, and would
     otherwise hold its memory until the next request. An idle rank is in
     no step, so leaving is safe.
   - tells its own page: the progress marker gets `phase="peer_gone"`
     and `why`. A phase change is written at once (`jobs.Marker.set`).
     `Watch.check` returns that reason, and the page stops the job through
     today's `stop()`: SIGTERM, the `mid_step` spare, then SIGKILL after
     `GRACE_S`. This is what frees a rank that `shutdown` cannot wake: a
     jaccl transfer, or a ring job whose sockets were not found. It is the
     path the page timer drove until now, so no case gets worse.
     Today the page reaches that path by a timer on another page. Now it
     reaches it on evidence from its own rank. The watcher thread runs
     while the main thread is blocked in a collective (rig: it woke once a
     second during a blocked `all_sum`).
4. **Every bell operation raises `ConnectionError`.** With keepalive, a bell
   `send` or `recv` can now fail with `TimeoutError` or a plain `OSError`
   (EHOSTUNREACH, ENETUNREACH, EHOSTDOWN). Neither is a `ConnectionError`,
   so neither is a `ring_error`. On a follower, such an error escapes as a
   traceback. On rank 0 it lands in "scheduler tick failed", which neither
   leaves nor relaunches. The places are `Link.sleep`, both branches of
   `Link.align` (also reached during load, `weights.py:181`/`:218`, and
   through `gather_ints`), and `_exchange`'s `sendall(b"w")`. So `Link`
   gets one helper that every bell `send`/`recv` goes through and that
   re-raises any `OSError` as `ConnectionError`, naming the rank. Separately,
   `align`'s rank-0 timeout message formats `timeout:.0f`. With
   `timeout=None` (every call but the step's `ALIGN_S` one) that is a
   `TypeError`, unreachable today and reachable with keepalive. The message
   must handle `None`.

5. **Only kernel silence counts, never a hang-up or reset.** A normal stop
   closes the bell too: rank 0 sends the `stop` op and exits. That is
   ordinarily a FIN (rank 0 reads every byte it is sent); a reset is
   possible only if a byte is unread (a guess, not seen). If the watcher acted on
   that, its `shutdown` could discard the `stop` op still unread in the
   follower's buffer, and a normal stop would become a crash. So the
   watcher acts only on `SO_ERROR` in {ETIMEDOUT, EHOSTUNREACH,
   EHOSTDOWN, ENETUNREACH, ENETDOWN}: the peer's kernel is not answering.
   After a hang-up or reset the watcher stops checking that socket: there
   is nothing more to learn from it.
   A peer process that ended needs no watcher: mlx sees the closed ring
   sockets itself, after any buffered data, and both pages see the rank
   exit (part A).

**Finding mlx's ring sockets.** Right after `mx.distributed.init`, the
rank lists `/dev/fd` (works on macOS and Linux). It keeps each socket whose
peer or local address is one of the job's ring `host:port`s, deduplicated
by (local, peer) address and skipping descriptors it opened itself.
Without the dedupe, the rig matched one socket twice: it listed a `dup` it
had just made, and `shutdown` on the duplicate raised. The rank holds those
socket objects (dups of mlx's descriptors) for the life of the job, so a
descriptor number reused later is never touched. If a ring job finds no
ring socket, the rank logs an error saying liveness is not armed (a future
mlx may open its sockets differently) and runs as today. The unit test
below fails on such an mlx before it ships.

**A world of 3 or more.** The bell is a star around rank 0. If rank 1's
machine dies, rank 0's watcher closes rank 0's ring and rank 0 leaves. Its
ring sockets then close, and rank 2's mlx sees that like any peer process
that ended. If rank 0's machine dies, every follower's bell keepalive fails.

**Settled on a further pass:**
- **The bell runs over the ring's own link.** It connects to rank 0's ring
  host (`launch.bell_address`: `hosts[0]`; `Link.bell`: `_rank0_host`), so
  bell silence means rank 0's ring link is silent too. In a world of 2 that
  is the only link. In a world of 3 or more, a cut link between two
  followers (rank 1 to rank 2, a separate cable) leaves every bell healthy.
  That case is still caught only as a stall (the progress marker's
  `STALL_S`, as today). This is a limit of this design, stated, not hidden.
- **jaccl jobs have no ring sockets to find.** The finder runs only on the
  ring link, so a jaccl job does not log "liveness not armed". Its watcher
  still logs the dead peer, wakes an idle rank 0, and marks `peer_gone`,
  so its page stops the job as the timer did before (item 3).
- **Raising out of a collective mid-step is not new.** After the
  `shutdown`, a rank blocked in a collective fails exactly as it does today
  when a peer process dies (rig: SIGKILL). The same `_ring_fatal` /
  `ring_error` paths and their mid-step guards apply, so this design adds
  no new way to leave mid-GPU-step.
- **A peer Mac that goes to sleep** stops answering keepalives, and the job
  ends about 15 s later. Today the page timer ends it at 20 s, so nothing
  gets worse. `cluster/checks.py` already warns when a machine may sleep on
  AC power.
- **Reading `SO_ERROR` clears it, for whoever reads first.** If the
  watcher reads it first, a parked follower's `recv` sees EOF instead of
  ETIMEDOUT, and `Link.sleep` treats both as "rank 0 left". If the main
  thread's `recv` takes it first (parked follower, rank 0 in `align`), the
  watcher sees `SO_ERROR` 0 and, by item 5, does nothing. That is fine,
  because the main thread already holds the error and leaves through item
  4. Nothing relies on the watcher having acted.

**Values:** see the decision below: about 30 s proposed, from a peer
machine's last acknowledgement to the ranks leaving. These set how fast a
dead machine is noticed, not how long work may take. A peer that is busy,
stuck in a long GPU step, SIGSTOPped, or not reading its ring sockets still
answers keepalives on the bell.

**This is stricter than today on the ring link. It is a decision, not a
side effect.** Today a ring-link outage shorter than TCP's retransmit
horizon (minutes) is survived, and the job resumes, though the page timer
on the page link may end it anyway. With the bell's keepalive, a link
silent for longer than the window ends the job. A silent link and a dead
machine look the same from this side, so some window must exist, and it is
the only number in this design that can end a job whose ranks are all
alive. So it should be generous: idle 10 s, probe every 5 s, 4 probes,
about 30 s. That is slower to notice a dead machine than the 15 s used
above, and far more tolerant of a flap. Noah picks the number.

**Why not options on the ring sockets themselves.** This was tried first.
Keepalive does not probe a connection with unacked data. The option that
limits unacked time breaks the principle on Linux: `TCP_USER_TIMEOUT` also
counts zero-window time. mlx reads a ring socket only while a receive is
queued, so a healthy peer in a long step shows a zero window. On the rig, a
peer that was healthy but not reading for 45 s had its connection dropped
(errno 110). macOS's `TCP_RXT_CONNDROPTIME` did not drop it, but one
mechanism on both OSes beats two that differ. The bell carries no bulk
data, so it has neither problem.

**Measured on a two-rank loopback rig, mlx 0.32.3** (idle 3 s, probe every
1 s, 3 probes):

| Case | Linux (container, iptables drop) | macOS |
|---|---|---|
| peer SIGKILLed | — | raised at once, no options needed |
| peer silent, ring options, rank waiting to receive | raised in 6.1 s | raised (pf drop, errno 60) |
| peer silent, ring options, rank mid-send (32 MB) | raised in 5.3 s; blocked 90 s+ without the unacked option | raised in 5.5 s (errno 60) |
| peer healthy, not reading 45 s, ring unacked option | dropped (wrong) | completed |
| peer healthy, not reading 45 s, keepalive only | completed | completed |
| **bell watcher (poll prototype)**, peer silent, rank mid-send, no ring options | bell `SO_ERROR` 110, ring shut, raised in 5.6 s; watcher woke once a second with an unread byte on the bell | **watcher never fired**: `poll(events=0)` silent; mlx raised on its own after 18 s through loopback retransmit, which says nothing about a real link |
| keepalive drop, seen by: `poll(events=0)` / `poll(POLLIN)` / `SO_ERROR` / TCP state | `0x18` / `0x19` / 110 once / `TCP_CLOSE` (7), stays | nothing / `0x11` / 60 once / `TCPS_CLOSED` (0), stays |
| `shutdown` on ring sockets wakes a blocked rank | (used above) | raised in 3.0 s |
| **built code** (`link.init`, 30 s window), four cases at once | — | rank 0 in a collective: raised; follower parked: `ConnectionError` (ETIMEDOUT); idle rank 0: `on_dead` called; follower in a collective: raised -- all within 1 s of each other, 30 s after the drop |

**jaccl (RDMA).** The watcher cannot unblock a jaccl transfer, since those
are not sockets. But its `peer_gone` mark lets the page stop the job
exactly as the timer did: SIGTERM, the `mid_step` spare, SIGKILL. A rank
blocked mid-GPU-step is spared and stays until its step ends, as today.
The `JACCL_COLLECTIVE_TIMEOUT_MS` patch on Noah's mlx fork (branch
`jaccl-selfheal`, not in the stock 0.32.3 wheels; `deps.jaccl_selfheal`
detects it, and `launch.rank_env` arms it only when present) would let
jaccl raise on its own thread too.

## Part A: peer_verdict without a clock

`peer_verdict` stops a job only on what a peer's page reports:

- **The page says the job ended** (`ended`): stop with its reason, as now.
- **A new process answers and holds no rank** (`boot_id` changed, no rank,
  not prepared): stop, as now.
- **The page holds no record of the rank and it is not prepared:** stop at
  once.
- **The page holds a record whose process has exited** (not stopping):
  stop at once, with that page's reason. `JobState` gains `exited`: for each
  such rank, the same reason that page's own watcher would give, "rank N on
  M exited" plus its log tail (link init failure, refusal, out of memory).
  This means `watch_once`'s log-tail decoration moves into one function,
  `exit_reason(job, recs, why) -> (why, kind)`, called from both places.
  So the precise reason does not depend on that page's watcher having run
  first, or running at all.
- **The page is unreachable or slow:** no verdict. Log it once per episode
  (`_PEER_SLOW`) and show it on the card. If the machine itself is gone,
  part B ends the ranks and every page stops the job by the rules above.

These are deleted: `PEER_GONE_S`, `_PEER_OK`, `_page_up`, and reading the
peer list's `last_seen` in this decision. `_PEER_BOOT` stays.

**The Start window has to close first.** `start()` pops the job from
`PREPARED` before `_start`. `_start` can then wait up to `START_WAIT_S`
(20 s) for an exiting rank, and again on a slot conflict, before it writes
the registry record. Meanwhile `job_state` answers "not prepared, no rank".
Starts go to every page in parallel, so a page whose rank is already up can
ask in that window. Today's 20 s clock hides this; part A without the fix
would stop every launch that hits it.

The fix:
- `start()` reads `PREPARED` without popping it.
- `start()` pops it in a `finally` around `_start`, so every exit pops it:
  success, each refusal, a spawn failure (`_start` returns 500 at
  `launch.py:983`), and an exception. On success `_start` has already saved
  the registry record, so the pop always comes after the save.
- `job_state` reads `PREPARED` before it reads the registry. With "save,
  then pop" on the writing side, an unlocked reader can see both (harmless)
  but never neither.
- Under `_START_LOCK`, `start()` checks again whether the job is still
  prepared, and goes to the "already running here" path if not, so a
  retried Start that queued behind the first one cannot spawn twice.
- `_remember`'s `PREPARED_S` expiry skips an entry whose Start is under
  way. The `finally` guarantees such an entry is popped, so it cannot be
  held forever.

**Still a race, not a hang:** when a rank dies, a page may stop the job
with "connection to a peer was lost" from its own rank (part B) before the
precise reason arrives. `stop()` keeps the first reason recorded
(`stop_reason or reason`). That is today's behaviour, unchanged here.

## Found while building

- **`stop()` told the other pages last.** It killed its own ranks first, so
  their ranks' ring failed before the Stop arrived, and their pages
  recorded a crash. With "[ring] ... lost" also read as a link init
  failure, an unload from a follower's page could be recovered or moved to
  another cable. Now `stop()` tells the other pages first, in parallel
  (`_tell_peers`), and a lost peer is not a link init failure (`RING_LOST`);
  a rank that exited with it is kind `machine`.
- **`job_state` read `PREPARED`, the registry and `ENDED` apart.** A peer
  could see "no record, not ended" for an instant during a stop. They are
  now read as one snapshot under `_LOCK`, and the "ended" verdict carries
  `ended_kind`.

## What stays

- **Bring-up deadlines** (`BELL_S`, the join, `START_WAIT_S`, `PREPARED_S`)
  bound a handshake, not a running job.
- **A rank wedged inside a GPU step** still answers keepalives. It is
  caught as a stall by the progress marker, as today.
- **A rank mid-GPU-step whose peer powered off** is spared by `mid_step`
  until its step ends, as today.

## Tests

- **Part B, unit:**
  - the socket finder picks exactly the ring sockets out of socketpairs,
    listeners and its own dups, deduplicated;
  - the watcher, given a closed TCP state and `SO_ERROR` ETIMEDOUT, shuts the given ring sockets,
    sets `dead` and calls `on_dead`; on a hang-up or a reset it does
    nothing; it never wakes on unread data;
  - every bell op (`sleep`, both `align` branches, `_exchange`'s wake)
    turns `TimeoutError` and `OSError(EHOSTUNREACH)` into
    `ConnectionError`, and `align(timeout=None)` names no `None`;
  - the watcher's `peer_gone` mark makes `Watch.check` return its reason;
  - `start()` pops `PREPARED` on success, refusal, spawn failure and
    exception, and only after the registry save;
  - the scheduler's loop calls `_ring_fatal` once when woken with
    `link.dead`.
- **Part B, against the installed mlx:** a real two-rank ring on loopback.
  Each rank finds exactly its ring sockets, and `shutdown` makes a blocked
  `all_sum` raise. This is the test that fails first on an mlx bump.
- **Part B, rig:** the loopback rigs above, including the
  healthy-but-not-reading case, kept as a test marked to run only by hand.
  Run against the built code, not the prototype: a follower parked on the
  bell, an idle rank 0, and a rank 0 in `align` during a load, each with
  the link dropped. The prototype covered only rank 0 blocked in
  `all_sum`.
  Dropping packets needs root (pf) or a container (iptables).
- **Part A, `peer_verdict`:**
  - a page unreachable for 10x the old `PEER_GONE_S` never stops a job;
  - no record and not prepared stops it at once;
  - `exited` stops it at once with the owning page's reason;
  - a new boot holding no rank stops it.
- **Part A, `start`:** `job_state` says prepared through a `_start` blocked
  in `wait_gone`, and a retried Start queued behind it spawns nothing.
- **End to end (FakeCluster):** keep
  `test_a_redeployed_page_keeps_the_job_its_rank_still_runs`, and add a
  page that stays down for 60 s while its rank runs. The job survives.
