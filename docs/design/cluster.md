# Cluster jobs

## knurlogic/cluster/jobs.py

Stock mlx has no collective timeouts, so nothing inside a rank can
interrupt an eval blocked on a peer that died. The failure path is out of
band instead:

- **each rank** writes `~/.cache/knurlogic/jobs/<job>/rank<r>.json` -- its
  phase, its step counter, whether work is in flight (only rank 0 knows),
  and the time -- every `MARK_S` seconds from a daemon thread, and at every
  phase change.
- **its page** watches the ranks it started (`verdict()`): a pid gone, a
  rank that never joined, or rank 0 busy with a step counter that has not
  moved for `STALL_S`. Idle is not stalled: a ring with nothing in flight
  sits in its exchange forever, correctly. On any of them the page tears
  the whole job down.

With the jaccl self-heal fork (machine/deps.py), a wedged RDMA collective
also throws inside the rank -- but only once `JACCL_COLLECTIVE_TIMEOUT_MS`
is set, which happens after the load (0 while loading: a cold 400 GB read
is not a hang).

## knurlogic/cluster/launch.py

The coordinator is the page someone pressed Launch on with two or more
machines picked. It never starts a rank on another machine itself: every
page starts its own ranks, after checking the request against its own
disk, memory, software and links. Two phases, so nothing starts unless
everything can:

- **plan** -- gather each machine's facts (its status' `cluster` block:
  chip, working set under its allowance, memory bandwidth, Thunderbolt
  addresses, RDMA, versions), order the ranks (`tuning/resolve.rank_order`)
  and place the model (tensor: an equal share each; pipeline:
  `tuning/resolve.pipeline_shares`). Deterministic, and shown before
  anything loads.
- **prepare** -- POST `/peer/cluster/prepare` to every page (this one
  directly): the job's fixed schema. Each checks the artifact by identity,
  the fit of its share, knurlogic/mlx versions against the coordinator's,
  and its link. Any refusal: nothing starts, the prepared pages are told to
  forget it, the refusal is shown.
- **start** -- POST `/peer/cluster/start`: each page spawns its own rank
  (`knurlogic serve` with the hidden ring flags), records it, and watches
  it.

Every page that runs a rank watches it (cluster/jobs.py); any rank dying or
stalling stops the whole job: its own ranks SIGTERM then SIGKILL, and
`/peer/cluster/stop` to every other page of the job. It also asks the
job's other pages (`/peer/cluster/job`) whether they still run their
ranks: one unreachable, or answering without its rank, for `PEER_GONE_S`
-- or saying the job ended there -- stops the job here too (a rank idle in
a collective on a peer that vanished never exits and is never "stalled").

A stop is done when the ranks' processes are gone, not when they were
signalled: until then their records stay, marked stopping. A prepare
refuses a share that does not fit beside what this machine's other ranks
and servers hold now, or while another job's rank is still loading here; a
start waits (`START_WAIT_S`) for stopped ranks to be gone, else refuses.
Unloading the job from any page does the same. Rank 0's HTTP port is where
chat goes, through the existing relay.

Peer routes are gated exactly like `/peer/loaded.json`
(`ui.peer_refusal`).

## knurlogic/cluster/recovery.py

- **Who** -- the page that coordinated a cluster launch
  (`cluster_jobs.launch` registers the job here), and the page that
  started a one-Mac server (ui's Launch registers the port). A relaunch is
  the same launch again: the same identity, machines, rank order, split,
  link, port, tune and settings, through `cluster_jobs.launch` (its prepare
  checks fit, versions, links and one load at a time on every page; the
  cable failover still follows it) or `mcp.load` for one Mac (fit and
  memory-still-moving refusals).
- **Never** -- a requested stop (an unload from any page or the MCP, a page
  closing); a stop because the model does not fit or a machine ran out of
  memory -- relaunching into the same memory can reboot the machine --
  which is `failed` at once, with the reason.
- **Waits** -- a machine that went away or stopped answering: the relaunch
  waits until every machine of the job answers its page again, within the
  window; else `failed`.
- **Limits** -- `MAX_ATTEMPTS` relaunches of a model within `WINDOW_S`,
  `BACKOFF_S` apart; after that the model is `failed`, with the last
  reason, until someone loads it again. A relaunch starts only once no rank
  of the old job is left on any of its machines (`knurlogic serve` for that
  job, by process), and a one-Mac server only once its old process is
  gone.
- **Switch** -- `KNURLOGIC_RECOVER=off` turns it off (on by default):
  failures stop the job and are reported.

What is reported, per model: `view()` -> `{attempts, last_reason, last_at,
next_at, state}`, state `recovering | recovered | failed`; None when there
is nothing to report. The page also writes each record to this machine's
`recovery.json` under the serving port, and a relaunch carries it to the
page of the machine running rank 0, so that model's own `/v1/residency` row
says it too. What it takes to relaunch -- each tracked model's record, its
launch request and attempts -- is kept in `recovery-models.json` beside it,
so a page restarted mid-recovery picks up where it was (`restore()`).
