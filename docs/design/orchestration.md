# Orchestration: the control plane between pages

The control plane is how knurlogic pages on 2..16 Macs find each other, agree
on a job, start and watch its ranks, and stop it. Model data (tensors, ring,
jaccl) stays on MLX's backends and is not part of it.

Today the plane exists but is implicit: JSON dicts posted to `/peer/*` routes,
parsed by hand at each end, with the shape spread over `cluster/launch.py`,
`cluster/peers.py` and `interfaces/page/server.py`. This design makes that
protocol an explicit, typed, versioned contract in one module, so a node
could be reimplemented in another language by reading one file.

## 1. Inventory: the protocol as it is

All calls are HTTP+JSON. "Gate" is `peer_refusal` (no Origin header; arrived
on loopback, Thunderbolt, or a `--peer` address) on every `/peer/*` route.

| Route | Direction | Payload (request -> reply) | Caller -> handler |
|---|---|---|---|
| `GET /status.json` | page -> peer page | header `X-Knurlogic-Peer: <id> <port>` (introduction) -> status doc: `schema`, `nodes[]` (own entry: id, name, machine facts), `peers[]`, `cluster` block | `Peers._one` -> status builder |
| `GET /loaded.json` | page -> peer page | -> residency: resident models, `where`, `jobs`, `recovery` | `peer_residency` -> `/loaded.json` |
| `POST /peer/loaded.json` | coordinator -> peer | single-machine `{action: load, identity, name, port, tune, sets, force}` or `{action: unload, port}` -> `{loaded, refused, note}` | `forward_launch` -> `peer_launch` |
| `POST /peer/cluster/prepare` | coordinator -> every rank page (and itself) | `SPEC_KEYS` spec: job, rank, world, split, link, identity, hosts, nodes, versions, port, auto_port, ... -> `{ok, machine, refused?, free_port?, alert?, note?}` | `launch()` `ask()` -> `prepare` |
| `POST /peer/cluster/start` | coordinator -> every rank page | `{job}` -> `{started, rank, pid, log}` or `{error}` | `launch()` -> `start` |
| `POST /peer/cluster/stop` | any rank page -> every other page of the job | `{job, reason}` -> stop result | `stop(propagate)`, `_abandon` -> `peer_route` |
| `POST /peer/cluster/job` | any rank page -> every other page of the job | `{job}` -> `{ranks_here, prepared, stopping, phase, processes, ended}` | `_ask_job` (`peer_verdict`), `_job_end` -> `job_state` |
| `POST /peer/cluster/shape` | coordinator -> peer | `{identity, name, world, split}` -> model shape | planner -> `shape_of` |
| `POST /peer/machine.json` | page -> peer | `{allowance_gib?, strategy?}` | `/machine.json?where=` -> `peer_machine` |
| `GET/POST /peer/settings.json?port=N` | page -> peer | a server's live knobs | `peer_settings` |
| `GET/POST /peer/v1/*` | page -> peer | chat/models relay to a server on the peer's loopback | `peer_relay` |
| mDNS `_knurlogic` | node <-> LAN | service + TXT (id, port) | `cluster/discovery.py` |

Rank to page is not HTTP: each rank writes `~/.cache/knurlogic/jobs/<job>/rank<r>.json`
(phase, step, in-flight, time, every `MARK_S`=2 s); its page reads it
(`jobs.Watch.verdict`, `phase_of`). Rank 0 also serves the model on `port`.

Not control-plane: browser -> page (`/loaded.json` POST load/unload, `/machine.json`,
`/peek`) and MCP `load/state/unload` -> page over loopback. They are the
front door; they must produce and consume the same typed messages, not a
parallel shape.

Gaps the contract fixes: replies mix `ok`/`refused`/`error`/HTTP status for
"no"; failure kind is inferred from free text (`recovery.kind`, regexes);
peer liveness is two different clocks (`Peers` 4 s refresh with 3 s timeout;
`peer_verdict` 20 s); version is only a status `schema` int checked by the
refresher; `failover` and `_pair` assume exactly two ranks (`a, b = order`).

## 2. The contract

One module, `knurlogic/cluster/protocol.py`: stdlib only, no imports from
`launch`, `peers` or the page. Frozen dataclasses, each with
`to_wire() -> dict` and `from_wire(dict)`. Unknown fields ignored, missing
required fields raise `ProtocolError` (plain message). Size cap stays
`PEER_MAX`. Paths never appear (`PATH_KEYS` refusal moves into `from_wire`).

**Envelope** (every message, both directions):

| Field | Meaning |
|---|---|
| `v` | `[major, minor]`, e.g. `[1, 0]` |
| `kind` | message kind (below) |
| `from` | sender node id (stable, see 3) |
| `job` | job id (hex nonce, `JOB_RX`), when the message is about a job |
| `seq` | sender's per-process counter; replies echo the request's `seq` as `re` |
| `ts` | sender wall clock, seconds (information only, never ordered on) |
| `body` | the kind's fields |

**Kinds** (replaces in brackets):

| Kind | Body | Replaces |
|---|---|---|
| `Hello` / `NodeInfo` | node id, name, build hash, knurlogic + mlx versions, protocol version, machine facts (chip, working set, bandwidth, Thunderbolt, RDMA, selfheal), addresses, `boot_id` | status `nodes[]` own entry + `launch.node_info` + `X-Knurlogic-Peer` |
| `Heartbeat` | `boot_id`, resident summary hash, jobs here `[{job, phase}]` | the `Peers._one` status poll (the heavy status stays for the browser) |
| `Survey` / `Residency` | request: empty. reply: resident models (`where`, identity, port, job), recovery rows | `peer_residency`, `/loaded.json` |
| `Prepare` | the job's fixed spec (`SPEC_KEYS` as typed fields; `nodes[]` of any length) | `/peer/cluster/prepare` |
| `PrepareReply` | `ok`, `refused`, `free_port`, `alert`, `note` | its `{ok, refused, free_port...}` |
| `Start` / `Started` | `job` / `rank, pid` | `/peer/cluster/start` |
| `RankStatus` | `job, rank, phase` (`spawned, loading, warming, ready, serving, stopping`), `step`, `busy`, `pid` | `rank<r>.json`; also the body of `JobState` per rank |
| `JobState` | `job, ranks_here[], prepared, stopping, phase, ended` | `/peer/cluster/job`, `job_state` |
| `Stop` / `Stopped` | `job, reason` / processes gone | `/peer/cluster/stop`, `stop` |
| `Load` / `Unload` (single machine) | identity, port, tune, sets, force / port | `/peer/loaded.json` |
| `Failure` | `job?, node, kind` = `refusal|memory|machine|failure` (+ `requested` for an asked stop), `reason` (plain text, <=300) | `recovery.kind(reason)` regexes; the sender now states the kind, regexes stay only for raw rank log tails |
| `Shape`, `MachineSet`, `Settings` | as today's bodies | `/peer/cluster/shape`, `/peer/machine.json`, `/peer/settings.json` |

**Versioning rule.** `v = [major, minor]`. A node that receives a higher
major replies with `Failure(kind=refusal, reason="<name> speaks protocol 2,
this machine 1: update knurlogic on the older one")` and treats the sender as
`version_mismatch` (no jobs, still listed). Same major, any minor: parse what
is known, ignore the rest; a new minor may only add optional fields or new
kinds, and an unknown kind gets a `Failure(refusal)` reply, never a crash.
Removing or re-typing a field is a major bump. Job-start additionally keeps
today's "every rank runs the same build" check (`prepare`): protocol
compatibility is not build compatibility.

## 3. Membership

- **Discovery.** Unchanged: Bonjour (`discovery.py`) plus `--peer` manual and
  the introduction header. A discovered address is only a candidate; a node
  is real when its `Hello` answers with an id.
- **Identity.** Node id is today's `machine/identity` id (persisted), carried
  in every envelope `from`. Peers are keyed by id (`peers.json`, as now), so
  one machine with several addresses stays one node. `boot_id` (random per
  process start) in `Hello` and `Heartbeat` lets a peer see "same id, new
  process": rank records from the old boot are gone.
- **Liveness.** Every node posts/answers `Heartbeat` to every known peer each
  `HEARTBEAT_S = 2` s (one thread, parallel, short timeout 1.5 s; today's
  `TIMEOUT_S=3`, `REFRESH_S=4` are replaced). States per peer:
  `answering` (heard within 3 intervals = 6 s), `stale` (6..20 s silent:
  shown, no new jobs placed), `gone` (>20 s, `PEER_GONE_S`: a job that has a
  rank there stops). `version_mismatch` is a fourth, separate state.
  One clock now serves both `Peers` and `peer_verdict`.
- Membership is a full mesh of heartbeats; at 16 nodes that is 240 tiny
  requests per 2 s, about 120 per second across the cluster, about 8 per node
  per second: fine over HTTP. No leader election, no gossip in this pass.

## 4. Job coordination

- **One coordinator per job**: the page the user pressed Launch on (it is
  always one of the job's machines: `order[r]["page"] is None` for itself). It
  runs plan -> `Prepare` to all -> `Start` to all, owns the recovery record
  (`recovery.track_cluster`) and, for link-init failure, the cable failover
  (`_follow`, `failover`). It gives no orders after `Start` except `Stop`.
- **After Start every rank page is equal.** Each watches its own ranks
  (`watch_once`) and every other page of the job by `JobState`/`Heartbeat`
  (`peer_verdict`). Any failure -> `Stop(reason)` to all (`stop(propagate)`).
  No pairwise assumptions: loops run over `nodes[]`; `failover`/`_pair`
  keep a bad-cable record per pair but the relaunch picks the next cable for
  the failing pair only, for any world size.
- **Coordinator's page dies**: the job stops. The other pages see it `gone`
  within 20 s and stop (what `peer_verdict` does today for any page); its own
  rank process is reaped when that page restarts (`_leftovers`). No takeover.
  Reason: the coordinator is also a rank's machine, so its death already
  breaks the collective; a takeover needs election, a replicated recovery
  record and a restart of the ranks anyway, so it buys nothing over a relaunch.
  The persisted recovery record (`recovery.json`, `restore()`) relaunches it
  when that page returns; if the machine never returns, the job stays
  stopped and the `Failure(kind=machine)` is shown on every surviving page.
  A non-coordinator may issue `Stop` (Unload from any page, as today).
- **A rank's machine drops** (heartbeat `gone`, or its page answers without the
  rank): every other page stops the job within `PEER_GONE_S`; recovery rules
  are today's (`recovery._tick_cluster`: `machine` kind waits for the node
  to be `answering`, relaunches the same order; `refusal` does not retry;
  `memory` after the user changes something). A stop completes when
  processes are gone, not when signalled (today's rule, kept).
- **Idempotence.** `Prepare`, `Start`, `Stop` for a job id are idempotent:
  re-sending after a timeout returns the same reply. `Start` for an unprepared
  or expired (`PREPARED_S`) job is a `Failure(refusal)`.

## 5. Transport

HTTP+JSON between pages stays, same paths, same gate (`peer_refusal`: no
Origin, loopback/Thunderbolt/`--peer` source), same body limit. The request
and reply bodies become the envelope above; HTTP status is only transport
(200 for any typed reply including refusals, 400 bad envelope, 413 too big).

One client, `knurlogic/cluster/transport.py`: `send(peer, msg, timeout) ->
Message` (encodes, posts, decodes, raises `PeerUnreachable` / `ProtocolError`
/ `PeerRefused`), `send_all(peers, msg, timeout)` (parallel, per-peer result),
and one server-side dispatcher `handle(path, headers, body) -> (status, doc)`
that calls the typed handler table. `launch._post`, `peers._http`,
`peer_residency`'s fetch and `forward_launch`'s post all become `send`. Default
timeouts live in one table (heartbeat 1.5 s, survey 3 s, prepare/start 30 s,
stop 50 s), not scattered constants. A new `/peer/v1/msg` route accepts any
envelope; old per-kind paths remain as aliases while step 6 lands.

## 6. Migration (each step leaves the suite green)

1. `protocol.py`: dataclasses, envelope, version check, `ProtocolError`. Tests
   only; nothing calls it. Round-trip test per kind.
2. `transport.py`: `send`/`send_all`/`handle` over a fake `http` callable.
   Swap `launch._post` and `_stop_post` to it; bodies unchanged on the wire
   (envelope optional when absent: a legacy body is wrapped as `v=[1,0]`).
3. Typed replies: `prepare`, `start`, `stop`, `job_state`, `peer_route`
   build/return `PrepareReply`, `Started`, `Stopped`, `JobState`. Wire keys
   for old fields are kept.
4. `Failure(kind)`: stop reasons carry an explicit kind; `recovery.kind`
   reads it first, regexes only for raw rank log lines.
5. `Hello`/`Heartbeat`: add `boot_id`, version, states `answering/stale/gone`
   in `Peers`; `peer_verdict` reads the same state; status poll stays for the
   browser.
6. Move `forward_launch`/`peer_launch`, `peer_residency`, `peer_machine`,
   `peer_settings` to typed messages on `/peer/v1/msg`; aliases stay one minor.
7. Remove the pair assumption in `failover`; add 3/4/8-node tests.
8. Enforce the envelope (legacy bodies refused with the upgrade message).

## 7. Tests

- **Fake nodes**: extend `tests/support/cluster_fake_page.py` (real handler,
  own cache dir, own identity, fake artifact/shape/rank) to start N pages in
  one test helper `fake_cluster(n)` for n in 3, 4, 8, each knowing all others;
  `cluster_fake_rank.py` writes markers and can be told to exit, stall, or
  refuse.
- Protocol round-trip for every kind (`from_wire(to_wire(x)) == x`), unknown
  field ignored, unknown kind refused, missing required field plain message.
- Version mismatch: higher major refused with the update text and listed
  `version_mismatch`; higher minor accepted.
- Heartbeat timeout: a paused fake page goes `answering -> stale -> gone` on
  the injected clock; resumed page returns to `answering`; new `boot_id`
  detected.
- Coordinator death (kill the launching fake page at world 3, 4, 8): every
  other page stops the job within `PEER_GONE_S`; restart resumes through
  `restore()`.
- Rank death (kill one rank, then one whole page, in a world of 4 and 8): all
  pages stop; `Failure.kind` is right; recovery relaunches in the same order.
- Refusal in a world of 8 at rank 5: nothing starts anywhere, all prepared
  pages forget the job.
- Idempotence: duplicate `Prepare`/`Start`/`Stop`.

## 8. Not in this pass

- Model data path, collective backends, placement/shard planning, RDMA
  topology beyond the existing cable logic.
- Coordinator takeover, leader election, replicated job state, gossip
  membership, encryption or authentication beyond today's gate.
- Replacing HTTP (websockets, gRPC), multi-LAN or internet clusters, more
  than 16 nodes.
- Changing the browser or MCP front door (they only reuse the typed shapes).
- Non-Python implementations (this pass only makes one possible).
