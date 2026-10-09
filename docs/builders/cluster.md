# Building on the cluster

Several Macs as one: finding each other, launching one model across them,
watching the ranks, recovering. The why: [cluster](../design/cluster.md),
[discovery](../design/discovery.md), and the page-to-page contract in
[orchestration](../design/orchestration.md). The split itself (how ranks
share a model) is [splits](splits.md).

## Where the code is

`src/knurlogic/cluster/`:

| file | what |
|---|---|
| `discovery.py` | `Discovery`: Bonjour (`_knurlogic._tcp`) through `dns_sd.h`; register one advertisement, browse, resolve |
| `peers.py` | `Peers` / `Peer`: named, remembered or introduced peers and their state (`answering`, `stale`, `gone`, `version_mismatch`); `peers.json` |
| `links.py` | which link reaches a peer (`link_of`, `kind_of_iface`, `thunderbolt`, `rdma`), and `Gate`: answer only on loopback and Thunderbolt under `--host cluster` |
| `protocol.py` | the control-plane messages: the envelope `{v, kind, from, job?, seq, ts, body}`, one frozen dataclass per kind (`Hello`, `Prepare`, `Start`, `Stop`, `Failure`, ...), `check_version` |
| `transport.py` | the one client (`send`, `send_all`) and server half (`handle`) of the control plane; `PeerUnreachable`, `PeerRefused` |
| `launch.py` | a job across machines: `check_spec`, `placement`, `prepare` then `start` (two phases), `launch`, `stop`, `watch_once`, `peer_verdict`, `failover`, `jobs_document`, `rank_argv` / `rank_env` |
| `jobs.py` | a job's files and its ranks' progress markers (`Marker`, `read_marker`; a rank writes its own through engine/split/marker.py: `progress`, `chunk_done`, `after_load`), `Watch`, `phase_of`, the job registry, `terminate`, `wait_gone`. Stdlib only |
| `recovery.py` | bounded auto-relaunch: `track_cluster`, `track_single`, `tick`, `cancel_job`, `view`; persisted in `recovery.json` and `recovery-models.json` |
| `checks.py` | `knurlogic doctor --cluster`: interfaces, firewall, sleep, Bonjour browse (`report`) |

Hooks outside it:

- `interfaces/page/`: the page is a node. `loads.py` (`cluster_launch`,
  `forward_launch`, `peer_launch`), `peers.py` (`peer_refusal`, the gate
  on peer routes; `peer_residency`; `peer_machine`; the one control-plane
  route `MSG_PATH`, `/peer/v1/msg`), `messages.py` (`peer_table`),
  `peek.py` (`peer_settings`) and `relay.py` (the model relay under
  `/peer/v1/`, `peer_relay`).
- `interfaces/mcp/lifecycle.py`: `load` with `machines`, `unload` with `job`.
- `interfaces/serve.py`: starts a rank (`run` with a ring); `cluster/`
  calls back into it with a lazy import.
- `machine/identity.py`: a node is its `id`, never its name.

## Rules that keep it correct

- **One route between pages.** Every page-to-page message is a
  `protocol` envelope POSTed to `MSG_PATH` through `transport.send`. A
  different major version is refused with text that names the machine to
  update. A handler never reads an address out of a message.
- **Nothing starts unless everything can.** `prepare` sends `Prepare` to
  every page; each checks its own disk, memory, software and links; any
  refusal and nothing starts. Then `Start`.
- **Each page starts only its own ranks.**
- **One rank dies, the whole job stops.** Each rank writes
  `~/.cache/knurlogic/jobs/<job>/rank<r>.json`; the page tears the job
  down on a gone pid, a rank that never joined, or rank 0 busy with an
  unmoved step counter (`STALL_S`). Idle is not stalled.
- **A stop is done when the processes are gone**, not when they were
  signalled (`terminate`, `wait_gone`).
- **Recovery is bounded and never after a requested stop** or a stop for
  not fitting or running out of memory. `KNURLOGIC_RECOVER=off` turns it
  off.
- **A peer that stops answering is kept and shown**, never dropped
  silently.

## Extending

- A new message kind: a `Body` subclass in `protocol.py` registered with
  `_register`, a handler in the page's table (`peer_table` in
  `interfaces/page/messages.py`), and an entry in `transport.TIMEOUTS` if
  it needs its own timeout.
- A new launch check: in `launch.check_spec` / `prepare` so it refuses in
  the first phase, with the reason.

## Tests

`tests/cluster/` (`test_cluster.py`, `test_cluster_jobs.py`,
`test_cluster_nmachines.py`, `test_cluster_orchestration.py`,
`test_protocol.py`, `test_peers.py`, `test_discovery.py`, `test_links.py`,
`test_recovery.py`, `test_orderly_stop.py`, `test_leader_link.py`,
`test_fresh_placement.py`, `test_mcp_cluster.py`,
`test_artifact_identity.py`), `tests/integration/test_peer_launch.py`.
They use `tests/support/fake_cluster.py`: real page processes with fake
ranks (see [testing](testing.md)).
