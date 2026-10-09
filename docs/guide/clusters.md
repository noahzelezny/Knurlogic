# Two or more Macs

One model can be split across 2 to 16 Macs joined by Thunderbolt. Each Mac
runs its own knurlogic page; the pages find each other, check the job, and
each starts its own part.

## Set up

On every Mac:

1. Install the **same knurlogic version**.
2. Put the **same model** on its disk.
3. Run `knurlogic`.

The page answers on the Thunderbolt link(s) and on 127.0.0.1 only, never on
Wi-Fi or Ethernet (a Mac with no Thunderbolt link answers on 127.0.0.1).
`knurlogic --host 127.0.0.1` keeps a page to its own Mac.

### Finding each other

Pages find each other over Bonjour (`_knurlogic._tcp`). Found machines
appear in the Memory panel. Where multicast is blocked, name a peer:

```bash
knurlogic --peer 169.254.12.34          # another Mac's page, HOST[:PORT]
```

Naming one side lists it on both. Over Ethernet or Wi-Fi each side's gate
needs the other named: name each Mac on the other, or link them with
Thunderbolt. A named peer is remembered once it answers.

If they do not see each other, run on **each** Mac:

```bash
knurlogic doctor --cluster
```

It checks addresses, the firewall, Bonjour and peers, and says the fix. It
only reads; it never changes the firewall or privacy settings.

## Launch

On the page: click two or more machines in the Memory panel, choose the
model, pick **Sharding** and **Interconnect**, press Launch. The page you
pressed Launch on coordinates.

| choice | options |
|---|---|
| Sharding | **Pipeline**: each Mac holds a run of whole layers, sized to its memory and bandwidth. **Tensor**: every layer split, the same share on each Mac. |
| Interconnect | **TCP/IP**: the ring, any link, any number of Macs. **RDMA**: jaccl over Thunderbolt 5; every pair must be cabled with Thunderbolt 5 with RDMA up. Beyond two Macs it is experimental and untested. |

Only the splits a model can take are offered; one that cannot be split
says so.

| | pipeline | tensor |
|---|---|---|
| shares | sized to each Mac (uneven Macs are fine) | equal; refused when the heads or quantization groups do not divide |
| MTP drafting | yes (the head is on rank 0) | no |
| Gemma 4 | refused (it shares KV across layers) | |

Measured on an M4 Max 128 GB + M3 Ultra 96 GB, decode tok/s:

| model | one Mac | tensor TCP | tensor RDMA | pipeline TCP | pipeline RDMA |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B VQ 3.4bpw | 71.5 / 55.6 | 38.5 | 51.5 | 57.8 | 59.1 |
| Qwen3.5-397B-A17B VQ 2.4bpw | does not fit | 19.9 | 25.6 | -- | 27.3 |

### What happens

1. **Plan.** The coordinator reads each Mac's chip, working set (under its
   allowance), bandwidth, Thunderbolt addresses and versions, orders the
   ranks and places the model. This is shown before anything loads.
2. **Prepare.** Every page checks the model is the same, its share fits
   beside what it already runs, the versions match, and its link works.
   One refusal and nothing starts; the refusal is shown.
3. **Start.** Each page starts its own rank. Rank 0 (the leader) answers
   HTTP; chat goes to its port.

With two Macs and several Thunderbolt subnets, the fastest shared one is
used, moving to the next if the link fails to come up.

A split model is one entry in Instances and one model to clients. Its
output is not token-identical to one process (the split sums rounded
partial results), but it is reproducible run to run.

### From an agent

MCP `load` with `machines` (names as `state` lists them), `split`
(`tensor` | `pipeline`) and `link` (`tcp` | `rdma`); optional `cable` with
two Macs. It needs the page running on this Mac. See [mcp.md](mcp.md).

## Failure

Each rank writes a heartbeat; the page that started it watches it. A rank
whose process is gone, that never joins (300 s), or whose leader is busy
without progress for 120 s stops the **whole job**: every page stops its
ranks. An idle job is not stalled. Requests in flight get a 503
`cluster_failed` (`Retry-After: 30`); a stream already under way ends with
one error event.

Unloading the job from any page stops it everywhere.

## Recovery

A model that dies or stalls without being asked to stop is relaunched by
the page that launched it, with the same machines, split, link, port and
settings. This also covers a one-Mac server started from the page.

- At most 3 relaunches in 15 minutes, after 10 s, 30 s and 90 s. Then the
  model is `failed`, with the last reason, until someone loads it again.
- Never after an unload, a page closing, or running out of memory (a
  relaunch into the same memory can take the Mac down): those are
  `failed` at once.
- A Mac that went away: the relaunch waits for every Mac of the job to
  answer again, within the window.
- Shown as `recovery: {attempts, last_reason, last_at, next_at, state}`
  (`recovering`, `recovered`, `failed`) on the page, in MCP `state`, and in
  the server's `/v1/residency`.
- `KNURLOGIC_RECOVER=off` in the page's environment turns it off.
- Not covered: a one-Mac server that hangs without exiting (shown as
  stalled, not acted on).

## Per-chip rounding

Macs with different GPU generations can produce different (equally valid)
tokens for the same prompt. Settings -> Knurlogic -> Per-chip rounding off
makes them identical at a small speed cost. See
[settings-and-memory.md](settings-and-memory.md).

## Prompt cache across Macs

Saving and restoring works on a split model; each Mac keeps its own part.
A saved session comes back only under the same split. See
[prompt-cache.md](prompt-cache.md).

Design: [../design/cluster.md](../design/cluster.md),
[../design/discovery.md](../design/discovery.md),
[../design/server.md](../design/server.md) (Cluster, Auto-recovery).
