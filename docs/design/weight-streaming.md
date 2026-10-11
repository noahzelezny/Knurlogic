# Weight streaming for split loads

A split (cluster) job's ranks each read their share of the weights. A
machine whose copy of the model sits on a shared network mount (SMB) read it
through that mount until now, and every rank of a job read at once.

Measured (8 GB reads of one safetensors file): the machine holding the
drive reads it cold at 1.76 GB/s; the other machine reads it over SMB cold
at 0.18 GB/s, and at 2.35 GB/s when the file is already in the holder's page
cache. The array and the Thunderbolt link are fast; the file server's cold
read is the bottleneck. With every rank reading at once, the holder's own
reads compete with the file server's and evict what it could have served
from cache.

Two changes, one feature (`cluster/weights.py` for the wire and the plan,
`engine/split/weights.py` for what a rank does with it; engine never imports
cluster, so the starter hands the engine its reader, as with the progress
marker):

1. a rank whose copy is on a network mount gets its share's tensors from
   the page of a machine that has the files on its own disk, over the
   Thunderbolt link, instead of through the mount;
2. the ranks that read remotely load first; the machines with local copies
   read their own share after the others have theirs.

## Rule: no copy on the receiver's disk

Streamed tensors go straight into the rank's memory. The receiving Mac
never writes them to disk: no staging files, no local mirror, and no
"copy the model locally" fallback. Keeping a duplicate of a 100 GB model
wastes space that belongs to the prompt cache, among other things. The
only fallback is the shared mount, read as before.

## 1. Tensors from a peer's page

### Who serves, who fetches

At **Prepare** each page already resolves the job's identity to a local
directory (`machine/artifact.resolve_identity`). Its reply now also says
`local_copy`: whether that directory is on this machine's own disk
(`artifact.on_network` is false). The coordinator then makes the job's
**read plan** (`cluster/weights.read_plan`), one entry per rank:

- `from`: for a rank whose copy is on a network mount, the page of a
  machine with a local copy, at that machine's **ring (Thunderbolt)
  address** and its page port -- the address the job's link already uses,
  so the request arrives on Thunderbolt and passes the peer gate. "" when
  the rank reads its own disk, or when no machine of the job has a local
  copy (everyone reads the mount, as before).
- `source`: that machine's name, for the page to show.
- `turn`: 0 or 1 (section 2).

The plan travels in the **Start** message (a new `reads` field: it is
known only after every Prepare answered). Each page puts its own rank's
entry in the rank's environment (`KNURLOGIC_WEIGHTS`, JSON). A path never
appears: the request names the artifact by identity, the file by its shard
name.

### The message

One new protocol kind, `Weights`, POSTed to the one route
(`/peer/v1/msg`) like every other, so it goes through the same gate
(`interfaces/page/peers.peer_refusal`: no Origin; loopback, Thunderbolt, or
a `--peer` address) and the same envelope (`cluster/transport`):

    Weights {identity, name, file, offset, length}

- `identity`: the artifact's identity; `name` its directory name (a name,
  never a path) to pick among copies, exactly as Prepare does.
- `file`: a shard's file name. It must be one of the artifact's own
  `*.safetensors` files (no `/`, no `..`): anything else is refused.
- `offset`, `length`: a byte range inside that file.

The serving page refuses when it has no copy with that identity, or when
its copy is itself on a network mount (it would only relay SMB). The reply
to a good request is **not** a JSON envelope: it is the bytes,
`Content-Type: application/octet-stream`, `Content-Length: length`, sent
with `socket.sendfile` (no copy through Python; the page never imports
mlx). A refusal is the usual typed `Failure` envelope. `transport.fetch`
is the client: it posts the envelope and reads the body straight into the
caller's buffer.

### On the fetching rank: substitution, with bounded memory

A rank loads lazily (`mlx_lm` with `lazy=True`: each array is a promise to
read a byte range of a shard file), cuts the model to its share (pipeline:
its layers; tensor: its slice of each array), and only then evaluates
(`runtime/model_host.evaluate_everything`). The fetch slots in before
that evaluation:

1. **Record** -- while the model is built lazily, `mx.load` is wrapped
   (`engine/split/weights.recording`) so every array it hands out is
   remembered with its shard file and tensor name. The offsets come from
   the shard's safetensors header, read once from the mount (a few MB,
   not the weights).
2. **Derive** -- the tensor split's cut (`engine/split/tensor`) builds new
   arrays from loaded ones (a slice, a repeated KV head). It records each
   as "derived from source X by f" (`weights.derived`), so the rank can
   rebuild it from X's bytes.
3. **Fill** -- for each array the model still holds (its share only: a
   pipeline rank dropped the other layers, a tensor rank holds slices),
   in the model's order: fetch the source tensor's bytes from the peer
   into a buffer, build the array, apply the derivation if any, evaluate
   it, and put it in place of the lazy one. Then the buffer is dropped.
4. Whatever is left lazy -- an array no record covers (a family's
   `sanitize` that stacks or renames, a bundled runtime's own tables), or
   every array after a fetch failed -- is read from the mount by the
   normal evaluation. Nothing breaks without the peer; it is only slower.

Memory is bounded: at most one source tensor's bytes and its built array
are in flight (two while the next tensor is fetched as the current one is
built), on top of what the rank holds anyway. A tensor split rank fetches
a whole source tensor to keep its slice -- the same bytes the mount read
moved before.

Each fetched GiB is written to the rank's marker (`reading_from`,
`fetched_bytes`), and the share's percentage rises as before (the page
measures what the rank's process holds).

**Fallback.** No `from` in the plan, or the peer refuses or goes away
mid-load: the remaining arrays are read from the mount, and the log says
how many bytes came each way.

## 2. Load order

Today every rank reads at once (`cluster/launch` only refuses a second
load on one machine). Now:

- **turn 0** -- ranks that read remotely (from a peer, or from the mount
  when nobody has a local copy) read immediately;
- **turn 1** -- ranks whose machine has a local copy read their own share
  only after every turn-0 rank has its share in.

Ordering is on only when both turns have ranks; otherwise every rank reads
at once, as before.

The wait is a rendezvous over the bell -- the plain TCP channel every
rank already holds to rank 0 (`engine/split/link.Link.align`) -- not a
collective: nothing spins on a Thunderbolt queue or the GPU while a rank
waits half an hour. Turn-1 ranks align *before* reading, turn-0 ranks
*after*; each rank aligns exactly once, so they meet when the last turn-0
rank is done. While a turn-1 rank waits, the serving machine's disk and
page cache work only for the remote ranks.

No overlap: the holder's disk is the shared bottleneck (it reads for the
peer and for itself), so reading in turn costs nothing against reading at
once, and keeps the page cache for the reads that need it. If a measured
load shows the holder idle at the end of the remote ranks' reads, a
bounded overlap (start at N% of the remote share) is the next step.

## 3. Progress on the page

The marker's `phase` stays `loading` through the wait: the job watcher
(`cluster/jobs.phase_of`) reads any phase it does not know as `joining`,
and a rank "joining" for `JOIN_S` is stopped. The wait is a separate
marker field instead:

- `turn: "waiting"` while a turn-1 rank waits; removed when it reads;
- `reading_from: <machine>` while a rank fetches from a peer;
- `fetched_bytes`, `mount_bytes`: what came each way.

`interfaces/page/loads.load_progress` turns them into each load entry:
phase `waiting` with `waiting: "turn"` (the existing `waiting` --
share in, the others still reading -- now says `waiting: "others"`), and
`reading_from`. A waiting rank is never `stalled`: its log is quiet and
its memory flat on purpose. The Load panel's per-machine line says
"waiting its turn" / "waiting for the others" / "loading from <machine>".

## Not changed

- A rank must still resolve the identity on its own machine at Prepare
  (the mount counts): the headers and the code it runs come from there.
  A machine with no copy at all is a separate change.
- One-Mac loads are untouched.
