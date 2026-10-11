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
`Content-Type: application/octet-stream`, `Content-Length: length`, read
with `pread` at most 64 MiB at a time and sent (the page never imports
mlx). Not `socket.sendfile`: on a cold SoftRAID array 8 sendfiles at once
moved 0.08 GB/s against 0.76 GB/s for 8 preads of 64 MiB, and a 397B
VQ-3.5bpw share came at 0.48 GiB/s with it. A refusal is the usual typed `Failure` envelope. `transport.fetch`
is the client: it posts the envelope and reads the body straight into the
caller's buffer.

### On the fetching rank: one opener, the share by its parameters

Every tensor file of an artifact is opened by one function,
`engine/model/weight_files.load` -- the trunk through knurlogic's copy of
mlx-lm's `load_model` (`engine/model/load_model.py`, pinned to the
installed mlx-lm by `tests/engine/test_load_model_pin.py`), a family's
drafting head and vision tower directly. While a split rank has a peer
(`weight_files.use_peer`, set at the rank's start from its plan):

1. **The trunk is opened deferred.** Its files are opened lazily, as
   before, and each array is recorded with its file and tensor name
   (`weight_files.origin`). Names alone would not do: a bundled VQ
   runtime renames tensors in its `sanitize` (the 397B's `sz_codes` is a
   parameter named `codes`), so the record follows the array itself.
2. **The model is cut to the share,** as before. The tensor split's cut
   (`engine/split/tensor`) records each new array as "made from source X
   by f" (`split/weights.derived`), so the rank can redo it on X's bytes.
3. **The share is filled** (`split/weights.fill`): every array
   `model.parameters()` still holds whose origin is a file's tensor (or a
   cut of one) is fetched over the link, built, cut if it was, and put in
   place with `model.update`. On the 35B and 397B models measured, every
   array a lazily built model holds is a parameter.
4. **The rest is read from the files** by the normal evaluation: a
   parameter no file gives as-is (a runtime's computed tables), or
   everything after a failed fetch.

**The head and the tower** are fetched as their family opens them:
`weight_files.load(path)` fetches the file's tensors (only `keys=` when
the family asks for some of a trunk shard) and hands back built arrays.
A file outside the artifact's folder is read from disk.

**How it is fetched** (`weight_files.ranges`): tensors in file order,
neighbours in one file merged into one request up to 64 MiB, a larger
tensor in 64 MiB pieces, 8 requests in flight (`PIECE`, `PARALLEL`).
Measured from the M4 against the M3's SoftRAID array, cold: 0.50 GB/s
with one request in flight, 2.04 GB/s with eight. At most `BUDGET`
(4 GiB) of fetched bytes wait to be built; a tensor larger than that
goes alone. A tensor split rank fetches a whole source tensor to keep its
slice: the bytes the file read moved before.

**Failures.** An `Exception` while fetching is logged and the files are
read instead: a fetch never fails a load, it is only slower.
`KeyboardInterrupt` and `SystemExit` are never caught; a stopped load
(`LOAD_STOP`) stops the fetch between ranges.

Each fetched GiB is written to the rank's marker (`reading_from`,
`fetched_bytes`), and the share's percentage rises as before.

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
waits several minutes. Turn-1 ranks align *before* reading; turn-0 ranks
align once their share, head and tower are all in (`split/weights.done`, after
binding -- not after the share alone, or the holder's own read competes
with the remote rank's last fetches). Each rank aligns exactly once.

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

## Measured (2026-10-11, M4 rank fetching from the M3 over Thunderbolt 4)

From the ranks' logs. Before: the M4 read its share over SMB cold at
0.18-0.33 GB/s (a 397B VQ-3.5bpw pipeline share: 90.9 GiB in 451 s).

| model, split | fetched over the link | read from the files | rate |
|---|---|---|---|
| 397B VQ-3.5bpw, pipeline | 84.4 GiB in 77.6 s | 68 arrays (row tables) | 1.09 GiB/s |
| 397B VQ-3.1bpw, pipeline, cold | 75.3 GiB in 52.9 s | 74 arrays | 1.42 GiB/s |
| 397B VQ-2.6bpw, pipeline | 65.3 GiB in 33.0 s | -- | 1.98 GiB/s |
| 35B 8-bit, tensor | 34.3 GiB in 24.9 s | 0 | 1.38 GiB/s |
| 35B VQ-4.6bpw, pipeline, warm | 11.1 GiB in 3.0 s | 0 | 3.7 GiB/s |
| DeepSeek-V4-Flash VQ-3.2bpw, pipeline | 62.9 GiB in 45.3 s | 189 arrays, 0.34 GiB | 1.39 GiB/s |
| GLM-5.3-Flash VQ-2.7bpw, pipeline | 62.5 GiB in 43.4 s | 0 | 1.44 GiB/s |
| Qwen3.8-Flash-Next VQ-4.4bpw, tensor | 80.2 GiB in 56.6 s | 0 | 1.42 GiB/s |

Read from the files: on DeepSeek-V4 the attention compressor's
`wkv_gate`, which its loader builds from file tensors rather than taking
one as-is, so no file range gives it. 0.34 GiB, but scattered small reads
over SMB: about 19 s on a cold load, 0.0 s once that Mac has them cached.
On the 397B, its VQ runtime's row tables. Gemma 4 does not split yet (both
splits refuse it), so it never streams; its tower loads through the same
opener on one Mac.

The 397B's drafting head (5.41 GiB) and vision tower (0.85 GiB) come
over the link in about 5 s; read over SMB they took 129 s and 149 s. The
rate is the holder's array: it read 1.1-1.5 GB/s while serving, the same
as for its own share. With socket.sendfile on the serving side the
3.5bpw share came at 0.46-0.49 GiB/s.
