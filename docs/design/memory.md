# Memory accounting

## knurlogic/machine/loaded.py

`discover` answers what is on the disk. This answers what is in memory,
and they are different questions with different answers: forty artifacts
on a volume, none of them loaded, is a normal state.

Nobody should have to open exo to see exo's models, or run `ollama ps` in a
terminal to see ollama's. A machine has one pool of memory and every
runtime is spending from it, so one page shows all of them.

Four runtimes, three channels, none of them a guess:

- **knurlogic** -- its own `/status.json`, which names the artifact and its
  resolved settings.
- **exo** -- `GET /state` -> `instances` (what was asked for) and `runners`
  (what is actually up). Both matter: an instance with every runner in
  `RunnerShuttingDown` is not loaded, and reading only the instance list
  would report it as though it were.
- **ollama** -- `GET /api/ps`, which is precisely "what is resident", as
  distinct from `/api/tags`, which is what is downloaded.
- **mlx-lm / mlx-vlm** -- `GET /v1/models` on a port that answers it. Their
  servers have no "what is loaded" endpoint -- `/v1/models` lists what they
  could serve -- so what is reported is the port and what it offers, marked
  as such rather than dressed up as residency.

Everything is an HTTP read with a short timeout. A runtime that is not
running is not an error; it is the ordinary case and it reports as absent.

## knurlogic/machine/metrics.py

GPU busy, CPU busy, memory pressure, swap and thermal state are the
things that quietly move decode speed, and each is readable without sudo:

- **gpu** -- `ioreg -c AGXAccelerator` PerformanceStatistics, the same
  "Device Utilization %" Activity Monitor's GPU History draws.
- **cpu** -- `host_statistics(HOST_CPU_LOAD_INFO)` tick deltas, through
  ctypes.
- **swap** -- `sysctl vm.swapusage`.
- **pressure** -- `sysctl kern.memorystatus_vm_pressure_level` (1 normal,
  2 warn, 4 critical); with swap growth, whether swap is happening now.
- **thermal** -- `NSProcessInfo.thermalState` (nominal / fair / serious /
  critical), through the Objective-C runtime -- what throttling follows.
- **temp_c** -- the hottest die sensor, from IOHIDEventSystemClient,
  through ctypes so nothing extra is installed. Undocumented API: if Apple
  moves it, this reads None and the line goes blank; it does not break. A
  °C line shows heat building before the state changes.

History is kept in this process, sampled when status is asked for and no
more often than `MIN_INTERVAL_S`, so a page that polls fast does not make
the machine work harder to report how hard it is working. Every probe
fails to None, never to 0: a missing reading drawn as idle is a lie.

## knurlogic/machine/status.py

A runtime shows up as one opaque number in Activity Monitor --
"python3.13, 45 GB" -- and you cannot tell weights from reclaimable cache
from a transient peak, or see which architecture actually loaded. Every one
of those is available; this surfaces them.

Two cautions, both measured and both printed next to the numbers:

- `ps` RSS and the framework's own accounting agree on a small resident
  model (12.09 GiB vs 11.61 on a 27B) and diverge under pressure -- a probe
  read 11.7 GiB from `ps` while the process held ~60. Neither number alone
  is trustworthy, so both are shown.
- Cache memory is reclaimable. Counting it as usage makes a runtime look
  like it is eating the machine when it is holding freed buffers it will
  hand back.

**One process or several.** `snapshot()` answers for the process it runs in
and that is all it can honestly do -- a remote node's numbers have to come
off that node. `aggregate()` is the shape a cluster arrives in: a list of
per-node snapshots plus the rollup, and it is what `/status.json` serves
even for one node, so a client written against one machine does not have to
be rewritten when a second appears. The single-node keys stay at the top
level for the same reason.

A rollup sums memory and does not average it: two nodes each 40 GiB active
are 80 GiB of weights held, not 40. It also reports how many nodes
answered, because a sum over nodes that did not reply is a smaller number
that looks like good news.

## knurlogic/machine/wired.py

On Apple Silicon `iogpu.wired_limit_mb` caps how much memory the GPU may
wire, and it is what the framework's "recommended working set" follows.
Measured on a 96 GB machine:

```
iogpu.wired_limit_mb: 86016        -> 84.0 GiB
framework working set:                84.0 GiB   of 96 GiB installed
```

So an artifact that "does not fit" often fits perfectly well -- the machine
was simply never told it could use its own memory. That is the most common
way somebody concludes local inference does not work on their Mac.

**What this module will not do.** It does not set the value. Changing it
needs root, it is a system-wide setting, and a package that quietly raises
how much memory the GPU may wire is not a package anyone should install.
knurlogic works out the number, prints the command, says what it costs, and
the person runs it.

**The reserve is a judgement, not a measurement**, and is labelled as one
everywhere it is used. macOS still has to run: window server, browser,
editor. Leaving too little does not OOM the model; it wedges the machine.
