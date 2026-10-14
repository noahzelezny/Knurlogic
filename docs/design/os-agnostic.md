# knurlogic on Linux and Windows (0.2.0)

Status: survey and proposal (2026-10-13). Nothing built. Decisions marked
**DECIDE** are Noah's.

## Goal

Someone without a Mac can `pip install knurlogic`, open the page, load a
model and chat, from the same code that runs the Macs. The Mac stays the
first-class target: nothing it does today gets slower or loses a feature.

## What MLX supports (checked 2026-10-13)

From MLX's install docs (MLX 0.32.3, the version knurlogic pins):

| platform | package | requirements |
|---|---|---|
| macOS, Apple silicon | `mlx` | macOS >= 14 |
| Linux, Nvidia GPU | `mlx[cuda12]` / `mlx[cuda13]` | SM >= 7.5 (Turing and newer), driver >= 550.54.14 (580 for CUDA 13), glibc >= 2.35 |
| Linux, CPU | `mlx[cpu]` | glibc >= 2.35 |
| Windows | `mlx[cpu]`, `mlx[cuda12]` / `mlx[cuda13]` | not in the docs; wheels on PyPI (below) |

- Distributed: MLX's docs give NCCL as the backend for CUDA, beside ring
  (TCP), MPI and JACCL (RDMA over Thunderbolt, Macs only).
- `mx.fast.cuda_kernel` is the CUDA counterpart of `mx.fast.metal_kernel`
  (and `precompiled_cuda_kernel` for PTX/cubin).
- An MLX maintainer reported all MLX tests passing on CUDA (May 2026). An
  independent write-up calls the CUDA backend aimed at development more
  than throughput. Not measured by us: no Linux/Nvidia machine here.
- PyPI is ahead of the docs. MLX 0.32.3's wheels: `mlx` for macOS arm64,
  Linux x86_64 and aarch64, Windows amd64 and arm64; `mlx-cuda-12` and
  `mlx-cuda-13` for Linux x86_64 and aarch64 and Windows amd64; `mlx-cpu`
  for Linux and Windows. So native Windows, with CUDA, has a build; how
  complete it is is not documented and not tried.
- Linux aarch64 CUDA wheels cover NVIDIA's DGX Spark (GB10: Arm CPU and a
  Blackwell GPU sharing 128 GB, Ubuntu-based DGX OS): unified memory like
  a Mac, so knurlogic's memory model fits it better than a discrete GPU.

So: one engine (MLX) on all targets, no second ML backend.

## What ties knurlogic to the Mac

From a full read of `src/` (222 files). Only `engine/` imports mlx; cluster,
interfaces, machine and tuning are already mlx-free.

### 1. Metal kernels (large)

43 `mx.fast.metal_kernel` uses in knurlogic:

| where | kernels | fallback today |
|---|---|---|
| `engine/kvattn.py` | 2 (KV attention, combine) | none found |
| `families/deepseek/architecture/deepseek_v4.py` | 11 | none found |
| `families/glm5/.../_mlx_vlm/turboquant.py` | 26 | `mx.metal.is_available()` guard |
| `_mlx_vlm/models/rope_utils.py`, `gated_delta.py`, `deepseek_v4/hyper_connection.py` | 4 | portable fallbacks exist |

Plus **every VQ artifact's own `model.py`: 4 Metal kernels, 0 CUDA**
(checked across all local TheDrainFlorist builds). VQ decoding is VQ
Lab's runtime, not knurlogic's code.

Per kernel, two ways: a portable fallback in plain `mx` ops (correct on
every backend, slower), or a CUDA twin through `mx.fast.cuda_kernel`
(fast, a second source to keep equal). Recommended: fallback first for
every kernel (correctness everywhere), CUDA twins only where a measured
Linux run shows the fallback is the bottleneck. Goldens prove each.

### 2. Memory model (medium)

knurlogic assumes unified memory: GPU working set = the wired limit
(`sysctl iogpu.wired_limit_mb`), system RAM = model room. Affects
`machine/memory/*`, `tuning/fit.py`, `tuning/checks.py`,
`engine/runtime/memory_guard.py`, `cluster/launch.py`. On Nvidia: model
room is VRAM (`mx.device_info()` on CUDA, or NVML), separate from host
RAM. Needs a "device memory" notion with two implementations.

### 3. Machine readouts (medium)

`machine/metrics.py`, `machine/identity.py`, `machine/memory/{wired,
footprint,pressure}.py`, `cluster/launch._chip`, `tuning/pipeline_split.py`
(bandwidth table by Apple chip name): sysctl, vm_stat, ioreg,
system_profiler, IOKit/ObjC through ctypes, mach calls, scutil, pmset.
Mostly fail-soft already (they return None), so the page would run but
show nothing. Proposed: one `machine/platform/` package with a `darwin`
and a `linux` module behind the same functions (memory, GPU use,
temperature, chip, identity, sleep check); everything else calls only
those. Linux via `/proc`, NVML (`nvidia-ml-py`) and `psutil`.

### 4. Processes and files (small)

- ps / pgrep / lsof -> `psutil` (also BSD vs GNU flag differences).
- `fcntl.flock` in `machine/loadlock.py` -> works on Linux; native
  Windows would need `msvcrt.locking` or `filelock`.
- `os.pread` (page range reads) -> Linux fine; Windows native not.
- `start_new_session`, SIGTERM/SIGKILL -> Linux fine.
- `mount` output parsing (network-mount detection) -> Linux has
  `/proc/mounts`.
- Cache dir `~/.cache/knurlogic` (XDG) -> already right on Linux.
- Menu bar (`rumps`, AppKit) -> already darwin-only; Linux has none.

Native Windows turns this from small into medium (no `fcntl`, no
`os.pread`, no POSIX signals or sessions); WSL2 makes them Linux.

### 5. Clusters (medium-large)

- Discovery: Bonjour through `libSystem.dylib` -> `zeroconf` (pure
  Python) on Linux, or Avahi.
- Link detection: `ifconfig`, `route -n get`, `networksetup`,
  `system_profiler SPThunderboltDataType` -> `ip`/`/sys/class/net`.
- Transport: ring (TCP) should run on Linux; JACCL is Macs only; NCCL is
  the CUDA path. Weight streaming is HTTP + pread: portable.
- A mixed cluster (Mac + Linux box) is possible in MLX's model (the docs
  show a Mac launching onto a CUDA host) but knurlogic's split code
  assumes one backend per job. Out of 0.2.0 unless Noah wants it.

### 6. Packaging and CI (small)

- `pyproject`: `mlx==0.32.3; platform_system=='Darwin'` exists; add
  `mlx[cuda12]==0.32.3; platform_system=='Linux'` (or an extra:
  `pip install knurlogic[cuda]`, with `[cpu]` beside it); `mlx-lm` needs
  the same treatment. Classifiers, description and README lose
  "Apple Silicon only".
- CI: `test.yml` runs only on `macos-14`. A Linux job (`mlx[cpu]`) can
  run the non-GPU suite for free on GitHub's runners; GPU tests need a
  GPU runner (paid; see the no-CI-without-asking rule).
- Tests: 10 files import mlx at top level without a skip; one explicit
  darwin skip (`test_discovery`).

## Proposed phases

1. **Platform layer** (no behaviour change on the Mac): `machine/platform/`
   with `darwin` (today's code moved, not rewritten) and `linux`;
   psutil for processes; device memory abstraction; tests run on both
   with fakes. Linux CI job on `mlx[cpu]`.
2. **Engine on CUDA**: portable fallback for every Metal kernel, selected
   by backend, golden-tested against the Metal output on the Mac; memory
   guard on device memory. Done = a non-VQ model (an mlx-community quant)
   loads and chats on a Linux Nvidia machine, with the page.
3. **Clusters on Linux**: zeroconf discovery, Linux link detection, ring
   over TCP (and NCCL where it applies). Done = a 2-machine Linux split.
4. **Release 0.2.0**: README install per OS, classifiers, changelog.

VQ models on Linux need VQ Lab's runtime kernels ported (CUDA twins or
fallbacks in each artifact's `model.py`): VQ Lab's call and schedule,
not part of these phases unless Noah decides otherwise.

## Progress (branch os-agnostic)

The suite runs in a Linux arm64 container on the Mac (python:3.12-slim,
`mlx[cpu]` 0.32.3, `docker run --init`, 4 CPUs / 6 GB).

| step | Linux suite |
|---|---|
| main as it was | 130 failed, 2048 passed |
| kvattn off without Metal (`engine/backend`) | 43 failed |
| `machine/processes` (psutil) for ps / pgrep / lsof, the test leak sweep | 41 failed |
| test markers for CPU numerics, installed memory via psutil, test fixes | 1 failed, 2037 passed |
| machine readouts per OS (below) | 1 failed, 2049 passed |
| gated-delta's one home, discovery through Avahi | 1 failed, 2055 passed |

The Mac suite is unchanged at every step (2223 passed with the new tests).

Machine readouts off macOS: cpu and swap from psutil (on every OS now);
gpu use, GPU bytes and the chip from `nvidia-smi` (`machine/nvidia.py`;
a unified-memory GPU such as GB10 reports no bytes of its own); temperature
from the kernel's sensors; the id from `/etc/machine-id` (Linux) or
`MachineGuid` (Windows); available memory from the kernel's own estimate
(MemAvailable); per-process memory as resident size. macOS's pressure
level, thermal state and wired limit have no counterpart and read None
(the wired limit: total only, `known` False). Read live in the container:
real cpu, swap and memory figures, the rest None. A container has no
`/etc/machine-id`, so its id falls back to the hostname's hash.

- MLX's CPU back end has no bfloat16 `gather_mm` (MoE experts) and rounds
  bfloat16 / FP8 differently from a GPU. Tests that need either are marked
  (`tests/support/backends.py`: `bf16_gather`, `gpu_rounding`) and skip
  on the CPU back end only; they run on Metal and must pass on CUDA
  before the tag. Measured on the Mac: DeepSeek-V4's fallbacks for its 11
  Metal kernels give the kernels' numbers on the GPU; the CPU difference
  is MLX's.
- First Linux chat (Qwen3.5-2B 4-bit, CPU, container): the server
  installs, loads (1.6 GiB in 1.9 s from the container's disk; 167 s
  through Docker's share of a macOS folder) and answers /v1/models, then
  the warm-up prefill is OOM-killed (peak 7.7 GB at a 12 GB cap). Cause,
  from a py-spy dump at 3 GB: without Metal, gated-delta (Qwen3.5,
  Qwen4-exp, GLM-5-next) takes mlx-lm's ops fallback, which steps the
  prompt one token at a time into ONE lazy graph, a float32 state of
  heads x Dv x Dk per layer per token kept alive until the chunk is
  evaluated. Plain mlx-lm on the same model stays at 1.5 GB for a short
  prompt but decodes ~10 tokens in 176 s. The same fallback runs on CUDA,
  whose MLX has no gated-delta kernel. Noah: a working CUDA kernel
  first, speed later. `engine/gated_delta.recurrence` is now the one home
  the three families call: mlx-lm's (or GLM-5's vendored) Metal kernel,
  unchanged; a CUDA port of it (one warp per value row, untested until
  an Nvidia machine); elsewhere mlx-lm's ops evaluated every 64 tokens,
  bit for bit the same numbers. Rerun: no OOM (3.7 GB of 6), but the
  warm-up had not finished in 15 min on one core. OPEN: CPU speed (plain
  mlx-lm was ~17 s a token on this model). `prefill_step_size` was the
  wrong key (the knob is KNURLOGIC_PREFILL_CHUNK; unknown keys pass by
  design). With `--set KNURLOGIC_PREFILL_CHUNK=64` the first Linux chat
  answered: warm-up 586 s (89 tokens), the reply's 12 tokens in 883 s,
  ~13 s a token, prefill chunks of 1-2 tokens in the log. Usable on a
  GPU, not on MLX's CPU back end.
- Peer discovery off macOS (Noah: the service the OS already runs):
  `cluster/discovery/avahi.py` asks avahi-daemon through libavahi-client,
  the same interface as Bonjour's. Two containers found each other. Windows
  (dnsapi's DNS-SD) is still to write.
- OPEN: the two-page cluster tests (`tests/cluster/test_recovery.py`,
  `test_cluster_jobs.py`: a killed rank, then `watch_once`) fail now and
  then in the container, and pass alone and on the Mac. A theory that page
  B stops the job first did not hold (slowing B's watcher did not fix
  it). Not yet understood.

## Decided (Noah, 2026-10-13)

- **Docker first.** Everything that can run without an Nvidia GPU is built
  and tested on the Mac in Linux arm64 containers (`mlx[cpu]`): the
  platform layer, kernel fallbacks against the Metal output, the page, a
  two-container TCP cluster. CUDA is checked on a friend's Nvidia machine
  before 0.2.0 is tagged; no paid compute.
- **VQ models are in scope.** The 128 GB unified-memory boxes (DGX Spark
  and the like) are the audience for the 397B. Each VQ artifact's 4 Metal
  kernels need a CUDA twin or a fallback: VQ Lab's runtime, so planned
  with VQ Lab when the engine side is ready.
- **Windows**: native, after Linux (MLX ships Windows wheels).
