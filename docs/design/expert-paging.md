# Expert paging

Status: idea, not designed or built (first written 2026-10-01).

Goal: a MoE model that does not fit gets slower, never crashes. A completed
agent run is worth more than a fast one.

## Observation

Qwen3.5-397B 2.4 stage-1, MTP off, alone on an M4 Max 128 GB (2026-10-01): the fit
check passed by 2 GiB, macOS swapped ~15 GiB of the model while it loaded,
then paged most of it back in (3.7 GiB left). Decode stayed ~18 tok/s.
It worked because only ~17B of 397B parameters are active per token and
routing is skewed: the hot experts stayed resident, idle ones went to swap.

So oversubscribed MoE already runs -- by accident, through swap.

## Why swap is the wrong mechanism

- Swap must compress and write an expert before its memory is free;
  weights read from the model file (file-backed) can be dropped and
  re-read with no write.
- Uncontrolled: a routing burst to cold experts stalls decode with no
  warning, and the fit check can only say "fits" or "refused".

## The idea

- Hot experts resident; cold experts loaded from the safetensors file on
  demand (mmap / pread into a small expert cache), evicted by use.
- Per-layer routing counts decide hot vs cold (and can be warmed from a
  profile saved per model).
- Fit check gains a third answer: "runs, paged" -- weights over the budget
  but the dense part + a minimum expert cache fit; the page and MCP say
  the expected slowdown instead of refusing.
- Under pressure at run time, shrink the expert cache instead of letting
  macOS swap or Metal abort.

## Open questions

- How MLX/Metal holds weights: wired GPU buffers vs file-backed arrays;
  whether an expert can be swapped in without a copy.
- Cost of a cold-expert read from local SSD vs SMB; per-token stall bound.
- VQ artifacts: codebooks shared, indices per expert -- page indices only?
- Interaction with MTP drafting (drafts route too) and batching (more rows
  touch more experts).
