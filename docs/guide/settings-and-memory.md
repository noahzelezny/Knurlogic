# Settings and memory

Every setting has a value worked out from the model's `config.json` and the
memory free, with the reason beside it. You change one only when you want
something different. A setting you set always beats the worked-out value.

## The Settings panel

Changes are **staged**, not sent as you make them. Closing Settings with
something staged asks once: **Apply**, **Discard** or **Keep editing**.
A setting that can change live goes to the running server; one read at
launch becomes that model's launch setting for next time, and nothing is
restarted.

| tab | what is in it | applies to |
|---|---|---|
| **Knurlogic** -> Presets | the preset every model launches with, each setting it sets, and per-chip rounding | every machine |
| **Knurlogic** -> Compaction | [compaction.md](compaction.md) | every model on every machine, from the next request |
| **Cluster** | each machine: what it is, its memory, its wired limit and its knurlogic allowance | that machine |
| **Models** | each base model's own launch settings, and a running one's live values | that model |

## Presets

| preset | what it is |
|---|---|
| `default` | the measured settings; fastest |
| `lean` | most context and most agents: 8-bit KV where the family takes it, 512-token prompt chunks, MTP off, 1 GiB cache reserve |

Pick one in Settings -> Knurlogic, per model in Settings -> Models
(Preset), with `knurlogic serve --tune lean`, or `tune` in the MCP. A
setting changed beside a preset beats the preset's value for that setting.
The names `safe`, `fast`, `stable` and `balanced` are still accepted
(`safe` is `lean`; the others are `default`).

## The settings

| setting (page name) | variable | values | |
|---|---|---|---|
| Prompt chunk | `KNURLOGIC_PREFILL_CHUNK` | auto, 512, 1024, 2048, 4096 | prompt tokens read per step. Wider reads long prompts faster but each step's memory spike grows. Auto picks the widest that fits in the room left, else 512. Output is the same at every width. |
| Cache reserve | `KNURLOGIC_CACHE_LIMIT_GB` | 1, 2, 4, 8 GiB (default 4) | freed memory held for reuse. No measured speed difference; less leaves more free. |
| MTP | `KNURLOGIC_MTP`, `KNURLOGIC_MTP_DYNAMIC` | dynamic, every step, off | [drafting.md](drafting.md) |
| KV cache | `KNURLOGIC_KV_BITS` | bf16, 8-bit (6 and 4 by `--set` only) | below |
| Context length | `KNURLOGIC_CONTEXT_LENGTH` | tokens | the longest prompt + answer a request may use. A cap: nothing is reserved. A longer prompt is refused. Default: the model's window. |
| Long context | `KNURLOGIC_LONG_CONTEXT` | off, yarn | below |
| Vision | `KNURLOGIC_VISION` | on, off | [vision.md](vision.md) |
| Thinking default | `KNURLOGIC_THINKING_DEFAULT` | default (the model's own), or a level the model's template has | the level for requests that name none. Live. |
| KV kernel | `KNURLOGIC_KV_KERNEL` | on, off | reads an 8-bit cache directly during decode. Leave on; off is for comparison. |
| Per-chip rounding | `KNURLOGIC_CROSS_CHIP` | page: on, off; `--set` also takes `auto` | for a split across different Mac chips: off pads some matmuls so every chip gives the same tokens (+2-6% time on those calls). |
| Preset | `KNURLOGIC_PRESET` | default, lean | per model |

The MCP `settings` tool and `knurlogic doctor <model>` list every value with
the measurement behind it and whether it can change live or only at launch.

### KV cache bits

The attention cache holds every token of every conversation. 8-bit takes
about 53% of bf16's memory, so roughly twice the context fits. Decode is
measured about 5% slower at 6k tokens of context and 6% at 16k (M4 Max); prefill
is not slower. 6 and 4 bits take less (41%, 28%) but are not measured on a
real model and are not offered on the page. GLM takes 8 only; DeepSeek-V4
takes none. Recurrent state and sliding windows are not quantized.

### Long context (YaRN)

For Qwen 3.5, 3.6 and 3.8 only. `yarn` raises the context cap to 1,048,576
tokens with the rope scaling Qwen documents, applied when the model loads
(the files are not changed). Setting a context length past the model's
window turns it on. The cost, per Qwen: short prompts may get slightly
worse, so leave it off unless you need more than 262k. Every token holds
KV memory; the load is refused when the machine cannot hold the chosen
context's KV.

## Memory

### The wired limit

macOS caps how much memory the GPU may use (`iogpu.wired_limit_mb`). Often a
model that "does not fit" fits once the Mac is told it may use its own
memory. Knurlogic never changes this setting: it needs `sudo` and is
system-wide. It works out the number and gives you the command, for
example:

```bash
sudo sysctl iogpu.wired_limit_mb=<MB>
```

Where to see it: `knurlogic doctor <model>` (it also says when raising the
limit will not help: the model needs a smaller quantization or another
Mac), and Settings -> Cluster, per machine (type a value and it gives the
command and a warning for that machine). The reserve left for macOS is a
judgement: too little wedges the Mac, not just the model.

### The knurlogic allowance

The most memory knurlogic may use on this Mac, for a Mac that also runs
other things. It only ever lowers what knurlogic would take: the fit check
for a load and the running server's memory guard both count against it.
Set it per machine in Settings -> Cluster. It is kept in
`~/.config/knurlogic/allowance.json`.

### What fits

A load is refused when the weights do not fit the working set (under the
allowance). The MCP `fit` tool and the picker answer with the arithmetic:
`fits`, `fits, low headroom` (it still loads, with a narrower prompt chunk)
or `will not fit`. Free memory is counted as free + inactive (the file
cache macOS hands over on demand).

While serving, a request that would run the machine out of memory waits,
or gets a 503 `insufficient_memory` with `Retry-After: 10`.

### Where the memory went

The page's Memory panel and `knurlogic loaded` show every runtime on the
Mac (knurlogic, ollama, exo, other OpenAI ports), swap and pressure. The
model server's `/status.json` has its own numbers.

## Where settings are kept

| what | file |
|---|---|
| the default preset, compaction, per-chip rounding | `~/.config/knurlogic/settings.json` (`XDG_CONFIG_HOME` honoured) |
| the allowance | `~/.config/knurlogic/allowance.json` |
| per-model launch settings | chosen in Settings -> Models, applied when that model loads |

`--set KEY=VALUE` on `knurlogic serve` beats a saved value for that launch.
A model server shows its resolved settings at `/settings.json`.

Design: [../design/settings.md](../design/settings.md),
[../design/memory.md](../design/memory.md),
[../design/kv-cache.md](../design/kv-cache.md).
