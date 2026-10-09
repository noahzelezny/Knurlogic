# Drafting (MTP)

Some models ship a multi-token-prediction (MTP) head beside their weights.
With it, a decode step drafts one token ahead and checks it in a two-wide
pass, so a drafting step emits two tokens instead of one (a wrong draft
costs a replay): usually faster replies, with the same output
distribution. The cost is the head's
memory.

Drafting is not bit-identical to plain decoding: at a near-tie between two
tokens either can come out.

## Which models have a head

| family | drafting |
|---|---|
| Qwen 3.5 / 3.6, Qwen 3.8, GLM-5 | if the model has a head packed beside it |
| Gemma 4, DeepSeek-V4 | no |

```bash
knurlogic mtp                 # every model on this Mac
knurlogic mtp <model>         # one
```

The survey sorts models into three groups: a head **built** beside the
weights (it will draft), raw `mtp.*` weights a head **could be built from**,
and models that only **declare** a head in their config. Declaring without
shipping one is common and nothing is missing. `knurlogic doctor <model>`
and the MCP `drafting` tool say the same for one model.

## Settings

| where | setting | values |
|---|---|---|
| Load model (page) | MTP | On / Off |
| Load model (page) | Dynamic MTP | On / Off |
| Settings -> Knurlogic -> Presets | MTP | dynamic / every step / off |
| `knurlogic serve` | `--no-draft`, `--mtp-dynamic on\|off` | |
| launch setting | `KNURLOGIC_MTP`, `KNURLOGIC_MTP_DYNAMIC` | on / off |
| MCP | `load(..., draft=false)`, `fit(..., draft=false)` | |

- **Dynamic** (the default): each batch width's cost is measured with and
  without drafting and the cheaper is used. Usually faster; timing varies
  as it switches.
- **Every step**: draft whenever a head is bound. Steady timing.
- **Off**: the head is not loaded; its memory is free.

The `lean` preset turns MTP off. When a model will not fit with its head
but would without it, `load` says so instead of only refusing.

## On a cluster

A pipeline split drafts: the head lives on rank 0. A tensor split does not
draft. See [clusters.md](clusters.md).

Design: [../design/drafting.md](../design/drafting.md),
[../design/tensor-mtp.md](../design/tensor-mtp.md) (MTP on a tensor split,
planned).
