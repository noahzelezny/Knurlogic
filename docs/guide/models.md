# Getting models

Knurlogic runs models in MLX format (safetensors with a `config.json`).
GGUF is not supported.

## Supported families

| family | `model_type` | vision | drafting (MTP) | 8-bit KV |
|---|---|---|---|---|
| Qwen 3.5 / 3.6 | `qwen3_5`, `qwen3_5_moe` | yes | if the model has a head | yes |
| Qwen 3.8 | `qwen4_exp` | yes | if the model has a head | yes |
| Gemma 4 | `gemma4`, `gemma4_text` | yes | no | yes |
| GLM-5 | `glm5_next` | yes | if the model has a head | yes |
| DeepSeek-V4 | `deepseek_v4` | no | no | no |

VQ-quantized models of these families are supported. A VQ model ships its
own `model.py`, and that file is what runs it; one without it is refused
("re-download it").

## Download from the page (Hugging Face)

Open the model picker (Load model -> choose a model) and pick the
**Hugging Face** entry beside the families. Search; results are MLX models,
most downloaded first, at most 50.

Each row says:

- whether it **runs here**: it has safetensors and a chat `model_type`
  knurlogic knows. If not, the row is greyed with the reason.
- whether it **fits**: its weight size against the memory of the machines
  you picked.
- whether it is **gated**: without access the row says "needs access".
  Run `hf auth login` on this Mac; knurlogic uses whatever login
  `huggingface_hub` finds. There is no token field.

The download goes into the standard Hugging Face cache
(`~/.cache/huggingface/hub`). When it finishes, the model appears in the
picker under its family, ready to launch. Downloads go to this Mac only,
even with other machines picked.

### The Downloads button

The top bar's **Downloads** button has a badge with the number running.
It lists every download:

| state | what you can do |
|---|---|
| running | a bar and a stop |
| stopped or failed | resume (it continues from the files on disk) or delete |
| finished | clear the row (the files stay) |

Delete asks first, with the size it frees, and removes the model from the
Hugging Face cache. It is refused while a running server has that model
loaded: unload it first. A download running when the page stopped comes
back as stopped.

### Updates

A downloaded model is tagged **update** when the Hub has a newer revision
(checked once when the page starts; `knurlogic ui --offline` or
`HF_HUB_OFFLINE=1` skips the check). Clicking the tag downloads the new
revision into the same cache.

## Download with the hf CLI

```bash
pip install -U huggingface_hub
hf download mlx-community/gemma-4-e4b-it-8bit \
  --local-dir ~/Knurlogic/Models/gemma-4-e4b-it-8bit
```

## Where knurlogic looks

`knurlogic models` lists every model it finds and whether each can run
here. It reads these folders:

| store | folder (environment variable that moves it) |
|---|---|
| knurlogic | `~/Knurlogic/Models` (`KNURLOGIC_MODELS`) |
| Hugging Face | `~/.cache/huggingface/hub` (`HF_HUB_CACHE`, or `HF_HOME`) |
| LM Studio | `~/.lmstudio/models`, `~/.cache/lm-studio/models` (`LMSTUDIO_MODELS`) |
| Ollama | `~/.ollama/models` (`OLLAMA_MODELS`) |
| exo | exo's own model folders |
| saved folders | any folder you added (below) |

`knurlogic models` options: `--servable` (only what this engine can load),
`--store NAME` (one store), `--path DIR` (also scan a folder), `--only-path`,
`--json`.

## A folder on an external drive

```bash
knurlogic models add "/Volumes/My SSD/Models"
knurlogic models folders           # list; an unmounted one says (not mounted)
knurlogic models remove "/Volumes/My SSD/Models"
```

The folder is remembered (in `~/.knurlogic/model_folders.json`). While the
drive is not mounted it is skipped, and found again once it is back.
Removing a folder does not touch the models in it. The MCP
`model_folders` tool does the same.

## Which models can draft

`knurlogic mtp` surveys every model found: which have a drafting head built
beside the weights, which carry raw `mtp.*` weights a head could be built
from, and which only declare one. `knurlogic mtp <model>` answers for one.
See [drafting.md](drafting.md).
