# Hugging Face in the model picker

The picker has a "Hugging Face" entry beside the families. It searches the
Hub for MLX-format models, says whether this machine can run one, and
downloads it into the standard Hugging Face cache, where discovery
(machine/discover.py) already reads. When the download finishes the model is
an ordinary local model under its family, ready to launch.

Code: `interfaces/page/hub.py` (the page's server side),
`views/picker.js` (search and rows), `views/memory.js` (INSTANCES cards),
`engine/arch.py` `supported()` (the architecture question).

## Endpoints (the page's own server)

- `GET /hub/search.json?q=` MLX models matching q, most downloaded first,
  50 at most. The Hub matches one substring; further words narrow the result.
- `GET /hub/repo.json?id=org/name` fetched when a row opens: size of the
  weights, `model_type` from its config.json, `supported` and `why`,
  `gated` and `access`.
- `GET /hub/downloads.json` downloads: state, bytes on disk, total, and why
  a failed one failed. The list is kept in `downloads.json` in Knurlogic's
  cache folder, so it survives a restart of the page.
- `POST /hub/download.json {action: download | cancel | dismiss | delete, id}`.
  `id` is `org/name` and nothing else.

## How it decides

- Runs here: the repo has safetensors, a `model_type`, it is not a non-chat
  type, and `arch.supported()` finds it, either claimed by a family manifest
  or as a module of an installed host package. It reads files and imports no
  mlx. Otherwise the row is greyed with the reason.
- Fits: the repo's weight bytes against the picked machines' working sets,
  the same number local rows use.
- Gated: the Hub says the repo is gated and the login on this machine is
  checked. Without access the row says "needs access" and to run
  `hf auth login`. The token is whatever huggingface_hub finds; there is no
  token field.

## Download

One thread per repo in the page's server, `snapshot_download` of the config,
tokenizer and safetensors files. Progress is the bytes on disk in the repo's
cache folder against the total. INSTANCES shows a card with a bar and a
cancel; a failure stays as a FAILED card with the reason until dismissed. A
second download of the same repo while one runs does nothing. Cancelling
stops at the next chunk and keeps the partial files so a retry resumes.
A download that was running when the page stopped comes back as stopped.
Stopped is not failed: downloading it again resumes from the files already
on disk. Done: the server forgets its model scan and the page reads /models.json
again.

## Downloads overlay

The nav has a Downloads button with a badge counting the downloads running.
It opens an overlay listing every download in the file above: a bar and a
stop for a running one; resume and delete for a stopped or failed one; clear
for a finished one (the files stay). Delete asks first, with the size it
frees. The list refreshes every two seconds while the overlay is open.

Delete removes the repo's folder from the Hugging Face cache and its row, and
the model leaves the picker. It is refused while a running server has the
model loaded from that folder (the servers registry says which): unload it
first. A download still running is stopped first.

## Out of scope

- GGUF and any format that is not MLX safetensors.
- A token field.
- Downloading to another machine: with several machines picked the row says
  "downloads to this Mac".
