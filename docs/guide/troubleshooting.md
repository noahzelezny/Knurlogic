# Troubleshooting

Start with `knurlogic doctor <model>` for a model, and
`knurlogic doctor --cluster` (on each Mac) for machines that do not see
each other. Both say what is wrong and what to do.

## A model will not load

| you see | what to do |
|---|---|
| "more space required" / `will not fit` | The weights are bigger than the working set. Check `doctor`: if it prints a `sudo sysctl iogpu.wired_limit_mb=...` command, the Mac has not been told it may use its own memory; run it. If it says raising the limit will not help, pick a smaller quantization, or split it across Macs ([clusters.md](clusters.md)). |
| fits only with MTP or vision off | Switch MTP or Vision off in Load model, or `load(draft=false)` / `load(vision=false)`. |
| `fits, low headroom` | It loads, with a narrower prompt chunk. Little room is left for long conversations. |
| the room line is amber | Little left to talk in. Try the `lean` preset or 8-bit KV ([settings-and-memory.md](settings-and-memory.md)). |
| refused: another load is moving memory | Wait; MCP `ready` names what is still loading. `force` overrides this one check. |
| an allowance is set | Settings -> Cluster: the allowance lowers what knurlogic may use. |
| `XX ... NOT INSTALLED` in doctor | The model's architecture is missing; the model cannot load. |
| "this VQ model does not ship its runtime (model.py)" | Re-download the model. |
| a model you downloaded is not in the picker | Only published models of a known architecture are offered. `knurlogic models` says why one is not servable. A folder on an unmounted drive is skipped. |
| a context length past the window is refused | Long context (YaRN) exists for Qwen 3.5 / 3.6 / 3.8 only, and the load is refused when the KV for that context does not fit. |

## Requests fail

| status | meaning | what to do |
|---|---|---|
| 400 "messages must be ..." / "invalid JSON" | a malformed request | fix the body |
| 400 "n > 1 is not supported" | | send the request n times |
| 400 `reasoning_effort ... is not one of` | a typo in the level | none, minimal, low, medium, high, xhigh |
| 400 "this request has images but ..." | the model has no vision, or it was launched with vision off | send text, or relaunch with Vision On |
| 400 "image URLs are not fetched" | | send the image as a `data:` URL |
| 400 a prompt past the context length | | compact ([compaction.md](compaction.md)) or raise the context length |
| 413 | the body is over `--max-request-mib` (512), or an image is too large | |
| 403 on `/v1/usage` or `/v1/prompt-cache` | only answered on the Mac itself | call it via 127.0.0.1 on that Mac |
| 503, `Retry-After: 5` | no model loaded, or still loading | retry; `GET /v1/models` shows `status` |
| 503 `insufficient_memory`, `Retry-After: 10` | not enough memory for this request now | retry, unload something, or use less context |
| 503 `cluster_failed`, `Retry-After: 30` | a rank of the split model died or stalled | the page relaunches it unless recovery is off ([clusters.md](clusters.md#recovery)) |
| a browser page gets refused | its origin is not allowed | `--allow-origin http://localhost:3000` |
| a request by DNS name gets refused | the Host is not one this Mac knows | `--allow-host studio.example.ts.net` |

## Slow first answer

The prompt is being read ("prefill"). A long prompt can take minutes.
Streaming clients get `knurlogic.progress` events with `done` / `total`.
The next turn of the same conversation reads only what is new
([prompt-cache.md](prompt-cache.md)); if it does not, see
`usage.knurlogic.cache.diverged`.

A wider prompt chunk reads long prompts faster if memory allows (Settings ->
Knurlogic -> Prompt chunk).

## Claude Code times out

Set `API_TIMEOUT_MS=3000000` (the Connect panel includes it). A client that
gives up and resends leaves nothing running: a request whose client hangs
up is cancelled.

## Tool calls come back as prose

`knurlogic doctor <model>` says whether the chat template's tool-call
dialect has a parser. If it says none was inferred, the model's calls will
read as text.

## Thinking runs too long

Send `reasoning_effort` (or `thinking: {"type": "disabled"}` on
`/v1/messages`), or set the model's **Thinking default** in Settings ->
Models. `usage.knurlogic.thinking` says what was applied.

## Macs do not see each other

Run `knurlogic doctor --cluster` on each. Common causes: no Thunderbolt
link (the page then answers on 127.0.0.1 only), multicast blocked (use
`--peer`), different knurlogic versions (shown on the page), the firewall
or macOS Local Network permission (named by doctor, never changed by it).

## A cluster launch is refused

The refusal names the machine and the reason: a share that does not fit,
the model not on that Mac, a version mismatch, RDMA without Thunderbolt 5
cables, a model that cannot be split, or another load in progress.

## Where things are

| | |
|---|---|
| settings | `~/.config/knurlogic/settings.json`, `allowance.json` |
| model folders, ledger | `~/.knurlogic/` (`KNURLOGIC_HOME`) |
| prompt-cache files | `~/.cache/knurlogic/prompt-cache/` |
| cluster job heartbeats | `~/.cache/knurlogic/jobs/` |
| recovery records | `~/.cache/knurlogic/recovery.json` |
| knurlogic's own model folder | `~/Knurlogic/Models` (`KNURLOGIC_MODELS`) |
