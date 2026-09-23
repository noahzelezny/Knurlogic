# knurlogic chat panel: feature spec, drawn from exo's dashboard

I only read code; nothing was written or run. Line numbers are from the files as read. "Not verified" marks an inference.

## 0. Findings that shape the design

1. **The page's `/v1/chat/completions` is mlx-lm's own handler, and it rejects images today.** knurlogic's routes are added to the engine's handler on the same port (`src/knurlogic/engine/seam.py:210-262`; `interfaces/web.py:309-355`). `web.py` has no `/v1/chat/completions` route, so the POST falls through to mlx-lm. The mlx-lm I read (`/opt/anaconda3/lib/python3.12/site-packages/mlx_lm/server.py:134-141`) throws `ValueError("Only 'text' content type is supported.")` on any non-text content part.
   - I did not verify that knurlogic's runtime uses this same mlx-lm install.
   - Conclusion: the UI cannot send `image_url` until the engine handles vision. The engine must either replace that content-flattening step or pull the image parts out before mlx-lm sees them.
2. **mlx-lm already streams prefill progress.** It writes `: keepalive {processed}/{total}\n\n` SSE comments (`server.py:1410-1413`), fed from the prompt progress callback (`:926,986,1040-1044`). This works like exo's `: prefill_progress {json}`, with a different key and format. The chat panel can parse it with no server change.
3. **mlx-lm already reports cache hits.** `usage.prompt_tokens_details.cached_tokens` is set at `server.py:1346`, and the bench already reads it (`index.html:1609`). That makes "turn N costs only turn N's new tokens" directly visible in the UI: show `new of total` per turn. This is the acceptance meter for images as real context.
4. **exo resends every image on every turn, as a full base64 data URL.** See `exo/dashboard/src/lib/stores/app.svelte.ts:2402-2433`: for each earlier message, image parts go first and text last. knurlogic should keep the same OpenAI wire format, so it stays compatible with agents and MCP, and make the reuse happen server-side:
   - key the encoded image by content hash (not verified that no such cache exists in engine/ yet);
   - build the prefix trie over token ids in which the image's placeholder tokens stand for that hash.
   - The UI's job is to send byte-identical data URLs every turn. So it downscales and re-encodes once at attach time, stores the result, and never re-encodes (§3.4).

## 1. exo dashboard map (`~/exo/dashboard/src/lib`)

| Piece | File:lines | What it does | Port? |
|---|---|---|---|
| Data model | `stores/app.svelte.ts:266-319` | `MessageAttachment{type:image\|text\|file\|generated-image\|pdf, name, content?, preview?(dataURL), mimeType?, pageImages?[]}`. `Message{id, role, content, timestamp, thinking?, attachments?, ttftMs?, tps?, tokens?}`. `Conversation{id, name, messages, createdAt, updatedAt, modelId, …, enableThinking}` | Shape nearly verbatim, as plain JS objects |
| Persistence | `app.svelte.ts:321, 627-670` | localStorage key `exo-conversations`, whole array as JSON; `tokens` (logprobs) stripped before save. Images stored inline as data URLs, so it will hit localStorage's ~5 MB quota quickly (the error is only `console.error`, line 669) | Reimplement: use IndexedDB for image blobs (§3.2) |
| Send path | `app.svelte.ts:2279-2530` | Builds attachments (2308-2360). Text/PDF text is appended as `\n\n[File: name]\n```…```` (2322, 2334). The fixed system prompt is at 2394-2398. History = `messages.slice(0,-1)` mapped to multimodal parts; `reasoning_content` is echoed back for assistant turns (2455-2459). Body: `{model, messages, temperature:0.7, stream:true, logprobs:true, top_logprobs:5, enable_thinking?}` (2509-2525) | Logic nearly verbatim. Drop logprobs and the fixed system prompt |
| SSE parser | `app.svelte.ts:2148-2220` `parseSSEStream` | Line-buffered. `: key json` comments go to `onEvent[key]`. `data: [DONE]` is skipped. The tail buffer is flushed at the end. The loop aborts if the conversation was deleted mid-stream | Nearly verbatim (about 50 lines). Add a parser for `keepalive P/T` |
| Thinking split | `app.svelte.ts:2107-2140` | Strips `<think>…</think>` (including a still-open block while streaming) into `thinking`, merged with `delta.reasoning_content` | Port verbatim; qwen3_5 and glm5 emit thinking |
| Prefill progress | `app.svelte.ts:2564-2566, 2641-2656` | Set from the `prefill_progress` event; cleared on the first token and at stream end | Reimplement against the keepalive format |
| Stats | `app.svelte.ts:2657-2663` | `generation_stats` event gives TPS; TTFT is measured client-side | knurlogic already does this better (`index.html:1603-1618`) |
| Abort | `app.svelte.ts:2505-2506` | `AbortController` held on the store | Port |
| Files / PDF | `types/files.ts:5,34-38,212-333` | pdfjs-dist, worker from jsdelivr. Max 20 pages, scale 2.0, each page rendered to an OffscreenCanvas as JPEG q0.8 into `pageImages[]`, plus extracted text capped at 100k chars and marked `[truncated]` / `[showing 20 of N pages]` | Port the logic. Load `pdfjs-dist` ESM from jsdelivr lazily, only when a PDF is attached |
| ChatForm | `components/ChatForm.svelte:71,155-200,265-280,471` | Accepts image/text/pdf. Paste of image files, or of text over 2500 chars, turns into a `pasted-text.txt` attachment. Drag and drop. Enter sends; Shift+Enter makes a newline; IME guard (`isComposing\|\|keyCode===229`) | Port behaviour. knurlogic's `index.html:1554` already guards `isComposing` |
| ChatAttachments | `ChatAttachments.svelte` (94 lines) | Chips with an icon by category, name truncated around the extension, size, remove × | Reimplement (trivial) |
| ChatMessages | `ChatMessages.svelte:29-210` | Scroll-follow with a "jump to bottom" button when not at the bottom; auto-scroll on send; copy, edit-and-regenerate, delete with confirm, regenerate last | Behaviour spec only |
| MarkdownContent | `MarkdownContent.svelte:2-100` (1074 lines) | marked + highlight.js + KaTeX. Code blocks get a language header and a copy button (`data-code` URI-encoded). Code blocks are protected by placeholders before math processing (93-97) | Reimplement lighter (§3.6) |
| ImageLightbox | `ImageLightbox.svelte` (96 lines) | Fullscreen overlay; Esc or click closes; download button derives the extension from the data-URL MIME type | Nearly verbatim as about 30 lines of vanilla JS |
| PrefillProgressBar | `PrefillProgressBar.svelte:12-40` | percent = processed/total; ETA from the measured rate after a 200 ms minimum window; k-format token counts | Port the math verbatim |
| ChatSidebar | `ChatSidebar.svelte:41-116,307` | Search by name, inline rename, delete with confirm, delete all | knurlogic's runs list (`index.html:1499-1516`) already covers most of this |
| ModelSelector | `ChatModelSelector.svelte:55-212` | Recommends the largest model that fits memory, per category | knurlogic's picker already does fit-by-working-set (`index.html:1140-1190`). Skip |
| Favorites / recents | `stores/favorites.svelte.ts:7`, `recents.svelte.ts:7-8` | localStorage `exo-favorite-models` (a set) and `exo-recent-models` (max 20, `{modelId, launchedAt}`) | Optional: add a ★ and a recents section to the picker. Trivial |

Licence: exo is Apache-2.0. Near-verbatim ports need its copyright and licence header in a comment block; single-file inline is fine.

## 2. knurlogic page map (`src/knurlogic/interfaces/web/index.html`, 1629 lines, vanilla JS, one file)

- **Layout** (`450-570`):
  - left `aside.side` holds `#newchat` and `#chats` (the runs list);
  - `main` holds `#memory` and `#try` (Bench: `#log`, `form#f > input#q`, `#again`, `#fresh`);
  - right `aside.right` holds `#runningpanel` (served model, `#badges`, `#resident`) and `#diskpanel` (the picker button `#openpick`, `#loadopts`, `#launch`);
  - there is a modal `#picker` and a sheet `#sheet` (Settings/Connect nav, `572-599`).
- **Theme**: CSS tokens on `:root` (`10`) with `prefers-color-scheme:light` overrides (`20`, …). Chat styles must follow the same token pattern.
- **Polling**: `setInterval(tick,2000)` to `/status.json`; `setInterval(loadResident,5000)` to `/loaded.json` (`887, 1368, 1409`).
- **`act(payload)`** (`1342-1352`): POST `/loaded.json` for load/unload; refusal shown via alert; then `loadResident()`.
- **Picker** (`1112-1240`): groups by family and base name, fit against `LASTWS`, VQ and MTP tags.
- **Run memory** (`1479-1528`): localStorage `kn.runs`, capped at 40, `{id, title, at, tune, msgs:[{role, content, met}]}`.
- **Bench send** (`1572-1627`):
  - sends only `[{role:'user', content}]`, with no history, `max_tokens:256`;
  - `stream_options.include_usage`;
  - the nonce prefix `withNonce` busts the prefix cache on purpose (`1566-1570`);
  - metrics are computed as prompt − cached (`1603-1618`);
  - the SSE loop ignores `:` comment lines (`1597`).

Relevant gap: Bench deliberately defeats the prefix cache, sends no history, and is text-only. Chat is the opposite on all three counts, so it must be a separate panel, not a mode of Bench.

## 3. Chat feature spec

### 3.1 Placement and UI
- Add a `section.panel#chat` in `main`, and turn `#try` into a tab pair, **Chat | Bench**, sharing the same slot. Chat is the default when a model is served. Two segmented buttons toggle `hidden`. Do not persist the tab (same reasoning as `576-580`).
- The left `aside.side` list shows chats or runs depending on the active tab. There are two stores, and `#newchat` relabels to "+ New chat" or "+ New run".
- Model: chat always targets the served model (`served_path()`, `seam.py:285`). The header shows its name from `#title`. If nothing is served, the input is disabled and says "choose a model", with a button to `#openpick`. No per-chat model selector; each conversation records `model` for display only.
- Vision gate: disable the attach button unless the served model supports vision.
  - Proposed field: `/loaded.json` or `/status.json` exposes `vision: true`. This needs a server change (§4).

### 3.2 Storage
- **localStorage `kn.chats`**: `[{id, title, model, at, updated, msgs:[{id, role, content, thinking?, att?:[{kind:'image'|'pdf'|'text', name, hash, mime, w, h, bytes, text?, pages?:[hash]}], met?}]}]`. Keep 100 chats, with titles taken from the first user message (52 chars, as `record()` does at `1530`).
- **IndexedDB `kn-img`** (store `blobs`, key = SHA-256 hex of the final data URL; value = `{dataURL, thumbDataURL, w, h, bytes}`).
  - This avoids exo's quota failure (`app.svelte.ts:667`).
  - The hash doubles as the stable image id, and the server could use the same key.
- Wrap every access in try/catch. If IndexedDB is unavailable, fall back to in-memory for the session and show "images won't survive reload".

### 3.3 API call (per send)
```
POST /v1/chat/completions
{ messages: [ ...(system prompt only if the user set one),
              ...history.map(toWire), newUser ],
  stream: true, stream_options:{include_usage:true},
  max_tokens: <setting, default 2048>, temperature: <default model's> ,
  enable_thinking?: bool }            // exo sends this top-level (2521); mlx-lm uses chat_template_kwargs — not verified which the seam honours
```
- `toWire(user with images)` = `content:[{type:'image_url', image_url:{url:dataURL}}…, {type:'text', text: content + fileText}]`. The order matches exo (images first, `app.svelte.ts:2418-2446`).
- The order and bytes must be **identical every turn** for prefix reuse.
- Assistant turns: `content` plus `reasoning_content` if thinking was captured (exo `2455-2459`). Whether the template re-renders thinking into the prefix affects cache hits; not verified per family.
- **No nonce, no fixed system prompt.** Anything that varies per request breaks the prefix, so the chat has no timestamps in the prompt.
- Abort: `AbortController`. A Stop button replaces Send while the request is in flight.

### 3.4 Attachments
- Inputs: file button, paste, drag-drop (ChatForm behaviour). Paste of text over 2500 chars becomes a `pasted-text.txt` attachment.
- **Images**: decode, downscale to a cap, re-encode once, then hash and store.
  - The cap is the model's native max, or a default long edge of 1536 (the per-family value comes from the vision config; not verified).
  - Re-encode as JPEG q0.85, or PNG if alpha.
  - Also make a 160 px thumbnail.
  - After this step the stored data URL is what goes on the wire forever. Never re-encode from the original file.
- **PDF**:
  - lazy `import('https://cdn.jsdelivr.net/npm/pdfjs-dist@<pinned>/build/pdf.min.mjs')` with the worker from the same place, matching exo `files.ts:34`;
  - render up to 20 pages at scale 2 as JPEG q0.8, extract text capped at 100k chars (exo constants `files.ts:36-38`);
  - pages are stored as images by hash; the text goes in the text part.
  - Warn before sending if pages × cost-per-image exceeds the context.
- **Text files**: inline as `[File: name]` fenced block (exo format).

### 3.5 Token cost per image
- Before sending, show on each chip "≈N tok", computed client-side from `w, h` and per-family patch math. Proposed source: the server exposes `vision: {patch, merge, min_pixels, max_pixels, tokens_per_image?}` for the served model; gemma4 is a fixed count (not verified).
- After sending, the turn's `met` shows `prompt: <new> new of <total>` (reuse `metHTML`, `index.html:1450-1466`) and flags an image turn. On turn ≥2, a cache miss on an unchanged image history is highlighted with the `warnv` class. This is the user-visible test that image reuse works.

### 3.6 Streaming parse
- Port exo `parseSSEStream` (`app.svelte.ts:2148-2220`) into the page, shared by Bench and Chat, replacing `index.html:1589-1601`. Handle:
  - `data:` JSON, including `usage`, `delta.content` and `delta.reasoning_content`;
  - `: keepalive P/T` → prefill progress;
  - `: prefill_progress {json}` as well, which keeps it compatible with exo's format if knurlogic ever emits it;
  - `[DONE]`;
  - the tail-buffer flush.
- `<think>` splitting: port exo `2107-2140`. Thinking renders in a collapsed `<details>` that is open while streaming.
- Render: throttle markdown re-render to one per `requestAnimationFrame`. During streaming, render only the last block as plain text if needed.

### 3.7 Prefill progress
- A bar above the pending assistant bubble: `processed/total` in k-format, percent, and an ETA from exo's `PrefillProgressBar.svelte:17-30` math (verbatim).
- Hidden on the first content token or at stream end. When cached tokens make the prefill instant, no keepalive arrives, so no bar shows, which is correct.

### 3.8 Markdown (no build step)
- Option A, recommended: `marked` plus `DOMPurify` from jsdelivr, loaded with lazy `<script>`. Mandatory sanitisation, since the model's output is untrusted HTML. KaTeX from jsdelivr, loaded only when `$…$` or `\(` appears.
- Option B: a roughly 150-line in-page renderer that escapes first and then handles fences, inline code, bold/italic, lists, headings, links (http/https only) and tables. It needs no CDN and fits the page's "one file, no deps" style.
- Either way: code blocks get a header with the language and a copy button (exo `MarkdownContent.svelte:27-45` pattern, reimplemented), and a copy button per message.
- Skip highlight.js for now, or lazy-load it from cdnjs.

### 3.9 Message actions and images
- Per message: copy; edit the last user message and regenerate (truncate after it); regenerate the last assistant turn; delete.
- Thumbnails show in user bubbles (from the IndexedDB `thumbDataURL`). Click opens the lightbox (exo `ImageLightbox.svelte` behaviour: Esc or backdrop closes; download with the extension from the MIME type).
- Scroll-follow plus a jump-to-bottom button (exo `ChatMessages.svelte:31-90`).

## 4. Changes to `index.html` (and the minimal server support it needs)

| Region | Change |
|---|---|
| `<style>` (3-420) | Add styles for `.tabs`, `#chat`, `.bubble`, `.thumb`, `.chip`, `.pfbar`, `.md` (code/pre/table), `.lightbox`, `details.think`, with dark/light token overrides in the existing pattern |
| `aside.side` (456-459) | No markup change; `#chats` rendering becomes tab-aware |
| `main` (460-478) | Wrap `#try` in a tab host; add `section#chat`: `#clog`, attachment chips row, `textarea#cq` (Enter sends, Shift+Enter newline), attach button with hidden `<input type=file multiple accept="image/*,.pdf,text/*,.md,.json,.py">`, Send/Stop, and a small `<details>` with max_tokens, temperature, thinking toggle and an optional system prompt |
| New top-level node | `div#lightbox hidden` beside `#picker` and `#sheet` |
| Script 1409 | Polling unchanged; `tick()` also enables or disables chat and attach from served/vision state |
| Script 1431-1528 | Keep the Bench code; generalise `renderRuns` / `newRun` to dispatch on the active tab; `metHTML` / `delta` reused as-is by chat |
| Script 1572-1627 | Pull the SSE loop out into a shared `streamChat(body, {onDelta, onThink, onProgress, onUsage, signal})`; Bench calls it unchanged in behaviour (nonce kept) |
| New script blocks | `kn.chats` store and IndexedDB image store; attachment pipeline (downscale/hash/thumbnail, lazy pdf.js); `toWire`; chat `send`; markdown renderer; think-splitter; prefill bar; lightbox |
| Picker (1140-1190) | Optional: a ★ favourite and recents (localStorage `kn.fav`, `kn.recent` max 20, as exo's stores); a VISION tag next to MTP/VQ when `/models.json` reports it |

Server-side changes the UI depends on (outside `index.html`):
- **`/v1/chat/completions` must accept `image_url` parts.** Today mlx-lm throws (`server.py:136-141`). The engine must intercept the parts, encode each image once keyed by content hash, and splice the embeddings, with the prefix cache keyed on tokens plus image hash. Not verified where in engine/ this lands.
- **Expose vision capability and patch math** on `/loaded.json` or `/status.json`, plus a `vision` flag on `/models.json` entries (`web.py:59-124`).
- **Request body cap**: data URLs are large. I read no Content-Length limit (`server.py:1123-1132` only checks presence); not verified for proxies.

## 5. Port verbatim vs reimplement
- **Near-verbatim** (keep Apache-2.0 notice): `parseSSEStream`; the `<think>` extractor; the PDF-to-page-images routine and its constants; the PrefillProgressBar ETA math; the history-to-OpenAI-multimodal mapping; the ImageLightbox download-extension logic; the paste-to-file rule (2500 chars).
- **Reimplement**: storage (IndexedDB, not base64 in localStorage); the markdown renderer (vanilla plus sanitiser); all Svelte components as DOM templates in the page's `esc()` / `innerHTML` style; the model selector (knurlogic's picker is already better fitted to the working set).
- **Drop**: logprobs/top_logprobs and the token heatmap; exo's fixed system prompt; image generation and editing; per-chat model, sharding and instance fields.