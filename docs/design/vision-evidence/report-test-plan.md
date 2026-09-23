# Knurlogic vision verification plan

Everything below comes from reading files, with no writes, loads or POSTs. Anything marked "not verified" is inferred and was not read.

## 0. Ground truth read

**Test pattern to copy.** `tests/test_batch_drafting.py:25-49` builds a tiny random `qwen3_5` with these steps:
- `register.register("qwen3_5")`, then `arch.Model(arch.ModelArgs(model_type=..., text_config=tc))`
- `mx.random.seed(0)` and `set_dtype(mx.float32)`
- the gate is greedy token identity against mlx-lm's own `BatchGenerator` (`:70-97`)
- the docstring at `:7-13` warns against near-tie vocabs. Keep that rule: vocab ≥ 512 and compare tokens, not raw logits. Logits get a tolerance.

**Existing gate.** `interfaces/mcp.py:72-133` has `ready()`. It already blocks on `ui.loading()`, knurlogic's own in-flight loads (`:84-92`), and on exo runners in transit. It returns `ready` and `blockers`. It is a check, not a mutex: two agents can both see `ready: true` and then both load. Section 4 closes that race.

**Other files.**
- `machine/servers.py:38` defines the registry `~/.cache/knurlogic/servers.json`. `~/.cache/knurlogic/` already exists and holds `servers.json`, `serve-809x.log` and `exo_watch.json`.
- `engine/seam.py:101 load()`, `:289 switch()`, `:323 unload()`. `:621-641 _stream` passes `prompt_cache` through. Image caching hooks in there.
- `tools/mtp_probe.py` is the only tool. It is a read-only single-load probe (docstring at `:1-12`), and it imports `exo.worker...` at `:40`. Vision probes should follow its style but not depend on exo.

**Reference implementation.** mlx-vlm 0.6.17 has all five families: `/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm/models/{qwen3_5,qwen3_5_moe,qwen4_exp,glm5_next,gemma4}`, with `qwen3_5/vision.py` and `qwen3_5/qwen3_5.py`. `/tmp/knur_clean/.../mlx_vlm` also exists (listing shows `evals`, `generate`); its per-family contents are not verified.

**Vision configs** (read from each artifact's `config.json`):

| family | artifact read | vision `model_type` | depth / layers | hidden | out_hidden | patch | merge | t_patch | heads | ffn | where the weights are |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_5 | Qwen3.8-27B-VQ-3.9bpw | qwen3_5 | 27 | 1152 | 5120 | 16 | 2 | 2 | 16 | 4304 | `model-vision-graft.safetensors` |
| qwen3_5_moe | Qwen3.6-35B-A3B-VQ-3.4bpw | qwen3_5_moe | 27 | 1152 | 2048 | 16 | 2 | 2 | 16 | 4304 | graft |
| qwen3_5_moe | Qwen3.5-397B-A17B-VQ-2.2bpw | qwen3_5_moe | 27 | 1152 | 4096 | 16 | 2 | 2 | 16 | 4304 | graft |
| qwen4_exp | Qwen3.8-Flash-Next-VQ-2.1bpw | qwen4_exp | 27 | 1152 | 2560 | 16 | 2 | 2 | 16 | 4304 | graft |
| glm5_next | GLM-5.3-Flash-VQ-2.7bpw | glm5_next_vision | 24 | 1024 | 4096 | **14** | 2 | 2 | 16 | 4096 | main shards |
| gemma4 | gemma-4-e4b-it-VQ-PLE | gemma4_vision | 16 layers | 768 | – | 16 | – | – | 12 (`num_attention_heads`) | 3072 | main shards |
| gemma4 | gemma-4-26b-a4b-it-VQ-6.2bpw | gemma4_vision | 27 layers | 1152 | – | 16 | – | – | 16 | 4304 | graft |

Gemma uses different key names (`num_hidden_layers`, `num_attention_heads`, no merge or temporal patch). Its fixture needs its own branch.

## 1. Tiny random fixtures with a vision tower

New file `tests/fixtures_vision.py`. It holds no weights; everything is built in memory from a config.

```python
FAMILIES = {
  "qwen3_5":     dict(src="Qwen3.8-27B-VQ-3.9bpw"),
  "qwen3_5_moe": dict(src="Qwen3.6-35B-A3B-VQ-3.4bpw"),
  "qwen4_exp":   dict(src="Qwen3.8-Flash-Next-VQ-2.1bpw"),
  "glm5_next":   dict(src="GLM-5.3-Flash-VQ-2.7bpw"),
  "gemma4":      dict(src="gemma-4-e4b-it-VQ-PLE"),
}
def tiny_config(family, real_cfg=None) -> dict
def tiny_model(family, impl="knurlogic"|"mlx_vlm", seed=0) -> (model, processor_cfg)
```

**Scaling rule.**
- Start from the real `config.json`. Read only `config.json` from `~/.exo/models`, never the weights. Fall back to a copy of the table above embedded in the test so CI works without artifacts.
- Keep the structural keys unchanged: `patch_size` (14 vs 16 matters for GLM), `spatial_merge_size`, `temporal_patch_size`, `model_type`, `image_token_id`, `video_token_id`, `vision_start_token_id`, rope and mrope sections, `deepstack_visual_indexes` if present (not verified whether present), and gemma's pooling and soft-token count.
- Scale these down:
  - vision: depth or `num_hidden_layers` → 2, hidden 64, heads 4, ffn 128
  - `out_hidden_size` → the text hidden size, 128
  - text: same shape as `_tiny` in `test_batch_drafting.py:34-38`
  - MoE: `num_experts` → 4, `num_experts_per_tok` → 2
- Keep the vocab ≥ 512 but ≥ `max(image_token_id, ...)+1`. The real special IDs are around 150k–260k (not verified). Two options:
  - (a) remap the special IDs into the small vocab inside `tiny_config`. This is the recommended option.
  - (b) keep vocab = max_id+1 with hidden 128; that is about 33M params in float32 per embedding (not verified).
- Everything runs in float32 with `mx.random.seed(0)`, and every parameter is evaluated.
- Images are deterministic `mx.random.uniform` pixel arrays at two sizes: 56×56 (the patch-14 grid for GLM) and 64×64 (patch 16). Include one non-square (64×96) so grid_thw and mrope positions get exercised.

**Test files** (all `pytest.importorskip("mlx.core")`, CPU/GPU, finishing in seconds each):
- `tests/test_vision_fixtures.py`: for each family the tiny model builds; vision forward shape is `(n_merged_patches, text_hidden)`; the number of image tokens after template expansion equals the number of patches after merge.
- `tests/test_vision_identity.py`: the gates in section 2.
- `tests/test_image_cache.py`: the cache gates in section 2.

## 2. Identity gates (tiny random, no real model)

Each gate is parametrized over the five families.

| # | gate | method | tolerance |
|---|---|---|---|
| G1 | the vendored vision tower equals mlx-vlm | load the same random weights into both: build `impl="mlx_vlm"`, `tree_flatten(params)`, then `load_weights` into the knurlogic model after sanitize/key mapping. Assert the key sets are equal after mapping. Compare `vision_tower(pixels, grid_thw)` | `allclose(atol=1e-5)` in float32 |
| G2 | preprocessing equals mlx-vlm | the knurlogic image processor versus the mlx-vlm processor on the same PIL image: `pixel_values`, `image_grid_thw`, and the number of expanded tokens | exact for grid and count; atol 1e-6 for pixels |
| G3 | merged input embeddings and positions equal mlx-vlm | `get_input_embeddings(ids, pixels)` and the mrope position ids (Qwen, GLM) | exact positions; embeddings atol 1e-5 |
| G4 | end-to-end greedy tokens equal mlx-vlm | 40 greedy tokens from an image+text prompt | token-identical, as in `test_batch_drafting.py:97` |
| G5 | a text-only request through the vision model equals the plain mlx-lm text model | same random text weights | token-identical. This proves vision support does not regress text or drafting |
| G6 | the image embedding is computed once | wrap `vision_tower.__call__` with a counter. Run a turn-1 image prompt, then turn 2 appended; the counter stays at 1. Key = sha256(pixel bytes + processor config) | counter == 1 |
| G7 | image-cached turn 2 equals uncached turn 2 | (a) cold: a fresh model, the full turn-2 prompt, no cache. (b) warm: turn 1 then turn 2 through the prompt cache and embedding cache | token-identical; logits of the last prefill position atol 1e-4 |
| G8 | the prefix KV cache survives images | after turn 1, the prompt-cache hit length for turn 2 is ≥ the turn-1 length. Assert on `trunk_offset(cache)`, as `test_batch_drafting.py:84` does | hit_len == len(turn1 tokens) |
| G9 | a different image with an identical placeholder token stream does not hit | two images with the same size, so the same token IDs, but different pixels | cache miss, and outputs differ from the cached run. **This is the critical false-hit gate:** image placeholder tokens are identical, so the cache key must include the image hash |
| G10 | MTP drafting stays identical with images | `MTPBatchGenerator` versus `BatchGenerator` on an image prompt; only for families with a head (`seam.py:473 keeps_mtp_weights`) | token-identical |
| G11 | the batch mixes image and text rows | three rows, one with an image, admitted one per call | each row equals its solo run |

The cache key for G8 and G9 must be `(tokens, image_hashes at positions)`. The recommended layout is to keep the LRU key as token IDs but replace each image's placeholder span with a per-image sentinel derived from the hash. This is inferred; the current key is `all_tokens` per `test_batch_drafting.py:81-84`.

**Command:**
```
cd ~/Documents/AgenicAI/knurlogic && python -m pytest tests/test_vision_*.py tests/test_image_cache.py -q
```
Use the interpreter that has mlx-vlm for G1–G4: `/opt/anaconda3/envs/exo/bin/python` (0.6.17) or the `/tmp/knur_clean` env (0.7.1). Where mlx-vlm is absent, `importorskip("mlx_vlm")` skips G1–G4; the other gates still run. Pin the reference: record `mlx_vlm.__version__` in the test's output. Pick 0.6.17 as the reference because it is the version exo currently uses (inferred).

## 3. Real-model gate per family

The host has 96 GB and shares it with exo. Budget: artifact size plus about 15% must fit what `fit` or `ready()` says is free (inferred).

| family | smallest artifact (`du -sh`) | where it runs |
|---|---|---|
| gemma4 | `gemma-4-e4b-it-VQ-PLE`, 7.4G (vision in main shards) | local, run first |
| gemma4 (graft path) | `gemma-4-26b-a4b-it-VQ-6.2bpw`, 19G | local; covers gemma's graft loader |
| qwen3_5 | `Qwen3.8-27B-VQ-3.9bpw`, 12G | local |
| qwen3_5_moe | `Qwen3.6-35B-A3B-VQ-3.4bpw`, 14G | local |
| qwen3_5_moe (397B) | `Qwen3.5-397B-A17B-VQ-2.2bpw`, 112G | **cluster** (> 96 GB); optional once 35B passes, same code path |
| qwen4_exp | `Qwen3.8-Flash-Next-VQ-2.1bpw`, 47G | local, but only when exo has freed memory; `ready()` must be true and fit must be ≥ 55G |
| glm5_next | `GLM-5.3-Flash-VQ-2.7bpw`, 108G | **cluster only**; also the only patch-14 tower and the only one with vision in main shards among the large models |

New file `tools/vision_gate.py`. It follows the read-only style of `tools/mtp_probe.py`: one artifact per invocation, it takes the lock (section 4), and it prints JSON to stdout.

1. **Load:** `seam.load(path)` with vision. Record `load_s` and peak memory from `seam.memory()` (`seam.py:68`). Assert the vision tensor count: 333 for graft artifacts (`docs/PLAN.md:236-238`). For GLM and gemma e4b, count the vision keys in the main shards.
2. **Text answer:** "The capital of France is", 16 greedy tokens; assert "Paris".
3. **Image answer:** a fixed fixture image committed at `tests/data/vision_red_square.png`, a solid red square on white, generated by a test helper rather than downloaded. Ask "What color is the square? One word." and assert "red" in lowercase. Add a second fixture with the digits "42" rendered by PIL to test OCR.
4. **Reuse and latency test:** 5 turns through the real server path (`seam.serve` with the chat endpoint on an ephemeral port, loopback only). The image is in turn 1; turns 2–5 are short text questions of about 10 tokens.
   - Record per turn: `prompt_tokens`, `cached_tokens` (the prompt-cache hit length), `prefill_s` = time to first token, and the vision-tower call count from the G6 counter exposed in `/status.json`.
   - Pass conditions:
     - vision calls == 1 across all 5 turns
     - `prompt_tokens - cached_tokens` ≤ the turn's new tokens plus the template overhead (≤ 32)
     - `ttft(turn5)` ≤ 1.5 × `ttft(text-only turn with the same new-token count)`
     - `ttft(turn5)` ≪ a cold prefill of the whole turn-5 conversation. Also measure the cold case once, uncached, as the control, in the spirit of mtp_probe's "control matters" rule.
   - Correctness: turn 5 asks "what color was the square?" again and must answer red. This proves the image is real context, not just a cache hit.
5. **Unload:** `seam.unload()` (`seam.py:323`); confirm memory returns to baseline ±1 GB.

**Commands** (one at a time, sequentially, each behind the lock):
```
python tools/vision_gate.py ~/.exo/models/TheDrainFlorist--gemma-4-e4b-it-VQ-PLE
python tools/vision_gate.py ~/.exo/models/TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw
python tools/vision_gate.py ~/.exo/models/TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw
python tools/vision_gate.py ~/.exo/models/TheDrainFlorist--gemma-4-26b-a4b-it-VQ-6.2bpw
python tools/vision_gate.py ~/.exo/models/TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw   # only if ready() and fit >= 55G
# cluster: GLM-5.3-Flash-VQ-2.7bpw (and optionally Qwen3.5-397B-2.2bpw) -- via exo placement, needs Noah's go-ahead and ready_for_exo_placement
```
Results go as JSON lines to `~/.cache/knurlogic/vision_gate.jsonl`, which the Workstreams "test windows" can read.

**GLM on the cluster.** Knurlogic's own server is a single box, per the note at `mcp.py:97-98`. The path for GLM is exo's fork (`~/exo`, branch mtp-stage1) with vendored vision, or it waits until knurlogic has distributed serving. Until then, GLM gets only the tiny gates G1–G11 plus a vision-tower-only real check: load only the vision keys from GLM's main shards via a safetensors key filter, about 1 GB (not verified). Compare the tower output against mlx-vlm's tower on the same image, running one tower at a time. That is safe locally and covers the patch-14 path. Do the same tower-only check for the 397B graft.

## 4. Model-load lock protocol

The lockfile is `~/.cache/knurlogic/load.lock`, next to `servers.json` (`machine/servers.py:38`).

- **Mechanism:** `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on an open fd. The kernel releases it when the process dies, so a crashed agent never leaves a stale lock. Write a JSON record inside for display only: `{pid, host, agent/session, artifact, purpose, started}`.
- **Scope:** hold the lock from before `ready()` through load, the test, and unload for gates. For a long-lived serve, hold it only until the phase is `serving`; `ui.loading()` already represents in-flight loads (`mcp.py:84-92`).
- **New module:** `src/knurlogic/machine/loadlock.py` (in machine/, so no mlx import):
  ```python
  @contextmanager
  def model_load(artifact, purpose, wait_s=0): ...   # raises Busy(holder_record)
  def holder() -> dict | None                         # parse the record; for ready()
  ```
- **Callers:** `seam.load` (`engine/seam.py:101`), `seam.switch` (`:289`), MCP `load` (`interfaces/mcp.py:316`), `interfaces/serve.py`, and `tools/*.py`. Tiny-fixture tests do not take it, since they use under 1 GB.
- **`ready()` change:** add a blocker when `holder()` is not None: `{"what": "model-load lock held", "detail": record}`. This moves the check-then-act race into the atomic flock.
- **Protocol for parallel build agents:**
  1. Run all tiny tests freely.
  2. For a real-model gate, call MCP `ready()`. If it is not ready, stop and report; do not poll-load.
  3. Run `tools/vision_gate.py`, which does `model_load(..., wait_s=0)` and exits with code 75 (EX_TEMPFAIL) and the holder record if the lock is busy.
  4. Always unload before exit (`try/finally`).
  5. Never POST to exo.
- **Test:** `tests/test_loadlock.py` covers four cases:
  - a second `flock` from a subprocess gets `Busy`
  - after `kill -9` of the holder, the lock is acquirable again
  - `ready()` reports the holder, with `ui.loading` and `_exo_state` monkeypatched
  - there is no mlx import in `machine/`

## 5. Order and exit criteria

1. `test_loadlock.py`.
2. Fixtures, then G1–G5 per family, starting with qwen3_5 because the fixture code already exists.
3. G6–G11, the image cache.
4. Real gates, sequentially: gemma-e4b, Qwen3.8-27B, Qwen3.6-35B, gemma-26b, Flash-Next.
5. GLM and 397B: tower-only checks locally; the full gate on the cluster.

**Done** when all tiny gates are green for all five families and each local artifact passes text, image and turn-5 reuse. Then set the arc's tracker tasks to `review` per CLAUDE.md.

**Key paths:**
- `~/Documents/AgenicAI/knurlogic/tests/test_batch_drafting.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/interfaces/mcp.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/engine/seam.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/machine/servers.py`
- `~/Documents/AgenicAI/knurlogic/tools/mtp_probe.py`
- `/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm/models/`