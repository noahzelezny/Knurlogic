# Testing

The suite runs on any Mac with tiny random-weight models and needs no
downloads. Real models are tested by hand, through the gates in
`tools/`. Why the gates exist: [tools](../design/tools.md).

## Running

    pip install -e '.[dev]'
    pytest -q -n 8 tests                      # the whole suite, 8 workers
    pytest -q tests/engine/test_scheduler.py  # one file

`pyproject.toml` puts `src` and `tests/support`, `tests/cluster`,
`tests/engine`, `tests/interfaces` on the path, so each checkout tests its
own source and a test module can import a helper from a sibling by bare
name.

## The layout

`tests/` mirrors the packages: `engine/`, `interfaces/`, `machine/`,
`tuning/`, `cluster/`, `context_management/`. Also:

- `tests/integration/`: cross-cutting tests, among them
  `test_layers.py` (the layer rules) and `test_model_launch.py`,
  `test_peer_launch.py`, `test_knurlogic_wide.py`.
- `tests/api/test_conformance.py`: the HTTP API's shapes.
- `tests/test_no_ai_attribution.py`.
- `tests/support/`: helpers that are not tests.

## The support helpers

| file | what |
|---|---|
| `fake_cluster.py` | `fake_cluster(n)`: n real page processes on this Mac, each with its own cache dir and identity, all knowing each other; only the machine facts are faked |
| `cluster_fake_page.py` | one such page: the real handler and cluster routes |
| `cluster_fake_rank.py` | what a page spawns in place of `knurlogic serve --rank r`: writes the real progress marker; rank 0 serves chat through the real scheduler's submit and abort. No model |
| `tensor_ring_worker.py`, `pipeline_ring_worker.py` | one rank of a two-process ring over a tiny model; rank 0 also runs the unsplit model and writes both logits |
| `fixtures_vision.py` | the shared tiny-vision builder (`tiny_config`); per family in `fixtures_vision_<family>.py` |
| `fixtures_thunderbolt.py` | captured outputs from two Macs joined by Thunderbolt, for `cluster/links` |
| `procs.py` | `Owned`: every process a test starts is killed and waited on |
| `goldens/` | builders for golden outputs (`build_deepseek_v4.py`, `build_gemma4.py`, ...) |
| `artifacts/`, `fixtures_deepseek_v4*/`, `glm5_template/` | small artifact stand-ins |

## What `tests/conftest.py` guarantees

Every test, automatically:

- gets its own `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` and `KNURLOGIC_HOME`,
  so the server and job registries, the load lock, recovery.json, saved
  settings and the ledger are never the real ones;
- points `KNURLOGIC_PAGE` at a port nothing listens on, so the MCP never
  reaches the real page;
- starts no cluster watcher thread and no recovery thread (tests call
  `watch_once` and `recovery.tick` by hand);
- kills what it started: every process carries a per-session token
  (`KNURLOGIC_TEST_SESSION`; each xdist worker its own) and is reaped at
  teardown; the session fails if any survives. Register a `Popen` with
  the `owned_procs` fixture.

## Real models and real machines

- A test that needs real weights, a second Mac, Bonjour or a Thunderbolt
  link skips itself (`pytest.mark.skipif` on the file or path it needs;
  some name the variable to set, e.g. `KNURLOGIC_TEST_DEEPSEEK_V4`).
- Real-model gates are scripts in `tools/` (`vision_gate.py`), run by
  hand, one artifact at a time, behind the load lock. Never call them from
  `tests/`.
- Tiny-fixture tests do not take the load lock.

## Writing a test

- Put it in the folder of the package it covers.
- Use a tiny fixture, not a real model.
- A test that spawns a process registers it with `owned_procs`.
- A test that needs a page starts one (or `fake_cluster`) and says where;
  it never relies on `127.0.0.1:8899`.
- For a split model, use the ring workers; for a cluster, `fake_cluster`.

## Notes

Feature tests follow the package map, so a feature spread across
packages has its tests spread too: prompt cache in `tests/engine/` and
`tests/interfaces/`, cluster MCP in `tests/cluster/test_mcp_cluster.py`.
