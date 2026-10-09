# Building knurlogic

For people who change knurlogic, build on it, or add a model. What a user
sees is in [the user guide](../guide/). Why each piece is shaped the way
it is lives in [docs/design/](../design/); these pages say where the code
is and what keeps it correct, and link the design notes instead of
repeating them.

## Setup

    git clone https://github.com/noahzelezny/Knurlogic
    cd Knurlogic
    python -m venv .venv && . .venv/bin/activate
    pip install -e '.[dev]'
    git config core.hooksPath scripts/git-hooks

The `dev` extra installs pytest, pytest-xdist, ruff and mypy. The git hooks (`scripts/git-hooks/commit-msg`, `pre-push`)
strip and refuse AI attribution in commit messages.

Before a commit (from [CONTRIBUTING.md](../../CONTRIBUTING.md)):

    ruff check .
    mypy
    pytest -q -n 8 tests

The suite uses tiny random-weight models and needs no downloads. Tests
that need real weights, a second Mac or a Thunderbolt link skip
themselves. See [testing](testing.md).

## The package map

One package, `src/knurlogic/`, in six parts: `engine/`, `interfaces/`,
`machine/`, `tuning/`, `cluster/`, `context_management/`. What each holds
and how a request and a cluster job flow through them:
[docs/architecture.md](../architecture.md).

## The layer rules

Enforced by `tests/integration/test_layers.py`, which parses every source
file, imports inside functions included:

1. Only `engine/` imports `mlx`, `mlx_lm` or `mlx_vlm`.
2. Only `interfaces/` creates an HTTP server (`http.server`,
   `socketserver`).
3. `engine/`, `machine/`, `tuning/` and `context_management/` never import
   `knurlogic.interfaces`.

The rest of the import diagram in architecture.md is convention. A new
exception to a rule is made in `test_layers.py`, with the reason.

Other rules from CONTRIBUTING.md: vendored architectures under
`engine/families/*/architecture/` are pinned in the `PROVENANCE.md` beside
them and re-vendored, never edited in place (`knurlogic vendor`,
`engine/vendor.py`); tests live in a folder that mirrors the package they
cover.

## The pages

- [engine.md](engine.md): the model server's core: the scheduler thread,
  the executor, the model host, prompts and request text, the memory
  guard, the KV cache.
- [models.md](models.md): model families, the manifest, and adding a model.
- [splits.md](splits.md): one model across several Macs: the tensor and
  pipeline splits, the step plan, the link.
- [http-and-page.md](http-and-page.md): the model server's HTTP API and the
  page (routes, the router, the peer relay).
- [mcp-and-cli.md](mcp-and-cli.md): the MCP server and the CLI commands.
- [settings.md](settings.md): knobs, presets, `resolve()`, and saved
  preferences.
- [machine.md](machine.md): memory, the allowance, the wired limit,
  installed and loaded models, the server registry, the load lock.
- [cluster.md](cluster.md): discovery, peers, the control plane, launch,
  jobs, recovery.
- [vision.md](vision.md): images as context.
- [drafting.md](drafting.md): MTP heads and the batched drafting loop.
- [compaction.md](compaction.md): server-side context management.
- [telemetry.md](telemetry.md): request ids, the ledger, timing spans,
  progress events.
- [prompt-cache.md](prompt-cache.md): the prompt cache, in memory and on
  disk, on one Mac and on a split model.
- [testing.md](testing.md): the test layout, the support helpers, xdist,
  and the real-model gates.
