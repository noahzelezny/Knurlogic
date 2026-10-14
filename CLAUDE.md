# CLAUDE.md — knurlogic

Read `docs/architecture.md` first: one package in six parts, and who may
import whom. The layer rules are enforced by `tests/integration/test_layers.py`;
the quality ratchets (ruff hit counts, duplicate code) by `tests/integration/test_quality.py` and its
`quality_baseline.json`. A red ratchet means the code moved, not the test.

## Shape rules (enforced by `test_layers.py`)
- `interfaces` -> `cluster`, `context_management`, `engine`, `machine`, `tuning`;
  `cluster` -> `engine`, `machine`, `tuning`; `machine` never imports `tuning`;
  nothing below `interfaces` imports it; nothing below `cluster` imports it.
- Only `engine/` imports `mlx`, `mlx_lm`, `mlx_vlm`. Only `interfaces/` opens an
  HTTP server. A new edge is a design decision: change `docs/architecture.md`
  and `test_layers.py` in the same commit.
- Vendored code (`engine/families/*/architecture`, `*/vision`; see `NOTICE`)
  stays as upstream wrote it. ruff and the ratchets exclude it.

## Code quality (enforced by `test_quality.py`)
- `ruff check .` is green. Rules with pre-existing hits are listed under
  `[tool.ruff.lint.extend-per-file-ignores]` in `pyproject.toml`; the ratchet
  counts them with that list switched off and fails if any count rises. When
  you touch a file, clear its hits for the rules you can see.
- Duplicate code: the pylint `duplicate-code` block count never rises. Before
  writing a helper, `grep -rn "def <verb>_" src/knurlogic`; widen the existing
  one rather than add a sibling. One home per concept.
- Group code by responsibility, not line count: a long module that does one
  job stays long; a module doing two jobs is split along the seam.
- Lowered a count? `python tests/integration/test_quality.py --write-baseline`
  in the same commit. It refuses to raise anything.

## The check (CI runs exactly this on 3.11 and 3.12)
```bash
ruff check . && mypy && python -m pytest -q tests
```
`pip install -e '.[dev]'` once per clone. `pytest -n auto` for speed.

## Commits
- No AI attribution anywhere: no co-author lines, no session links, in commits
  or tracked files (`tests/test_no_ai_attribution.py` and
  `.github/workflows/attribution.yml` refuse them).
- Stage only the paths you changed, by name. Never `git add -A` / `.`.
