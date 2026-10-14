"""Code-quality ratchets: numbers that only go down.

Two gates, each measured against ``quality_baseline.json`` next to this file.

1. ruff, with every family in pyproject enabled and the baseline ignore list
   (``lint.extend-per-file-ignores``) switched off: the hit count per rule may
   not rise. ``ruff check`` on the tree stays green because pyproject ignores
   the baselined rules; this test is what stops them growing.
2. pylint ``duplicate-code``: the number of similar-code blocks across the
   source may not rise. Two functions that differ in a literal are one
   function with a parameter.

A count *below* the baseline fails too, with instructions: lower the baseline
in the same commit so it stays honest. To rewrite the baseline from the
current tree (it refuses to raise any number):

    python knurlogic/tests/integration/test_quality.py --write-baseline

When a baselined ruff rule reaches zero hits, delete it from both the baseline
and ``lint.extend-per-file-ignores`` in pyproject: the rule is then fully on.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("quality_baseline.json")
SRC = ROOT / "src" / "knurlogic"
RUFF_TARGETS = ["."]
# Directories (any path part) that are vendored or generated: not ours to shape.
# (pylint skips them; ruff has its own exclude list in pyproject.)
EXCLUDE_PARTS: set[str] = {"architecture", "vision"}
MIN_SIMILARITY_LINES = 6


def _run(*argv: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", *argv], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout


def _toml(table: dict[str, Any], prefix: str = "") -> str:
    """Serialize the nested dict tomllib gave us (strings, numbers, bools, lists,
    dicts) back to TOML; JSON scalars and arrays are valid TOML literals."""
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    out = f"[{prefix}]\n" if prefix else ""
    out += "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in scalars.items())
    for k, v in table.items():
        if isinstance(v, dict):
            out += "\n" + _toml(v, f"{prefix}.{k}" if prefix else k)
    return out


def ruff_counts() -> dict[str, int]:
    """Hit count per rule with the baseline ignore list switched off.

    A `--config KEY=VALUE` override merges with pyproject instead of replacing
    it, so the switch-off is a temporary copy of the ruff settings minus
    `lint.extend-per-file-ignores`, written next to pyproject so that relative
    paths in it resolve the same way.
    """
    import tomllib

    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]
    settings.get("lint", {}).pop("extend-per-file-ignores", None)
    config = ROOT / ".ruff-ratchet.toml"
    config.write_text(_toml(settings))
    try:
        out = _run(
            "ruff", "check", *RUFF_TARGETS, "--config", config.name,
            "--statistics", "--output-format", "json", "--exit-zero",
        )
    finally:
        config.unlink()
    return {row["code"]: row["count"] for row in json.loads(out or "[]")}


def duplicate_blocks() -> int:
    ignore = ",".join(f".*/{part}/.*" for part in EXCLUDE_PARTS)
    out = _run(
        "pylint", "--disable=all", "--enable=duplicate-code",
        f"--min-similarity-lines={MIN_SIMILARITY_LINES}", "--recursive=y",
        "--output-format=json", *([f"--ignore-paths={ignore}"] if ignore else []),
        str(SRC),
    )
    return sum(1 for m in json.loads(out or "[]") if m["message-id"] == "R0801")


def measure() -> dict[str, Any]:
    return {"ruff": dict(sorted(ruff_counts().items())), "duplicate_blocks": duplicate_blocks()}


def baseline() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(BASELINE.read_text())
    return data


def test_ruff_hit_counts_only_fall() -> None:
    now, then = ruff_counts(), baseline()["ruff"]
    rose = {c: (then.get(c, 0), n) for c, n in now.items() if n > then.get(c, 0)}
    assert not rose, (
        f"new ruff violations (rule: baseline -> now): {rose}. See them with "
        f"`ruff check {' '.join(RUFF_TARGETS)} --isolated --select {','.join(rose)}` "
        "and fix them; the baseline does not rise."
    )
    fell = {c: (n, now.get(c, 0)) for c, n in then.items() if now.get(c, 0) < n}
    assert not fell, (
        f"ruff counts fell (rule: baseline -> now): {fell}. Good: lower the baseline "
        f"(`python {Path(__file__).name} --write-baseline`); a rule at zero also leaves "
        "lint.extend-per-file-ignores in pyproject."
    )


def test_duplicate_blocks_only_fall() -> None:
    now, then = duplicate_blocks(), baseline()["duplicate_blocks"]
    assert now <= then, (
        f"{now - then} new duplicate-code block(s) (pylint R0801, >= {MIN_SIMILARITY_LINES} "
        "similar lines). Make it one function and call it from both places. "
        f"List them: `pylint --disable=all --enable=duplicate-code "
        f"--min-similarity-lines={MIN_SIMILARITY_LINES} --recursive=y {SRC.relative_to(ROOT)}`"
    )
    assert now == then, (
        f"duplicate blocks fell {then} -> {now}. Good: lower the baseline "
        f"(`python {Path(__file__).name} --write-baseline`)."
    )


if __name__ == "__main__":
    if "--write-baseline" not in sys.argv:
        print(json.dumps(measure(), indent=2))
        sys.exit(0)
    new = measure()
    if BASELINE.exists() and "--allow-raise" not in sys.argv:
        old = baseline()
        raised = {c: (old["ruff"].get(c, 0), n) for c, n in new["ruff"].items()
                  if n > old["ruff"].get(c, 0)}
        if new["duplicate_blocks"] > old["duplicate_blocks"]:
            raised["duplicate_blocks"] = (old["duplicate_blocks"], new["duplicate_blocks"])
        if raised:
            sys.exit(f"refusing to raise the baseline: {raised} (fix the code, or --allow-raise)")
    BASELINE.write_text(json.dumps(new, indent=2) + "\n")
    print(f"wrote {BASELINE.relative_to(ROOT)}; baselined ruff rules for pyproject:")
    print(json.dumps(sorted(new["ruff"]), indent=4))
