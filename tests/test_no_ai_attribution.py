"""No commit carries AI attribution, and the hook that strips it is on."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = re.compile(
    r"^co-authored-by:.*(claude|anthropic)|generated with \[?claude",
    re.I | re.M)


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def checkout():
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout (an sdist or wheel)")


def test_no_commit_message_has_ai_attribution(checkout):
    log = _git("log", "--format=%h%n%B%x00", "HEAD").stdout
    bad = [c.split("\n", 1)[0] for c in log.split("\x00")
           if ATTRIBUTION.search(c)]
    assert not bad, f"AI attribution in commit messages: {bad}"


def test_the_commit_msg_hook_is_installed(checkout):
    # relative, or absolute to this checkout's (or the main checkout's) copy
    path = _git("config", "core.hooksPath").stdout.strip()
    if not path:
        pytest.skip("hook not enabled in this clone: "
                    "git config core.hooksPath scripts/git-hooks")
    assert path.rstrip("/").endswith("scripts/git-hooks") and (
        (ROOT / path) if not Path(path).is_absolute() else Path(path)
    ).joinpath("commit-msg").is_file(), (
        "run: git config core.hooksPath scripts/git-hooks")


def test_the_hook_strips_a_trailer(tmp_path):
    msg = tmp_path / "MSG"
    msg.write_text("fix a thing\n\nbody\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    subprocess.run(["sh", str(ROOT / "scripts/git-hooks/commit-msg"),
                    str(msg)], check=True, capture_output=True)
    assert msg.read_text() == "fix a thing\n\nbody\n"


def _ai_patterns():
    """AI_IDENT and AI_NAMED from the hooks' own file."""
    src = (ROOT / "scripts/git-hooks/ai_identity.sh").read_text()
    out = {}
    for name in ("AI_IDENT", "AI_NAMED"):
        line = next(ln for ln in src.splitlines() if ln.startswith(name + "="))
        # shell quoting: '...' pieces, with '"'"' for a literal quote
        out[name] = re.compile(
            line.split("=", 1)[1].replace("'\"'\"'", "'").strip("'"), re.I)
    return out


def test_no_commit_is_by_an_ai_identity(checkout):
    pat = _ai_patterns()["AI_IDENT"]
    log = _git("log", "--format=%h %an <%ae> / %cn <%ce>", "HEAD").stdout
    bad = [ln for ln in log.splitlines() if pat.search(ln.split(" ", 1)[1])]
    assert not bad, f"commits by an AI identity: {bad[:5]}"


def test_no_commit_message_names_an_ai_model(checkout):
    pat = _ai_patterns()["AI_NAMED"]
    log = _git("log", "--format=%B", "HEAD").stdout
    bad = [ln for ln in log.splitlines() if pat.search(ln)]
    assert not bad, f"messages naming an AI model: {bad[:5]}"


def test_the_pre_commit_hook_refuses_an_ai_author(tmp_path):
    import os
    env = dict(os.environ, GIT_AUTHOR_NAME="Claude",
               GIT_AUTHOR_EMAIL="noreply@anthropic.com")
    r = subprocess.run(["sh", str(ROOT / "scripts/git-hooks/pre-commit")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "AI identity" in r.stderr
