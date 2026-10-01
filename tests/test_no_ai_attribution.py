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
    path = _git("config", "core.hooksPath").stdout.strip()
    assert path == "scripts/git-hooks", (
        "run: git config core.hooksPath scripts/git-hooks")


def test_the_hook_strips_a_trailer(tmp_path):
    msg = tmp_path / "MSG"
    msg.write_text("fix a thing\n\nbody\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    subprocess.run(["sh", str(ROOT / "scripts/git-hooks/commit-msg"),
                    str(msg)], check=True, capture_output=True)
    assert msg.read_text() == "fix a thing\n\nbody\n"
