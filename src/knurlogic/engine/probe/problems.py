"""The onboarding set: problems with one integer answer each, so a
forced answer can be checked (problems.json). Their answers were computed
by code when the set was written -- famous problems with their constants
changed, so an answer is worked out, not recalled -- and every probe in
docs/design/auto-effort.md was fitted on traces of these."""

from __future__ import annotations

import json
from pathlib import Path

_DATA = json.loads((Path(__file__).with_name("problems.json")).read_text())

#: the system prompt every onboarding problem is asked under
SYSTEM: str = _DATA["system"]
#: (question, answer)
PROBLEMS: list[tuple[str, int]] = [(p["question"], int(p["answer"]))
                                   for p in _DATA["problems"]]
