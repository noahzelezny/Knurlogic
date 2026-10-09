"""No tracked file names a particular machine (scripts/check_machine_specifics.py
holds the patterns and the examples they allow). The self-test builds its
samples at runtime so this file holds none."""
import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "check_machine_specifics.py"
_spec = importlib.util.spec_from_file_location("check_machine_specifics", _P)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_no_tracked_file_names_a_machine():
    bad = C.scan_tree()
    assert not bad, ("machine-specific strings in tracked files (use config, "
                     "an env var or a placeholder):\n" + "\n".join(bad))


def _ip(*parts) -> str:
    return ".".join(str(p) for p in parts)


def test_the_patterns_catch_what_they_are_for():
    assert C.hits("ssh " + _ip(10, 1, 2, 3))
    assert C.hits("inet " + _ip(192, 168, 7, 7) + " netmask")
    assert C.hits(_ip(172, 20, 1, 4))
    assert not C.hits(_ip(192, 0, 2, 2) + " " + _ip(169, 254, 1, 1) + " "
                      + _ip(127, 0, 0, 1) + " version " + _ip(110, 0, 0, 12))
    assert C.hits("/" + "Users" + "/someone/models/")
    assert not C.hits("/" + "Users" + "/x/models/")
    assert C.hits('"/' + "Volumes" + '/Some Drive/Models"')
    assert not C.hits('"/' + "Volumes" + '/My SSD/Models"')
