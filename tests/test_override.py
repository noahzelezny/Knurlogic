"""Overrides, tested across the boundary they exist to cross.

The claim is not "a finder can load a file" -- that is trivially true. The
claim is that an override reaches a process exo starts with `mp.Process`
under start method "spawn", in an environment where knurlogic is NOT
importable, because that is where inference actually runs.

So every test here runs a real subprocess with a clean environment, and
every one has a CONTROL ARM without the override. An arm that silently did
nothing would return the upstream value and fail, rather than pass quietly.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic import override                              # noqa: E402

UPSTREAM = "upstream-module-as-installed"
OVERLAID = "knurlogic-override-served-this"


def _world(tmp_path):
    """An installed package, an override for one of its modules, and a probe.

    The probe reports from BOTH the process it starts in and a spawn()ed
    child, because those are different questions: the first is what
    `register.py` could already do, the second is the one that was failing.
    """
    site = tmp_path / "site"
    (site / "fakepkg").mkdir(parents=True)
    (site / "fakepkg" / "__init__.py").write_text("")
    (site / "fakepkg" / "mod.py").write_text(f"VALUE = {UPSTREAM!r}\n")

    src = tmp_path / "overrides" / "fakepkg" / "mod.py"
    src.parent.mkdir(parents=True)
    src.write_text(f"VALUE = {OVERLAID!r}\n")

    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import json, multiprocessing as mp, os, sys

        def _report():
            import fakepkg.mod as m
            return {"pid": os.getpid(), "value": m.VALUE,
                    "file": getattr(m, "__file__", None),
                    "knurlogic_imports": _imports("knurlogic")}

        def _imports(name):
            try:
                __import__(name)
                return True
            except Exception:
                return False

        def child(q):
            q.put(_report())

        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            here = _report()
            q = mp.Queue()
            p = mp.Process(target=child, args=(q,))
            p.start()
            there = q.get()
            p.join()
            print(json.dumps({"parent": here, "spawned_child": there}))
        """))
    return site, src, probe


def _run(probe, env_extra, site, tmp_path):
    """A clean environment on purpose: knurlogic must NOT be importable.

    exo runs in its own env (python 3.13, its own rust bindings) and
    knurlogic is not installed there. An installer that needed knurlogic
    would pass here and fail on the only box that matters.
    """
    env = {"PATH": os.environ.get("PATH", ""),
           "HOME": os.environ.get("HOME", ""),
           "PYTHONPATH": str(site)}
    for k, v in env_extra.items():
        if k == "PYTHONPATH":
            env["PYTHONPATH"] = f"{v}{os.pathsep}{site}"
        else:
            env[k] = v
    out = subprocess.run([sys.executable, str(probe)], env=env,
                         capture_output=True, text=True, timeout=120,
                         cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_override_reaches_a_spawned_child_and_needs_no_knurlogic(tmp_path):
    """The whole design in one test, with its control arm.

    exo's runner is a spawned process, so an override that only reaches the
    launcher reaches nothing that matters.
    """
    site, src, probe = _world(tmp_path)
    # knurlogic IS installed in the interpreter running these tests, so
    # "it worked" would not prove the installer did not lean on it. Shadow
    # it with a module that refuses to import: that is the env exo is in.
    block = tmp_path / "block"
    block.mkdir()
    (block / "knurlogic.py").write_text(
        "raise ImportError('knurlogic is deliberately absent here')\n")

    o = override.Override(module="fakepkg.mod", path=src, against="fakepkg 1.0",
                        why="test fixture")
    env = override.install([o], root=tmp_path / "stage",
                          log=tmp_path / "override.log")
    env = dict(env, PYTHONPATH=f"{block}{os.pathsep}{env['PYTHONPATH']}")

    control = _run(probe, {"PYTHONPATH": str(block)}, site, tmp_path)
    applied = _run(probe, env, site, tmp_path)

    # The control proves the arms are not the same configuration.
    assert control["parent"]["value"] == UPSTREAM
    assert control["spawned_child"]["value"] == UPSTREAM

    assert applied["parent"]["value"] == OVERLAID
    assert applied["spawned_child"]["value"] == OVERLAID, (
        "the spawned child is the one that runs the model; an override that "
        "stops at the launcher is the bug this mechanism exists to fix")
    assert applied["spawned_child"]["pid"] != applied["parent"]["pid"]
    # Self-containment: it worked in a process where knurlogic cannot import.
    assert applied["parent"]["knurlogic_imports"] is False
    assert applied["spawned_child"]["knurlogic_imports"] is False


def test_the_log_records_each_process_that_applied_it(tmp_path):
    """A spawned runner cannot be asked what it imported. Without this
    channel, `--cluster` would claim an effect it cannot see."""
    site, src, probe = _world(tmp_path)
    log = tmp_path / "override.log"
    env = override.install([override.Override("fakepkg.mod", src)],
                          root=tmp_path / "stage", log=log)
    _run(probe, env, site, tmp_path)

    acts = override.activations(log)
    applied = [a for a in acts if a["event"] == "applied"]
    assert {a["module"] for a in applied} == {"fakepkg.mod"}
    assert len({a["pid"] for a in applied}) == 2, (
        "two processes imported it -- the launcher and the spawned child")

    st = override.status([override.Override("fakepkg.mod", src)], log)
    assert st["applied_in_launched_processes"] == ["fakepkg.mod"]


def test_a_drifted_override_is_refused_not_warned(tmp_path):
    """The claim an override makes is 'this exact arithmetic'. Serving a file
    that is not the pinned one makes every later measurement unciteable."""
    site, src, probe = _world(tmp_path)
    o = override.Override("fakepkg.mod", src, sha256=override.sha256_of(src))
    env = override.install([o], root=tmp_path / "stage",
                          log=tmp_path / "override.log")
    assert _run(probe, env, site, tmp_path)["parent"]["value"] == OVERLAID

    src.write_text("VALUE = 'edited-after-pinning'\n")
    out = subprocess.run([sys.executable, str(probe)],
                         env={**{"PATH": os.environ.get("PATH", ""),
                                 "HOME": os.environ.get("HOME", "")}, **env,
                              "PYTHONPATH": f"{env['PYTHONPATH']}"
                                            f"{os.pathsep}{site}"},
                         capture_output=True, text=True, timeout=120,
                         cwd=str(tmp_path))
    assert out.returncode != 0 and "refusing to import" in out.stderr

    # And `install` refuses before it ever gets that far.
    o2 = override.Override("fakepkg.mod", src, sha256="0" * 64)
    try:
        override.install([o2], root=tmp_path / "stage2")
    except ValueError as e:
        assert "digest does not match" in str(e)
    else:
        raise AssertionError("install must refuse a drifted override")


def test_it_does_not_silently_disable_an_existing_sitecustomize(tmp_path):
    """Ours is first on PYTHONPATH and python imports exactly one module by
    that name. Conda ships one; breaking somebody's environment setup would
    be a rude way to install a memory knob."""
    site, src, probe = _world(tmp_path)
    (site / "sitecustomize.py").write_text(
        "import os; os.environ['THEIR_SITECUSTOMIZE_RAN'] = '1'\n")
    old = '"knurlogic_imports": _imports("knurlogic")'
    assert old in probe.read_text(), "probe changed; this patch is stale"
    probe.write_text(probe.read_text().replace(
        old, old + ', "theirs": os.environ.get("THEIR_SITECUSTOMIZE_RAN")'))

    env = override.install([override.Override("fakepkg.mod", src)],
                          root=tmp_path / "stage", log=tmp_path / "log")
    got = _run(probe, env, site, tmp_path)
    assert got["parent"]["value"] == OVERLAID, "ours still applied"
    assert got["parent"]["theirs"] == "1", "and theirs still ran"


def test_the_staging_dir_holds_only_the_installer(tmp_path):
    """It goes on PYTHONPATH, so anything else in it shadows a real module
    for every process that inherits the environment."""
    _site, src, _probe = _world(tmp_path)
    root = tmp_path / "stage"
    override.install([override.Override("fakepkg.mod", src)], root=root)
    importable = [p.name for p in root.iterdir()
                  if p.suffix == ".py" or p.is_dir()]
    assert importable == ["sitecustomize.py"]
