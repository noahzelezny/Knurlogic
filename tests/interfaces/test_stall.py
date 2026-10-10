"""interfaces/spawn.stalled: a quiet log is a stall only while memory does
not move -- a load reading weights silently over a network share is not."""
from knurlogic.interfaces import spawn


def test_quiet_log_with_growing_memory_is_loading():
    S = spawn.STALL_QUIET_S
    pid = 990001
    assert spawn.stalled(pid, S + 1, 10, 0.0)           # first sight: by the log
    assert not spawn.stalled(pid, S + 100, 20, 100.0)   # grew: loading
    assert not spawn.stalled(pid, S + 200, 20, 100.0 + S)
    assert spawn.stalled(pid, S + 300, 20, 101.0 + S)   # still, quiet: stalled


def test_a_talking_log_is_never_stalled():
    pid = 990002
    spawn.stalled(pid, 1, 5, 0.0)
    assert not spawn.stalled(pid, 1, 5, 10_000.0)
    assert not spawn.stalled(pid, None, 5, 20_000.0)
