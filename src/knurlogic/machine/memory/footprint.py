"""Where this machine's memory went, from the OS: `vm_stat` for what is
used and what a model could still have (available_memory), and every
process's phys_footprint attributed to a runtime (memory_map). The
runtimes' own accounts of what they hold are machine/loaded.py's.

Design: docs/design/memory.md (loaded).
"""

from __future__ import annotations

# --- where the RAM went -----------------------------------------------------
# The runtimes are unreliable narrators about their own size: exo reports none
# per instance, and mlx-lm reports none at all. The OS knows, so ask it.
#
# WHICH NUMBER, and the obvious one is wrong: `ps -o rss` undercounts badly.
# Measured on one live process here, 1.93 GiB RSS against a 4.40 GiB phys
# footprint -- 2.3x, and that is a process holding no Metal buffers. RSS
# misses what a model runtime mostly IS. `top -l 1 -stats pid,mem` reports
# phys_footprint, the same number Activity Monitor shows as Memory, and costs
# 0.31s.

#: command-line fragment -> runtime. Order matters: knurlogic before the
#: plain mlx names, because ours is an mlx server too.
_RUNTIME_MARKS = (
    ("knurlogic", "knurlogic"),
    ("exo", "exo"),
    ("ollama", "ollama"),
    ("mlx_vlm", "mlx-vlm"),
    ("mlx_lm", "mlx-lm"),
    ("vqlab", "vqlab"),
)


_UNIT = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def available_memory() -> dict:
    """How much memory a model could actually have, from `vm_stat`.

    Used is what cannot be handed back without swapping: anonymous pages
    (a model's weights, active OR inactive -- an MoE's idle experts sit in
    inactive anonymous pages), wired pages and what the compressor
    occupies. Available is what can: free, file-backed pages (cache,
    droppable; speculative read-ahead is among them) and purgeable ones.

    Earlier versions counted `inactive` as available because psutil does.
    On a 128 GiB Mac holding a 109 GiB model that read 66.9 GiB used when
    real use was ~125: inactive anonymous pages are reclaimable only by
    swapping, and the fit check then admitted a model that did not fit.
    File-backed pages (active or inactive) are ONE bucket here, so nothing
    is counted twice; purgeable pages are anonymous and are moved from used
    to available.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    m = re.search(r"page size of (\d+)", out)
    if not m:
        return {}
    page = int(m.group(1))
    st = {}
    for line in out.splitlines()[1:]:
        g = re.match(r'"?([^":]+)"?:\s+(\d+)', line.strip())
        if g:
            st[g.group(1).strip()] = int(g.group(2)) * page

    free = st.get("Pages free", 0)
    # Speculative pages (read-ahead) are already inside "File-backed pages":
    # active + inactive + speculative == file-backed + anonymous. Adding
    # them again overstated what a launch could have.
    cache = st.get("File-backed pages", 0)
    purgeable = st.get("Pages purgeable", 0)
    wired = st.get("Pages wired down", 0)
    comp = st.get("Pages occupied by compressor", 0)
    res = {
        "available_bytes": free + cache + purgeable,
        "free_bytes": free,
        "cached_bytes": cache,
        "purgeable_bytes": purgeable,
        "wired_bytes": wired,
        "compressed_bytes": comp,
    }
    if "Anonymous pages" in st:
        res["used_bytes"] = max(
            st["Anonymous pages"] - purgeable, 0) + wired + comp
    return res


def _footprints() -> tuple:
    """({pid: bytes}, physmem) from `top`, which reports phys_footprint."""
    import re
    import subprocess
    try:
        out = subprocess.run(
            # `-o mem` is load-bearing: without it `-n` takes the first N
            # processes in top's default order, not the N biggest, and the
            # multi-gigabyte ones are simply not in the output. The first
            # pass here saw 1.9 GiB on a box holding 15.
            #
            # No `-n` at all: the whole table costs the same 0.31s as sixty
            # rows, and a cut loses exactly the case that matters -- an IDLE
            # runtime sits near the bottom, and "exo is holding 163 MiB
            # because nothing is loaded" is an answer, not noise.
            ["top", "-l", "1", "-o", "mem", "-stats", "pid,mem"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return {}, {}
    found, phys = {}, available_memory()
    for line in out.splitlines():
        if line.startswith("PhysMem:"):
            continue
        m = re.match(r"\s*(\d+)\s+([\d.]+)([BKMGT])\s*$", line)
        if m:
            found[int(m.group(1))] = int(float(m.group(2)) * _UNIT[m.group(3)])
    return found, phys


def _commands() -> dict:
    """{pid: command line}. `ps` is the only place the full argv lives, and
    the argv is what says which runtime a bare `python3.12` belongs to."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="],
                             capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cmds = {}
    for line in out.splitlines():
        line = line.strip()
        pid, _, rest = line.partition(" ")
        if pid.isdigit():
            cmds[int(pid)] = rest.strip()
    return cmds


#: Never a model runtime, however the command line reads. A shell sitting in
#: a directory named after a runtime would match, and so would a `tail`
#: watching an exo log -- both reported as runtimes holding memory.
_NOT_A_RUNTIME = {
    "zsh", "bash", "sh", "fish", "tail", "head", "less", "more", "grep",
    "cat", "vim", "nvim", "nano", "code", "git", "ssh", "tmux", "screen",
    "make", "watch", "sed", "awk", "find", "rg", "fd", "top", "ps",
}


def _runtime_of(cmd: str) -> str:
    """Which runtime a process belongs to, from its EXECUTABLE and its
    `-m module`, never from the command line as a whole.

    Matching anywhere in the line over-attributes badly: it catches a
    shell whose working directory is named after a project and a `tail`
    following a log, and reports both as runtimes holding memory. The
    executable path
    and the module being run are the two places that actually say what the
    process IS.
    """
    if not cmd:
        return ""
    from knurlogic.machine.servers import is_test_process
    if is_test_process(cmd):
        return ""             # a test's fake rank is no runtime of this Mac
    parts = cmd.split()
    exe = parts[0]
    base = exe.rsplit("/", 1)[-1].lstrip("-").lower()
    if base in _NOT_A_RUNTIME:
        return ""

    # The module after `-m`, which is what names a python process -- or,
    # for an interpreter running a script, the script: a console-script
    # launch (`.../Python .../venv/bin/knurlogic serve`) names itself only
    # there, and would otherwise count as "everything else".
    module = ""
    for i, tok in enumerate(parts[:-1]):
        if tok == "-m":
            module = parts[i + 1].lower()
            break
    places = [exe]
    if not module and base.startswith("python") and len(parts) > 1 \
            and not parts[1].startswith("-"):
        places.append(parts[1])

    # The module wins over the interpreter's path: `envs/exo/bin/python -m
    # vqlab.cli` is vqlab borrowing exo's env, not exo.
    for mark, name in _RUNTIME_MARKS:
        if module == mark or module.startswith(mark + "."):
            return name
    if module:
        places = places[1:]
    # The script before the interpreter, as the module is: `envs/exo/bin/
    # python .../vqlab/bench/speed_pair.py` is vqlab borrowing exo's env.
    for place in reversed(places):
        low = place.lower()
        for mark, name in _RUNTIME_MARKS:
            if low.rsplit("/", 1)[-1].lstrip("-") == mark:
                return name
            # A path component, i.e. an env or install directory belonging
            # to it -- `.../envs/exo/bin/python3.13` is exo's interpreter.
            if f"/{mark}/" in low:
                return name
    return ""


def memory_map(floor: int = 256 << 20) -> dict:
    """Every process above `floor`, attributed to a runtime where possible.

    The unattributed remainder is REPORTED, not hidden. "Where did the RAM
    go" is not answered by a list that sums to less than the machine and
    does not say so.
    """
    (foot, phys), cmds = _footprints(), _commands()
    rows: list = []
    by_runtime: dict = {}
    for pid, b in foot.items():
        cmd = cmds.get(pid, "")
        rt = _runtime_of(cmd)
        # The floor is for everything else. A runtime process is reported
        # whatever it weighs: the question is where the memory went, and
        # "exo has four processes totalling 200 MiB" answers it -- nothing
        # is loaded. Dropping it under a floor answers nothing and looks
        # identical to exo not running.
        if not rt and b < floor:
            continue
        name = (cmd.split(" ")[0].rsplit("/", 1)[-1] or f"pid {pid}")[:40]
        rows.append({"pid": pid, "bytes": b, "runtime": rt, "name": name,
                     "cmd": cmd[:200]})
        if rt:
            by_runtime[rt] = by_runtime.get(rt, 0) + b
    rows.sort(key=lambda r: -r["bytes"])

    total = 0
    try:
        import subprocess
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True,
                                   timeout=5).stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    seen = sum(r["bytes"] for r in rows)
    avail = phys.get("available_bytes")
    used = phys.get("used_bytes")
    if used is None:
        used = (total - avail) if (total and avail is not None) else seen
    # A footprint counts pages that were swapped out; `used` does not. What
    # the footprints add up to beyond `used` is therefore in swap (at most
    # what the OS says is swapped), shared among the runtimes by size.
    # Each runtime's row is then its RESIDENT part, so the rows add up to
    # the machine from this one sample.
    from knurlogic.machine import metrics as _metrics
    swap = _metrics._swap() or 0
    foot_rt = dict(by_runtime)
    fp = sum(foot_rt.values())
    in_swap = min(swap, max(sum(foot.values()) - used, 0), fp) \
        if phys.get("used_bytes") is not None else 0
    swapped = {k: int(in_swap * v / fp) for k, v in foot_rt.items()} \
        if fp and in_swap else {}
    by_runtime = {k: v - swapped.get(k, 0) for k, v in foot_rt.items()}
    rt = sum(by_runtime.values())
    # "Everything else" is what the OS says is spent MINUS what we could put
    # a name to -- the kernel, the file cache, compressed pages and every
    # process under the floor. Deriving it from the footprints instead made
    # the free figure wrong by 74 GiB on this machine.
    return {
        "installed_bytes": total,
        "seen_bytes": seen,
        "used_bytes": used,
        # What a model could have: the free pages PLUS the file cache macOS
        # will hand over on demand. Not top's "unused", which is only the
        # first of those.
        "free_bytes": phys.get("available_bytes", max(total - used, 0)),
        "truly_free_bytes": phys.get("free_bytes", 0),
        "cached_bytes": phys.get("cached_bytes", 0),
        "wired_bytes": phys.get("wired_bytes", 0),
        # RESIDENT bytes per runtime (footprint minus what is swapped out);
        # the footprints themselves and the swapped part ride beside it
        "by_runtime": by_runtime,
        "footprint_by_runtime": foot_rt,
        "swapped_by_runtime": swapped,
        "swap_bytes": swap,
        "runtime_bytes": rt,
        "other_bytes": max(used - rt, 0),
        "processes": rows[:25],
        "floor_bytes": floor,
        "from_os": bool(phys),
        "metric": "phys_footprint (what Activity Monitor calls Memory), not "
                  "RSS -- measured 2.3x apart on one process here. Used and "
                  "free come from the OS, not from summing these.",
    }
