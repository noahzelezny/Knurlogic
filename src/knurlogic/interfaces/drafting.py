"""`knurlogic mtp` -- which artifacts can draft, across the whole machine.

The per-artifact answer lives in the engine (`knurlogic.engine.mtp.status`,
stdlib only). Surveying every model on the machine needs `machine.discover`,
and the engine depends on nothing above it -- so the survey is here, beside
the MCP `drafting` tool that answers the same question for an agent.

It lived in the engine until the reorganisation made the layering visible;
it had been broken since mtp became a package (`from . import discover`
stopped resolving) and nothing had run it.
"""
from __future__ import annotations

from knurlogic.engine.mtp import BUILT, DECLARED, GRAFTABLE, status
from knurlogic.machine import discover
from knurlogic.machine.artifact import Artifact


def survey(rows) -> str:
    """The three states across a set of discovered models.

    A per-artifact answer is what `doctor` gives. This one exists because the
    useful question is comparative: an artifact that declares a head reads as
    broken alone, and as completely ordinary next to the 38 others that do
    the same thing.
    """
    groups: dict = {BUILT: [], GRAFTABLE: [], DECLARED: []}
    for f in rows:
        if not f.servable:
            continue
        try:
            st = status(Artifact.load(str(f.path)))
        except Exception:
            continue
        if st.state in groups:
            groups[st.state].append((f, st))

    L = []
    built = groups[BUILT]
    if built:
        L.append(f"{len(built)} artifacts have a BUILT drafting head beside "
                 f"the weights:")
        for f, st in built:
            L.append(f"  {f.name[:44]:<46}{st.head.gib:>6.2f} GiB  "
                     f"{st.head.family or '?'}  {st.head.bits or '?'}-bit")
        L.append("")
    if groups[GRAFTABLE]:
        L.append(f"{len(groups[GRAFTABLE])} carry raw `mtp.*` weights a head "
                 f"can be built FROM:")
        for f, st in groups[GRAFTABLE]:
            L.append(f"  {f.name[:44]:<46}{st.graft_tensors:>6d} tensors")
        L.append("")
    dec = groups[DECLARED]
    if dec:
        vq = [x for x in dec if x[1].is_vq]
        L.append(f"{len(dec)} declare a head and have none packed beside "
                 f"them.")
        if vq:
            L.append(f"  {len(vq)} are VQ artifacts, where packing one is "
                     f"the VQ toolchain's job.")
        if len(dec) - len(vq):
            L.append(f"  {len(dec) - len(vq)} are community rungs that "
                     f"inherited the config key; no")
            L.append("  publisher ships those weights, so nothing is missing "
                     "and there is nothing")
            L.append("  to do about it.")
    return "\n".join(L) or "no artifact here declares or ships an MTP head."


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="knurlogic mtp",
        description="which artifacts have a multi-token-prediction head, "
                    "which could have one, and which only say they do")
    p.add_argument("artifact", nargs="?",
                   help="one artifact; omit to survey every model found")
    a = p.parse_args(argv)

    if a.artifact:
        st = status(Artifact.load(a.artifact))
        print(st.render() or "no MTP head, and none declared.")
        return 0
    print(survey(discover.find()))
    return 0
