"""Which machine is rank 0 and the order of the ring: the leader is the
fastest single core, each next rank the one the last reaches fastest.
"""

from __future__ import annotations

# ------------------------------------------------------------- rank order
#
# Nobody types --rank. The cluster page asks this for the order, passes
# --rank/--world to each machine's `serve`, and lets the user drag a
# machine to the front (an explicit order, which wins).
#
# Rank 0 does the CPU and Python side -- HTTP, the scheduler, tokenizing,
# sampling, encoding the plan -- while under tensor every rank's GPU work
# is equal. So the leader is the fastest single core: the newest chip
# generation, then the higher P-core clock; then the most free memory (it
# may host extras, like the vision tower); then the order given.

#: link kinds, fastest first
LINK_SPEED = ("rdma", "tb5", "tb4", "ethernet", "wifi")


def _link_rank(kind) -> int:
    k = str(kind or "").lower()
    return LINK_SPEED.index(k) if k in LINK_SPEED else len(LINK_SPEED)


def chip_generation(chip) -> int:
    """"Apple M4 Max" -> 4; 0 when it does not say."""
    import re
    m = re.search(r"\bM(\d+)\b", str(chip or ""))
    return int(m.group(1)) if m else 0


def leader_key(m: dict, position: int = 0) -> tuple:
    """Sort key for the leader, smallest first: newest chip generation,
    higher P-core clock, more free memory, earlier in the given order."""
    free = m.get("free_bytes")
    if free is None:
        free = m.get("working_set_bytes") or 0
    return (-chip_generation(m.get("chip")),
            -float(m.get("p_core_ghz") or 0.0), -int(free), position)


def rank_order(machines: list, explicit: list | None = None) -> list:
    """Machine names in rank order.

    `machines`: [{"name", "chip" ("Apple M4 Max"), "p_core_ghz",
    "free_bytes" (else "working_set_bytes"), "links": {other name: link
    kind}, ...}]; anything else (memory bandwidth, ...) rides along for
    later placement and is ignored here. Rank 0 is `leader_key`'s first.
    Each next rank is the unplaced machine the previous one reaches over
    the fastest link (LINK_SPEED; the ring follows the links), ties by
    `leader_key`.

    `explicit`: names in the order wanted (the page's drag). Every machine
    must appear once; it is returned as is."""
    names = [m["name"] for m in machines]
    if len(set(names)) != len(names):
        raise ValueError(f"machine names repeat: {names}")
    if explicit is not None:
        if sorted(explicit) != sorted(names):
            raise ValueError(f"explicit order {list(explicit)} is not a "
                             f"permutation of the machines {names}")
        return list(explicit)
    if not machines:
        return []
    pos = {n: i for i, n in enumerate(names)}
    by = {m["name"]: m for m in machines}
    order = [min(machines, key=lambda m: leader_key(m, pos[m["name"]]))["name"]]
    left = [n for n in names if n != order[0]]
    while left:
        here = by[order[-1]].get("links") or {}
        nxt = min(left, key=lambda n: (_link_rank(here.get(n)),)
                  + leader_key(by[n], pos[n]))
        order.append(nxt)
        left.remove(nxt)
    return order
