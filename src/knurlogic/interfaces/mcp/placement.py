"""Where `load` runs, without machine-specific names: `machines` may hold
role words as well as names and ids, so an agent on anyone's Mac can say
where without knowing what the Macs are called.

  here    this Mac (the page's `me`)
  peers   every peer answering this Mac's page
  all     here + peers
  fit     this Mac if the model fits here (the `fit` tool's check), else the
          smallest set of answering Macs it fits on (cluster/launch's
          placement, the decision the page's Launch makes); alone only

resolve_machines turns the list into page node ids (names and ids pass
through page_client.node_ids, unchanged); everything else in `load` stays
as it was.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from knurlogic.interfaces.mcp import page_client

ROLES = ("here", "peers", "all", "fit")
#: the page's Launch picks pipeline first (assets/views/picker.js MULTI)
DEFAULT_SPLIT = "pipeline"
GIB = 1024 ** 3


def _no(why: str, **more) -> dict[str, Any]:
    return {"loaded": False, "refused": why, **more}


def _machines(st: dict) -> list:
    """[{id, name, state, cluster}] the page sees: itself first, then its
    peers in its order; `cluster` is the block placement reads (the node
    entry the page holds for that machine)."""
    nodes = [n for n in st.get("nodes") or [] if isinstance(n, dict)]
    own = next((n for n in nodes if n.get("role") == "local"), {})
    by_addr = {n.get("address"): n for n in nodes if n.get("address")}
    me = st.get("me") or {}
    out = [{"id": me.get("id"), "name": me.get("name"), "state": "answering",
            "cluster": own.get("cluster")}]
    for p in st.get("peers") or []:
        if isinstance(p, dict):
            out.append({"id": p.get("id"), "name": p.get("name"),
                        "state": p.get("state"),
                        "cluster": (by_addr.get(p.get("address")) or {})
                        .get("cluster")})
    return out


def is_role_list(words: list[str]) -> bool:
    return any(str(w).lower() in ROLES for w in words)


def resolve_machines(words: list[str], artifact: str = "", split: str = "",
                     draft: bool = True, vision: bool = True) -> tuple:
    """(ids, split, refusal): `load`'s `machines` -- role words mixed with
    names or ids -- as page node ids, here first then peers in page order,
    duplicates collapsed. `split` is the one asked, or the page's default
    when a role made it a cluster. Reads the page's /status.json."""
    low = [str(w).lower() for w in words]
    if "fit" in low:
        if len(low) != 1:
            return [], split, _no("fit stands alone: it picks the machines "
                                  "itself, so it is not mixed with names "
                                  "or other roles")
        return fit_machines(artifact, split, draft, vision)
    st = page_client.page_get("/status.json")
    ms = _machines(st)
    me, peers = ms[0], ms[1:]
    up = [p for p in peers if p["state"] == "answering"]
    picked: list = []
    for w, lw in zip(words, low):
        if lw == "here":
            picked.append(me["id"])
        elif lw in ("peers", "all"):
            if not up:
                seen = [f"{p['name']} ({p['state']})" for p in peers]
                return [], split, _no(
                    f"{lw}: no peer is answering this Mac's page",
                    machines=([me["name"]] + seen))
            picked += ([me["id"]] if lw == "all" else []) \
                + [p["id"] for p in up]
        else:
            ids, no = page_client.node_ids([w])
            if no:
                return [], split, no
            picked += ids
    order = [m["id"] for m in ms]
    ids = sorted(dict.fromkeys(picked),
                 key=lambda i: order.index(i) if i in order else len(order))
    return ids, (split or DEFAULT_SPLIT) if len(ids) >= 2 else split, None


def fit_machines(artifact: str, split: str = "", draft: bool = True,
                 vision: bool = True) -> tuple:
    """(ids, split, refusal) for `fit`: [] (this Mac, no page needed) when
    the `fit` tool says it fits here; else the smallest set of answering
    Macs cluster/launch.placement places it on -- this Mac and the peers
    in page order, the first set of each size tried first."""
    from knurlogic.interfaces.mcp import inspection
    split = split or DEFAULT_SPLIT
    if split not in page_client.SPLITS:
        return [], split, _no(f"split is tensor | pipeline, not {split!r}")
    here = inspection.fit(artifact=artifact, draft=draft, vision=vision)
    if here.get("refused"):
        return [], split, here if "loaded" in here else \
            {"loaded": False, **here}
    if here.get("fits"):
        return [], split, None
    path = _local_path(artifact)
    if not path:
        return [], split, _no("fit measures the model from this Mac's "
                              "copy, and this Mac has none",
                              note=f"name the machines for {artifact!r}")
    st = page_client.page_get("/status.json")
    up = [m for m in _machines(st) if m["state"] == "answering"
          and isinstance(m["cluster"], dict)
          and m["cluster"].get("working_set_bytes")]
    per = [{"machine": m["name"],
            "working_set_gib": round(int(m["cluster"]["working_set_bytes"])
                                     / GIB, 1),
            **({"available_gib": round(int(m["cluster"]["available_bytes"])
                                       / GIB, 1)}
               if m["cluster"].get("available_bytes") else {})}
           for m in up]
    from knurlogic.cluster import launch
    from knurlogic.tuning.knobs import mtp_of, vision_of
    sets = {**({} if draft else {"KNURLOGIC_MTP": "off"}),
            **({} if vision else {"KNURLOGIC_VISION": "off"})}
    last = f"{here.get('verdict', 'will not fit')} on this Mac"
    for k in range(2, len(up) + 1):
        try:
            shape = launch.shape_of(path, k, split, vision=vision_of(sets),
                                    mtp=mtp_of(sets))
        except (OSError, ValueError, KeyError) as e:
            last = f"{split} across {k}: {e}"
            continue
        if shape.get("refusals"):
            last = f"{split} across {k}: " + "; ".join(shape["refusals"])
            continue
        for group in combinations(up, k):
            infos = [{**m["cluster"], "name": m["name"]} for m in group]
            try:
                launch.placement(infos, shape, split)
            except ValueError as e:
                last = str(e)
                continue
            return [m["id"] for m in group], split, None
    return [], split, _no(
        f"fit: {artifact} fits on no set of the answering Macs "
        f"({split} split): {last}", machines=per,
        here={k: here[k] for k in ("verdict", "size_gib", "budget_gib",
                                   "headroom_gib") if k in here})


def _local_path(artifact: str) -> str:
    from knurlogic.interfaces.load_checks import NotLoadable, resolve_name
    try:
        return str(resolve_name(artifact, None))
    except NotLoadable:
        return ""
