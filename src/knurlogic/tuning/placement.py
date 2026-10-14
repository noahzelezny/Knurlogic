"""Where a split model's ranks go: the rank order, each rank's share of
the weights against its machine's budget, and the prompt chunk a rank's
own room allows. Pure over the machines' cluster blocks and the
artifact's shape (cluster/launch.shape_of), so every page that places the
same job places it the same way.
"""

from __future__ import annotations

import logging

from knurlogic.machine.memory.wired import budget_of
from knurlogic.machine.units import gib

logger = logging.getLogger(__name__)


def placement(machines: list, shape: dict, split: str,
              order: list | None = None) -> dict:
    """Rank order and each rank's share. `machines`: [{"name", "chip",
    "working_set_bytes", "available_bytes"?, "bandwidth_gbs", "links"}];
    `shape`: the artifact's {"layer_bytes", "other_bytes", "leader_bytes"
    (rank 0's alone: the MTP head and the vision tower),
    "tensor_per_rank_bytes", "refusals"}. Pure: the same inputs give the
    same answer on every page.
    -> {"order": [names], "leader", "split", "shares": [{"rank", "machine",
        "bytes", "layers"?, "bounds"?}], "layers": [counts] | [],
        "reason"}. Raises ValueError with the arithmetic when it cannot."""
    from knurlogic.tuning import fit, pipeline_split, rank_order
    if shape.get("refusals"):
        raise ValueError("; ".join(shape["refusals"]))
    names = rank_order.rank_order(machines, order)
    by = {m["name"]: m for m in machines}
    n = len(names)
    lead = int(shape.get("leader_bytes") or 0)
    if split == "tensor":
        # the ranks' shares are equal; rank 0 alone adds the head and tower
        per = int(shape["tensor_per_rank_bytes"])
        shares, left = [], []
        for r, nm in enumerate(names):
            budget = budget_of(by[nm])
            own = fit.rank_share_bytes(shape, r)
            if fit.share_refusal(own, budget):
                plus = (f" + {gib(lead)} rank 0 alone holds (the "
                        f"MTP head and vision tower)" if r == 0 and lead
                        else "")
                raise ValueError(f"{nm}: its tensor share {gib(per)}{plus} "
                                 f"does not fit its working set "
                                 f"{gib(budget)} less the "
                                 f"{gib(fit.step_margin(budget))} step "
                                 f"margin: more space required")
            shares.append({"rank": r, "machine": nm, "bytes": own})
            left.append(f"{nm} leaves {gib(budget - own)}")
        reason = (f"tensor split {n} ways: every rank holds "
                  f"~{gib(per)}"
                  + (f", rank 0 {gib(lead)} more (the MTP head and "
                     f"vision tower)" if lead else "")
                  + f" ({', '.join(left)}); rank 0 "
                  f"{names[0]} leads (newest chip, then free memory) and "
                  f"samples")
        return {"order": names, "leader": names[0], "split": split,
                "shares": shares, "layers": [], "reason": reason}
    ranks = [{"name": nm,
              "working_set_bytes": budget_of(by[nm]),
              "memory_bandwidth_gbs": by[nm].get("bandwidth_gbs")}
             for nm in names]
    layers, other = list(shape["layer_bytes"]), int(shape.get("other_bytes") or 0)
    try:
        sh = pipeline_split.pipeline_shares(layers, ranks, other, lead,
                                            reserve=shape.get("reserve"))
    except ValueError:
        if not shape.get("reserve"):
            raise
        # the layers fit with only the step margin kept free: the fit
        # line is the step margin (fit.single_fit_check), so it fits
        sh = pipeline_split.pipeline_shares(layers, ranks, other, lead,
                                            reserve=None)
    shares = [{"rank": r, "machine": nm,
               "bytes": fit.rank_share_bytes(shape, r, sh["layers"]),
               "layers": sh["layers"][r], "bounds": list(sh["bounds"][r])}
              for r, nm in enumerate(names)]
    return {"order": names, "leader": names[0], "split": split,
            "shares": shares, "layers": sh["layers"],
            "reason": f"rank 0 {names[0]} leads; " + sh["reason"]}


def rank_room_chunk(path, spec: dict, budget: int, share: int):
    """The prompt chunk this rank's own room allows -- its share of the
    weights against its own budget, the resolver's rule -- or None when it
    cannot be worked out (the ring then runs its floor). `spec`: the
    rank's job spec ("tune", "sets")."""
    try:
        from knurlogic.machine.artifact import Artifact
        from knurlogic.tuning import knobs
        from knurlogic.tuning.resolve import launch_settings, resolve
        a = Artifact.load(path)
        tune = spec.get("tune") or "default"
        sets = knobs.canonical_sets(dict(spec.get("sets") or {}))
        launch = launch_settings(a, tune, sets)
        r = resolve(a, int(budget), tune=tune, holds_bytes=int(share),
                    kv_bits=launch.get("kv_bits"),
                    long_context=launch.get("long_context", "off"),
                    vision=launch.get("vision", True))
        v = knobs.engine_settings(r.env).get("prefill_step_size")
        return int(v) if v else None
    except Exception:  # a rank that cannot say leaves the ring on the floor
        logger.exception("could not work out this rank's prompt chunk")
        return None
