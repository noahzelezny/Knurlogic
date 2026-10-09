"""The pipeline split's arithmetic: which models pipeline, the bytes of
each layer, how many layers each rank holds, and what the leader holds
besides (engine/split/pipeline.py does the split).
"""

from __future__ import annotations

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import fit, measured

# ---------------------------------------------------------- pipeline split
#
# One model served by N ranks, each holding a contiguous run of layers
# (engine/split/pipeline.py). Rank 0 -- the leader, which samples -- holds
# the LAST layers, so the logits are born where they are used and nothing
# is gathered; rank N-1 holds the first layers and embeds. Pure arithmetic,
# so the split is said, with its reason, before anything loads.

#: model types engine/split/pipeline.py knows how to slice (each has its
#: own index fixups there); anything else is refused with a reason
PIPELINE_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text",
                  "qwen3_5_moe_text", "glm5_next", "qwen4_exp",
                  "qwen4_exp_text", "deepseek_v4")

_PIPELINE_WHY_NOT = {
    "gemma4": "gemma4 shares KV across layers (a layer reads a cache another "
              "layer wrote), so a cut between them would need that cache on "
              "two ranks; not built",
}

#: memory bandwidth (GB/s) of chips that come in ONE bandwidth only. A Max
#: that is sold binned (M3 Max 300/400, M4 Max 410/546) is not here:
#: unknown is said, never guessed.
CHIP_BANDWIDTH_GBS = {
    "Apple M1 Max": 400.0, "Apple M1 Ultra": 800.0,
    "Apple M2 Max": 400.0, "Apple M2 Ultra": 800.0,
    "Apple M3 Ultra": 819.0, "Apple M4 Pro": 273.0,
}


def chip_bandwidth_gbs(chip) -> float | None:
    return CHIP_BANDWIDTH_GBS.get(str(chip or "").strip())


def pipeline_refusals(cfg: dict, n: int) -> list:
    """Why this config cannot be pipelined `n` ways; [] when it can."""
    if n < 2:
        return []
    tc = cfg.get("text_config", cfg) or {}
    types = {cfg.get("model_type"), tc.get("model_type")}
    if not types & set(PIPELINE_TYPES):
        mt = cfg.get("model_type")
        why = next((w for k, w in _PIPELINE_WHY_NOT.items()
                    if any(str(t or "").startswith(k) for t in types)), None)
        return [f"pipeline split knows {', '.join(PIPELINE_TYPES[:2])}, "
                f"glm5_next, qwen4_exp and deepseek_v4; this is {mt!r}"
                + (f": {why}" if why else "")]
    L = tc.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if L is not None and int(L) < n:
        return [f"num_hidden_layers = {L} is fewer than {n} ranks: every "
                f"rank holds at least one layer"]
    return []


def layer_bytes_of(tensors: dict, n_layers: int) -> tuple:
    """{name: bytes} -> ([bytes of layer i for i < n_layers], bytes outside
    the layers). A tensor of layer i is one whose name holds `.layers.i.`;
    an index >= n_layers, or a name under `mtp.` (a grafted head), counts
    as outside."""
    import re
    per = [0] * n_layers
    other = 0
    rx = re.compile(r"\.layers\.(\d+)\.")
    for k, b in tensors.items():
        m = None if k.split(".")[0] == "mtp" else rx.search(k)
        i = int(m.group(1)) if m else -1
        if 0 <= i < n_layers:
            per[i] += int(b)
        else:
            other += int(b)
    return per, other


def _largest_remainder(n: int, weights: list, floor: list, cap: list) -> list:
    """n items shared by weight, each share within [floor, cap]; ties go
    to the lower index. Deterministic: plain arithmetic, no sort instability
    (keys carry the index)."""
    k = len(weights)
    out = list(floor)
    left = n - sum(out)
    free = [i for i in range(k) if out[i] < cap[i]]
    while left > 0 and free:
        tot = sum(weights[i] for i in free)
        if tot <= 0:
            want = {i: left / len(free) for i in free}
        else:
            want = {i: left * weights[i] / tot for i in free}
        give = {i: min(int(want[i]), cap[i] - out[i]) for i in free}
        if sum(give.values()) == 0:
            # remainders: the largest fractional part first, lower rank on a
            # tie
            order = sorted(free, key=lambda i: (-(want[i] - int(want[i])), i))
            give = {i: 0 for i in free}
            for i in order[:left]:
                give[i] = 1
        for i, g in give.items():
            out[i] += g
            left -= g
        free = [i for i in range(k) if out[i] < cap[i]]
    return out


def _byte_bounds(layer_bytes: list, weights: list, cap: list):
    """Contiguous runs by the real bytes: rank n-1 takes the first layers,
    each rank's run as near its weight's share of the bytes still unplaced
    as whole layers allow, within what it can hold, leaving every later rank
    one layer; rank 0 takes the rest. [(start, end) per rank], or None when
    a rank would hold more than it can."""
    n, L = len(weights), len(layer_bytes)
    bounds: list = [None] * n
    at, left = 0, float(sum(layer_bytes))
    for r in range(n - 1, 0, -1):
        wsum = sum(weights[:r + 1])
        target = left * (weights[r] / wsum if wsum > 0 else 1.0 / (r + 1))
        end, acc = at, 0
        while end < L - r:            # leave ranks r-1 .. 0 a layer each
            nxt = acc + layer_bytes[end]
            if nxt > cap[r] or (end > at and
                                abs(nxt - target) >= abs(acc - target)):
                # a tie leaves the layer to the lower ranks (rank 0 last)
                break
            acc, end = nxt, end + 1
        if end == at:
            return None
        bounds[r] = (at, end)
        left -= acc
        at = end
    bounds[0] = (at, L)
    if any(sum(layer_bytes[a:b]) > cap[r] for r, (a, b) in enumerate(bounds)):
        return None
    return bounds


def pipeline_shares(layer_bytes: list, ranks: list, other_bytes: int = 0,
                    leader_bytes: int = 0, reserve: dict | None = None) -> dict:
    """Which layers each rank holds.

    `layer_bytes`: bytes of each layer, in order. `ranks`: in rank order,
    [{"name", "working_set_bytes", "memory_bandwidth_gbs" (None: unknown)}].
    `other_bytes`: what every rank holds besides its layers (embeddings,
    final norm, lm_head -- replicated). `leader_bytes`: what rank 0 alone
    holds besides (the MTP head and the vision tower:
    `leader_bytes`), so rank 0 takes fewer layers for them.

    A rank's weight is what it can hold (working set less the replicated
    bytes), times its memory bandwidth when EVERY rank's is known (decode
    reads each layer's weights once per step, so a faster rank should read
    more of them); a mix of known and unknown bandwidths weighs by capacity
    only, and says so. What a rank can hold leaves its step margin
    (step_margin) free; the reason says what each rank leaves. Every rank holds
    at least one layer and no rank more
    than fits; `reserve` (fit_reserve) is what each rank also keeps free:
    the first request's transient and a minimum context's KV, so an uneven
    split respects what the fit check does; rank 0 holds the LAST run of
    layers, rank N-1 the first.

    -> {"layers": [count per rank], "bounds": [(start, end) per rank],
        "bytes": [layer bytes per rank], "weights": [...], "reason": str}
    Raises ValueError, with the arithmetic, when it cannot be done."""
    n, L = len(ranks), len(layer_bytes)
    if n < 1:
        raise ValueError("no ranks")
    if L < n:
        raise ValueError(f"{L} layers cannot give each of {n} ranks one")
    names = [str(r.get("name", f"rank{i}")) for i, r in enumerate(ranks)]
    wss = [int(r.get("working_set_bytes") or 0) for r in ranks]
    # a rank's layers leave its step margin free, as a single machine's
    # weights do (the scheduler's floor: 5%, at least 4 GiB)
    held = [int(other_bytes) + (int(leader_bytes) if i == 0 else 0)
            for i in range(n)]
    cap = [w - held[i] - fit.rank_margin(w, reserve) for i, w in enumerate(wss)]
    for i, c in enumerate(cap):
        if c <= 0:
            raise ValueError(
                f"{names[i]}: working set {wss[i] / fit.GIB:.1f} GiB "
                f"holds none of the layers after the {held[i] / fit.GIB:.1f} "
                f"GiB it keeps besides them (embeddings, norm, lm_head"
                + (", and on rank 0 the MTP head and vision tower"
                   if i == 0 and leader_bytes else "")
                + f") and the {fit.rank_margin(wss[i], reserve) / fit.GIB:.1f} GiB "
                f"it keeps free (step margin or first-request transient, "
                f"plus a minimum context's KV)")
    bws = [r.get("memory_bandwidth_gbs") for r in ranks]
    known = all(b for b in bws)
    weights = [cap[i] * (float(bws[i]) if known else 1.0) for i in range(n)]
    avg = sum(layer_bytes) / L if L else 0
    # a rank's ceiling in layers, by the average layer; checked exactly below
    ceil = [max(1, min(L, int(cap[i] // avg))) if avg else L for i in range(n)]
    if sum(ceil) < L:
        raise ValueError(
            f"{L} layers x {avg / fit.GIB:.2f} GiB average = "
            f"{sum(layer_bytes) / fit.GIB:.1f} GiB; the ranks hold "
            + " + ".join(f"{names[i]} {cap[i] / fit.GIB:.1f}" for i in range(n))
            + f" = {sum(cap) / fit.GIB:.1f} GiB of layers")
    counts = _largest_remainder(L, weights, [1] * n, ceil)
    # stage order: rank n-1 first ... rank 0 last
    bounds: list = [None] * n
    at = 0
    for r in range(n - 1, -1, -1):
        bounds[r] = (at, at + counts[r])
        at += counts[r]
    # Cut by the real bytes, not by counting layers: layers are not alike
    # (Qwen3.8 Flash's layer 1 carries a 42 GiB n-gram embedding). Counted,
    # an M3 Ultra rank took layers 0..18 -- 63.5 GiB of 110 -- and fit, but
    # with 13 GiB left for every prompt's KV while the M4 Max rank kept
    # 70 GiB free, and long prompts were refused. The count is the fallback,
    # not the rule.
    alt = _byte_bounds(layer_bytes, weights, cap)
    if alt is not None:
        bounds = alt
        counts = [b - a for a, b in bounds]
    got = [sum(layer_bytes[a:b]) for a, b in bounds]
    over = [i for i in range(n) if got[i] > cap[i]]
    if over:
        i = over[0]
        raise ValueError(
            f"{names[i]}: layers {bounds[i][0]}..{bounds[i][1] - 1} are "
            f"{got[i] / fit.GIB:.1f} GiB against {cap[i] / fit.GIB:.1f} GiB it can "
            f"hold")
    how = ("capacity x memory bandwidth" if known else
           "capacity only (memory bandwidth unknown on "
           + ", ".join(names[i] for i in range(n) if not bws[i]) + ")")
    reason = (f"{L} layers by {how}: " + "; ".join(
        f"rank {i} {names[i]} holds {counts[i]} (layers {bounds[i][0]}.."
        f"{bounds[i][1] - 1}, {got[i] / fit.GIB:.1f} of {cap[i] / fit.GIB:.1f} GiB"
        f", leaves {(wss[i] - held[i] - got[i]) / fit.GIB:.1f} GiB"
        + (f", {float(bws[i]):g} GB/s" if bws[i] else "") + ")"
        for i in range(n)) + "; rank 0 holds the last layers and samples")
    return {"layers": counts, "bounds": [tuple(b) for b in bounds],
            "bytes": got, "weights": weights, "reason": reason}


def pipeline_layer_bytes(artifact: Artifact) -> tuple:
    """layer_bytes_of over the artifact's top-level safetensors headers.
    Neither the tower nor an MTP head is the trunk, and neither is on every
    rank: rank 0 alone holds them (`leader_bytes`)."""
    import json
    import struct

    cfg = artifact.raw_config or {}
    tc = cfg.get("text_config", cfg) or {}
    L = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    sizes = {}
    for f in sorted(artifact.path.glob("*.safetensors")):
        if f.name.startswith("model-vision"):
            continue
        try:
            with open(f, "rb") as fh:
                (hn,) = struct.unpack("<Q", fh.read(8))
                if hn <= 0 or hn > (1 << 28):
                    continue
                header = json.loads(fh.read(hn))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.startswith(measured.VISION_TOWER_PREFIXES):
                continue
            if f.name.startswith("mtp") or k.split(".")[0] == "mtp":
                continue
            a, b = v.get("data_offsets", (0, 0))
            sizes[k] = int(b) - int(a)
    return layer_bytes_of(sizes, L)


def leader_bytes(artifact: Artifact, vision: bool = True,
                 mtp: bool = True) -> int:
    """What rank 0 of a split model (pipeline or tensor) holds and no other
    rank does: the MTP head when it drafts (`mtp`; the followers only run
    the verify rows) and the vision tower when it serves images (`vision`;
    it encodes at tokenize and ships the image rows, a follower binds the
    family without one). Read off the safetensors headers."""
    import json
    import struct

    total = 0
    for f in sorted(artifact.path.glob("*.safetensors")) if mtp else ():
        try:
            with open(f, "rb") as fh:
                (hn,) = struct.unpack("<Q", fh.read(8))
                if hn <= 0 or hn > (1 << 28):
                    continue
                header = json.loads(fh.read(hn))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if f.name.startswith("mtp") or k.split(".")[0] == "mtp":
                a, b = v.get("data_offsets", (0, 0))
                total += int(b) - int(a)
    return total + (fit.tower_bytes(artifact)[0] if vision else 0)
