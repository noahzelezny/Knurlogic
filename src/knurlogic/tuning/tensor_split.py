"""The tensor split's arithmetic: which models split, what each rank
holds, and why a split is refused -- from the config and the safetensors
headers, before anything loads (engine/split/tensor.py does the split).
"""

from __future__ import annotations

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import measured

# ------------------------------------------------------------ tensor split
#
# One model served by N ranks, every layer's weights split N ways (the
# qwen3_5 families, qwen4_exp and deepseek_v4: engine/split/tensor.py does
# the split).
# Pure arithmetic over the config and the safetensors headers, so a refusal is
# said -- with its numbers -- before anything loads.

def _tensor_maps() -> tuple:
    from knurlogic.engine import families
    m = families.build_maps()
    return m["tensor"], m["tensor_archs"]


#: model type -> its family's `tensor` entry (engine/families/<family>):
#: the model types engine/split/tensor.py knows how to split, and the
#: config keys each needs divisible by the ranks; and those architectures
_TENSOR, _TENSOR_ARCHS = _tensor_maps()
TENSOR_TYPES = tuple(_TENSOR)


def tensor_sharded(name: str) -> bool:
    """Is this weight split across ranks under tensor? (tensor_rules: a VQ
    codebook never is.)"""
    from knurlogic.engine.split.tensor_rules import sharded
    return sharded(name)


def _block_width(tc: dict, inp: dict):
    """An input's width: the product of its config keys; None when one is
    absent and has no default."""
    w = 1
    for k in inp["keys"]:
        v = tc.get(k)
        if v is None:
            v = (inp.get("defaults") or {}).get(k)
        elif not v and k in (inp.get("defaults") or {}):
            v = inp["defaults"][k]
        if v is None:
            return None
        w *= int(v)
    return w


def _block_refusals(types: set, tc: dict, n: int) -> list:
    """A family whose activations are rounded in blocks along a linear's
    input (its manifest's `tensor_split`: DeepSeek-V4's act_quant, block
    128, architecture edit 21): a split that cuts such an input must leave
    each rank whole blocks, or the split model rounds other blocks than
    the whole one."""
    from knurlogic.engine import families
    rules = families.build_maps()["tensor_split"]
    out: list = []
    seen: set = set()
    for t in sorted(t for t in types if t in rules):
        rule = rules[t]
        b = int(rule["act_quant_block"])
        for inp in rule["inputs"]:
            if inp["what"] in seen:
                continue
            seen.add(inp["what"])
            v = _block_width(tc, inp)
            if v is None or v % n:
                continue
            if (v // n) % b:
                out.append(
                    f"{inp['what']} = {v}: a rank's {v // n} is not whole "
                    f"{b}-blocks ({v // n} % {b} = {v // n % b})"
                    f": its activation rounding (act_quant, block {b}) "
                    f"would differ from the whole model's")
    return out


def tensor_refusals(cfg: dict, n: int) -> list:
    """Why this config cannot be split `n` ways, one line per reason with
    its arithmetic; [] when it can."""
    out: list = []
    if n < 2:
        return out
    tc = cfg.get("text_config", cfg)
    types = {cfg.get("model_type"), tc.get("model_type")}
    if not types & set(TENSOR_TYPES):
        out.append(f"tensor split knows {', '.join(_TENSOR_ARCHS)}; this "
                   f"is {cfg.get('model_type')!r}")
        return out

    def div(what, v, why=""):
        if v is None:
            return
        if int(v) % n:
            out.append(f"{what} = {v} is not divisible by {n} ranks "
                       f"({v} / {n} = {int(v) / n:g}){why}")

    div("num_attention_heads", tc.get("num_attention_heads"))
    kv = tc.get("num_key_value_heads")
    if kv:
        if kv >= n:
            div("num_key_value_heads", kv)
        elif n % kv:
            out.append(f"num_key_value_heads = {kv} is fewer than {n} ranks "
                       f"and does not divide them ({n} % {kv} = {n % kv}): "
                       f"the heads cannot be repeated evenly")
    div("linear_num_key_heads", tc.get("linear_num_key_heads"))
    div("linear_num_value_heads", tc.get("linear_num_value_heads"))
    # a family's own axes a rank must hold whole (its manifest's
    # `tensor.divisible`, with why)
    for k in dict.fromkeys(k for t in sorted(types & set(_TENSOR))
                           for k in _TENSOR[t].get("divisible", ())):
        div(k, tc.get(k))
    out += _block_refusals(types, tc, n)
    # the arrays' own axes (intermediate sizes, quantization groups, VQ
    # code rows) are tensor_header_refusals': the headers answer them
    if cfg.get("vq_linear"):
        out.append(f"{len(cfg['vq_linear'])} VQ dense linear(s) (vq_linear): "
                   f"not split by tensor in this build")
    if cfg.get("vq_embed"):
        out.append(f"{len(cfg['vq_embed'])} VQ embedding(s) (vq_embed): not "
                   f"split by tensor in this build")
    for path, m in sorted((cfg.get("vq_modules") or {}).items()):
        IN = int(m.get("in", 0))
        G, D = int(m.get("group", 64)), int(m.get("dim", 1))
        if path.endswith("down_proj") and m.get("pack_bits"):
            # sharded-to-all: codes split on their input axis. Packed codes
            # are uint32 words holding 32 codes per BITS words, so a slice
            # must hold whole 32-code blocks -- 32*dim inputs (the headers
            # see words, not codes: this one the config answers)
            unit = max(G, 32 * D)
            if IN % n or (IN // n) % unit:
                out.append(
                    f"{path}: input {IN} / {n} = {IN / n:g}, not a multiple "
                    f"of {unit} (max(group {G}, 32 x dim {D})): a rank's "
                    f"slice would cut a packed code word")
        if len(out) > 12:
            out.append("... (and more)")
            break
    return out


def tensor_placement_of(tensors: dict, n: int) -> dict:
    """{name: bytes} -> what each of `n` ranks holds under tensor: the
    sharded weights' Nth plus every replicated one."""
    sharded = sum(b for k, b in tensors.items() if tensor_sharded(k))
    replicated = sum(b for k, b in tensors.items() if not tensor_sharded(k))
    return {"ranks": n, "sharded_bytes": sharded,
            "replicated_bytes": replicated,
            "per_rank_bytes": -(-sharded // max(n, 1)) + replicated}


#: path -> (stat stamp of its shards, trunk headers): the picker asks
#: every model's headers on each listing, ~80 of them, many over SMB
_HEADERS: dict = {}


def trunk_headers(path) -> dict:
    """{name: (shape, bytes)} from the artifact's top-level safetensors
    headers -- 8 bytes and a JSON each, no weights -- minus the tower and a
    packed MTP head (neither is split: rank 0 alone holds them,
    `leader_bytes`). Cached on each shard's
    (size, mtime_ns, ctime_ns)."""
    import json
    import struct
    from pathlib import Path

    files = sorted(f for f in Path(path).glob("*.safetensors")
                   if not f.name.startswith(("mtp", "model-vision")))
    try:
        stamp = tuple((f.name, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                      for f in files for st in (f.stat(),))
    except OSError:
        stamp = None
    hit = _HEADERS.get(str(path))
    if stamp is not None and hit and hit[0] == stamp:
        return hit[1]
    out = {}
    for f in files:
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
            a, b = v.get("data_offsets", (0, 0))
            out[k] = (tuple(v.get("shape") or ()), int(b) - int(a))
    if stamp is not None:
        _HEADERS[str(path)] = (stamp, out)
    return out


def tensor_placement(artifact: Artifact, n: int) -> dict:
    """tensor_placement_of over the artifact's trunk headers."""
    return tensor_placement_of(
        {k: b for k, (_, b) in trunk_headers(artifact.path).items()}, n)


def tensor_header_refusals(path, cfg: dict, n: int) -> list:
    """Why the arrays on disk cannot be cut `n` ways by the split's own
    rules (engine/split/tensor_rules), each with its numbers."""
    from knurlogic.engine.split.tensor_rules import refusals, skipzero_split
    if n < 2:
        return []
    tc = cfg.get("text_config", cfg)
    kd = (tc.get("linear_num_key_heads") or 0) * \
        (tc.get("linear_key_head_dim") or 0)
    shapes = {k: s for k, (s, _) in trunk_headers(path).items()}
    sz = skipzero_split(path)
    out = []
    if sz:
        # the runtime cuts these by output row, per expert: the rows must
        # divide (it refuses at load; said here before any rank starts)
        for p, m in sorted(((cfg.get("vq_skipzero") or {}).get("modules")
                            or {}).items()):
            OUT = int(m.get("out") or ((cfg.get("vq_modules") or {})
                                         .get(p) or {}).get("out") or 0)
            if p.endswith(("gate_proj", "up_proj")) and OUT % n:
                out.append(f"{p}: {OUT} output rows do not divide by {n}")
    return out + refusals(shapes, n, kd, int(tc.get("num_key_value_heads")
                                            or 0), sz)


def tensor_unverified(path) -> dict:
    """{(rule path, unknown parameters): a layer holding them} -- the
    modules whose layout no split rule knows, for a launch to run
    (engine/split/viability)."""
    from knurlogic.engine.split.tensor_rules import skipzero_split, unverified
    return unverified({k: s for k, (s, _) in trunk_headers(path).items()},
                      skipzero_split(path))


def tensor_split_refusals(path, n: int, cfg: dict | None = None) -> list:
    """Everything the config and the headers say against splitting the
    artifact at `path` `n` ways; [] when nothing does."""
    import json
    from pathlib import Path
    if cfg is None:
        cfg = json.loads((Path(path) / "config.json").read_text())
    why = tensor_refusals(cfg, n)
    if why or n < 2:
        return why
    return tensor_header_refusals(path, cfg, n)
