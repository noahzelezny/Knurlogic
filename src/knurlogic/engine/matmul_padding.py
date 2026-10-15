"""Quantized-matmul row padding: the same rounding on every chip, for
per-chip rounding off (KNURLOGIC_PER_CHIP_ROUNDING).

mlx 0.31.2's `mx.quantized_matmul` switches from its matrix-vector kernel
(qmv) to its matrix-matrix kernel (qmm) at a row count that depends on the
GPU architecture (applegpu_g15d vs applegpu_g16s), so a 10-31 row forward
rounds differently on M3- and M4-generation chips. A call with 9-31 rows
against a 2-D weight is therefore zero-padded to 32 rows, run (qmm on every
chip) and sliced back; 1-8 and >=32 rows are untouched.

Per-chip rounding is on by default: each chip runs its own fastest kernel
and nothing is padded (rank 0 samples every token, so rounding cannot
desync a cluster). `off` pads on every machine, so a split across chip
generations gives the same tokens as one over a single generation. mlx
is imported only when the wrap is installed or the chip is asked.
Design: docs/design/server.md (per-chip rounding).
"""
from __future__ import annotations

#: rows in [LO, HI) are padded to HI
LO, HI = 9, 32
#: what KNURLOGIC_PER_CHIP_ROUNDING takes
VALUES = ("on", "off")
#: the prompt cache key's own words for a setting (the `setting` of
#: resolve's dict): they say whether to pad, so rounding off is "on"
KEY_SETTING: dict[object, str] = {"on": "off", "off": "on"}

_STATE = {"installed": False}


def parse(v) -> str:
    """'on' | 'off' ('' / None -> on; 1/0, true/false too)."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "on", "1", "true", "yes"):
        return "on"
    if s in ("off", "0", "false", "no"):
        return "off"
    raise ValueError(f"KNURLOGIC_PER_CHIP_ROUNDING {v!r}: one of "
                     f"{list(VALUES)}")


def resolve(setting, chips=None) -> dict:
    """{on, setting, chips, why, label} for per-chip rounding `setting`.
    `chips`: one {name, arch} per machine of the job (rank order), or
    None/one for a single machine. Machines differ by GPU architecture; one
    that did not report it is compared by name.

    The dict is the prompt cache key's own (engine/prompt_cache:
    disk.identity and layers.rounding_of hash it into key ids, so its shape
    and values never change or every saved entry is orphaned): `on` is
    whether calls are padded (per-chip rounding OFF) and `setting` is in
    the key's words (KEY_SETTING). `describe` and `public` give it in the
    setting's words."""
    s = KEY_SETTING[parse(setting)]
    chips = [c for c in (chips or []) if isinstance(c, dict)]
    names = [str(c.get("name") or c.get("arch") or "?") for c in chips]
    label = " + ".join(dict.fromkeys(names))
    if s == "on":
        on, why = True, "forced on"
    else:
        on, why = False, "off"
    return {"on": on, "setting": s, "chips": names, "why": why,
            "label": label}


def public(r: dict) -> dict:
    """A resolution in per-chip rounding's words, for status and the page:
    {per_chip_rounding: on|off, setting, padded, chips, why} (plus
    `installed` when the host recorded it)."""
    setting = KEY_SETTING.get(r.get("setting"), r.get("setting"))
    padded_ = bool(r.get("on"))
    why = "set " + str(setting)
    out = {"per_chip_rounding": "off" if padded_ else "on",
           "setting": setting, "padded": padded_,
           "chips": list(r.get("chips") or []), "why": why}
    if "installed" in r:
        out["installed"] = r["installed"]
    return out


def describe(r: dict) -> str:
    """'off (M3 Ultra + M4 Max)' / 'on (set on)': per-chip
    rounding off means the same rounding on every chip."""
    p = public(r)
    if p["padded"] and r.get("label"):
        return f"off ({r['label']})"
    return f"{p['per_chip_rounding']} ({p['why']})"


def padded(orig):
    """The wrapper around `orig` (mx.quantized_matmul)."""
    import mlx.core as mx

    def quantized_matmul(x, w, scales=None, biases=None, transpose=True,
                         group_size=None, bits=None, mode="affine", **kw):
        m = x.size // x.shape[-1] if x.ndim >= 2 else 1
        if transpose and w.ndim == 2 and LO <= m < HI:
            x2 = x.reshape(m, x.shape[-1])
            x2 = mx.concatenate(
                [x2, mx.zeros((HI - m, x.shape[-1]), x.dtype)])
            y = orig(x2, w, scales, biases, transpose=transpose,
                     group_size=group_size, bits=bits, mode=mode, **kw)[:m]
            return y.reshape(*x.shape[:-1], y.shape[-1])
        return orig(x, w, scales, biases, transpose=transpose,
                    group_size=group_size, bits=bits, mode=mode, **kw)

    quantized_matmul._knurlogic_padded = True
    return quantized_matmul


def install() -> bool:
    """Wrap mx.quantized_matmul, once per process. True if it is wrapped."""
    import mlx.core as mx
    cur = mx.quantized_matmul
    if getattr(cur, "_knurlogic_padded", False):
        _STATE["installed"] = True
        return True
    mx.quantized_matmul = padded(cur)
    _STATE["installed"] = True
    return True


def installed() -> bool:
    return bool(_STATE["installed"])


def gpu_architecture() -> str:
    """This machine's GPU architecture ('applegpu_g15d'), '' if unknown."""
    try:
        from knurlogic.engine import backend
        info = backend.device_info()
        return str(info.get("architecture") or "")
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return ""
