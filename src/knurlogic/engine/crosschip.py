"""Identical results across chips (KNURLOGIC_CROSS_CHIP).

mlx 0.31.2's `mx.quantized_matmul` switches from its matrix-vector kernel
(qmv) to its matrix-matrix kernel (qmm) at a row count that depends on the
GPU architecture (applegpu_g15d vs applegpu_g16s), so a 10-31 row forward
rounds differently on M3- and M4-generation chips. A call with 9-31 rows
against a 2-D weight is therefore zero-padded to 32 rows, run (qmm on every
chip) and sliced back; 1-8 and >=32 rows are untouched.

Off by default (rank 0 samples every token, so rounding cannot desync a
cluster). `on` forces it; `auto` turns it on for a cluster job whose
machines have different GPU architectures. mlx is imported only by
`install`. Design: docs/design/server.md (cross-chip rounding).
"""
from __future__ import annotations

#: rows in [LO, HI) are padded to HI
LO, HI = 9, 32
VALUES = ("auto", "on", "off")

_STATE = {"orig": None, "installed": False}


def parse(v) -> str:
    """'auto' | 'on' | 'off' ('' / None -> off; 1/0, true/false too)."""
    s = str(v if v is not None else "").strip().lower()
    if s == "auto":
        return "auto"
    if s in ("on", "1", "true", "yes"):
        return "on"
    if s in ("", "off", "0", "false", "no"):
        return "off"
    raise ValueError(f"KNURLOGIC_CROSS_CHIP {v!r}: one of {list(VALUES)}")


def resolve(setting, chips=None) -> dict:
    """{on, setting, chips, why}. `chips`: one {name, arch} per machine of
    the job (rank order), or None/one for a single machine. Machines differ
    by GPU architecture; one that did not report it is compared by name."""
    s = parse(setting)
    chips = [c for c in (chips or []) if isinstance(c, dict)]
    names = [str(c.get("name") or c.get("arch") or "?") for c in chips]
    keys = {str(c.get("arch") or c.get("name") or "") for c in chips} - {""}
    mixed = len(chips) > 1 and len(keys) > 1
    label = " + ".join(dict.fromkeys(names))
    if s == "on":
        on, why = True, "forced on"
    elif s == "off":
        on, why = False, "off"
    elif mixed:
        on, why = True, f"auto: machines differ ({label})"
    else:
        on, why = False, ("auto: one GPU architecture" if len(chips) > 1
                          else "auto: one machine")
    return {"on": on, "setting": s, "chips": names, "why": why,
            "label": label}


def describe(r: dict) -> str:
    """'on (M3 Ultra + M4 Max)' / 'off (auto: one machine)'."""
    if r["on"] and r.get("label"):
        return f"on ({r['label']})"
    return f"{'on' if r['on'] else 'off'} ({r['why']})"


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

    quantized_matmul._knurlogic_cross_chip = True
    return quantized_matmul


def install() -> bool:
    """Wrap mx.quantized_matmul, once per process. True if it is wrapped."""
    import mlx.core as mx
    cur = mx.quantized_matmul
    if getattr(cur, "_knurlogic_cross_chip", False):
        _STATE["installed"] = True
        return True
    _STATE["orig"] = cur
    mx.quantized_matmul = padded(cur)
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    """Put the original back (tests)."""
    import mlx.core as mx
    if _STATE["orig"] is not None:
        mx.quantized_matmul = _STATE["orig"]
    _STATE.update(orig=None, installed=False)


def installed() -> bool:
    return bool(_STATE["installed"])


def gpu_architecture() -> str:
    """This machine's GPU architecture ('applegpu_g15d'), '' if unknown."""
    try:
        import mlx.core as mx
        info = (mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())
        return str(info.get("architecture") or "")
    except Exception:
        return ""
