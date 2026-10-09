"""The context a model can run: the window it was trained for, running
past it with YaRN where the model card documents it, and whether this box
has room for the KV of a long context.
"""

from __future__ import annotations

from typing import Any

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import fit


# --- the longest context a model was built for -------------------------------
# max_position_embeddings is the trained window. A config that declares YaRN
# rope scaling was built to run past it: the window is then
# original_max_position_embeddings * factor. Anything else -- rope_type
# "default", no scaling at all -- stops at max_position_embeddings. The
# engine does not refuse a position past it (rope is computed for any
# position), it just runs a model past the length it was trained on, which
# is quietly worse output rather than an error. So the cap is refused above
# this, not honoured.
def model_window(cfg: dict) -> tuple:
    """(tokens, why) for an artifact's config.json; (0, why) when it does
    not say."""
    cfg = cfg or {}
    text: Any = cfg.get("text_config") if isinstance(
        cfg.get("text_config"), dict) else {}
    mpe = int(text.get("max_position_embeddings")
              or cfg.get("max_position_embeddings") or 0)
    rope: Any = None
    for c in (text, cfg):
        for key in ("rope_scaling", "rope_parameters"):
            if isinstance(c.get(key), dict):
                rope = c[key]
                break
        if rope:
            break
    kind = str((rope or {}).get("rope_type") or (rope or {}).get("type")
               or "").lower()
    if kind == "yarn" and rope.get("factor"):
        orig = int(rope.get("original_max_position_embeddings") or mpe or 0)
        yarn = int(orig * float(rope["factor"]))
        if yarn > mpe:
            return yarn, (f"{orig:,} trained x YaRN factor "
                          f"{float(rope['factor']):g} (rope_scaling)")
    if mpe:
        return mpe, ("max_position_embeddings; no YaRN rope scaling in its "
                     "config, so nothing longer was trained")
    return 0, "its config does not say (no max_position_embeddings)"


# --- running past the trained window: YaRN, where the model card says so ----
# Qwen's model cards for the hybrid families knurlogic serves document one
# recipe for ~1M tokens: rope_parameters gains rope_type "yarn", factor 4.0,
# original_max_position_embeddings 262144 (mrope/partial rotary unchanged),
# and vLLM/sglang raise their max length (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# --max-model-len 1010000; SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# --context-length 1010000). Static YaRN: "potentially impacting performance
# on shorter texts". Sources:
#   https://huggingface.co/Qwen/Qwen3.5-397B-A17B   (1,010,000)
#   https://huggingface.co/Qwen/Qwen3.6-35B-A3B     (1,010,000)
#   https://huggingface.co/Qwen/Qwen3.8-27B         (1,000,000)
#   https://huggingface.co/Qwen/Qwen3.8-Flash-Next  (1,000,000)
# The cap here is the YaRN window itself (262144 x 4 = 1,048,576, what
# model_window reads off a yarn config); Qwen's tested lengths are above.
LONG_CONTEXT_VALUES = ("off", "yarn")
#: model_type -> (factor, original_max_position_embeddings, documented tokens)
LONG_CONTEXT_YARN = {
    "qwen3_5": (4.0, 262144, 1_010_000),
    "qwen3_5_moe": (4.0, 262144, 1_010_000),
    "qwen4_exp": (4.0, 262144, 1_000_000),
}


def long_context_family(model_type: str) -> str | None:
    """The LONG_CONTEXT_YARN key for a model_type (a text config's
    `qwen3_5_text` is its wrapper's), or None where no card documents it."""
    mt = str(model_type or "")
    mt = mt[:-5] if mt.endswith("_text") else mt
    return mt if mt in LONG_CONTEXT_YARN else None


def long_context_of(v) -> str:
    """'off' | 'yarn'; ''/None is off."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "off", "0", "false", "no", "none"):
        return "off"
    if s not in LONG_CONTEXT_VALUES:
        raise ValueError(f"Long context: {v!r} isn't "
                         f"{' or '.join(LONG_CONTEXT_VALUES)}")
    return s


def long_context_refusal(model_type: str, mode) -> str | None:
    """Why `mode` cannot be taken for this family, or None."""
    if long_context_of(mode) == "off":
        return None
    if long_context_family(model_type) is None:
        return (f"KNURLOGIC_LONG_CONTEXT=yarn is refused for "
                f"{model_type or 'this model'}: only the Qwen families "
                f"whose model cards document YaRN take it "
                f"({', '.join(sorted(LONG_CONTEXT_YARN))})")
    return None


def context_ceiling(model_type: str, cfg: dict) -> int:
    """The most context a launch of this model can ask for: its YaRN
    window where the family documents YaRN, else its native window (0 when
    the config does not say)."""
    native, _ = model_window(cfg or {})
    if long_context_family(model_type):
        top, _ = model_window(with_long_context(
            {"model_type": model_type, **(cfg or {})}, "yarn"))
        return max(top, native)
    return native


def settle_context(model_type: str, cfg: dict, sets: dict) -> tuple:
    """(sets, notes): a launch's KNURLOGIC_CONTEXT_LENGTH made one the model
    can take, never a refusal -- a saved per-model value must not brick a
    launch.

    Asking for more than the model's native window IS asking for long
    context: where the family's card documents YaRN, KNURLOGIC_LONG_CONTEXT
    is turned on for the launch (its KV-room check still applies), and a
    value past even the YaRN window is lowered to it. A family without YaRN
    has the value lowered to its native window. Each change is one note."""
    out = dict(sets or {})
    raw = out.get("KNURLOGIC_CONTEXT_LENGTH")
    try:
        want = int(str(raw).strip())
    except (TypeError, ValueError):
        return out, []            # check_knob says what is wrong with it
    native, _ = model_window(cfg or {})
    if not native or want <= native:
        return out, []
    notes = []
    try:
        mode = long_context_of(out.get("KNURLOGIC_LONG_CONTEXT"))
    except ValueError:
        return out, []            # refused with its own reason
    if long_context_family(model_type):
        if mode == "off":
            out["KNURLOGIC_LONG_CONTEXT"] = "yarn"
            notes.append(f"context {want:,} is past the native "
                         f"{native:,}: long context (YaRN) is on for this "
                         f"launch")
        top, _ = model_window(with_long_context(
            {"model_type": model_type, **(cfg or {})}, "yarn"))
        if top and want > top:
            out["KNURLOGIC_CONTEXT_LENGTH"] = str(top)
            notes.append(f"context {want:,} lowered to {top:,}, the most "
                         f"long context (YaRN) reaches")
        return out, notes
    out["KNURLOGIC_CONTEXT_LENGTH"] = str(native)
    notes.append(f"context {want:,} lowered to {native:,}, this model's "
                 f"maximum ({model_type or 'this family'} has no documented "
                 f"long context)")
    return out, notes


def long_context_config(cfg: dict, mode) -> dict:
    """The top-level config keys to overlay at load (mlx-lm's
    `model_config`, a shallow update) for `mode`: {} when off, else the
    text config with rope_parameters carrying Qwen's YaRN. The artifact's
    config.json is never written. Raises ValueError where refused."""
    mode = long_context_of(mode)
    if mode == "off":
        return {}
    cfg = cfg or {}
    mt = str(cfg.get("model_type") or "")
    why = long_context_refusal(mt, mode)
    if why:
        raise ValueError(why)
    factor, orig, _doc = LONG_CONTEXT_YARN[long_context_family(mt) or ""]
    nested = isinstance(cfg.get("text_config"), dict)
    tc = dict(cfg["text_config"]) if nested else dict(cfg)
    key = "rope_parameters" if isinstance(tc.get("rope_parameters"), dict) \
        or not isinstance(tc.get("rope_scaling"), dict) else "rope_scaling"
    rp = dict(tc.get(key) or {})
    rp.pop("type", None)
    rp.update(rope_type="yarn", factor=factor,
              original_max_position_embeddings=orig)
    tc[key] = rp
    if nested:
        return {"text_config": tc}
    return {key: rp}


def with_long_context(cfg: dict, mode) -> dict:
    """`cfg` as the model will load under `mode` (a copy)."""
    out = dict(cfg or {})
    out.update(long_context_config(cfg, mode))
    return out


def long_context_room(artifact: Artifact, working_set_bytes: int,
                      holds_bytes: int, context: int, kv_bits=None):
    """None when this box can hold the KV of `context` tokens beside what
    it holds of the model and the step margin, else the sentence saying
    how short it is and what would fit."""
    tc = (artifact.raw_config or {}).get("text_config") \
        or artifact.raw_config or {}
    per, why = fit.kv_bytes_per_token(tc, kv_bits)
    if not per or not context:
        return None
    need = per * int(context)
    left = max(int(working_set_bytes) - int(holds_bytes)
               - fit.step_margin(working_set_bytes), 0)
    if need <= left:
        return None
    fits = left // per
    half = ("" if kv_bits is not None else
            f"; 8-bit KV (KNURLOGIC_KV_BITS=8) needs "
            f"{fit.kv_bytes_per_token(tc, 8)[0] * int(context) / fit.GIB:.1f} GiB")
    return (f"{int(context):,} tokens of context need {need / fit.GIB:.1f} GiB "
            f"of KV ({per:,} bytes per token: {why}) and this box leaves "
            f"{left / fit.GIB:.1f} GiB after the model and the step margin -- "
            f"about {fits:,} tokens. Lower KNURLOGIC_CONTEXT_LENGTH{half}")
