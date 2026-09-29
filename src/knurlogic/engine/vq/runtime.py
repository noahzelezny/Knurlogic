"""knurlogic's own VQ runtime: the vendored kernels plus the attach step.

WHAT A PUBLISHED BUNDLE IS. Every released rung's `model.py` is three texts
concatenated by vqlab's bundlers: `vq_switch.py` (MoE expert + PLE kernels),
`vq_dense.py` on the dense rungs (VQLinear / VQEmbedding), and a loader shim
that builds the registry architecture and swaps each VQ-coded module for its
drop-in before weights load. The rungs differ from each other (and from vqlab
HEAD) in the DEFAULTS of a few env flags baked into that text -- measured
against the Hub on 2026-09-23: 27B == HEAD byte for byte; Flash-Next 2.1 ==
HEAD but for three default lines; see docs/design/vq-rung-knobs.md.

So this module reproduces a bundle without its text: the two runtime files
are vendored VERBATIM (PROVENANCE.md pins commit and digest), executed into
one fresh namespace per knob set -- concatenated, exactly as a bundle is, so
vq_dense finds vq_switch's kernels in its own globals the way it does inside
a model.py -- with the rung's knobs in the environment while the flags are
read. The shim's job (attach) is done by `model_classes` below.

WHY A FRESH NAMESPACE PER KNOB SET, NOT ONE IMPORT. The flags are module
globals read ONCE at import (`_GEMMSEG_BF16IO = os.environ.get(...)` and
~30 more). One shared import would freeze the first rung's numerics into
every rung loaded after it in the process -- the v1.5-overrides-v2 bug of
design D1 again, moved from the resolver into the import system.

ENV PRECEDENCE: a flag already set in the process environment wins over the
rung's knob. The resolver emits the rung's own values by default, so they
agree; a value differing from the rung's is a person asking (a runtime
profile, a debugging override), and the person wins.

Text only. The vision path is the family packages' (P1-P3): they build the
multimodal model and call `attach_vq` on its language model.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import threading
import types
from contextlib import contextmanager
from pathlib import Path

from . import rungs as _rungs

HERE = Path(__file__).parent
#: Vendored from vqlab at this commit (HEAD 42df84f; last functional change
#: to vq_switch.py ef4e8dc). The digests are the pin: tests hold the files to
#: them, so an edit here is a visible re-vendor, not drift.
VQLAB_COMMIT = "42df84f"
RUNTIME_FILES = {
    "vq_switch.py":
        "40870875499de7cb937be412dd63f9e5cc40e0d58d5adcf4c595661d2b21b67b",
    "vq_dense.py":
        "ac62e4e43f4accefeb6761303d16acc3d60f154561ddab3e24ceb3850e89a235",
}

_lock = threading.Lock()
_modules: dict = {}


def source() -> str:
    """The runtime text as a dense bundle carries it: vq_switch + vq_dense.
    MoE bundles carry vq_switch alone; the extra classes are inert there."""
    return "\n\n".join((HERE / f).read_text() for f in RUNTIME_FILES)


def head_defaults() -> dict:
    """Every flag default the vendored runtime bakes in."""
    return _rungs.flag_defaults(source())


@contextmanager
def _env_overlay(knobs: dict):
    """Set each knob not already in the environment; restore after. Under a
    lock: os.environ is process-global and two loads must not interleave."""
    added = [k for k in knobs if k not in os.environ]
    for k in added:
        os.environ[k] = str(knobs[k])
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


def effective_flags(knobs: dict) -> dict:
    """What each flag will read as, for this knob set in this environment."""
    out = head_defaults()
    out.update({k: str(v) for k, v in knobs.items()})
    out.update({k: os.environ[k] for k in out if k in os.environ})
    return out


def runtime_module(knobs: dict | None = None) -> types.ModuleType:
    """The runtime, executed with these knobs. One module per distinct set
    of EFFECTIVE flag values, cached: the kernels compile once per process
    per numerics, not once per load."""
    knobs = dict(knobs or {})
    with _lock:
        # read under the lock: _env_overlay below sets os.environ
        # temporarily, and effective_flags reads it
        eff = effective_flags(knobs)
        key = hashlib.sha256(
            repr(sorted(eff.items())).encode()).hexdigest()[:16]
        mod = _modules.get(key)
        if mod is not None:
            return mod
        name = f"knurlogic.engine.vq._rt_{key}"
        mod = types.ModuleType(name)
        mod.__file__ = str(HERE / "vq_switch.py")
        code = compile(source(), f"<knurlogic vq runtime {VQLAB_COMMIT}>",
                       "exec")
        with _env_overlay(knobs):
            exec(code, mod.__dict__)
        mod.KNURLOGIC_FLAGS = eff
        _modules[key] = mod
        return mod


# --- the attach step (the bundles' loader shim, as code) --------------------

def _reach_vq(root, path):
    """(owner, leaf) for a config path, tolerating both module layouts.

    Same candidates as the bundles' `_reach_vq` (vqlab arch_resolve.py):
    as written, with `language_model.` added or removed, and PLE `shard_N`
    respelled `shards.N`. A VQ module that silently fails to attach leaves a
    random-init dense layer in the graph that loads clean and generates
    plausible garbage -- so a miss raises, naming every candidate."""
    import re
    cands = [path]
    if path.startswith("language_model."):
        cands.append(path[len("language_model."):])
    else:
        cands.append("language_model." + path)
    shard = re.compile(r"\.ngram_embedding\.shard_(\d+)$")
    for c in list(cands):
        if shard.search(c):
            cands.append(shard.sub(r".ngram_embedding.shards.\1", c))
    for cand in cands:
        obj, parts, ok = root, cand.split("."), True
        for c in parts[:-1]:
            try:
                obj = obj[int(c)] if c.isdigit() else getattr(obj, c)
            except (AttributeError, IndexError, KeyError, TypeError):
                ok = False
                break
        leaf = parts[-1]
        if ok and leaf.isdigit() and isinstance(obj, (list, tuple)) \
                and int(leaf) < len(obj):
            return obj, leaf
        if ok and hasattr(obj, leaf):
            return obj, leaf
    raise AttributeError(
        f"vq module {path!r} does not resolve on {type(root).__name__}; "
        f"tried {cands}")


def _put(owner, leaf, module):
    if leaf.isdigit() and isinstance(owner, list):
        owner[int(leaf)] = module
    else:
        setattr(owner, leaf, module)


def _dtype(cfg: dict):
    """The stack's compute dtype, as the dense shim reads it: VQEmbedding
    decodes in fp16 and casts to this, because fp16 into a bf16 model
    promotes everything downstream to fp32."""
    import mlx.core as mx
    name = (cfg.get("dtype") or cfg.get("torch_dtype")
            or cfg.get("text_config", {}).get("dtype") or "bfloat16")
    return {"bfloat16": mx.bfloat16, "float16": mx.float16,
            "float32": mx.float32}[name]


def attach_vq(model, cfg: dict, rt: types.ModuleType) -> int:
    """Swap every VQ-coded module the config names for its drop-in, with
    zero tensors of the shipped shapes (weights load over them). Returns how
    many were attached. Shapes follow each bundle shim exactly -- including
    that the MoE shim rounds a packed row UP to whole 32-code words and the
    dense shim does not; they are different packers."""
    import mlx.core as mx
    n = 0
    for path, m in (cfg.get("vq_modules") or {}).items():
        owner, leaf = _reach_vq(model, path)
        pb = m.get("pack_bits", 0)
        if pb:
            ncol = (m["in"] // m["dim"] + 31) // 32 * pb
            ct = mx.uint32
        else:
            ncol = m["in"] // m["dim"]
            ct = mx.uint8 if m["k"] <= 256 else mx.uint16
        _put(owner, leaf, rt.VQSwitchLinear(
            mx.zeros((m["experts"], m["out"], ncol), dtype=ct),
            mx.zeros((m["k"], m["dim"]), dtype=mx.float16),
            mx.zeros((m["experts"], m["out"], m["in"] // m["group"]),
                     dtype=mx.float16),
            group_size=m["group"], pack_bits=pb,
            in_features=m["in"] if pb else None))
        n += 1
    ple = cfg.get("vq_ple")
    if ple:
        g = ple["geometry"]
        for key in ple["keys"]:
            rows, cols = ple["shapes"][key]
            owner, leaf = _reach_vq(model, key)
            rb = g.get("row_bytes")
            codes0 = (mx.zeros((rows, rb), dtype=mx.uint8) if rb else
                      mx.zeros((rows, cols // g["dim"]), dtype=mx.uint16))
            _put(owner, leaf, rt.VQPLEEmbedding(
                codes0,
                mx.zeros((g["k"], g["dim"]), dtype=mx.float16),
                mx.zeros((rows, cols // g["group"]), dtype=mx.float16),
                group_size=g["group"],
                packed_nsub=(cols // g["dim"]) if rb else 0))
            n += 1
    for path, m in (cfg.get("vq_linear") or {}).items():
        owner, leaf = _reach_vq(model, path)
        pb = m.get("pack_bits", 0)
        ct = mx.uint32 if pb else (mx.uint8 if m["k"] <= 256 else mx.uint16)
        cols = (m["in"] // m["dim"] // 32 * pb) if pb else m["in"] // m["dim"]
        _put(owner, leaf, rt.VQLinear(
            mx.zeros((m["out"], cols), dtype=ct),
            mx.zeros((m["k"], m["dim"]), dtype=mx.float16),
            mx.zeros((m["out"], m["in"] // m["group"]), dtype=mx.float16),
            group_size=m["group"], pack_bits=pb,
            in_features=m["in"] if pb else None))
        n += 1
    for path, m in (cfg.get("vq_embed") or {}).items():
        owner, leaf = _reach_vq(model, path)
        pb = m.get("pack_bits", 0)
        ct = mx.uint32 if pb else (mx.uint8 if m["k"] <= 256 else mx.uint16)
        cols = (m["in"] // m["dim"] // 32 * pb) if pb else m["in"] // m["dim"]
        _put(owner, leaf, rt.VQEmbedding(
            mx.zeros((m["rows"], cols), dtype=ct),
            mx.zeros((m["k"], m["dim"]), dtype=mx.float16),
            mx.zeros((m["rows"], m["in"] // m["group"]), dtype=mx.float16),
            group_size=m["group"], pack_bits=pb,
            in_features=m["in"] if pb else None, out_dtype=_dtype(cfg)))
        n += 1
    return n


def model_classes(cfg: dict, knobs: dict | None = None):
    """(Model, ModelArgs) for mlx-lm's `load_model(get_model_classes=...)`.

    The base is `mlx_lm.models.<model_type>` -- through the import system,
    so knurlogic's registered architectures win exactly as they do for a
    bundle (which imports the same name). Text only, mlx-lm's tree: the
    bundles also take mlx-lm's arch when mlx-lm is the loader."""
    rt = runtime_module(knobs)
    base = importlib.import_module(f"mlx_lm.models.{cfg['model_type']}")
    args_cls = getattr(base, "ModelArgs", None)

    if args_cls is not None:
        class Model(base.Model):
            def __init__(self, args):
                super().__init__(args)
                self._vq_attached = attach_vq(self, cfg, rt)
    else:
        # An mlx-vlm-shaped arch (glm5_next): a ModelConfig that builds its
        # nested configs itself, and a __call__ that takes input_ids and
        # returns an output object. mlx-lm's loader and server call
        # model(inputs, cache=..., input_embeddings=...) and read logits, so
        # the call goes straight to the language model -- images come in
        # as input_embeddings from the vision family, never as pixels.
        class args_cls:
            """mlx-lm calls ModelArgs.from_dict(config). mlx-vlm's loader
            builds the nested configs itself (utils.update_module_configs:
            `<name>_config` -> `<Name>Config.from_dict`), and 0.6.17's
            ModelConfig -- the one the GLM rungs are built on -- does not,
            so it is done here, the same way. Generation defaults, the
            other step of mlx-vlm's loader, set sampling fields only."""

            @staticmethod
            def from_dict(config):
                mc = base.ModelConfig.from_dict(config)
                for name in ("text", "vision", "perceiver", "projector",
                             "audio"):
                    cls = getattr(base, f"{name.title()}Config", None)
                    sub = config.get(f"{name}_config")
                    if cls is not None and isinstance(sub, dict) and \
                            hasattr(mc, f"{name}_config"):
                        setattr(mc, f"{name}_config", cls.from_dict(sub))
                return mc

        class Model(base.Model):
            def __init__(self, args):
                super().__init__(args)
                self._vq_attached = attach_vq(self, cfg, rt)

            def __call__(self, inputs, cache=None, input_embeddings=None,
                         **kwargs):
                out = self.language_model(inputs, cache=cache,
                                          inputs_embeds=input_embeddings,
                                          **kwargs)
                return out.logits

    Model.__qualname__ = Model.__name__ = f"KnurlogicVQ_{base.Model.__name__}"
    Model.__module__ = rt.__name__
    return Model, args_cls


def serves(path) -> bool:
    """Does knurlogic's runtime serve this artifact? Only a rung listed as
    VERIFIED in rungs.json (G-VQ passed against its published bundle).
    Every other artifact -- unlisted, or listed and not yet gated -- loads
    the model.py it ships, unchanged."""
    return _rungs.verified(path)


def load_model(path, knobs: dict | None = None, lazy: bool = False,
               strict: bool = True, model_config: dict | None = None):
    """mlx-lm's own `load_model`, with the bundle's `model_file` switched
    off and our classes in its place. `knobs` defaults to the rung's
    (rungs.json); pass {} to run HEAD's defaults."""
    from mlx_lm.utils import load_model as _load_model
    p = Path(path)
    if knobs is None:
        knobs = _rungs.knobs(p)

    def _classes(config):
        return model_classes(config, knobs)

    # config.update(model_config) runs before mlx-lm checks model_file, so
    # None here routes it to get_model_classes (mlx-lm 0.31.3 utils.py
    # load_model; the pin test guards that file's version).
    return _load_model(p, lazy=lazy, strict=strict,
                       model_config={**(model_config or {}),
                                     "model_file": None},
                       get_model_classes=_classes)


def load(path, knobs: dict | None = None):
    """(model, tokenizer), as `mlx_lm.utils.load` returns them."""
    from mlx_lm.utils import load_tokenizer
    model, config = load_model(path, knobs)
    tok = load_tokenizer(Path(path),
                         eos_token_ids=config.get("eos_token_id", None))
    from knurlogic.engine import templates
    templates.install(tok)
    return model, tok
