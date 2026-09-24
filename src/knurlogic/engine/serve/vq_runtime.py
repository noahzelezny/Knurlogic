"""Verified rungs load on knurlogic's own VQ runtime (engine/vq).

A rung rungs.json lists as VERIFIED -- tools/vq_gate.py proved it
bit-identical to its published model.py -- loads through
`engine.vq.runtime`; every other artifact loads exactly as before, bundled
model.py and all.
"""

from __future__ import annotations

from . import state


def install(srv) -> None:
    """Route the server's `load` through knurlogic's VQ runtime for a rung
    rungs.json lists as VERIFIED (G-VQ: bit-identical to its published
    model.py). Every other artifact loads exactly as before, bundled
    model.py and all. The server imported `load` by name, so the name in
    its module is what gets swapped."""
    if getattr(srv.load, "_knurlogic", False):
        return
    real = srv.load

    def load(path_or_hf_repo, tokenizer_config=None, model_config=None,
             adapter_path=None, lazy=False, return_config=False, **kw):
        from pathlib import Path
        from knurlogic.engine.vq import runtime
        p = Path(str(path_or_hf_repo))
        if (adapter_path is None and not model_config and p.is_dir()
                and runtime.serves(p)):
            from mlx_lm.utils import load_tokenizer
            model, config = runtime.load_model(p, lazy=lazy)
            tok = load_tokenizer(p, tokenizer_config,
                                 eos_token_ids=config.get("eos_token_id"))
            state.SERVED["runtime"] = "knurlogic"
            return (model, tok, config) if return_config else (model, tok)
        state.SERVED["runtime"] = "bundled"
        return real(path_or_hf_repo, tokenizer_config=tokenizer_config,
                    model_config=model_config, adapter_path=adapter_path,
                    lazy=lazy, return_config=return_config, **kw)

    load._knurlogic = True
    srv.load = load
