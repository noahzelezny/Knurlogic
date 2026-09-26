"""Drafting: an artifact's MTP head, bound to the loaded model.

The batch engine (engine/mtp/batch_generator.MTPBatchGenerator) drafts
with it; the scheduler hands it the head when one is bound.
"""

from __future__ import annotations

from . import state


# An artifact that ships a multi-token-prediction head carries weights mlx-lm
# will never run: it has no MTP path at all, and neither does upstream exo.
# knurlogic does, so a head beside the weights is simply used -- no flag, no
# environment variable, no mode file.



def drafting_status() -> dict:
    """What drafting is doing, for `/status.json` and the startup line."""
    d = dict(state.DRAFT)
    d.pop("head", None)
    spec = d.pop("spec", None)
    d.pop("batch_installed", None)
    d["family"] = getattr(spec, "name", "")
    d["acceptance"] = (d["accepted"] / d["steps"]) if d["steps"] else None
    d["drafts_now"] = bool(d["on"])
    return d


def load_head(model_path: str):
    """Load the drafting head beside this artifact, if there is one.

    Absent is the ordinary case and not an error: sidecars are named outside
    mlx-lm's `model*.safetensors` glob precisely so a directory carrying one
    still loads normally through the stock loader.
    """
    from knurlogic.engine.mtp import find_head

    found = find_head(model_path)
    if found is None:
        state.DRAFT.update(on=False, why="no drafting head beside the weights")
        return None
    prov = state.SERVED.get("provider")
    model = getattr(prov, "model", None) if prov else None
    if model is None:
        state.DRAFT.update(on=False, why="model not loaded yet")
        return None
    try:
        from knurlogic.engine.mtp.registry import load_head
        head, spec = load_head(model, sidecar=found.path)
    except Exception as e:
        # A head that will not bind is a fact worth printing, not a crash:
        # the model serves perfectly well without one.
        state.DRAFT.update(on=False, why=f"{type(e).__name__}: {e}")
        return None
    state.DRAFT.update(head=head, spec=spec, on=True,
                  why=f"{found.path.name}, {found.gib:.2f} GiB")
    return head
