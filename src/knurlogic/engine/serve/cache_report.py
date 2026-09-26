"""What the prompt cache actually did for one request, in its usage.

The trie OFFERS a prefix; the batch engine can still discard it (a
drafting row whose head is not aligned with it prefills from scratch), so
only the engine knows what was used. It writes the report onto the
request object admission names (engine/runtime/executor.Admission.report),
and the request's text stage puts it in usage:

    usage.knurlogic.cache = {offered, used, discarded, prefilled, via,
                             images: {total, in_cached_span, prefilled,
                                      encoded},
                             checkpoints_stored}

with `prompt_tokens_details.cached_tokens` = `used`.
"""
from __future__ import annotations

ATTR = "_knurlogic_cache"


def attach(request, report: dict) -> None:
    """Onto the request and every request it was copied from: a layer that
    rewrites the request before tokenizing (vision's placeholders) leaves a
    `_knurlogic_origin` link, and the handler holds the original."""
    seen = set()
    while request is not None and id(request) not in seen:
        seen.add(id(request))
        try:
            setattr(request, ATTR, report)
        except Exception:
            pass
        request = getattr(request, "_knurlogic_origin", None)


def of(request):
    return getattr(request, ATTR, None) if request is not None else None
