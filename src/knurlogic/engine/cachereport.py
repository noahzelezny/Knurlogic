"""What the prompt cache actually did for one request, in its usage.

mlx-lm reports `usage.prompt_tokens_details.cached_tokens` as what the
prefix trie OFFERED. The batch engine can still discard an offered prefix
(a drafting row whose head is not aligned with it prefills from scratch),
and then that number claims reuse that never happened. The engine is the
only place that knows, so it writes the report and the handler prints it:

    usage.knurlogic.cache = {offered, used, discarded, prefilled, via,
                             images: {total, in_cached_span, prefilled},
                             checkpoints_stored}

and `cached_tokens` becomes `used`. A request the engine did not admit
(mlx-lm's own generator, or the sequential path) carries no report.

Plumbing, all in-process: the server tokenizes a request and inserts it
into the batch engine on the SAME generation thread, one after the other,
so `_tokenize` records the request and `insert_segments` pairs the new uid
with it; admission writes the report onto the request object, which the
HTTP handler that built it still holds.
"""
from __future__ import annotations

import threading

_local = threading.local()
ATTR = "_knurlogic_cache"


def tokenizing(request) -> None:
    """The generation thread is about to insert this request."""
    _local.request = request


def claim():
    """The request just tokenized on this thread, once."""
    req = getattr(_local, "request", None)
    _local.request = None
    return req


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


def into_usage(usage: dict, report) -> dict:
    """Put the report into a usage dict; cached_tokens becomes what was used."""
    if not report or not isinstance(usage, dict):
        return usage
    usage.setdefault("prompt_tokens_details", {})["cached_tokens"] = \
        report["used"]
    usage["knurlogic"] = {"cache": report}
    return usage
