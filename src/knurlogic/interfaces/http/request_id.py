"""X-Request-Id as the page's router handles it.

The model server echoes a client's valid id exactly (and uses it as the
ledger row's id and usage.knurlogic.request_id); without one it answers
with a minted ULID (interfaces/http/telemetry.py, docs/design/telemetry.md).
The page's router passes a client's header on and returns the model
server's id, falling back to the client's own when an upstream sends none.

A value is accepted only if it is 1..128 visible ASCII characters (and
spaces): anything else is not an id this server will repeat into a
header, and is ignored rather than refused."""
from __future__ import annotations

HEADER = "X-Request-Id"
MAX_LEN = 128


def valid(value) -> str | None:
    """`value` if it may be echoed, else None."""
    if not isinstance(value, str) or not 0 < len(value) <= MAX_LEN:
        return None
    if any(not (0x20 <= ord(ch) <= 0x7E) for ch in value):
        return None
    return value


def of(headers) -> str | None:
    """The request's id from its headers (any case), or None."""
    try:
        return valid(headers.get(HEADER)) if headers is not None else None
    except AttributeError:
        return None
