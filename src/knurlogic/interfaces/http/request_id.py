"""X-Request-Id: a client's own id for a request, echoed back unchanged.

A harness that tags each call with its own id (the harness does) joins its trace
to the server's record of the request on that id. The server generates
none: a request without the header gets none back. The value is opaque
and echoed verbatim -- as a response header, and as `request_id` in the
answer's usage.knurlogic -- by the model server and by the page's router,
which passes it to the model server and the model server's echo back.

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
