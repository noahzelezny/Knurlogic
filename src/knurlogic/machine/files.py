"""Small files several processes read: written whole or not at all.

  write_atomic(path, data)   text or bytes to a temporary file beside
                             `path`, then renamed over it, so a reader sees
                             the old file or the new one, never half
  write_json(path, doc)      the same for a JSON document
  read_json(path, default)   the document, or `default` when the file is
                             missing, unreadable, or not of default's type

The temporary file is named `.tmp-<pid>-<thread>-<name>`: unique per
writer, so two processes saving at once do not write into one file, and
the prompt cache's sweeps of `.tmp-*` leftovers apply to it.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


def write_atomic(path, data: str | bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".tmp-{os.getpid()}-{threading.get_ident()}-{path.name}")
    try:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path, doc, end: str = "", **dumps_kw) -> None:
    """`doc` as json.dumps(doc, **dumps_kw) + `end`, atomically."""
    write_atomic(path, json.dumps(doc, **dumps_kw) + end)


def read_json(path, default=None):
    """The JSON document at `path`; `default` when it cannot be read or
    parsed, or (default not None) is not of default's type."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default
    if default is not None and not isinstance(doc, type(default)):
        return default
    return doc
