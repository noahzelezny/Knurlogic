"""Small files several processes read: written whole or not at all.

  atomic(path)               a context yielding a temporary path beside
                             `path` to write any way (a streamed fetch,
                             mx.save_safetensors), renamed over `path` on
                             success and removed on failure
  temp_path(path)            that temporary path, for a writer whose file
                             is not renamed over `path`
  write_atomic(path, data)   text or bytes to a temporary file beside
                             `path`, then renamed over it, so a reader sees
                             the old file or the new one, never half
  write_json(path, doc)      the same for a JSON document
  read_json(path, default)   the document, or `default` when the file is
                             missing, unreadable, or not of default's type
  update_json(path, change, default)
                             read, change and write the document as one
                             step against every other update_json of
                             `path`, in any thread or process: a writer
                             that read before another one saved never
                             writes its stale copy over that save

The temporary file is named `.tmp-<pid>-<thread>-<name>`: unique per
writer, so two processes saving at once do not write into one file, and
the prompt cache's sweeps of `.tmp-*` leftovers apply to it.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def temp_path(path) -> Path:
    """The temporary file beside `path`: `.tmp-<pid>-<thread>-<name>`."""
    path = Path(path)
    return path.with_name(
        f".tmp-{os.getpid()}-{threading.get_ident()}-{path.name}")


@contextmanager
def atomic(path) -> Iterator[Path]:
    """Yield `temp_path(path)` (its directory made) for the body to write;
    rename it over `path` when the body returns, remove it when it
    raises."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_atomic(path, data: str | bytes) -> None:
    """`data` (text or bytes) to `path`, atomically."""
    with atomic(path) as tmp:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data)


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


@contextmanager
def exclusive(path) -> Iterator[None]:
    """`path` held for one writer at a time, across threads and processes:
    an flock on `.lock-<name>` beside it (flock is per open file, so two
    threads of one process exclude each other too). The lock file stays;
    it is empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".lock-{path.name}").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def update_json(path, change: Callable, default=None, end: str = "",
                **dumps_kw):
    """Read the document at `path` (`default` as read_json), pass it to
    `change`, and write what it returns -- the document itself when it
    returns None, so `change` may edit it in place -- all under
    `exclusive(path)`. -> the document written."""
    with exclusive(path):
        doc = read_json(path, default)
        out = change(doc)
        doc = doc if out is None else out
        write_json(path, doc, end, **dumps_kw)
        return doc
