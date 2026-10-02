"""A small JSON cache in <cache_dir>, {key: [stamp, value]}, so a page or
serve start does not redo work whose inputs have not changed.

Loaded once per process (per cache dir); new entries are written once,
after a short quiet spell (a /models.json build writes once, not per model),
by temp file + rename -- page and serve may both write, last writer wins and
the entries are idempotent. A corrupt or unreadable file is ignored. Keys
are paths: a flush prunes the ones that no longer exist."""
import json
import threading
from pathlib import Path

#: seconds of quiet before new entries are written
DELAY = 1.0


class DiskCache:
    def __init__(self, name: str, valid=lambda v: True):
        self.name = name
        #: what a loaded entry's value must satisfy to be kept
        self.valid = valid
        self.file = None
        self.data: dict = {}
        self.timer = None
        self.dirty = False
        self.atexit = False
        #: guards every read and write (page handler threads and the flush
        #: timer share it)
        self.lock = threading.Lock()

    def path(self) -> Path | None:
        try:
            from knurlogic.machine.servers import cache_dir
            return cache_dir() / self.name
        except OSError:
            return None

    def _live(self) -> dict:
        """The live dict; call with the lock held."""
        f = self.path()
        if self.file != f:
            data = {}
            if f is not None:
                try:
                    raw = json.loads(f.read_text())
                    if isinstance(raw, dict):
                        data = {k: v for k, v in raw.items()
                                if isinstance(v, list) and len(v) == 2
                                and self.valid(v[1])}
                except (OSError, ValueError):
                    data = {}
            self.file, self.data = f, data
        return self.data

    def get(self, key: str, stamp):
        """The value stored for `key` under this same (JSON-shaped) stamp,
        else None."""
        with self.lock:
            hit = self._live().get(key)
        return hit[1] if hit and hit[0] == stamp else None

    def put(self, key: str, stamp, value) -> None:
        with self.lock:
            self._live()[key] = [stamp, value]
            self._schedule()

    def flush(self) -> None:
        """Write the file now (pruning gone paths). Handler threads keep
        writing while it runs: the dict is copied under the lock, pruned
        outside it, and only the pruned keys are dropped from the live
        dict, so a concurrent write is kept (and lands next flush)."""
        import os
        import tempfile
        with self.lock:
            t, self.timer = self.timer, None
            self.dirty = False
            f = self.file
            snap = dict(self.data)
        if t is not None:
            t.cancel()
        if f is None:
            return
        gone = [k for k in snap if not os.path.exists(k)]
        with self.lock:
            for k in gone:
                if self.data.get(k) is snap[k]:
                    del self.data[k]
                del snap[k]
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=f".{f.stem}.")
            with os.fdopen(fd, "w") as fh:
                json.dump(snap, fh)
            os.replace(tmp, f)
        except OSError:
            with self.lock:
                self.dirty = True       # atexit tries again
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _atexit_flush(self) -> None:
        if self.dirty:
            self.flush()

    def _schedule(self) -> None:
        """Call with the lock held."""
        import atexit
        if self.timer is not None:
            self.timer.cancel()
        if not self.atexit:
            atexit.register(self._atexit_flush)
            self.atexit = True
        self.dirty = True
        t = threading.Timer(DELAY, self.flush)
        t.daemon = True
        self.timer = t
        t.start()
