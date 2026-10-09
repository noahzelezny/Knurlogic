"""The request ledger: one row per served request, on this machine only
(docs/design/fleet.md, "The request record"; docs/design/telemetry.md).

Counts, never text: token counts, timings, the client's opaque labels
(X-Client, X-Client-Session, X-Client-Run, X-Client-Role), the outcome.
A row is written once, when its request closes.

Stored in SQLite at KNURLOGIC_HOME/ledger.db (default ~/.knurlogic), WAL,
one table. A ring: rows older than KNURLOGIC_LEDGER_DAYS (30) go, and the
oldest go while the live data is over KNURLOGIC_LEDGER_MIB (256).

This is the only module that opens the ledger file; engine/ never imports
it (the scheduler reports timings on the job, the HTTP layer writes)."""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DAYS = 30
MIB = 256
#: the retention check runs once per this many inserts (and at open)
PRUNE_EVERY = 200

#: the contract's request headers -> column, byte limit (telemetry.md)
LABELS = {"X-Client": ("client", 128), "X-Client-Session": ("session", 128),
          "X-Client-Run": ("run", 128), "X-Client-Role": ("role", 32)}

COLUMNS = ("id", "ts_start", "ts_end", "machine", "model", "api", "key_id",
           "client", "session", "run", "role", "prompt_tokens",
           "cached_tokens", "output_tokens", "queue_ms", "prefill_ms",
           "prefill_tps", "decode_ms", "decode_tps", "finish", "status",
           "disk_tokens")
GROUPS = ("key", "model", "client", "session", "run", "role", "api")
_GROUP_COL = {"key": "key_id"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY, ts_start REAL NOT NULL, ts_end REAL,
  machine TEXT, model TEXT, api TEXT, key_id TEXT NOT NULL,
  client TEXT, session TEXT, run TEXT, role TEXT,
  prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
  queue_ms REAL, prefill_ms REAL, prefill_tps REAL,
  decode_ms REAL, decode_tps REAL, finish TEXT, status INTEGER,
  disk_tokens INTEGER);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts_start);
CREATE INDEX IF NOT EXISTS requests_key ON requests (key_id, ts_start);
CREATE INDEX IF NOT EXISTS requests_model ON requests (model, ts_start);
"""

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"     # Crockford


def ulid(now: float | None = None) -> str:
    """A ULID: 48 bits of milliseconds then 80 random, 26 characters of
    Crockford base32 -- sorts by time, unique without coordination."""
    ms = int((time.time() if now is None else now) * 1000) & ((1 << 48) - 1)
    n = (ms << 80) | secrets.randbits(80)
    return "".join(_B32[(n >> (5 * i)) & 31] for i in range(25, -1, -1))


def label(value, limit: int) -> str | None:
    """A header value cut to `limit` UTF-8 bytes (never refused), on a
    character boundary; None for none. A folded header (a line starting
    with a space continues the one before it; Python's parser keeps the
    CRLF) reads as one space, as RFC 9112 says, and other control
    characters go: a label is one line."""
    if not isinstance(value, str):
        return None
    value = " ".join("".join(c if c.isprintable() or c in " \t\r\n" else ""
                             for c in value).split())
    if not value:
        return None
    b = value.encode("utf-8", "replace")
    return b[:limit].decode("utf-8", "ignore") if len(b) > limit else value


def labels(headers) -> dict:
    """The four X-Client-* labels from request headers (any case)."""
    out = {}
    for h, (col, limit) in LABELS.items():
        try:
            out[col] = label(headers.get(h), limit) if headers else None
        except AttributeError:
            out[col] = None
    return out


def path() -> Path:
    return Path(os.environ.get("KNURLOGIC_HOME",
                               Path.home() / ".knurlogic")) / "ledger.db"


def _env_num(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
        return v if v > 0 else default
    except ValueError:
        return default


class Ledger:
    def __init__(self, file: Path | str | None = None, *,
                 days: float | None = None, mib: float | None = None):
        self.file = Path(file) if file else path()
        self.days = days or _env_num("KNURLOGIC_LEDGER_DAYS", DAYS)
        self.mib = mib or _env_num("KNURLOGIC_LEDGER_MIB", MIB)
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._inserts = 0

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.file), check_same_thread=False,
                                 isolation_level=None, timeout=5)
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")   # before tables
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            # columns added since a ledger was made: added to it
            have = {r[1] for r in db.execute("PRAGMA table_info(requests)")}
            if "disk_tokens" not in have:
                db.execute("ALTER TABLE requests ADD COLUMN disk_tokens "
                           "INTEGER")
            self._db = db
            self._prune(db)
        return self._db

    def insert(self, row: dict) -> None:
        cols = [c for c in COLUMNS if c in row]
        with self._lock:
            db = self._conn()
            db.execute(f"INSERT OR REPLACE INTO requests ({','.join(cols)}) "
                       f"VALUES ({','.join('?' * len(cols))})",
                       [row[c] for c in cols])
            self._inserts += 1
            if self._inserts % PRUNE_EVERY == 0:
                self._prune(db)

    def prune(self) -> None:
        with self._lock:
            self._prune(self._conn())

    def _used(self, db) -> int:
        ps = db.execute("PRAGMA page_size").fetchone()[0]
        pc = db.execute("PRAGMA page_count").fetchone()[0]
        free = db.execute("PRAGMA freelist_count").fetchone()[0]
        return (pc - free) * ps

    def _prune(self, db) -> None:
        db.execute("DELETE FROM requests WHERE ts_start < ?",
                   (time.time() - self.days * 86400,))
        limit = self.mib * 1024 * 1024
        while self._used(db) > limit:
            n = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if not n:
                break
            db.execute("DELETE FROM requests WHERE id IN (SELECT id FROM "
                       "requests ORDER BY ts_start LIMIT ?)",
                       (max(n // 10, 1),))
        db.execute("PRAGMA incremental_vacuum")

    def rows(self, since: float = 0, until: float | None = None,
             limit: int = 100) -> list[dict]:
        with self._lock:
            cur = self._conn().execute(
                f"SELECT {','.join(COLUMNS)} FROM requests WHERE ts_start >= ?"
                " AND ts_start < ? ORDER BY ts_start DESC LIMIT ?",
                (since, until or 1e12, int(limit)))
            return [dict(zip(COLUMNS, r)) for r in cur.fetchall()]

    def summary(self, since: float = 0, until: float | None = None,
                group: str = "model", key: str | None = None) -> list[dict]:
        """Rollups by one label: requests, tokens, errors, mean rates."""
        if group not in GROUPS:
            raise ValueError(f"group is one of {', '.join(GROUPS)}")
        col = _GROUP_COL.get(group, group)
        where, args = "ts_start >= ? AND ts_start < ?", [since, until or 1e12]
        if key:
            where += " AND key_id = ?"
            args.append(key)
        with self._lock:
            cur = self._conn().execute(
                f"SELECT {col}, COUNT(*), SUM(prompt_tokens), "
                "SUM(cached_tokens), SUM(output_tokens), "
                "SUM(finish IN ('error','refused')), SUM(finish='cancelled'),"
                " AVG(queue_ms), AVG(prefill_tps), AVG(decode_tps) "
                f"FROM requests WHERE {where} GROUP BY {col} "
                "ORDER BY COUNT(*) DESC", args)
            names = (group, "requests", "prompt_tokens", "cached_tokens",
                     "output_tokens", "errors", "cancelled", "queue_ms_avg",
                     "prefill_tps_avg", "decode_tps_avg")
            return [dict(zip(names, r)) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


_LEDGER: list = []
_LEDGER_LOCK = threading.Lock()


def ledger() -> Ledger:
    """This process's ledger (KNURLOGIC_HOME/ledger.db), opened lazily;
    a new one when KNURLOGIC_HOME has moved (tests)."""
    with _LEDGER_LOCK:
        p = path()
        if not _LEDGER or _LEDGER[0].file != p:
            _LEDGER[:] = [Ledger(p)]
        return _LEDGER[0]


def record(row: dict) -> None:
    """Write a row; a ledger that cannot be written never fails a request
    (logged)."""
    try:
        ledger().insert(row)
    except (sqlite3.Error, OSError) as e:
        logger.warning("the request ledger could not be written: %s", e)
