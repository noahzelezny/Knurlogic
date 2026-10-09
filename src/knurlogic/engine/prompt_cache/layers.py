"""The prompt cache on disk, one file per layer, so any split reads it
(docs/design/prompt-cache-shards.md).

Layout, under a root (location.py):

    <root>/<key id>/<tokens hash>/
        entry.json            tokens, owner, pinned, saved_at, shared,
                              model, kind, gen/seq, the layers it holds,
                              whether it carries a drafting head's cache
        L000.safetensors      cache slot 0 (the model's first layer), ...
        L059.safetensors
        draft.safetensors     the drafting head's cache, when the entry
                              has one

A layer file holds one cache slot of the model's `make_cache()` list, by
its global index: on a pipeline every layer has one, so slot i is layer i
whichever rank runs it. Its header records the slot, the tokens' hash and
the object tree (disk.py's encoding); the arrays are in the file.

The key (`shard_key`) has no layout: the artifact's identity, the KV bits,
long-context rope, per-chip rounding and CACHE_FORMAT -- not knurlogic's
version, so a release that does not touch the format keeps the cache.
The drafting head is not in the key: only `draft.safetensors` depends on
it, and its header names the head; a reader whose head differs (or that
has none) restores the entry's layers without it.

Who writes what: a single machine writes every file. On a pipeline each
rank writes the layer files of the layers it holds; rank 0 (it holds the
drafting head) also writes `draft.safetensors`. `entry.json` is written
last, and only once every rank has written its part (`commit`: a vote
over the ring's link after the ranks' saves), so an entry is whole or
absent. An entry directory without `entry.json` is incomplete: never
read, and removed by the sweep once stale.

Reading: a rank reads only the layers it holds (`Part`); an entry missing
any of them is a miss on that rank, and the restore's vote (`agree`) keeps
only the entries every rank read. Tensor splits keep disk.py's per-split
format (a later phase).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import disk, location

logger = logging.getLogger(__name__)

#: the on-disk format's version: bump it when a file written by this code
#: would be read wrongly by the next (it is in the key)
CACHE_FORMAT = 2
ENTRY = "entry.json"
DRAFT = "draft.safetensors"
SUFFIX = disk.SUFFIX
_HASH = re.compile(r"^[0-9a-f]{24}$")
#: entries one restore vote can name (the newest); fixed, so every rank's
#: row has the same shape
MAX_VOTE = 256


def layer_name(i: int) -> str:
    return f"L{int(i):03d}{SUFFIX}"


def is_entry_name(name: str) -> bool:
    """A tokens-hash directory name (what a ring's ops name)."""
    return bool(_HASH.match(name or ""))


# ------------------------------------------------------------------ part

@dataclass(frozen=True)
class Part:
    """The cache slots one rank holds: [start, start + n) of `total`;
    `extras`: it also owns what follows them in an entry (the drafting
    head's cache)."""
    start: int
    n: int
    total: int
    extras: bool = True
    rank: int = 0
    world: int = 1

    @property
    def slots(self) -> range:
        return range(self.start, self.start + self.n)


def part_of(host) -> Part | None:
    """This rank's Part, or None when this format does not apply (a tensor
    split: its slices are a later phase; disk.py's format is used)."""
    from mlx_lm.models.cache import make_prompt_cache
    layout = getattr(host, "cache_layout", None) or {"split": "none"}
    split = layout.get("split", "none")
    if split not in ("none", "pipeline"):
        return None
    n = len(make_prompt_cache(host.model))
    if split == "none":
        return Part(0, n, n)
    bounds = [list(b) for b in layout.get("bounds") or []]
    rank = int(layout.get("rank", 0))
    if not bounds or rank >= len(bounds):
        return None
    s, e = bounds[rank]
    if n != e - s:
        # a stage whose cache list is not one slot per layer: no global
        # index to file it under
        logger.warning("prompt cache: rank %d holds layers [%d, %d) but %d "
                       "cache slots; saved per split", rank, s, e, n)
        return None
    return Part(s, n, max(b[1] for b in bounds), rank == 0, rank,
                int(layout.get("world", len(bounds))))


def uses_layers(host) -> bool:
    """This format, not the per-split one: a whole model or a pipeline.
    The same answer on every rank of a ring."""
    layout = getattr(host, "cache_layout", None) or {"split": "none"}
    return layout.get("split", "none") in ("none", "pipeline")


# ------------------------------------------------------------------- key

def shard_key(path, *, kv_bits=None, long_context=None,
              cross_chip=None) -> dict:
    """What a layer's K/V depends on: no layout, no knurlogic version."""
    from knurlogic.machine import artifact
    try:
        ident = artifact.identity(path) if path else ""
    except (OSError, ValueError):
        ident = ""
    return {"cache_format": CACHE_FORMAT,
            "artifact": ident or f"path:{path}",
            "kv_bits": None if kv_bits is None else int(kv_bits),
            "long_context": long_context or None,
            "cross_chip": rounding_of(cross_chip)}


def rounding_of(cross_chip) -> str | None:
    """What per-chip rounding changes in the numbers: whether it is on and,
    when on, its setting and the chips it rounds for. Off is off, whatever
    machines a split spans (their names would make a single machine's
    entries and a cluster's different keys for identical K/V)."""
    if cross_chip is None:
        return None
    if isinstance(cross_chip, dict):
        if not cross_chip.get("on"):
            return "off"
        return repr({"setting": cross_chip.get("setting"),
                     "chips": sorted(cross_chip.get("chips") or [])})
    return repr(cross_chip)


def host_shard_key(host) -> dict | None:
    if getattr(host, "model", None) is None or not getattr(host, "path", None):
        return None
    return shard_key(host.path, kv_bits=getattr(host, "kv_bits", None),
                     long_context=os.environ.get("KNURLOGIC_LONG_CONTEXT"),
                     cross_chip=getattr(host, "cross_chip", None))


# ------------------------------------------------------------ files

def _write(path: Path, objs: list, meta: dict) -> int:
    """One safetensors file of `objs` (encoded), atomically. Returns the
    arrays' bytes."""
    import mlx.core as mx
    arrays, tree = disk.encode_entry(objs)
    nbytes = sum(int(a.nbytes) for a in arrays.values())
    if not arrays:
        arrays = {"__none__": mx.zeros((1,), dtype=mx.int8)}
    head = dict(meta, tree=tree, nbytes=nbytes)
    tmp = path.parent / f".tmp-{os.getpid()}-{path.name}"
    try:
        mx.save_safetensors(str(tmp), arrays, {"knurlogic": json.dumps(head)})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return nbytes


def _read(path: Path, want: dict) -> list:
    """The objects of a file whose header says `want` (every key equal);
    raises ValueError for any mismatch."""
    import mlx.core as mx
    head = disk._header(path)
    if head is None:
        raise ValueError(f"{path.name}: unreadable header")
    for k, v in want.items():
        if head.get(k) != v:
            raise ValueError(f"{path.name}: {k} is {head.get(k)!r}, not {v!r}")
    arrays = mx.load(str(path))
    arrays.pop("__none__", None)
    if sum(int(a.nbytes) for a in arrays.values()) != head.get("nbytes"):
        raise ValueError(f"{path.name}: its arrays do not match the header")
    return list(disk._decode(head["tree"], arrays))


def entry_meta(d: Path) -> dict | None:
    """An entry directory's entry.json, or None (incomplete or unreadable)."""
    try:
        m = json.loads((Path(d) / ENTRY).read_text())
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def _dir_size(d: Path) -> int:
    total = 0
    try:
        for f in d.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def remove_dir(d: Path) -> bool:
    """Delete an entry directory: entry.json first, so no reader sees a
    half-deleted entry as whole."""
    d = Path(d)
    try:
        (d / ENTRY).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()


def discard(p) -> None:
    """Delete an entry's file (disk.py's format) or directory (this one)."""
    p = Path(p)
    try:
        if p.is_dir():
            remove_dir(p)
        else:
            p.unlink()
    except OSError:
        pass


def units(base: Path):
    """(dir, size, mtime) of every complete entry directory under `base`,
    any model's: mtime is entry.json's (touched on use)."""
    out = []
    try:
        keys = [p for p in Path(base).iterdir() if p.is_dir()]
    except OSError:
        return out
    for k in keys:
        try:
            subs = [p for p in k.iterdir() if p.is_dir()
                    and is_entry_name(p.name)]
        except OSError:
            continue
        for d in subs:
            try:
                mt = (d / ENTRY).stat().st_mtime
            except OSError:
                continue
            out.append((d, _dir_size(d), mt))
    return out


def clean(base: Path, now: float, stale_s: float) -> None:
    """Stale temp files in entry directories, and incomplete entries (no
    entry.json) nobody wrote to for `stale_s`."""
    try:
        keys = [p for p in Path(base).iterdir() if p.is_dir()]
    except OSError:
        return
    for k in keys:
        try:
            subs = [p for p in k.iterdir() if p.is_dir()
                    and is_entry_name(p.name)]
        except OSError:
            continue
        for d in subs:
            try:
                files = list(d.iterdir())
                newest = max([f.stat().st_mtime for f in files] or
                             [d.stat().st_mtime])
            except OSError:
                continue
            if not (d / ENTRY).exists():
                if now - newest > stale_s:
                    shutil.rmtree(d, ignore_errors=True)
                continue
            for f in files:
                if f.name.startswith(".tmp-"):
                    try:
                        if now - f.stat().st_mtime > stale_s:
                            f.unlink()
                    except OSError:
                        pass


def has_entries(key_dir: Path) -> bool:
    """A key directory still holds an entry directory (complete or being
    written)."""
    try:
        return any(p.is_dir() and is_entry_name(p.name)
                   for p in Path(key_dir).iterdir())
    except OSError:
        return False


def row(d: Path, size: int, base: Path) -> dict | None:
    """The registry row of an entry directory (disk.list_disk's shape)."""
    m = entry_meta(d)
    if m is None:
        return None
    o = m.get("owner") or {}
    return {"key_id": m.get("key_id"), "model": m.get("model"),
            "hash": m.get("tokens_hash"), "session": o.get("session"),
            "role": o.get("role"), "run": o.get("run"),
            "tokens": m.get("n_tokens"), "bytes": size,
            "saved_at": m.get("saved_at"), "pinned": bool(m.get("pinned")),
            "file": str(d), "location": str(base)}


# ----------------------------------------------------------------- store

class Store:
    """One rank's view of one model's entries in this format: where they
    are (`base`), the key, and which slots this rank holds."""

    def __init__(self, base: Path, key: dict, part: Part,
                 draft: dict | None = None):
        self.base = Path(base)
        self.key = key
        self.key_id = disk.key_id(key)
        self.dir = self.base / self.key_id
        self.part = part
        #: the drafting head this rank runs (disk.draft_of), or None
        self.draft = draft
        #: {hash: tokens} saved here and waiting for their entry.json (a
        #: ring's commit)
        self.pending: dict = {}
        #: rank 0's records of the entries saved since the last commit op
        #: (engine/prompt_cache/commands._journal_commit)
        self.outbox: list = []

    @classmethod
    def for_host(cls, host) -> Store | None:
        """The store for what `host` has loaded, or None (nothing loaded,
        a tensor split, or the cache is off)."""
        if not disk.enabled() or getattr(host, "model", None) is None \
                or not getattr(host, "path", None):
            return None
        part = part_of(host)
        if part is None:
            return None
        base, _ = location.root_for(host.path)
        return cls(base, host_shard_key(host), part,
                   disk.draft_of() if part.extras else None)

    @property
    def location(self) -> str:
        return str(self.base)

    # ------------------------------------------------------- one entry

    def write_part(self, tokens, cache: list) -> int:
        """This rank's files of one entry (every array encoded before any
        file is written: an unsaveable entry writes nothing). Returns the
        arrays' bytes."""
        toks = [int(t) for t in tokens]
        h = disk.tokens_hash(toks)
        p = self.part
        if len(cache) < p.n:
            raise disk.Unserializable(f"an entry of {len(cache)} caches; "
                                      f"this rank holds {p.n}")
        for c in cache:                      # fail before writing anything
            disk.encode_entry([c])
        d = self.dir / h
        d.mkdir(parents=True, exist_ok=True)
        common = {"cache_format": CACHE_FORMAT, "key_id": self.key_id,
                  "tokens_hash": h, "n_tokens": len(toks)}
        total = 0
        for i, c in enumerate(cache[:p.n]):
            idx = p.start + i
            total += _write(d / layer_name(idx), [c], dict(common, layer=idx))
        extra = list(cache[p.n:])
        if p.extras and extra:
            total += _write(d / DRAFT, extra,
                            dict(common, layer="draft", draft=self.draft))
        return total

    def holds(self, d: Path, meta: dict) -> bool:
        """Every file this rank needs of entry `d` is there (`meta`: its
        entry.json, or the commit's record of it)."""
        p = self.part
        if meta.get("layers") != list(range(p.total)):
            return False
        if meta.get("key_id") not in (None, self.key_id):
            return False
        return all((d / layer_name(i)).is_file() for i in p.slots) and \
            not (p.extras and meta.get("draft") and
                 not (d / DRAFT).is_file())

    def read_part(self, d: Path, meta: dict) -> list:
        """This rank's caches of entry `d`, its layers in order, then the
        drafting head's cache when this rank runs the same head. Raises
        ValueError for any file that fails its checks."""
        want = {"cache_format": CACHE_FORMAT, "key_id": self.key_id,
                "tokens_hash": meta["tokens_hash"],
                "n_tokens": meta["n_tokens"]}
        out = []
        for i in self.part.slots:
            got = _read(d / layer_name(i), dict(want, layer=i))
            if len(got) != 1:
                raise ValueError(f"{layer_name(i)} holds {len(got)} caches")
            out.append(got[0])
        f = d / DRAFT
        if self.part.extras and meta.get("draft") and self.draft and \
                f.is_file() and (disk._header(f) or {}).get("draft") == \
                self.draft:
            out += _read(f, dict(want, layer="draft"))
        return out

    def write_entry(self, meta: dict, tokens) -> Path:
        """entry.json, last: the entry is now whole. `meta`: the save's
        record (no tokens); the same bytes on every rank."""
        d = self.dir / meta["tokens_hash"]
        body = dict(meta, tokens=[int(t) for t in tokens])
        tmp = d / f".tmp-{os.getpid()}-{ENTRY}"
        tmp.write_text(json.dumps(body, sort_keys=True))
        os.replace(tmp, d / ENTRY)
        if not (self.dir / "key.json").exists():
            try:
                (self.dir / "key.json").write_text(json.dumps(self.key,
                                                              indent=1))
            except OSError:
                pass
        return d

    # ------------------------------------------------------- entries

    def entries(self) -> list:
        """(gen, seq, hash, dir, meta) of this model's complete entries,
        oldest save first."""
        out = []
        try:
            subs = [p for p in self.dir.iterdir()
                    if p.is_dir() and is_entry_name(p.name)]
        except OSError:
            return out
        for d in subs:
            m = entry_meta(d)
            if m is None or m.get("key") != self.key:
                continue
            out.append((int(m.get("gen", 0)), int(m.get("seq", 0)), d.name,
                        d, m))
        return sorted(out, key=lambda e: e[:3])

    def index(self) -> dict:
        """{tuple(tokens): {"file": dir, "owner"}}: the read-back's lookup
        (entries this rank can read whole)."""
        out = {}
        for _, _, _, d, m in self.entries():
            toks = m.get("tokens")
            if not isinstance(toks, list) or len(toks) != m.get("n_tokens") \
                    or not self.holds(d, m):
                continue
            out[tuple(toks)] = {"file": d, "owner": m.get("owner")}
        return out

    def candidates(self, max_n: int, max_bytes: int | None = None) -> list:
        """The entry directories a load restores: the newest `max_n` this
        rank can read whole, within `max_bytes` if given, oldest first."""
        if max_n <= 0:
            return []
        es = [e for e in self.entries() if self.holds(e[3], e[4])][-max_n:]
        if max_bytes:
            out, total = [], 0
            for e in reversed(es):
                s = _dir_size(e[3])
                if total + s > max_bytes:
                    break
                out.append(e)
                total += s
            es = list(reversed(out))
        return [e[3] for e in es]

    def read(self, dirs, drop_bad: bool | None = None) -> list:
        """Read entry directories into memory, in order: disk.read's shape,
        [(dir, tokens, cache, kind, read_ms, meta)]. One that fails a check
        is left out (and, on a single machine, deleted)."""
        import mlx.core as mx
        if drop_bad is None:
            drop_bad = self.part.world == 1
        out = []
        for d in dirs:
            d = Path(d)
            t0 = time.perf_counter()
            m = entry_meta(d)
            why = None
            try:
                if m is None:
                    why = "no entry.json (incomplete)"
                elif m.get("key") != self.key:
                    why = "its key is not this model's"
                elif not self.holds(d, m):
                    why = "a layer file this rank needs is missing"
                else:
                    toks = m.get("tokens") or []
                    if len(toks) != m.get("n_tokens") or \
                            disk.tokens_hash(toks) != m.get("tokens_hash"):
                        why = "its tokens do not match entry.json"
                    else:
                        cache = self.read_part(d, m)
                        mx.eval(list(disk._arrays_of(cache)))
                        out.append((d, list(toks), cache,
                                    m.get("kind", "assistant"),
                                    round((time.perf_counter() - t0) * 1000,
                                          1),
                                    {k: m.get(k) for k in
                                     ("owner", "pinned", "saved_at",
                                      "shared")}))
                        continue
            except Exception as e:  # any read or decode failure: a miss
                why = f"{type(e).__name__}: {e}"
            logger.warning("prompt cache entry %s not read: %s", d.name, why)
            if drop_bad and m is not None:
                remove_dir(d)
        return out

    def remove(self, names) -> int:
        """Delete this model's entries by hash name (a ring's drop_files)."""
        n = 0
        for name in names or ():
            if is_entry_name(name) and remove_dir(self.dir / name):
                n += 1
        return n

    def drop_sessionless(self, older_than_s=None,
                         now: float | None = None) -> tuple:
        """This model's entries no session owns (only those unused for
        `older_than_s`, when given). Returns (count, {dir paths})."""
        now = time.time() if now is None else now
        n, gone = 0, set()
        for _, _, _, d, m in self.entries():
            if (m.get("owner") or {}).get("session"):
                continue
            try:
                if older_than_s is not None and now - \
                        (d / ENTRY).stat().st_mtime < float(older_than_s):
                    continue
            except OSError:
                continue
            if remove_dir(d):
                n += 1
                gone.add(str(d))
        return n, gone

    # ---------------------------------------------------------- save

    def save(self, lru, *, owners=None, shared=None, only_new: bool = False,
             select=None, prefill_tps=None, model: str | None = None,
             commit: bool = True, sweep_after: bool = True,
             removed: list | None = None) -> dict:
        """disk.save's rules in this format: this rank's files of each
        selected entry. `commit`: write each entry.json now (a single
        machine); False on a ring, where `stats["pending"]` is rank 0's
        record of each entry for the commit op (and every rank keeps the
        tokens in `self.pending`)."""
        t0 = time.monotonic()
        stats = {"saved": 0, "kept": 0, "skipped": 0, "bytes": 0, "why": [],
                 "entries": 0, "not_worth": 0, "pending": [],
                 "location": self.location}
        if lru is None or not disk.enabled():
            return stats
        have = {e[2]: e for e in self.entries()}
        gen = max((e[0] for e in have.values()), default=0) + 1
        seq = 0
        for _m, tokens, e in disk.lru_entries(lru):
            own = owners.get(tuple(tokens)) if owners is not None else None
            common = shared is not None and tuple(tokens) in shared
            if owners is not None and not (own and own.get("session")) \
                    and not common:
                continue
            if select is not None and tuple(tokens) not in select:
                continue
            stats["entries"] += 1
            if disk.not_worth(len(tokens), int(e.nbytes), prefill_tps):
                stats["not_worth"] += 1
                continue
            try:
                if not all(isinstance(t, int) for t in tokens):
                    raise disk.Unserializable(
                        "an image key (its image store does not outlive "
                        "the model)")
                h = disk.tokens_hash(tokens)
                d = self.dir / h
                old = have.get(h)
                if only_new and old is not None and own is not None and \
                        own.get("file") == str(d) and self.holds(d, old[4]):
                    stats["kept"] += 1          # on disk and unchanged
                    continue
                cache = e.prompt_cache
                draft = bool(self.part.extras and len(cache) > self.part.n)
                meta = {"format": "knurlogic-prompt-cache",
                        "cache_format": CACHE_FORMAT, "key": self.key,
                        "key_id": self.key_id, "n_tokens": len(tokens),
                        "tokens_hash": h, "kind": e.cache_type,
                        "owner": disk._owner_of(own),
                        "pinned": bool(own and own.get("pinned")),
                        "shared": bool(common), "model": model,
                        "layers": list(range(self.part.total)),
                        "draft": draft, "gen": gen, "seq": seq,
                        "saved_at": round(time.time(), 3)}
                om = old[4] if old is not None else None
                if om is not None and om.get("owner") == meta["owner"] and \
                        om.get("n_tokens") == len(tokens) and \
                        (om.get("draft") or not draft) and \
                        self.holds(d, om):
                    # on disk: only its place in the LRU changes
                    meta["saved_at"] = om.get("saved_at", meta["saved_at"])
                    meta["draft"] = bool(om.get("draft"))
                    stats["kept"] += 1
                else:
                    stats["bytes"] += self.write_part(tokens, cache)
                    stats["saved"] += 1
                seq += 1
                self.pending[h] = [int(t) for t in tokens]
                if commit:
                    self.write_entry(meta, self.pending.pop(h))
                    try:
                        os.utime(d / ENTRY)
                    except OSError:
                        pass
                else:
                    stats["pending"].append(
                        {k: v for k, v in meta.items() if k != "key"})
                if own is not None:
                    own["file"] = str(d)
                    own["saved_at"] = meta["saved_at"]
            except disk.Unserializable as ex:
                stats["skipped"] += 1
                stats["why"].append(str(ex))
            except (OSError, ValueError, RuntimeError) as ex:
                stats["skipped"] += 1
                stats["why"].append(f"{type(ex).__name__}: {ex}")
        if stats["skipped"]:
            logger.warning("prompt cache: %d entr%s not saved: %s",
                           stats["skipped"],
                           "y" if stats["skipped"] == 1 else "ies",
                           "; ".join(sorted(set(stats["why"]))))
        stats["sweep"] = {}
        if sweep_after:
            stats["sweep"] = disk.sweep(self.base, removed=removed)
            fb = location.fallback()
            if fb != self.base and fb.exists():
                disk.sweep(fb)          # the old format ages out there
        stats["seconds"] = round(time.monotonic() - t0, 3)
        logger.info("prompt cache saved to %s: %d entries, %d written "
                    "(%d bytes, %.2f GiB), %d already there, %d not worth "
                    "it, %d skipped, %.3fs", self.dir, stats["entries"],
                    stats["saved"], stats["bytes"], stats["bytes"] / disk.GIB,
                    stats["kept"], stats["not_worth"], stats["skipped"],
                    stats["seconds"])
        return stats

    def ready(self, meta: dict) -> bool:
        """This rank saved its part of the entry `meta` names and holds
        every file of it (the commit's vote)."""
        h = meta.get("tokens_hash", "")
        return h in self.pending and is_entry_name(h) and \
            self.holds(self.dir / h, dict(meta, key_id=self.key_id))


# ------------------------------------------------- across a ring's ranks

def _gather(link, row: list) -> list:
    """Every rank's `row` (same length on every rank), rank order."""
    import mlx.core as mx
    link.align()
    got = mx.distributed.all_gather(mx.array(row, dtype=mx.int32),
                                    group=link.group, stream=mx.cpu).tolist()
    if got and not isinstance(got[0], list):
        got = [got[i:i + len(row)] for i in range(0, len(got), len(row))]
    return got


def commit(link, store: Store | None, entries: list) -> list:
    """A ring's commit op: every rank says which of rank 0's just-saved
    `entries` it holds its part of; each entry every rank holds gets its
    entry.json (on every rank: the same bytes, so a shared folder sees one
    file and a machine's own folder gets its copy). A collective: every
    rank calls it at the op, whatever it holds. Returns the hashes made
    whole."""
    if not entries:
        return []
    mine = [1 if store is not None and store.ready(e) else 0
            for e in entries]
    rows = _gather(link, mine) if link is not None else [mine]
    whole = []
    for i, e in enumerate(entries):
        h = e.get("tokens_hash", "")
        toks = store.pending.pop(h, None) if store is not None else None
        if not all(r[i] for r in rows):
            logger.info("prompt cache: entry %s not committed: a rank did "
                        "not write its part", h)
            continue
        if store is not None and toks is not None:
            try:
                store.write_entry(dict(e, key=store.key,
                                       key_id=store.key_id), toks)
            except OSError:
                logger.exception("prompt cache: writing %s's entry.json", h)
                continue
        whole.append(h)
    return whole


def _code(name: str) -> list:
    """Two 28-bit ints of a 24-hex name (what a vote carries)."""
    v = int(name[:14], 16)
    return [(v >> 28) & ((1 << 28) - 1), v & ((1 << 28) - 1)]


def vote_row(got: list) -> list:
    """What a rank says it read (disk.read's shape), as one fixed-width
    int32 row: the count, then two ints per entry name, newest MAX_VOTE."""
    names = [Path(g[0]).name for g in got][-MAX_VOTE:]
    row = [len(names)]
    for n in names:
        row += _code(n)
    return row + [0] * (1 + 2 * MAX_VOTE - len(row))


def agree(link, got: list) -> list:
    """The restore's vote: every rank says which entries it read (the
    newest MAX_VOTE); each rank keeps the ones every rank read, in rank
    0's order. A collective: every rank calls it at the same point."""
    rows = _gather(link, vote_row(got))

    def codes(r):
        return [tuple(r[1 + 2 * i:3 + 2 * i]) for i in range(r[0])]
    every = [set(codes(r)) for r in rows]
    order = [c for c in codes(rows[0]) if all(c in s for s in every)]
    mine = {tuple(_code(Path(g[0]).name)): g for g in got}
    out = [mine[c] for c in order if c in mine]
    if len(out) != len(got):
        logger.info("prompt cache: %d of the %d entries this rank read are "
                    "held whole by every rank", len(out), len(got))
    return out


def restore(max_n: int, store: Store | None, *, max_bytes=None, link=None,
            sweep_first: bool = True) -> list:
    """The load-time read of this format: the newest `max_n` entries this
    rank can read whole (within `max_bytes` on a single machine), agreed
    across the ranks when `link` is given (every rank calls this at the
    same point, whatever it holds; a ring's ranks do not sweep here, rank
    0's saves do). Returns disk.read's shape for disk.insert. Never
    raises."""
    got = []
    try:
        if store is not None and disk.enabled():
            if sweep_first and link is None:
                disk.sweep(store.base)
            got = store.read(store.candidates(
                max_n, max_bytes if link is None else None))
    except Exception:  # a restore that fails is a miss, never a failed load (logged)
        logger.exception("prompt cache: reading the saved entries failed")
        got = []
    if link is not None:
        got = agree(link, got)
    return got
