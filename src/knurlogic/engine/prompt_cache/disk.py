"""The prompt cache on disk: saved when a client asks (never on its own),
restored when the same model loads again (docs/design/prompt-cache-disk.md).

Layout: ~/.cache/knurlogic/prompt-cache/<key id>/ (XDG_CACHE_HOME
honoured), one safetensors file per prompt-cache entry, named
`<gen>-<seq>-<tokens hash>.safetensors`: `gen` is the save it belongs to
(1 + the newest in the directory), `seq` its place in the in-memory LRU at
that save. Restoring inserts entries in (gen, seq) order, so the LRU comes
back in the order it was saved -- on every rank of a ring alike.

The key (`identity`) is everything that changes the cached K/V: the
artifact's identity (weights, quant, shipped model.py), the KV bits, the
drafting head (its cache rides in the entry), long-context rope, per-chip
rounding, and the split this rank holds. A file whose key differs is never
read into the model: it is in another directory, and its own header must
say the same key.

A cache entry is serialized by its objects' attributes (`vars`), not by
mlx-lm's `state`: knurlogic's caches (quantized K/V, DeepSeek V4's
compressor pools, GLM's latent + indexer) keep state `state` does not
carry. Arrays go into the file, plain values (int, float, bool, str, None,
lists, tuples, str-keyed dicts) into its header, nested objects by class
(imported back by module and name). Anything else -- a function, a model
reference -- makes the entry unsaveable: skipped with the reason logged,
never written half.

Writes are atomic (a temp file, then rename); a read checks the key, the
token count and hash, and the array bytes; a file that fails any of them
is deleted and is a miss.

Settings (read at use, so live):
  KNURLOGIC_PROMPT_CACHE_DISK     on (default) / off
  KNURLOGIC_PROMPT_CACHE_DISK_GB  total budget across every model; LRU
                                  eviction (default: 20% of the disk's free
                                  space plus what the cache holds, at most
                                  64 GiB)
  KNURLOGIC_PROMPT_CACHE_TTL_H    hours an entry nobody used is kept (24)

Ownership: an entry belongs to the client session that made it
(X-Client-Session; telemetry.md). Its header carries the owner {session,
role, run}, `pinned` and `saved_at`. Only entries with a session are saved.
A pinned session's files are never swept (TTL nor budget): only an
explicit drop removes them. Pins live in one `pins.json` at the root
({session: true|false}), the truth for every key directory; a header's
`pinned` is what it was when the file was written.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import struct
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FORMAT = 1
GIB = 1 << 30
DEFAULT_TTL_H = 24.0
DEFAULT_CAP_GIB = 64.0
DEFAULT_FREE_SHARE = 0.20
ENV_ON = "KNURLOGIC_PROMPT_CACHE_DISK"
ENV_GB = "KNURLOGIC_PROMPT_CACHE_DISK_GB"
ENV_TTL = "KNURLOGIC_PROMPT_CACHE_TTL_H"
PINS = "pins.json"
#: bytes/s an entry is read back at, for the break-even rule (an entry
#: whose recompute -- tokens / the measured prefill rate -- is quicker than
#: its read is not written). Measured on the M3 Ultra: 70-128 MB read in
#: ~24 ms (and ~3 GB/s written); 3 GB/s is the conservative read side.
READ_BPS = 3e9
SUFFIX = ".safetensors"
_NAME = re.compile(r"^(\d{8})-(\d{6})-([0-9a-f]{24})\.safetensors$")
#: a temp file this old is a dead writer's
_TMP_STALE_S = 3600.0


class Unserializable(ValueError):
    """This entry holds something the format cannot carry."""


# ------------------------------------------------------------- settings

def root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache"))
    return base / "knurlogic" / "prompt-cache"


def _env() -> dict:
    """The settings as saved knurlogic-wide (tuning/preferences) over the
    environment."""
    try:
        from knurlogic.tuning import preferences
        return preferences.prompt_cache_env()
    except (OSError, ValueError, ImportError):
        return dict(os.environ)


def enabled() -> bool:
    v = _env().get(ENV_ON, "").strip().lower()
    return v not in ("off", "0", "false", "no")


def ttl_s() -> float:
    try:
        h = float(_env().get(ENV_TTL, "") or DEFAULT_TTL_H)
    except ValueError:
        h = DEFAULT_TTL_H
    return max(h, 0.0) * 3600.0


def not_worth(n_tokens: int, nbytes: int, prefill_tps) -> bool:
    """Recomputing the entry (n_tokens at the measured prefill rate) is
    quicker than reading it back (nbytes at READ_BPS). No rate measured
    yet: worth saving."""
    if not prefill_tps or prefill_tps <= 0:
        return False
    return n_tokens / float(prefill_tps) < nbytes / READ_BPS


def budget_bytes(base: Path | None = None) -> int:
    """KNURLOGIC_PROMPT_CACHE_DISK_GB, else 20% of (the disk's free space
    + what the cache already holds), at most 64 GiB."""
    v = _env().get(ENV_GB, "").strip()
    if v:
        try:
            return max(int(float(v) * GIB), 0)
        except ValueError:
            pass
    base = base or root()
    held = sum(s for _, s, _ in _all_files(base))
    probe = base
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = 0
    return int(min(DEFAULT_CAP_GIB * GIB, DEFAULT_FREE_SHARE * (free + held)))


# ------------------------------------------------------------------ key

def identity(path, *, kv_bits=None, draft=None, layout=None,
             long_context=None, cross_chip=None) -> dict:
    """What the K/V of an entry depends on. `layout`: this rank's part of a
    split ({"split": "pipeline", "world", "rank", "layers": [a, b]}), None
    for a whole model."""
    from knurlogic import __version__
    from knurlogic.machine import artifact
    try:
        ident = artifact.identity(path) if path else ""
    except (OSError, ValueError):
        ident = ""
    return {"format": FORMAT, "knurlogic": __version__,
            "artifact": ident or f"path:{path}",
            "kv_bits": None if kv_bits is None else int(kv_bits),
            "draft": draft, "layout": layout or {"split": "none"},
            "long_context": long_context or None,
            "cross_chip": None if cross_chip is None else repr(cross_chip)}


def key_id(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()
                          ).hexdigest()[:16]


def host_key(host) -> dict | None:
    """The key for what `host` (engine/runtime/model_host.ModelHost) has loaded
    now, or None when nothing is."""
    if getattr(host, "model", None) is None or not getattr(host, "path", None):
        return None
    from knurlogic.engine.mtp import binding
    draft = None
    if binding.DRAFT.get("on") and binding.DRAFT.get("head") is not None:
        h = binding.DRAFT["head"]
        draft = {"head": type(h).__module__ + ":" + type(h).__qualname__,
                 "block": int(getattr(h, "block_size", 0) or 0)}
    return identity(host.path, kv_bits=getattr(host, "kv_bits", None),
                    draft=draft, layout=getattr(host, "cache_layout", None),
                    long_context=os.environ.get("KNURLOGIC_LONG_CONTEXT"),
                    cross_chip=getattr(host, "cross_chip", None))


def tokens_hash(tokens) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()
                          ).hexdigest()[:24]


# -------------------------------------------------------- serialization

def _encode(obj, arrays: dict, seen: set):
    import mlx.core as mx
    if isinstance(obj, mx.array):
        name = f"a{len(arrays)}"
        arrays[name] = obj
        return {"t": "a", "k": name}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"t": "v", "v": obj}
    if isinstance(obj, (list, tuple)):
        return {"t": "l" if isinstance(obj, list) else "u",
                "v": [_encode(x, arrays, seen) for x in obj]}
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise Unserializable("a dict with non-string keys")
        return {"t": "d", "v": {k: _encode(v, arrays, seen)
                                for k, v in obj.items()}}
    cls = type(obj)
    slots = [s for c in cls.__mro__
             for s in ((c.__dict__.get("__slots__", ()),)
                       if isinstance(c.__dict__.get("__slots__", ()), str)
                       else c.__dict__.get("__slots__", ()))
             if s not in ("__dict__", "__weakref__")]
    if callable(obj) or isinstance(obj, type) or \
            not (hasattr(obj, "__dict__") or slots) or \
            type(obj).__module__ in ("builtins", "functools", "types"):
        raise Unserializable(f"a {cls.__name__}")
    if id(obj) in seen:
        raise Unserializable(f"a {cls.__name__} referenced twice")
    seen.add(id(obj))
    mod, qual = cls.__module__, cls.__qualname__
    if "_disk_class" in cls.__dict__:
        # a class a factory makes (engine/families/qwen/kvcache._classes):
        # (module, factory, index) says how to make it again
        m, f, i = cls._disk_class
        mod, qual = m, f"{f}()[{int(i)}]"
    if _resolve(mod, qual) is not cls:
        raise Unserializable(f"class {mod}.{qual} cannot be imported back")
    node = {"t": "o", "c": f"{mod}:{qual}",
            "v": {k: _encode(v, arrays, seen)
                  for k, v in getattr(obj, "__dict__", {}).items()}}
    if slots:
        node["s"] = {s: _encode(getattr(obj, s), arrays, seen)
                     for s in slots if hasattr(obj, s)}
    return node


_FACTORY = re.compile(r"^(\w+)\(\)\[(\d+)\]$")


def _resolve(mod: str, qual: str):
    try:
        o = importlib.import_module(mod)
        m = _FACTORY.match(qual)
        if m:
            return getattr(o, m[1])()[int(m[2])]
        if "<locals>" in qual:
            return None
        for part in qual.split("."):
            o = getattr(o, part)
        return o
    except (ImportError, AttributeError, IndexError, TypeError):
        return None


def _decode(node, arrays: dict):
    t = node["t"]
    if t == "a":
        return arrays[node["k"]]
    if t == "v":
        return node["v"]
    if t == "l":
        return [_decode(x, arrays) for x in node["v"]]
    if t == "u":
        return tuple(_decode(x, arrays) for x in node["v"])
    if t == "d":
        return {k: _decode(v, arrays) for k, v in node["v"].items()}
    if t == "o":
        mod, qual = node["c"].split(":", 1)
        cls = _resolve(mod, qual)
        if cls is None:
            raise ValueError(f"no class {node['c']}")
        obj = cls.__new__(cls)
        if node["v"]:
            obj.__dict__.update({k: _decode(v, arrays)
                                 for k, v in node["v"].items()})
        for k, v in node.get("s", {}).items():
            object.__setattr__(obj, k, _decode(v, arrays))
        return obj
    raise ValueError(f"unknown node {t!r}")


def encode_entry(cache: list) -> tuple[dict, dict]:
    """(arrays, tree) of one prompt-cache entry (a list of caches)."""
    arrays: dict = {}
    tree = _encode(list(cache), arrays, set())
    return arrays, tree


# ----------------------------------------------------------- one entry

def _header(f: Path) -> dict | None:
    """The knurlogic metadata of an entry file, read from its header only
    (no array bytes), or None."""
    try:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n <= 0 or n > (256 << 20):
                return None
            h = json.loads(fh.read(n))
        return json.loads(h["__metadata__"]["knurlogic"])
    except (OSError, ValueError, KeyError, struct.error, TypeError):
        return None


def save_entry(d: Path, key: dict, tokens, cache: list, kind: str,
               gen: int, seq: int, owner: dict | None = None,
               pinned: bool = False, model: str | None = None,
               shared: bool = False) -> Path:
    """Write one entry atomically; raises Unserializable for one the format
    cannot carry (nothing is written). `owner`: {session, role, run} of the
    request that made it; `model`: the served model's name (the
    registry's)."""
    import mlx.core as mx
    toks = [int(t) for t in tokens]
    arrays, tree = encode_entry(cache)
    nbytes = sum(int(a.nbytes) for a in arrays.values())
    arrays["__tokens__"] = mx.array(toks, dtype=mx.int32)
    meta = {"format": FORMAT, "key": key, "key_id": key_id(key),
            "n_tokens": len(toks), "tokens_hash": tokens_hash(toks),
            "kind": kind, "nbytes": nbytes, "tree": tree,
            "owner": _owner_of(owner), "pinned": bool(pinned),
            "saved_at": round(time.time(), 3), "model": model,
            "shared": bool(shared)}
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{gen:08d}-{seq:06d}-{tokens_hash(toks)}{SUFFIX}"
    tmp = d / f".tmp-{os.getpid()}-{final.stem}{SUFFIX}"
    try:
        mx.save_safetensors(str(tmp), arrays,
                            {"knurlogic": json.dumps(meta)})
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


def _owner_of(meta) -> dict | None:
    """{session, role, run} of an owner record, or None for no session."""
    if not meta or not meta.get("session"):
        return None
    return {k: meta.get(k) for k in ("session", "role", "run")}


def tokens_of(f: Path) -> list | None:
    """An entry file's tokens, read from its `__tokens__` bytes alone (the
    header says where): no other array is read."""
    try:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n <= 0 or n > (256 << 20):
                return None
            h = json.loads(fh.read(n))
            t = h["__tokens__"]
            if t.get("dtype") != "I32":
                return None
            a, b = t["data_offsets"]
            fh.seek(8 + n + a)
            raw = fh.read(b - a)
        return list(struct.unpack(f"<{len(raw) // 4}i", raw))
    except (OSError, ValueError, KeyError, struct.error, TypeError):
        return None


def load_entry(f: Path, key: dict):
    """(tokens, cache list, kind) of an entry file, or None: a file that
    does not match `key`, or fails its checks, is deleted and is a miss."""
    import mlx.core as mx
    why = None
    try:
        meta = _header(f)
        if meta is None:
            why = "unreadable header"
        elif meta.get("key") != key:
            # not this model's: never read in (it should not be in this
            # directory at all)
            why = "its key is not this model's"
        else:
            arrays = mx.load(str(f))
            toks = arrays.pop("__tokens__").tolist()
            if len(toks) != meta["n_tokens"] or \
                    tokens_hash(toks) != meta["tokens_hash"]:
                why = "its tokens do not match the header"
            elif sum(int(a.nbytes) for a in arrays.values()) != \
                    meta["nbytes"]:
                why = "its arrays do not match the header"
            else:
                cache = _decode(meta["tree"], arrays)
                return toks, list(cache), meta.get("kind", "assistant")
    except Exception as e:  # any read or decode failure is a corrupt file: a miss
        why = f"{type(e).__name__}: {e}"
    logger.warning("prompt cache file %s dropped: %s", f.name, why)
    try:
        f.unlink()
    except OSError:
        pass
    return None


# --------------------------------------------------------- directories

def _all_files(base: Path):
    """(path, size, mtime) of every entry file under `base`."""
    out = []
    try:
        dirs = [p for p in base.iterdir() if p.is_dir()]
    except OSError:
        return out
    for d in dirs:
        try:
            for f in d.iterdir():
                if f.name.endswith(SUFFIX) and not f.name.startswith("."):
                    st = f.stat()
                    out.append((f, st.st_size, st.st_mtime))
        except OSError:
            continue
    return out


def entries(d: Path) -> list:
    """(gen, seq, hash, path) of a key directory's entries, oldest save
    first."""
    out = []
    try:
        for f in d.iterdir():
            m = _NAME.match(f.name)
            if m:
                out.append((int(m[1]), int(m[2]), m[3], f))
    except OSError:
        pass
    return sorted(out)


# ----------------------------------------------------------------- pins

def pins(base: Path | None = None) -> dict:
    """{session: pinned} from the root's pins.json ({} when none)."""
    try:
        got = json.loads(((base or root()) / PINS).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(got, dict):
        return {}
    return {str(k): bool(v) for k, v in got.items()}


def set_pin(session: str, pinned: bool | None,
            base: Path | None = None) -> None:
    """Pin or unpin a session's files (None: forget it, as a drop does).
    Written atomically; one small file for every key directory, so a pin
    never rewrites an entry file."""
    base = base or root()
    p = pins(base)
    if pinned is None:
        if session not in p:
            return
        p.pop(session)
    elif p.get(session) is bool(pinned):
        return
    else:
        p[session] = bool(pinned)
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".tmp-{os.getpid()}-{PINS}"
    tmp.write_text(json.dumps(p, indent=1, sort_keys=True))
    os.replace(tmp, base / PINS)


def _pinned_files(files, base: Path) -> set:
    """The files among `files` whose session is pinned (headers are read
    only when some session is)."""
    on = {s for s, v in pins(base).items() if v}
    if not on:
        return set()
    out = set()
    for f, _, _ in files:
        o = (_header(f) or {}).get("owner") or {}
        if o.get("session") in on:
            out.add(f)
    return out


def sweep(base: Path | None = None, now: float | None = None,
          removed: list | None = None) -> dict:
    """The TTL, then the budget (least recently used first), over every
    model's unpinned entries; stale temp files too. A pinned session's
    files are exempt from both (only a drop removes them), and the budget
    counts unpinned files only. Returns what it removed; `removed`, when
    given, gets each entry file deleted (a ring's rank 0 names them to the
    other ranks, which never sweep on their own)."""
    base = base or root()
    now = time.time() if now is None else now
    ttl, gone_ttl, gone_lru = ttl_s(), 0, 0
    files = _all_files(base)
    held = _pinned_files(files, base)
    keep = []
    for f, size, mt in files:
        if f in held:
            continue
        if now - mt > ttl:
            try:
                f.unlink()
                gone_ttl += 1
                if removed is not None:
                    removed.append(f)
            except OSError:
                pass
        else:
            keep.append((f, size, mt))
    cap = budget_bytes(base)
    total = sum(s for _, s, _ in keep)
    for f, size, _ in sorted(keep, key=lambda x: x[2]):
        if total <= cap:
            break
        try:
            f.unlink()
            total -= size
            gone_lru += 1
            if removed is not None:
                removed.append(f)
        except OSError:
            pass
    try:
        for d in base.iterdir():
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.name.startswith(".tmp-"):
                    try:
                        if now - f.stat().st_mtime > _TMP_STALE_S:
                            f.unlink()
                    except OSError:
                        pass
            if not any(f.name.endswith(SUFFIX) for f in d.iterdir()):
                shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass
    return {"ttl": gone_ttl, "budget": gone_lru, "bytes": total, "cap": cap,
            "pinned": len(held)}


def drop_files(session: str, base: Path | None = None) -> int:
    """Delete every file of `session`, in every model's key directory, and
    forget its pin. Returns how many went."""
    base = base or root()
    n = 0
    for f, _, _ in _all_files(base):
        o = (_header(f) or {}).get("owner") or {}
        if o.get("session") == session:
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    set_pin(session, None, base)
    return n


def drop_sessionless_files(d: Path, older_than_s=None,
                           now: float | None = None) -> tuple:
    """Delete the files in key directory `d` no session owns (requests
    that named none, the shared system-prompt checkpoints) -- only those
    unused for `older_than_s`, when given. Returns (count, {paths})."""
    now = time.time() if now is None else now
    n, gone = 0, set()
    for _, _, _, f in entries(d):
        m = _header(f)
        if m is None or (m.get("owner") or {}).get("session"):
            continue
        try:
            if older_than_s is not None and \
                    now - f.stat().st_mtime < float(older_than_s):
                continue
            f.unlink()
            n += 1
            gone.add(str(f))
        except OSError:
            pass
    return n, gone


def list_disk(base: Path | None = None) -> list:
    """Every entry file's registry row, from its header alone: {key_id,
    model, hash, session, role, run, tokens, bytes, saved_at, pinned,
    file}. `pinned` is pins.json's word, else the header's."""
    base = base or root()
    p = pins(base)
    out = []
    for f, size, _ in _all_files(base):
        m = _header(f)
        if m is None:
            continue
        o = m.get("owner") or {}
        s = o.get("session")
        out.append({"key_id": m.get("key_id"), "model": m.get("model"),
                    "hash": m.get("tokens_hash"), "session": s,
                    "role": o.get("role"), "run": o.get("run"),
                    "tokens": m.get("n_tokens"), "bytes": size,
                    "saved_at": m.get("saved_at"),
                    "pinned": p.get(s, bool(m.get("pinned"))) if s else False,
                    "file": str(f)})
    return out


def index(d: Path) -> dict:
    """{tuple(tokens): {"file", "owner"}} of a key directory's entries: the
    on-demand read-back's lookup, built once (only the tokens' bytes are
    read, no other array). A later save of the same tokens wins."""
    out = {}
    for _, _, _, f in entries(d):
        toks = tokens_of(f)
        m = _header(f)
        if toks is None or m is None or len(toks) != m.get("n_tokens"):
            continue
        out[tuple(toks)] = {"file": f, "owner": m.get("owner")}
    return out


# ----------------------------------------------- the prompt cache's side

def _lru_entries(lru) -> list:
    """(model, tokens, CacheEntry) of an mlx-lm LRUPromptCache, in the
    order its per-type queues hold them (least recent first within a
    type)."""
    out = []
    order = lru._lru
    for kind in order._ordering:
        for model, tokens in list(order._lrus[kind]):
            try:
                e = lru._trie.get(model, tokens)
            except (KeyError, AttributeError, IndexError):
                continue
            if e is not None:
                out.append((model, tokens, e))
    return out


def save(lru, key: dict, base: Path | None = None, *, owners=None,
         only_new: bool = False, model: str | None = None,
         select=None, prefill_tps=None, shared=None,
         sweep_after: bool = True, removed: list | None = None) -> dict:
    """Write the entries of `lru` (the scheduler's mlx-lm LRUPromptCache)
    under `key`. Returns counts; never raises for one entry.

    `owners`: the prompt cache's side map {tuple(tokens): meta}. Given,
    only entries with a session are saved (anonymous ones never are), each
    file carries its owner, and each saved entry's meta gets "file" and
    "saved_at". None (a bare LRU): every entry, no owner.

    A full save (`only_new` False) renames an entry already on disk
    (same tokens, same owner) into this save, so the files' (gen, seq)
    order is the LRU's at that save. A session's save (`only_new`) writes
    only what is not on disk yet -- an entry whose meta names no file, or
    whose file is gone -- under one new gen, in LRU order, and touches
    nothing else: cheap, and a restore still inserts oldest save first.
    `select`: the token tuples to consider (None: all). `prefill_tps`: the
    measured prefill rate; an entry quicker to recompute than to read back
    is not written (counted "not_worth"). `shared`: token tuples of the
    shared system-prompt checkpoints (nobody's), saved with `owners` too:
    marked "shared" in their header, restored as shared."""
    t0 = time.monotonic()
    stats = {"saved": 0, "kept": 0, "skipped": 0, "bytes": 0, "why": [],
             "entries": 0, "not_worth": 0}
    if lru is None or not enabled() or key is None:
        return stats
    base = base or root()
    d = base / key_id(key)
    have = {h: f for _, _, h, f in entries(d)}
    gen = max((g for g, _, _, _ in entries(d)), default=0) + 1
    seq = 0
    for _m, tokens, e in _lru_entries(lru):
        own = owners.get(tuple(tokens)) if owners is not None else None
        common = shared is not None and tuple(tokens) in shared
        if owners is not None and not (own and own.get("session")) \
                and not common:
            continue
        if select is not None and tuple(tokens) not in select:
            continue
        stats["entries"] += 1
        if not_worth(len(tokens), int(e.nbytes), prefill_tps):
            stats["not_worth"] += 1
            continue
        try:
            if not all(isinstance(t, int) for t in tokens):
                raise Unserializable("an image key (its image store does "
                                     "not outlive the model)")
            h = tokens_hash(tokens)
            old = have.pop(h, None)
            if only_new and old is not None and own is not None and \
                    own.get("file") == str(old):
                stats["kept"] += 1          # on disk and unchanged
                continue
            name = d / f"{gen:08d}-{seq:06d}-{h}{SUFFIX}"
            meta = _header(old) if old is not None and not only_new \
                else None
            if meta is not None and meta.get("key") == key and \
                    meta.get("n_tokens") == len(tokens) and \
                    (own is None or meta.get("owner") == _owner_of(own)):
                os.replace(old, name)
                os.utime(name)
                seq += 1
                stats["kept"] += 1
                if own is not None:
                    own["file"] = str(name)
                continue
            f = save_entry(d, key, tokens, e.prompt_cache, e.cache_type,
                           gen, seq, owner=own,
                           pinned=bool(own and own.get("pinned")),
                           model=model, shared=common)
            seq += 1
            if old is not None and old != f:
                try:
                    old.unlink()        # the same tokens' older file
                except OSError:
                    pass
            if own is not None:
                own["file"], own["saved_at"] = str(f), round(time.time(), 3)
            stats["saved"] += 1
            stats["bytes"] += f.stat().st_size
        except Unserializable as ex:
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
    if d.exists() and not (d / "key.json").exists():
        try:
            (d / "key.json").write_text(json.dumps(key, indent=1))
        except OSError:
            pass
    stats["sweep"] = sweep(base, removed=removed) if sweep_after else {}
    stats["seconds"] = round(time.monotonic() - t0, 3)
    logger.info("prompt cache saved to %s: %d entries, %d written "
                "(%d bytes, %.2f GiB), %d already there, %d not worth it, "
                "%d skipped, %.3fs", d, stats["entries"], stats["saved"],
                stats["bytes"], stats["bytes"] / GIB, stats["kept"],
                stats["not_worth"], stats["skipped"], stats["seconds"])
    return stats


def remove_entry(lru, model, tokens) -> bool:
    """Take one entry out of an mlx-lm LRUPromptCache. mlx-lm 0.32 has no
    public single-entry removal (only trim_to, which pops least recently
    used), so this is what its own insert_cache does to a replaced entry:
    PromptTrie.pop, CacheOrder.remove, and the byte counters. False when
    the entry is not there."""
    tokens = list(tokens)       # CacheOrder holds lists: a tuple never matches
    try:
        if lru._trie.get(model, tokens) is None:
            return False
    except (KeyError, AttributeError, IndexError):
        return False
    e = lru._trie.pop(model, tokens)
    lru._lru.remove(model, tokens)
    lru._n_bytes -= e.nbytes
    lru._n_bytes_by_type[e.cache_type] -= e.nbytes
    return True


def candidates(key: dict, max_n: int, max_bytes: int | None = None,
               base: Path | None = None, sweep_first: bool = True) -> list:
    """The files a load restores: the newest `max_n` (by save, then LRU
    place), within `max_bytes` if given, oldest first (the order they are
    inserted). The TTL and budget are applied first."""
    if not enabled() or key is None or max_n <= 0:
        return []
    base = base or root()
    if sweep_first:
        sweep(base)
    es = entries(base / key_id(key))[-max_n:]
    if max_bytes:
        out, total = [], 0
        for e in reversed(es):
            try:
                s = e[3].stat().st_size
            except OSError:
                continue
            if total + s > max_bytes:
                break
            out.append(e)
            total += s
        es = list(reversed(out))
    return [e[3] for e in es]


def vote(files) -> list[int]:
    """What a rank says it can restore, as int32s for an all_gather: the
    count and 4 x 28 bits of the digest of the (gen, seq, hash) names."""
    h = hashlib.sha256("\n".join(Path(f).name for f in files).encode())
    n = int.from_bytes(h.digest()[:14], "big")
    return [len(files)] + [(n >> (28 * i)) & ((1 << 28) - 1)
                           for i in range(4)]


def agree(link, files) -> list:
    """All-or-none on a ring: every rank restores `files` only when every
    rank has the same list (names carry save and LRU place, so the same
    names are the same entries in the same order); otherwise none does --
    a miss, not an error. A collective: every rank calls it at the same
    point (ModelHost.after_bind, before the warm-up)."""
    import mlx.core as mx
    row = vote(files)
    link.align()
    got = mx.distributed.all_gather(mx.array(row, dtype=mx.int32),
                                    group=link.group, stream=mx.cpu).tolist()
    rows = [got[i:i + len(row)] for i in range(0, len(got), len(row))] \
        if got and not isinstance(got[0], list) else got
    if all(r == rows[0] for r in rows):
        return list(files)
    logger.info("prompt cache: the ranks hold different saved entries "
                "(%s); restoring none", [r[0] for r in rows])
    return []


def read(key: dict, files) -> list:
    """Read `files` (candidates()) into memory, in order: [(file, tokens,
    cache, kind, read_ms, meta)] for each that passes its checks (one that
    does not is deleted and left out); meta: {owner, pinned, saved_at}
    from its header."""
    import mlx.core as mx
    out = []
    for f in files:
        t0 = time.perf_counter()
        head = _header(Path(f)) or {}
        got = load_entry(Path(f), key)
        if got is None:
            continue
        toks, cache, kind = got
        mx.eval(list(_arrays_of(cache)))
        out.append((Path(f), toks, cache, kind,
                    round((time.perf_counter() - t0) * 1000, 1),
                    {k: head.get(k) for k in ("owner", "pinned",
                                              "saved_at", "shared")}))
    return out


def insert(lru, model_key, got: list) -> dict:
    """What read() returned, into `lru` in order (insert_cache: the
    in-memory trie's own prefix rules then serve them). Returns
    {tuple(tokens): {"tokens", "read_ms", "file", "owner", "saved_at"}}."""
    out = {}
    for f, toks, cache, kind, ms, *meta in got:
        lru.insert_cache(model_key, list(toks), cache, cache_type=kind)
        try:
            os.utime(f)
        except OSError:
            pass
        m = meta[0] if meta else {}
        out[tuple(toks)] = {"tokens": len(toks), "read_ms": ms,
                            "file": str(f), "owner": m.get("owner"),
                            "saved_at": m.get("saved_at"),
                            "shared": bool(m.get("shared"))}
    if out:
        logger.info("prompt cache: %d entr%s restored from disk (%d tokens, "
                    "%.1fs)", len(out), "y" if len(out) == 1 else "ies",
                    sum(v["tokens"] for v in out.values()),
                    sum(v["read_ms"] for v in out.values()) / 1000)
    return out


def restore(lru, model_key, key: dict | None, *, max_bytes=None,
            link=None, base: Path | None = None) -> dict:
    """The load-time restore: the candidates for `key`, read, agreed across
    the ranks when `link` is a ring's (every rank calls this at the same
    point, whatever it read), inserted. A file that cannot be read is a
    miss, never a failed load; returns insert()'s map."""
    got = []
    try:
        if key is not None:
            # a ring's ranks do not sweep here: each would by its own
            # disk's budget, and their lists would differ (rank 0's saves
            # sweep and name what they delete to the others)
            got = read(key, candidates(key, lru.max_size, max_bytes, base,
                                       sweep_first=link is None))
    except Exception:  # a restore that fails is a miss, never a failed load (logged)
        logger.exception("prompt cache: reading the saved entries failed")
        got = []
    if link is not None and not agree(link, [g[0] for g in got]):
        got = []
    return insert(lru, model_key, got) if got else {}


def adopt(owners: dict, pinned: set, restored: dict,
          base: Path | None = None, shared: set | None = None) -> None:
    """Restored entries' owners into the prompt cache's side map (and the
    pinned sessions into its sticky set), as the files said: an entry read
    back is on disk already. A shared system-prompt checkpoint goes back
    into `shared`, so the next save keeps it."""
    on = {s for s, v in pins(base).items() if v}
    for t, v in restored.items():
        if v.get("shared") and shared is not None:
            shared.add(t)
            continue
        o = v.get("owner") or {}
        if not o.get("session"):
            continue
        s = o["session"]
        if s in on:
            pinned.add(s)
        owners[t] = {"session": s, "role": o.get("role"),
                     "run": o.get("run"), "pinned": s in on,
                     "file": v.get("file"), "saved_at": v.get("saved_at")}


def _arrays_of(obj):
    import mlx.core as mx
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _arrays_of(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _arrays_of(x)
    elif hasattr(obj, "__dict__") and not callable(obj):
        for x in vars(obj).values():
            yield from _arrays_of(x)


def source_of(lru, model_key, prompt, used: int):
    """The token tuple of the entry fetch_nearest_cache handed out for
    `prompt` when it supplied `used` tokens (its own trie search, read
    back -- not a second matcher), or None."""
    if used <= 0:
        return None
    try:
        r = lru._trie.search(model_key, prompt)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    if r.exact is not None:
        return tuple(r.exact)
    short = len(r.shorter) if r.shorter is not None else 0
    if r.longer is not None and r.common_prefix > short:
        return tuple(r.longer)
    return tuple(r.shorter) if short else None
