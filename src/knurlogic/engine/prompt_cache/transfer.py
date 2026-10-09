"""An entry's missing pieces, fetched from a peer (option 2 of
docs/design/prompt-cache-shards.md, "Where it lives").

Machines that do not share model storage each keep their own cache root.
An entry saved under one split leaves each layer's file on the machine
that held that layer; under another split (or on one machine) a rank
needs files another machine wrote. Before the restore vote (and before a
read-back), a rank asks the peers which pieces of this model's entries
they hold (`listing`, served by each page) and fetches the ones it lacks
(`fill`): each into this machine's root, under a temp name, verified,
then renamed. A fetched file is then an ordinary local file (the sweep,
TTL and pins apply).

The engine does no networking: the interfaces side installs a fetcher
(`use`), an object with

    peers()                                  -> [peer, ...]
    listing(peer, key_id)                    -> listing()'s shape, or None
    get(peer, key_id, entry, piece, dest, size) -> bytes written to dest

(interfaces/page/prompt_cache.PageFetcher). With none installed, or no
peer answering, nothing is fetched; on shared storage the files are
already here, so nothing is either.

Budget: fetching an entry counts toward the break-even. Each peer's
transfer rate is measured on every fetch (the last one counts); an
entry whose missing bytes at those rates take longer than recomputing
its tokens at the measured prefill rate is not fetched. No rate measured
yet: fetched.

A piece is named `entry` (entry.json), `draft` (draft.safetensors) or a
layer's slot index. `piece_path` is the only way a request's names
become a file: each name must match its format, and the file must lie
inside one of this machine's cache roots.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from . import disk, layers, location

logger = logging.getLogger(__name__)

ENTRY_PIECE = "entry"
DRAFT_PIECE = "draft"
_KEY_ID = re.compile(r"^[0-9a-f]{16}$")
_LAYER = re.compile(r"^[0-9]{1,4}$")
_LAYER_FILE = re.compile(r"^L([0-9]{3,4})" + re.escape(layers.SUFFIX) + "$")
#: an entry.json is small; a larger one is refused
ENTRY_MAX = 64 << 20

#: the installed fetcher (`use`), or None
_FETCHER: list = [None]
#: {peer: bytes per second of its last fetch}
RATES: dict = {}


def use(fetcher) -> None:
    """Install the fetcher this process asks peers with (None: none)."""
    _FETCHER[0] = fetcher


def fetcher():
    return _FETCHER[0]


# ------------------------------------------------------------ names

def is_key_id(s) -> bool:
    return isinstance(s, str) and bool(_KEY_ID.match(s))


def parse_piece(s):
    """"entry", "draft" or a layer's slot index (int), from a request's
    string; None for anything else."""
    if s in (ENTRY_PIECE, DRAFT_PIECE):
        return s
    if isinstance(s, str) and _LAYER.match(s):
        return int(s)
    return None


def file_of(piece) -> str:
    if piece == ENTRY_PIECE:
        return layers.ENTRY
    if piece == DRAFT_PIECE:
        return layers.DRAFT
    return layers.layer_name(int(piece))


def _inside(p: Path, root: Path) -> bool:
    try:
        return p.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def piece_path(key_id, entry, piece, roots=None) -> Path | None:
    """The file of one piece in this machine's cache roots (`roots`:
    location.known() by default), or None: a name that does not match
    its format, a file that is not there, a symbolic link, or a path
    that leaves the root. Only entries that are whole here (their
    entry.json written) are served."""
    if not is_key_id(key_id) or not layers.is_entry_name(entry):
        return None
    piece = parse_piece(piece) if isinstance(piece, str) else piece
    if piece is None or (isinstance(piece, int) and not 0 <= piece < 10000):
        return None
    name = file_of(piece)
    for root in (location.known() if roots is None else roots):
        root = Path(root)
        d = root / key_id / entry
        f = d / name
        try:
            if not (d / layers.ENTRY).is_file() or f.is_symlink() or \
                    not f.is_file():
                continue
        except OSError:
            continue
        if _inside(f, root):
            return f
    return None


def listing(key_id, roots=None) -> dict:
    """What this machine holds of one model's whole entries:
    {tokens hash: {"gen", "seq", "n_tokens", "draft", "layers": [slot
    indexes held], "sizes": {piece: bytes}}}, over every cache root
    (merged; the first root holding a piece names its size)."""
    out: dict = {}
    if not is_key_id(key_id):
        return out
    for root in (location.known() if roots is None else roots):
        kd = Path(root) / key_id
        try:
            subs = [p for p in kd.iterdir()
                    if p.is_dir() and layers.is_entry_name(p.name)]
        except OSError:
            continue
        for d in subs:
            m = layers.entry_meta(d)
            if m is None or m.get("key_id") != key_id:
                continue
            row = out.setdefault(d.name, {
                "gen": int(m.get("gen", 0)), "seq": int(m.get("seq", 0)),
                "n_tokens": m.get("n_tokens"), "draft": False,
                "layers": [], "sizes": {}})
            try:
                files = list(d.iterdir())
            except OSError:
                continue
            for f in files:
                if f.is_symlink():
                    continue
                if f.name == layers.ENTRY:
                    piece = ENTRY_PIECE
                elif f.name == layers.DRAFT:
                    piece = DRAFT_PIECE
                    row["draft"] = True
                else:
                    g = _LAYER_FILE.match(f.name)
                    if g is None:
                        continue
                    piece = int(g.group(1))
                    if piece not in row["layers"]:
                        row["layers"].append(piece)
                try:
                    row["sizes"].setdefault(str(piece), f.stat().st_size)
                except OSError:
                    continue
            row["layers"].sort()
    return out


# ------------------------------------------------------------ fetching

def _has(info: dict, piece) -> bool:
    if piece == ENTRY_PIECE:
        return True
    if piece == DRAFT_PIECE:
        return bool(info.get("draft"))
    return piece in (info.get("layers") or [])


def _size(info: dict, piece) -> int | None:
    v = (info.get("sizes") or {}).get(str(piece))
    return int(v) if isinstance(v, int) and v >= 0 else None


def _check_entry(store, name: str, body: bytes) -> dict:
    """A fetched entry.json, parsed and checked against this model and
    this name; raises ValueError."""
    m = json.loads(body)
    if not isinstance(m, dict):
        raise ValueError("entry.json is not an object")
    if m.get("key") != store.key or m.get("key_id") != store.key_id:
        raise ValueError("entry.json is another model's")
    toks = m.get("tokens")
    if m.get("tokens_hash") != name or not isinstance(toks, list) or \
            len(toks) != m.get("n_tokens") or \
            not all(isinstance(t, int) for t in toks) or \
            disk.tokens_hash(toks) != name:
        raise ValueError("entry.json's tokens do not match its name")
    if m.get("layers") != list(range(store.part.total)):
        raise ValueError("entry.json's layers are not this model's")
    return m


def _verify(store, f: Path, meta: dict, piece) -> None:
    """A fetched layer or draft file: its header names this model, this
    entry and this piece, and its arrays match the header (layers._read);
    raises ValueError."""
    want = {"cache_format": layers.CACHE_FORMAT, "key_id": store.key_id,
            "tokens_hash": meta["tokens_hash"],
            "n_tokens": meta["n_tokens"],
            "layer": "draft" if piece == DRAFT_PIECE else int(piece)}
    layers._read(f, want)


def _get(f, peer, store, name, piece, dest: Path, size) -> int:
    t0 = time.perf_counter()
    n = int(f.get(peer, store.key_id, name, piece, dest, size))
    dt = time.perf_counter() - t0
    if n != size:
        raise ValueError(f"{file_of(piece)}: {n} bytes, the peer listed "
                         f"{size}")
    if dt > 0 and n > 0:
        RATES[peer] = n / dt
    return n


def _fetch_one(f, store, name, piece, sources, d: Path, meta) -> bytes | None:
    """One piece from the first source that delivers a verified copy:
    written into `d` under its own name (entry.json's bytes are returned
    instead, for the caller to write last). None when no source did."""
    for peer, info in sources:
        size = _size(info, piece)
        if size is None or (piece == ENTRY_PIECE and size > ENTRY_MAX):
            continue
        tmp = d / f".tmp-{os.getpid()}-fetch-{file_of(piece)}"
        try:
            _get(f, peer, store, name, piece, tmp, size)
            if piece == ENTRY_PIECE:
                body = tmp.read_bytes()
                _check_entry(store, name, body)
                return body
            _verify(store, tmp, meta, piece)
            os.replace(tmp, d / file_of(piece))
            return b""
        except Exception as e:  # any failed fetch or check: the next source
            logger.warning("prompt cache: %s of entry %s from %s refused: "
                           "%s: %s", file_of(piece), name, peer,
                           type(e).__name__, e)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    return None


def _missing(store, d: Path, meta: dict | None, info_draft: bool) -> list:
    p = store.part
    out = [] if meta is not None else [ENTRY_PIECE]
    out += [i for i in p.slots if not (d / layers.layer_name(i)).is_file()]
    draft = meta.get("draft") if meta is not None else info_draft
    if p.extras and draft and not (d / layers.DRAFT).is_file():
        out.append(DRAFT_PIECE)
    return out


def _complete(f, store, name: str, local: dict | None, lists: dict,
              prefill_tps, stats: dict) -> None:
    d = store.dir / name
    has = {peer: got[name] for peer, got in lists.items()
           if isinstance(got.get(name), dict)}
    first = next(iter(has.values()), {})
    if local is not None and (local.get("layers") !=
                              list(range(store.part.total))):
        return                          # not this model's layers
    need = _missing(store, d, local, bool(first.get("draft")))
    if not need:
        return
    stats["entries"] += 1
    sources = {pc: [(peer, info) for peer, info in has.items()
                    if _has(info, pc)] for pc in need}
    if not all(sources.values()):
        stats["unavailable"] += 1
        logger.info("prompt cache: entry %s lacks %s here and no peer "
                    "holds it", name,
                    ", ".join(file_of(pc) for pc, s in sources.items()
                              if not s))
        return
    n_tokens = (local or first).get("n_tokens") or 0
    if prefill_tps and prefill_tps > 0:
        est = 0.0
        for pc, s in sources.items():
            peer, info = s[0]
            rate = RATES.get(peer)
            if rate:
                est += (_size(info, pc) or 0) / rate
        if est > n_tokens / float(prefill_tps):
            stats["not_worth"] += 1
            logger.info("prompt cache: entry %s not fetched: %.1fs to "
                        "fetch, %.1fs to recompute", name, est,
                        n_tokens / float(prefill_tps))
            return
    d.mkdir(parents=True, exist_ok=True)
    meta, entry_body = local, None
    if ENTRY_PIECE in need:
        entry_body = _fetch_one(f, store, name, ENTRY_PIECE,
                                sources[ENTRY_PIECE], d, None)
        if entry_body is None:
            stats["rejected"] += 1
            return
        meta = json.loads(entry_body)
    for pc in need:
        if pc == ENTRY_PIECE:
            continue
        if _fetch_one(f, store, name, pc, sources[pc], d, meta) is None:
            stats["rejected"] += 1
            return
        stats["pieces"] += 1
    if entry_body is not None:
        # last: the entry is whole here only once every piece is
        tmp = d / f".tmp-{os.getpid()}-{layers.ENTRY}"
        tmp.write_bytes(entry_body)
        os.replace(tmp, d / layers.ENTRY)
        stats["pieces"] += 1
    stats["fetched"] += 1


#: the most one fill (a load's, a read-back's) spends fetching: a slow peer
#: must not hold a ring's ranks at the restore vote; what is not fetched by
#: then is a miss, recomputed as before
FILL_BUDGET_S = 120.0


def fill(store, max_n: int | None = None, *, dirs=None, prefill_tps=None,
         fetch=None, budget_s: float | None = None) -> dict:
    """Fetch what this rank lacks of this model's entries: the newest
    `max_n` (local and the peers' together), or the entry directories
    `dirs` (a read-back). Never raises; stats say what happened."""
    stats = {"entries": 0, "fetched": 0, "pieces": 0, "unavailable": 0,
             "not_worth": 0, "rejected": 0, "out_of_time": 0}
    end = time.monotonic() + (FILL_BUDGET_S if budget_s is None
                              else budget_s)
    f = fetch or _FETCHER[0]
    if f is None or store is None or not disk.enabled():
        return stats
    try:
        local = {e[2]: e[4] for e in store.entries()}
        if dirs is not None:
            names = [Path(x).name for x in dirs]
            names = [n for n in names if layers.is_entry_name(n)]
            if all(n in local and not _missing(store, store.dir / n,
                                               local[n], False)
                   for n in names):
                return stats            # all here: no peer is asked
        try:
            peers = list(f.peers() or [])
        except Exception:  # no peers known is no fetch (logged)
            logger.exception("prompt cache: listing the peers")
            peers = []
        lists = {}
        for peer in peers:
            try:
                got = f.listing(peer, store.key_id)
            except Exception as e:  # a peer that does not answer is skipped
                logger.info("prompt cache: %s did not list its entries: "
                            "%s: %s", peer, type(e).__name__, e)
                continue
            if isinstance(got, dict):
                lists[peer] = got
        if not lists:
            return stats
        if dirs is None:
            order = {}
            for n, m in local.items():
                order[n] = (int(m.get("gen", 0)), int(m.get("seq", 0)), n)
            for got in lists.values():
                for n, info in got.items():
                    if layers.is_entry_name(n) and isinstance(info, dict) \
                            and n not in order:
                        order[n] = (int(info.get("gen", 0)),
                                    int(info.get("seq", 0)), n)
            names = [o[2] for o in sorted(order.values())]
            if max_n is not None:
                names = names[-max_n:] if max_n > 0 else []
        for i, n in enumerate(names):
            if time.monotonic() > end:
                stats["out_of_time"] = len(names) - i
                logger.warning("prompt cache: fetching from the peers took "
                               "over %.0f s; %d entr%s left as misses",
                               FILL_BUDGET_S if budget_s is None else budget_s,
                               len(names) - i,
                               "y" if len(names) - i == 1 else "ies")
                break
            _complete(f, store, n, local.get(n), lists, prefill_tps, stats)
    except Exception:  # a fetch that fails is a miss, never a failed load
        logger.exception("prompt cache: fetching from the peers failed")
    if stats["entries"]:
        logger.info("prompt cache: from the peers: %d entr%s lacked "
                    "pieces here, %d fetched (%d files), %d held by no "
                    "peer, %d not worth it, %d refused", stats["entries"],
                    "y" if stats["entries"] == 1 else "ies",
                    stats["fetched"], stats["pieces"], stats["unavailable"],
                    stats["not_worth"], stats["rejected"])
    return stats
