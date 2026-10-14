"""Saved entries converted between KV bit widths (docs/design/
prompt-cache-shards.md, "Conversion").

The key (layers.shard_key) holds the KV bits, so a model switched between
16-bit and 8-bit KV (KNURLOGIC_KV_BITS) finds none of its saved sessions
under its own key. When a load's restore, or a read-back, finds fewer
entries than it wants under its key, it looks under the keys that differ
in `kv_bits` alone, reads an entry there, converts it, writes it under its
own key and deletes the other width's: a session is on disk once.

What kv_bits changes is what engine/kvquant.install's make_cache swaps:
an exact mlx-lm KVCache becomes a QuantKVCache, and a family cache its
manifest names (`kv_quant.caches`: qwen4_exp's `_AttnCache`, GLM's MLA
`_LatentCache`) becomes that factory's quantized class; a CacheList's
members the same way. Nothing else differs between the two widths'
caches: recurrent state (ArraysCache), sliding windows, indexer keys, the
drafting head's cache are bf16 in both and are carried as read. So the
conversion is per slot, against a fresh cache list of the loaded model
(its `template`): where the template's slot is a quantized cache and the
entry's is not, the K/V are quantized with the template's bits and the
group size the 8-bit cache picks (kvquant._group), with kvquant's own
quantize -- what an 8-bit prefill stores, up to the rounding of the bf16
values it was given; where the template's is not and the entry's is, they
are dequantized (8 -> 16 carries the 8-bit error into a 16-bit run, so it
is a separate setting). Quantization groups run along the head dim, so a
tensor rank's head range converts the same as the whole layer cut to it.

Setting (knurlogic-wide, tuning/groups.PROMPT_CACHE_KNOBS):
  KNURLOGIC_PROMPT_CACHE_CONVERT  down (default: 16-bit entries to a
                                  quantized width), both (also back to
                                  16-bit), off
A quantized entry is never converted to another quantized width.

On a ring every rank converts its own part at the restore (a collective):
the restore's vote keeps the entries every rank read, converted or not --
the setting is knurlogic-wide, so every rank decides the same, and a rank
that did not convert does not vote for it. Then a second vote: each rank
says it wrote its part under the new key; each entry every rank wrote gets
its entry.json, and every rank deletes the old width's directory by name,
as a journaled drop_files would (no rank keeps half of it). A ring's
read-back reads only entries restored at its load, so it never converts.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import disk

logger = logging.getLogger(__name__)

ENV = "KNURLOGIC_PROMPT_CACHE_CONVERT"
MODES = ("down", "both", "off")
DEFAULT = "down"
#: the widths a key's kv_bits can hold (None: 16-bit)
WIDTHS = (None, 8, 6, 4)
#: fields of a cache that are its K/V or their quantization, never carried
_KV = {"keys", "values", "offset", "kv_bits", "group", "dims",
       "kv8_kernel", "kv8_fetch"}


def mode(env: dict | None = None) -> str:
    v = str((env if env is not None else disk._env()).get(ENV, "")
            or DEFAULT).strip().lower()
    return v if v in MODES else DEFAULT


def allowed(from_bits, to_bits, how: str | None = None) -> bool:
    """Whether an entry of `from_bits` converts to `to_bits` under `how`
    (the setting when None)."""
    how = mode() if how is None else how
    if how == "off" or from_bits == to_bits:
        return False
    if from_bits is None:
        return how in ("down", "both")
    if to_bits is None:
        return how == "both"
    return False


def bits_of(meta: dict | None):
    return ((meta or {}).get("key") or {}).get("kv_bits")


# ------------------------------------------------------------ the arrays

def _quant(c) -> bool:
    from knurlogic.engine.kvquant import BatchQuantKVCache, QuantKVCache
    return isinstance(c, (QuantKVCache, BatchQuantKVCache))


def _carry(src, dst):
    """`dst` (a fresh template cache) with `src`'s fields that are not its
    K/V (qwen4_exp's indexer)."""
    for k, v in vars(src).items():
        if k not in _KV:
            setattr(dst, k, v)
    return dst


def _down(c, t):
    from knurlogic.engine import kvquant as kq
    n = int(c.offset or 0)
    if c.keys is None or n == 0:
        return _carry(c, t)
    k, v = c.keys[..., :n, :], c.values[..., :n, :]
    g = kq._group(k.shape[-1], v.shape[-1])
    t.group, t.dims = g, (k.shape[-1], v.shape[-1])
    t.keys = kq._q(k, g, t.kv_bits)
    t.values = kq._q(v, g, t.kv_bits)
    t.offset = n
    return _carry(c, t)


def _dq(parts, dim, g, bits, n):
    import mlx.core as mx

    from knurlogic.engine import kvquant as kq
    dt = parts[1].dtype
    if dim == 0:
        return mx.zeros(parts[0].shape[:2] + (n, 0), dtype=dt)
    return kq._dq(tuple(p[..., :n, :] for p in parts), g, bits, dt)


def _up(c, t):
    n = int(c.offset or 0)
    if c.keys is None or n == 0:
        return _carry(c, t)
    dk, dv = c.dims or (None, None)
    if dk is None:
        # from_state keeps no dims: from the packed width
        dk = c.keys[0].shape[-1] * 32 // c.kv_bits
        dv = c.values[0].shape[-1] * 32 // c.kv_bits
    t.keys = _dq(c.keys, dk, c.group, c.kv_bits, n)
    t.values = _dq(c.values, dv, c.group, c.kv_bits, n)
    t.offset = n
    return _carry(c, t)


def convert_one(c, t):
    """Entry cache `c` as the loaded model's template cache `t` stores it:
    quantized, dequantized, or `c` itself (kv_bits does not change it)."""
    sub, tsub = getattr(c, "caches", None), getattr(t, "caches", None)
    if isinstance(sub, (tuple, list)) and isinstance(tsub, (tuple, list)):
        if len(sub) != len(tsub):
            raise ValueError("a cache list of another length")
        out = [convert_one(a, b) for a, b in zip(sub, tsub)]
        if all(a is b for a, b in zip(out, sub)):
            return c
        return type(c)(*out)
    if _quant(t) and not _quant(c) and hasattr(c, "keys"):
        return _down(c, t)
    if _quant(c) and not _quant(t) and hasattr(t, "keys"):
        return _up(c, t)
    return c


def convert_caches(cache: list, template: list) -> list:
    """An entry's caches (a rank's slots, then any drafting head's cache)
    as the model with `template` (a fresh make_cache list of the rank's
    slots) stores them; what follows the slots is carried as read."""
    n = len(template)
    if len(cache) < n:
        raise ValueError(f"an entry of {len(cache)} caches; the model "
                         f"makes {n}")
    return [convert_one(c, t) for c, t in zip(cache[:n], template)] + \
        list(cache[n:])


# ----------------------------------------------------------- the entries

def siblings(store) -> list:
    """[(bits, Store)]: the stores of the keys that differ from `store`'s
    in kv_bits alone, whose entries the setting lets convert to it."""
    from .layers import Store
    if getattr(store, "template", None) is None:
        return []
    to = store.key.get("kv_bits")
    how = mode()
    out = []
    for b in WIDTHS:
        if allowed(b, to, how):
            s = Store(store.base, dict(store.key, kv_bits=b), store.part,
                      store.draft)
            out.append((b, s))
    return out


def candidates(store, n: int, max_bytes=None, skip=()) -> list:
    """Up to `n` entry directories of the other widths this store may
    convert, newest last, none whose tokens `skip` (hash names) holds."""
    if n <= 0:
        return []
    out = []
    for _b, s in siblings(store):
        out += [d for d in s.candidates(n + len(skip), max_bytes)
                if d.name not in skip]
    out = [d for d in out if (Path(d) / "entry.json").exists()]
    out.sort(key=lambda d: (int((disk_meta(d) or {}).get("gen", 0)),
                            int((disk_meta(d) or {}).get("seq", 0))))
    return out[-n:]


def disk_meta(d):
    from .layers import entry_meta
    return entry_meta(Path(d))


def read(store, dirs) -> list:
    """The entries of other widths in `dirs`, read and converted to this
    store's: layers.Store.read's shape, each meta with `converted`
    {from_bits, to_bits} and `old` (its directory). One that fails a check
    is left out (never deleted: another width's entry)."""
    if not dirs or getattr(store, "template", None) is None:
        return []
    import mlx.core as mx
    to = store.key.get("kv_bits")
    by = {b: s for b, s in siblings(store)}
    out = []
    for d in dirs:
        b = bits_of(disk_meta(d))
        s = by.get(b)
        if s is None:
            continue
        for d0, toks, cache, kind, ms, meta in s.read([d], drop_bad=False):
            try:
                tmpl = store.template()[:store.part.n]
                new = convert_caches(cache, tmpl)
                mx.eval(list(disk._arrays_of(new)))
            except Exception as e:  # a conversion that fails is a miss (logged)
                logger.warning("prompt cache entry %s not converted: %s: %s",
                               Path(d0).name, type(e).__name__, e)
                continue
            out.append((d0, toks, new, kind, ms,
                        dict(meta, converted={"from_bits": b, "to_bits": to},
                             old=str(d0), source=disk_meta(d0))))
    if out:
        logger.info("prompt cache: %d entr%s converted from %s-bit KV to "
                    "%s-bit", len(out), "y" if len(out) == 1 else "ies",
                    sorted({str(g[5]["converted"]["from_bits"] or 16)
                            for g in out}), to or 16)
    return out


def _meta_for(store, src: dict, draft: bool) -> dict:
    """The converted entry's entry.json body (no tokens): the old one's
    under this store's key."""
    m = {k: v for k, v in (src or {}).items() if k != "tokens"}
    m.update(key=store.key, key_id=store.key_id, stored=store.stored(),
             layers=list(range(store.part.total)), draft=bool(draft),
             converted_from=bits_of(src) or 16)
    if getattr(store, "fingerprint", None):
        m["fingerprint"] = store.fingerprint
    return m


def write_part(store, got) -> int:
    """This rank's files of a converted entry under `store`'s key: 0 not
    written, 1 written, 2 written with the drafting head's cache."""
    d0, toks, cache, kind, ms, meta = got
    try:
        store.write_part(toks, cache)
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("prompt cache: converted entry %s not written: %s",
                       Path(d0).name, e)
        return 0
    return 2 if store.part.extras and len(cache) > store.part.n else 1


def _done(store, got, draft: bool):
    """entry.json of the converted entry, then the old width's directory
    deleted. Returns `got` naming the new directory."""
    from . import layers
    d0, toks, cache, kind, ms, meta = got
    m = _meta_for(store, meta.get("source"), draft)
    d = store.write_entry(m, toks)
    layers.remove_dir(Path(meta["old"]))
    return (d, toks, cache, kind, ms,
            {k: v for k, v in meta.items() if k not in ("old", "source")})


def persist(store, got: list) -> list:
    """A single machine: each converted entry written under the store's
    key and the old width's deleted. An entry that cannot be written stays
    in memory, its old file kept."""
    out = []
    for g in got:
        w = write_part(store, g)
        if not w:
            out.append(_keep_old(g))
            continue
        try:
            out.append(_done(store, g, w == 2))
        except OSError:
            logger.exception("prompt cache: committing a converted entry")
            out.append(_keep_old(g))
    return out


def _keep_old(g):
    d0, toks, cache, kind, ms, meta = g
    return (d0, toks, cache, kind, ms,
            {k: v for k, v in meta.items() if k not in ("old", "source")})


def persist_ring(link, store, got: list) -> list:
    """A ring, after the restore's vote (a collective: every rank calls it
    with the agreed entries, the same names in the same order): each rank
    writes its part of each entry it converted, the ranks vote, and every
    entry every rank holds under the new key gets its entry.json and loses
    the old width's directory, on every rank that converted it."""
    from . import layers
    marks = []
    for g in got[:layers.MAX_VOTE]:
        if (g[5] or {}).get("old"):
            marks.append(write_part(store, g) if store is not None else 0)
        else:
            marks.append(1)         # held under this key already
    row = [len(marks)] + marks + [0] * (layers.MAX_VOTE - len(marks))
    rows = link.gather_ints(row)
    out = []
    for i, g in enumerate(got):
        if not (g[5] or {}).get("old"):
            out.append(g)
            continue
        col = [r[1 + i] if i < r[0] else 0 for r in rows]
        if i >= layers.MAX_VOTE or not all(col):
            logger.info("prompt cache: converted entry %s not committed: a "
                        "rank did not write its part", Path(g[0]).name)
            out.append(_keep_old(g))
            continue
        try:
            out.append(_done(store, g, any(v == 2 for v in col)))
        except OSError:
            logger.exception("prompt cache: committing a converted entry")
            out.append(_keep_old(g))
    return out


def index(store, have) -> dict:
    """The read-back's lookup of the entries of other widths a single
    machine may convert: {tuple(tokens): {"file", "owner", "convert":
    from_bits}}, none whose tokens `have` holds."""
    out = {}
    for b, s in siblings(store):
        for _, _, _, d, m in s.entries():
            toks = m.get("tokens")
            if not isinstance(toks, list) or len(toks) != m.get("n_tokens") \
                    or tuple(toks) in have or not s.holds(d, m):
                continue
            out[tuple(toks)] = {"file": d, "owner": m.get("owner"),
                                "convert": b}
    return out


def read_back(store, d) -> list:
    """A single machine's read-back of another width's entry `d`: read,
    converted, written under the store's key, the old one deleted."""
    return persist(store, read(store, [d]))
