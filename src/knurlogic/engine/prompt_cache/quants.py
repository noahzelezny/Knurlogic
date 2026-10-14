"""Saved entries borrowed from another quant of the same model
(docs/design/prompt-cache-shards.md, "Borrowing between quants").

The key (layers.shard_key) holds the artifact's identity, so a model
loaded at another quant (3.4 bpw, 5.4 bpw, 8-bit) finds none of the
sessions the others saved. Their caches have the same shape and meaning
when the two artifacts share what fixes them: the text architecture and
the tokenizer (`fingerprint`). Continuing from another quant's cache is an
approximation, as 16 -> 8 conversion is (measured on Qwen3.6-35B quants:
the next-token distribution differs from a re-read of the prompt by mean
KL 0.015-0.042, about 40-60% of the gap the two quants have anyway).

When a load's restore, or a read-back, finds fewer entries than it wants
under its key and its own other widths (convert.py), it looks under the
keys that differ in `artifact` (and, when the conversion setting allows,
in kv_bits) for entries with the loaded model's fingerprint, reads them,
and converts one of another width with convert.py's code. Order: own
entries, own converted widths, borrowed at the same width, borrowed and
converted.

A borrowed entry is the other quant's: never deleted, moved or written.
It is used in memory; the session's next save goes under the loaded
model's key, as any save does (`file` is not adopted, so a keep-latest
step never deletes it). On a ring every rank reads its own part, the
restore's vote keeps what every rank read, and nothing is persisted. A
ring's read-back reads only entries under its own key, so it never
borrows; peer fetch (transfer.py) does not borrow either.

Setting (knurlogic-wide, tuning/groups.PROMPT_CACHE_KNOBS):
  KNURLOGIC_PROMPT_CACHE_ANY_QUANT  on (default) / off
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from knurlogic.engine.arrays import arrays_of

from . import convert, disk

logger = logging.getLogger(__name__)

ENV = "KNURLOGIC_PROMPT_CACHE_ANY_QUANT"
#: the text config's fields that fix a cache's shape and meaning: layers,
#: widths, heads, which layers attend how, MLA and indexer dims, the
#: compressed layers, the vocabulary, and the rope settings that place
#: positions. Quantization is not among them.
FIELDS = (
    "model_type", "num_hidden_layers", "hidden_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "layer_types",
    "full_attention_interval", "sliding_window", "sliding_window_pattern",
    "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
    "v_head_dim", "index_n_heads", "index_head_dim", "index_topk",
    "compress_ratios", "linear_num_key_heads", "linear_num_value_heads",
    "linear_key_head_dim", "linear_value_head_dim",
    "linear_conv_kernel_dim", "vocab_size", "rope_theta", "rope_scaling",
    "rope_parameters", "partial_rotary_factor",
    "max_position_embeddings")
#: the files that map text to token ids (a chat template does not: the
#: cache is keyed by ids). With a tokenizer.json, what it DOES is compared
#: (`_tokenizer_meaning`), not its bytes: publishers of one model ship the
#: same vocabulary with other formatting and other side files (measured:
#: Qwen3.6-35B's VQ and mlx-community 8-bit copies). Without one, the
#: other files' bytes.
TOKENIZER = ("tokenizer.json", "tokenizer.model", "vocab.json",
             "merges.txt", "added_tokens.json", "special_tokens_map.json")
#: {path: (stamp, fingerprint)} and {artifact identity: fingerprint}
_BY_PATH: dict = {}
_BY_IDENT: dict = {}


def on(env: dict | None = None) -> bool:
    from knurlogic.tuning.knobs import on_off
    try:
        return on_off((env if env is not None else disk._env()).get(ENV))
    except ValueError:
        return True


# ----------------------------------------------------------- same model

def _text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def _rope(v):
    """A rope block with its type under one name: publishers write `type`
    or `rope_type` for the same thing."""
    if isinstance(v, dict) and "type" in v and "rope_type" not in v:
        v = {("rope_type" if k == "type" else k): x for k, x in v.items()}
    return v


def _tokenizer_meaning(raw: bytes) -> bytes:
    """What a tokenizer.json maps text to ids with: its model's type,
    vocabulary and merges (pairs as "a b"), the added tokens (id, text,
    special), the splitting rule and the normalizer. A different splitting
    rule tokenizes some text differently, so it counts (measured: the
    397B VQ copies split without combining marks, Qwen's own with)."""
    t = json.loads(raw)
    m = t.get("model") or {}
    merges = [" ".join(x) if isinstance(x, list) else x
              for x in m.get("merges") or []]
    added = sorted((a.get("id"), a.get("content"), bool(a.get("special")))
                   for a in t.get("added_tokens") or [])
    return json.dumps({"type": m.get("type"), "vocab": m.get("vocab"),
                       "merges": merges, "added": added,
                       "pre": t.get("pre_tokenizer"),
                       "normalizer": t.get("normalizer")},
                      sort_keys=True).encode()


def fingerprint(path) -> str | None:
    """16 hex of what fixes a cache's shape and meaning for the artifact at
    `path`: its text architecture (FIELDS, from text_config for a
    multimodal wrapper) and its tokenizer files' bytes. Two quants of one
    model share it; None when there is no config.json. Cached per path
    until config.json or a tokenizer file changes."""
    p = Path(path)
    files = [p / "config.json"] + [p / n for n in TOKENIZER]
    try:
        stamp = tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns)
                      for f in files if f.is_file())
    except OSError:
        return None
    if not stamp or stamp[0][0] != "config.json":
        return None
    hit = _BY_PATH.get(str(p))
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        cfg = json.loads(files[0].read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    tc = _text_config(cfg)
    arch = {k: _rope(tc.get(k, cfg.get(k))) for k in FIELDS}
    arch["model_type"] = [cfg.get("model_type"), tc.get("model_type")]
    h = hashlib.sha256(json.dumps(arch, sort_keys=True).encode())
    tj = p / "tokenizer.json"
    try:
        if tj.is_file():
            h.update(b"tokenizer.json\0" + _tokenizer_meaning(tj.read_bytes()))
        else:
            for f in files[1:]:
                try:
                    h.update(f.name.encode() + b"\0" + f.read_bytes())
                except FileNotFoundError:
                    continue
    except (OSError, ValueError):
        return None
    fp = h.hexdigest()[:16]
    _BY_PATH[str(p)] = (stamp, fp)
    return fp


def of_artifact(ident: str | None) -> str | None:
    """The fingerprint of the local artifact a key's `artifact` names (an
    identity, or `path:<dir>` when it had none), or None when it resolves
    to none here: an entry saved before fingerprints were recorded."""
    if not isinstance(ident, str) or not ident:
        return None
    if ident in _BY_IDENT:
        return _BY_IDENT[ident]
    fp = None
    try:
        if ident.startswith("path:"):
            fp = fingerprint(ident[5:])
        else:
            from knurlogic.machine import artifact
            p = artifact.resolve_identity(ident)
            fp = fingerprint(p) if p else None
    except OSError:
        fp = None
    _BY_IDENT[ident] = fp
    return fp


def of_entry(meta: dict | None) -> str | None:
    """An entry's fingerprint: the one its entry.json records, else its
    key's artifact's."""
    m = meta or {}
    return m.get("fingerprint") or \
        of_artifact((m.get("key") or {}).get("artifact"))


# ----------------------------------------------------------- the entries

def _key_of(d: Path) -> dict | None:
    """A key directory's key: its key.json, else its first entry's."""
    try:
        k = json.loads((d / "key.json").read_text())
        if isinstance(k, dict):
            return k
    except (OSError, ValueError):
        pass
    from .layers import entry_meta, is_entry_name
    try:
        subs = sorted(p for p in d.iterdir()
                      if p.is_dir() and is_entry_name(p.name))
    except OSError:
        return None
    for s in subs:
        k = (entry_meta(s) or {}).get("key")
        if isinstance(k, dict):
            return k
    return None


def lenders(store) -> list:
    """[(bits, Store)]: the stores of the keys under this store's root that
    differ from its key in `artifact` (and in kv_bits when convert.allowed
    says so), the same width first. None when the setting is off or the
    loaded model has no fingerprint."""
    from .layers import Store
    if not on() or not getattr(store, "fingerprint", None):
        return []
    mine = store.key
    to = mine.get("kv_bits")
    how = convert.mode()
    same = {k: v for k, v in mine.items() if k not in ("artifact", "kv_bits")}
    out = []
    try:
        dirs = sorted(p for p in store.base.iterdir() if p.is_dir())
    except OSError:
        return []
    for d in dirs:
        if d.name == store.key_id:
            continue
        k = _key_of(d)
        if not isinstance(k, dict) or k.get("artifact") == mine.get(
                "artifact") or \
                {a: b for a, b in k.items()
                 if a not in ("artifact", "kv_bits")} != same:
            continue
        b = k.get("kv_bits")
        if b != to and (getattr(store, "template", None) is None or
                        not convert.allowed(b, to, how)):
            continue
        s = Store(store.base, k, store.part, store.draft)
        if s.key_id != d.name:
            continue
        out.append((b, s))
    out.sort(key=lambda bs: bs[0] != to)
    return out


def _same(store, meta) -> bool:
    return of_entry(meta) == store.fingerprint


def candidates(store, n: int, max_bytes=None, skip=()) -> list:
    """Up to `n` entry directories of other quants this store may borrow,
    in restore order (lowest priority first: borrowed and converted, then
    borrowed at this width; newest last within each), none whose tokens
    `skip` (hash names) holds."""
    if n <= 0:
        return []
    to = store.key.get("kv_bits")
    seen = set(skip)
    same, other = [], []
    for b, s in lenders(store):
        for d in s.candidates(n + len(seen), max_bytes):
            if d.name in seen or not _same(store, convert.disk_meta(d)):
                continue
            seen.add(d.name)
            (same if b == to else other).append(d)

    def age(d):
        m = convert.disk_meta(d) or {}
        return int(m.get("gen", 0)), int(m.get("seq", 0))
    same = sorted(same, key=age)[-n:]
    other = sorted(other, key=age)[-(n - len(same)):] \
        if n > len(same) else []
    return other + same


def _name_of(meta: dict) -> str:
    return meta.get("model") or (meta.get("key") or {}).get("artifact") \
        or "another quant"


def read(store, dirs) -> list:
    """The other quants' entries in `dirs`, read as this store's rank holds
    them and converted to its width where that differs: layers.Store.read's
    shape, each meta with `borrowed_from` (the source artifact's name) and,
    when converted, `converted`. One that fails a check is left out; none
    is ever deleted."""
    if not dirs or not getattr(store, "fingerprint", None):
        return []
    import mlx.core as mx

    from .layers import Store
    to = store.key.get("kv_bits")
    out = []
    for d in dirs:
        m = convert.disk_meta(d)
        k = (m or {}).get("key")
        if not isinstance(k, dict) or not _same(store, m):
            continue
        b = k.get("kv_bits")
        s = Store(store.base, k, store.part, store.draft)
        for d0, toks, cache, kind, ms, meta in s.read([d], drop_bad=False):
            extra = {"borrowed_from": _name_of(m)}
            if b != to:
                try:
                    if store.template is None:
                        raise ValueError("no template to convert to")
                    cache = convert.convert_caches(
                        cache, store.template()[:store.part.n])
                    mx.eval(arrays_of(cache))
                except Exception as e:  # a conversion that fails is a miss (logged)
                    logger.warning("prompt cache entry %s not converted: "
                                   "%s: %s", Path(d0).name,
                                   type(e).__name__, e)
                    continue
                extra["converted"] = {"from_bits": b, "to_bits": to}
            out.append((d0, toks, cache, kind, ms, dict(meta, **extra)))
    if out:
        logger.info("prompt cache: %d entr%s borrowed from %s",
                    len(out), "y" if len(out) == 1 else "ies",
                    sorted({g[5]["borrowed_from"] for g in out}))
    return out


def index(store, have) -> dict:
    """The read-back's lookup of the other quants' entries a single machine
    may borrow: {tuple(tokens): {"file", "owner", "borrow": name}}, none
    whose tokens `have` holds; the same width before a converted one."""
    out = {}
    for _b, s in lenders(store):
        for _, _, _, d, m in s.entries():
            toks = m.get("tokens")
            if not isinstance(toks, list) or len(toks) != m.get("n_tokens") \
                    or tuple(toks) in have or tuple(toks) in out or \
                    not _same(store, m) or not s.holds(d, m):
                continue
            out[tuple(toks)] = {"file": d, "owner": m.get("owner"),
                                "borrow": _name_of(m)}
    return out


def read_back(store, d) -> list:
    """A single machine's read-back of another quant's entry `d`: read (and
    converted), nothing written or deleted."""
    return read(store, [d])
