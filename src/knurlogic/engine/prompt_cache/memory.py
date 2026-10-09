"""The prompt cache in memory: PromptCache (mlx-lm's LRU plus the side map
of who owns each entry) and entry_owner, the owner a request's entry gets."""

from __future__ import annotations


class PromptCache:
    """mlx-lm's LRUPromptCache, with the one rule it lacks: an exact hit is
    returned one token short (trimmed; a cache that cannot trim is a miss),
    so there is always a token to process -- an exact hit otherwise leaves
    nothing and kills the generation thread (seen with GLM, none then
    low effort)."""

    def __init__(self, max_size: int = 10, max_bytes: int | None = None):
        from mlx_lm.models.cache import LRUPromptCache
        self.lru = LRUPromptCache(max_size=max_size,
                                  **({"max_bytes": max_bytes}
                                     if max_bytes else {}))
        #: the side map: {tuple(tokens): {session, role, run, pinned, file,
        #: saved_at}} of the entries a session made. mlx-lm's LRU evicts
        #: silently (and drops prefixes on insert): pruned against it lazily
        self.owners: dict = {}
        #: sessions pinned (X-Cache-Retain: pin, or POST .../pin): sticky
        #: on this server, for their later entries too
        self.pinned: set = set()
        #: token tuples of the shared system-prompt checkpoints (a
        #: keep-latest session's, nobody's): saved and restored like a
        #: session's entries, so a reload's first worker skips that prefill
        self.shared: set = set()

    def fetch(self, key, tokens):
        from mlx_lm.models import cache as C
        cache, rest = self.lru.fetch_nearest_cache(key, tokens)
        if cache is None or rest or not tokens:
            return cache, rest
        if C.can_trim_prompt_cache(cache):
            C.trim_prompt_cache(cache, 1)
            return cache, list(tokens[-1:])
        return None, list(tokens)

    def hit_length(self, key, tokens) -> int:
        """Tokens fetch() would hand back cached, without the copy fetch
        makes (the admission prices the checkpoints past the hit before it
        fetches). 0 if the trie cannot be asked: every boundary priced, the
        safe side."""
        try:
            r = self.lru._trie.search(key, tokens)
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0
        if r.exact is not None:
            return max(len(tokens) - 1, 0)
        short = len(r.shorter) if r.shorter is not None else 0
        if r.longer is not None and r.common_prefix > short:
            return min(len(tokens) - 1, r.common_prefix)
        return short

    def insert(self, key, tokens, cache, kind: str, origin=None,
               owner: dict | None = None) -> list:
        """`origin`: (event, uid) the cache came from -- what a ring's
        journal names (engine/prompt_cache/ring.JournalPromptCache). `owner`:
        {session, role, run[, step, latest]} of the request that made it.
        Returns the files of the entries it superseded (owner["latest"]),
        for the caller to delete."""
        self.lru.insert_cache(key, list(tokens), cache, cache_type=kind)
        return self.own(tokens, owner, key)

    def own(self, tokens, owner: dict | None, key=None) -> list:
        """Record who an entry (just inserted) belongs to. A new insert of
        the same tokens is a new entry: not on disk until saved again.
        With owner["latest"] the session keeps only this step's entries
        (`step`: the row that made them): every earlier one leaves memory,
        and its file is returned to be deleted."""
        t = tuple(tokens)
        s = (owner or {}).get("session")
        if not s:
            self.owners.pop(t, None)
            if (owner or {}).get("shared"):
                self.shared.add(t)
            return []
        step = owner.get("step")
        self.owners[t] = {"session": s, "role": owner.get("role"),
                          "run": owner.get("run"),
                          "pinned": s in self.pinned, "file": None,
                          "saved_at": None, "step": step}
        if not owner.get("latest"):
            return []
        old = [(u, m) for u, m in self.owners.items()
               if m["session"] == s and u != t and m.get("step") != step]
        files = []
        for u, m in old:
            if m.get("file"):
                files.append(m["file"])
            self.owners.pop(u, None)
            if key is not None:
                from knurlogic.engine.prompt_cache import disk as prompt_disk
                prompt_disk.remove_entry(self.lru, key, list(u))
        return files

    def live(self) -> list:
        """(model, tokens, CacheEntry, meta or None) of every entry in the
        LRU, least recent first; prunes the side map of what is gone."""
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        out = [(m, t, e, self.owners.get(tuple(t)))
               for m, t, e in prompt_disk.lru_entries(self.lru)]
        here = {tuple(t) for _, t, _, _ in out}
        for t in [t for t in self.owners if t not in here]:
            del self.owners[t]
        self.shared &= here
        return out

    def of_session(self, session: str) -> list:
        """(model, tokens) of the live entries `session` owns: the one
        selection a drop, a park and a save of a session share."""
        return [(m, t) for m, t, _, meta in self.live()
                if meta is not None and meta["session"] == session]

    def remove(self, model, tokens) -> bool:
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        self.owners.pop(tuple(tokens), None)
        return prompt_disk.remove_entry(self.lru, model, tokens)

    def drop(self, session: str) -> int:
        """Every entry of `session` out of memory; forgets its pin."""
        n = sum(self.remove(m, t) for m, t in self.of_session(session))
        self.pinned.discard(session)
        return n

    def drop_sessionless(self) -> int:
        """Every entry no session owns (requests that named none, the
        shared system-prompt checkpoints) out of memory."""
        n = sum(self.remove(m, t) for m, t, _e, meta in self.live()
                if meta is None)
        self.shared.clear()
        return n

    def set_pinned(self, session: str, pinned: bool) -> int:
        """Pin or unpin `session` here (its live entries and its later
        ones). Returns its live entries."""
        (self.pinned.add if pinned else self.pinned.discard)(session)
        n = 0
        for meta in self.owners.values():
            if meta["session"] == session:
                meta["pinned"] = bool(pinned)
                n += 1
        return n

    def trim_to(self, n_bytes: int) -> None:
        self.lru.trim_to(n_bytes=n_bytes)

    @property
    def nbytes(self) -> int:
        return int(self.lru.nbytes)


def entry_owner(job, uid=None, kind: str | None = None) -> dict | None:
    """{session, role, run} of the request a cache entry came from, or
    None when it named no session (it owns nothing). X-Cache-Keep: latest
    adds the step (`uid`, the row) and `latest`: the entry replaces the
    session's earlier steps. Its system-prompt checkpoint belongs to no
    session: one copy per distinct prefix, shared by every session that
    starts from it."""
    s = getattr(job, "session", None)
    if not s:
        return None
    latest = bool(getattr(job, "keep_latest", False))
    if latest and kind == "system":
        return {"shared": True}
    out = {"session": s, "role": getattr(job, "role", None),
           "run": getattr(job, "run", None)}
    if latest:
        out.update(step=uid, latest=True)
    return out
