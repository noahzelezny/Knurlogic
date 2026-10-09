"""A ring's prompt cache: JournalPromptCache, the scheduler's PromptCache
on rank 0 with every change journaled, and apply_cache_op, the following
ranks' side of those ops (engine/split/tensor)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knurlogic.engine.split.ring import Journal


class JournalPromptCache:
    """The scheduler's PromptCache on rank 0 of a ring: count-based (no byte
    cap) and every change recorded for the other ranks. Byte trims (the
    memory guard) become pops of the least recently used entries."""

    def __init__(self, inner, journal: Journal):
        self.inner = inner
        self.journal = journal

    @property
    def lru(self):
        return self.inner.lru

    def fetch(self, key, tokens):
        # the admit op carries the hit; fetching changes nothing
        return self.inner.fetch(key, tokens)

    def insert(self, key, tokens, cache, kind: str, origin=None,
               owner: dict | None = None) -> list:
        if origin is None:
            raise ValueError("a ring's prompt cache inserts only from an "
                             "event (origin=(event, uid))")
        files = self.inner.insert(key, tokens, cache, kind, owner=owner)
        event, uid = origin
        # the owner rides along: every rank's side map mirrors rank 0's,
        # so a session's drop, pin or save selects the same entries there
        self.journal.add("insert", uid=int(uid), event=event, kind=kind,
                         **({"owner": owner} if owner else {}))
        return files

    # the side map is rank 0's to read; changes to it are journaled
    @property
    def owners(self) -> dict:
        return self.inner.owners

    @property
    def pinned(self) -> set:
        return self.inner.pinned

    @property
    def shared(self) -> set:
        return self.inner.shared

    def remove(self, model, tokens) -> bool:
        # rank 0's side of a park: the followers free theirs at the `park`
        # op, by the same rule
        return self.inner.remove(model, tokens)

    def live(self) -> list:
        return self.inner.live()

    def of_session(self, session: str) -> list:
        return self.inner.of_session(session)

    def hit_length(self, key, tokens) -> int:
        return self.inner.hit_length(key, tokens)

    def drop(self, session: str) -> int:
        n = self.inner.drop(session)
        self.journal.add("drop", session=session)
        return n

    def drop_sessionless(self) -> int:
        n = self.inner.drop_sessionless()
        self.journal.add("drop_sessionless")
        return n

    def set_pinned(self, session: str, pinned: bool) -> int:
        n = self.inner.set_pinned(session, pinned)
        self.journal.add("pin", session=session, pinned=bool(pinned))
        return n

    def trim_to(self, n_bytes: int) -> None:
        before = len(self.inner.lru)
        self.inner.trim_to(n_bytes)
        popped = before - len(self.inner.lru)
        if popped:
            self.journal.add("pop", n=popped)

    @property
    def nbytes(self) -> int:
        return self.inner.nbytes


def apply_cache_op(op: dict, cache, model_key, last: dict,
                   save_disk, disk_dir=None, disk_key=None) -> bool:
    """A following rank's prompt-cache ops, applied to its own part as rank
    0 applied them to its: insert (with the owner), pop, drop and pin of a
    session (memory and this rank's own files), and the saves. False for
    an op that is not one of them. `last`: the last step's checkpoint and
    finished caches by (event, uid); `save_disk(only_new, select)`: this
    rank's save."""
    from pathlib import Path

    from knurlogic.engine.prompt_cache import disk as prompt_disk
    from knurlogic.engine.split.link import Desync
    kind = op["op"]
    if kind == "insert":
        got = last.get((op["event"], op["uid"]))
        if got is None:
            raise Desync(f"no {op['event']} for row {op['uid']} in "
                         f"the last step")
        # a keep-latest step replaces the session's earlier entries here
        # as on rank 0 (the side maps match): this rank's files go too
        from pathlib import Path
        for f in cache.insert(model_key, got[0], got[1], op["kind"],
                              owner=op.get("owner")) or ():
            try:
                Path(f).unlink()
            except OSError:
                pass
    elif kind == "pop":
        cache.lru.trim_to(n_sequences=len(cache.lru) - op["n"])
    elif kind == "drop":
        cache.drop(op["session"])
        prompt_disk.drop_files(op["session"])
    elif kind == "drop_sessionless":
        cache.drop_sessionless()          # memory: the same entries as rank 0
    elif kind == "park_session":
        # rank 0 parked the session: the same entries saved, then freed --
        # those this rank's save wrote (all of them: a ring has no
        # not-worth-it skip)
        mine = {tuple(t) for _, t in cache.of_session(op["session"])}
        if mine:
            save_disk(only_new=True, select=mine)
        for m, t in cache.of_session(op["session"]):
            meta = cache.owners.get(tuple(t)) or {}
            if meta.get("file") and Path(meta["file"]).exists():
                cache.remove(m, t)
    elif kind == "read_back":
        # rank 0 read a parked or evicted entry back for the request it is
        # about to admit: this rank reads its own part, by the same name.
        # Missing, the ranks would prefill different lengths -- out of step
        got = prompt_disk.read(disk_key, [disk_dir / op["name"]]) \
            if disk_dir is not None and disk_key is not None \
            and "/" not in op["name"] else []
        if not got:
            raise Desync(f"read_back: no {op['name']} on this rank")
        prompt_disk.adopt(cache.owners, cache.pinned, prompt_disk.insert(
            cache.lru, model_key, got), shared=cache.shared)
    elif kind == "drop_files":
        # rank 0's files of a drop, by name: an entry's files carry the same
        # name on every rank, so this rank's part goes too and no rank keeps
        # half an entry (a restore would refuse it on every rank)
        if disk_dir is not None:
            for name in op.get("names") or ():
                if "/" in name or not name.endswith(prompt_disk.SUFFIX):
                    continue
                try:
                    (disk_dir / name).unlink()
                except OSError:
                    pass
    elif kind == "pin":
        cache.set_pinned(op["session"], op["pinned"])
        prompt_disk.set_pin(op["session"], op["pinned"])
    elif kind == "save_cache" and op.get("session"):
        # rank 0's choice, made the same way on the same side map: the
        # session's longest entry
        mine = [tuple(t) for _, t in cache.of_session(op["session"])]
        if mine:
            save_disk(only_new=True, select={max(mine, key=len)})
    elif kind == "save_cache":
        save_disk()
    else:
        return False
    return True
