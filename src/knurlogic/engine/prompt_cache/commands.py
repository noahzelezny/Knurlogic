"""The Scheduler's prompt-cache methods (PromptCacheCommands, a mixin
Scheduler inherits): the commands a client queues (save, drop, pin, park,
list) and what the scheduler thread does for them and for the cache on
disk (save, restore, read back). Command is what is queued."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knurlogic.engine.runtime.scheduler import Job

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """A load or unload for the scheduler thread; `done` is set when it has
    run (or was refused: `error`)."""
    kind: str
    path: str | None = None
    force: bool = True
    executes: bool = False
    #: set when the scheduler has taken it on (or refused it: then `done`)
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: str = ""
    #: what a "save" did (engine/prompt_cache/disk.save's counts)
    result: dict | None = None
    #: a "drop" or "pin"'s session, and a "pin"'s word
    session: str | None = None
    pinned: bool = True


class PromptCacheCommands:
    """Scheduler's prompt-cache methods (engine/runtime/scheduler)."""

    def save_prompt_cache(self, session: str | None = None) -> Command:
        """Queue a save of the prompt cache to disk (prompt_disk.save), on
        the scheduler thread between steps; on a ring every rank saves its
        own part. `session`: only that session's newest entry (its longest),
        if not on disk yet -- what a client asks for right after its context
        compacts. Command.result has the counts."""
        return self._command(Command("save", None, session=session))

    def drop_prompt_cache(self, session: str) -> Command:
        """Queue a drop of `session`'s prompt-cache entries: out of memory
        (on a ring every rank's part, a `drop` op) and its files off disk
        under every model's key. Command.result: {"memory", "disk"}."""
        return self._command(Command("drop", None, session=session))

    def drop_sessionless(self, older_than_s: float | None = None) -> Command:
        """Queue a drop of the loaded model's entries no session owns: its
        files off disk (only those unused for `older_than_s`, when given)
        and, with no age given, its in-memory ones too (every rank's part
        on a ring). Command.result: {"memory", "disk"}."""
        c = Command("drop", None, session=None)
        c.result = {"older_than_s": older_than_s}
        return self._command(c)

    def pin_prompt_cache(self, session: str, pinned: bool) -> Command:
        """Queue a pin (or unpin) of `session`: its files are then never
        swept, only dropped. Command.result: {"session", "pinned",
        "entries"}."""
        return self._command(Command("pin", None, session=session,
                                     pinned=bool(pinned)))

    def park_prompt_cache(self, session: str) -> Command:
        """Queue a park of `session`: its entries saved to disk (what is not
        there yet), then freed from memory; its next request reads them
        back (_read_back). The client's call -- a coordinator parking an
        agent that waits on others. Command.result: {"session", "saved",
        "bytes", "freed", "in_memory"}."""
        return self._command(Command("park", None, session=session))

    def list_prompt_cache(self) -> Command:
        """Queue a listing of the in-memory entries (read between steps).
        Command.result: {"entries": [...], "key_id", "model"}."""
        return self._command(Command("list", None))

    # ------------------------------------------- the prompt cache on disk

    def _save_disk(self, only_new: bool = False,
                   select=None) -> dict | None:
        """The sessions' prompt-cache entries to disk under the loaded
        model's key (engine/prompt_cache/disk); anonymous entries are not
        saved, nor one quicker to recompute than to read back (the
        break-even rule; on a single server only: a ring's ranks must all
        keep the same entries, and each has its own bytes). `only_new`:
        only what is not on disk yet; `select`: these token tuples only.
        Never raises: a save that fails is logged and the unload goes
        on."""
        if self.cache is None:
            return None
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        try:
            key = prompt_disk.host_key(self.host)
            if key is None:
                return None
            self.cache.live()               # prunes the side map
            removed: list = []
            got = prompt_disk.save(
                self.cache.lru, key, owners=self.cache.owners,
                shared=self.cache.shared,
                only_new=only_new, select=select,
                prefill_tps=self._prefill_tps if self.tensor is None
                else None,
                model=Path(self.host.path or "").name or None,
                removed=removed)
            mine = prompt_disk.root() / prompt_disk.key_id(key)
            gone = sorted(f.name for f in removed if f.parent == mine)
            for t in [t for t, v in self._disk_index.items()
                      if Path(v.get("file", "")).name in gone]:
                del self._disk_index[t]
            if self.tensor is not None and gone:
                # the other ranks never sweep: they delete these, by name
                self.tensor.journal.add("drop_files", names=gone)
            for t, meta in self.cache.owners.items():
                if meta.get("file"):
                    self._disk_index[t] = {
                        "file": Path(meta["file"]),
                        "owner": {k: meta[k] for k in
                                  ("session", "role", "run")}}
            return got
        except Exception:  # never fails an unload or a stop (logged)
            logger.exception("saving the prompt cache to disk failed")
            return None

    def _superseded(self, files) -> None:
        """The files of entries a keep-latest session's new step replaced:
        off disk, and out of the read-back index."""
        for f in files or ():
            try:
                Path(f).unlink()
            except OSError:
                pass
        if files:
            gone = {str(f) for f in files}
            for t in [t for t, v in self._disk_index.items()
                      if str(v.get("file")) in gone]:
                del self._disk_index[t]

    def _save_session(self, session: str) -> dict | None:
        """POST /v1/prompt-cache/save {"session"}: that session's newest
        entry -- its longest -- to disk if it is not there yet."""
        mine = [tuple(t) for _, t in self.cache.of_session(session)] \
            if self.cache is not None else []
        if not mine:
            return {"saved": 0, "kept": 0, "skipped": 0, "bytes": 0,
                    "why": [], "entries": 0, "not_worth": 0}
        return self._save_disk(only_new=True, select={max(mine, key=len)})

    def _park(self, session: str, why: str = "parked on request") -> dict:
        """`session`'s entries to disk (what is not there yet), then out of
        memory -- one the save did not write (not worth it, unsaveable)
        stays. Its next request reads them back (_read_back). Returns
        {"saved", "bytes", "freed", "in_memory"}: written now, their size,
        taken out of memory, still in it."""
        mine = {tuple(t) for _, t in self.cache.of_session(session)}
        got = self._save_disk(only_new=True, select=mine) if mine else None
        n = 0
        for m, t in self.cache.of_session(session):
            meta = self.cache.owners.get(tuple(t)) or {}
            if meta.get("file") and Path(meta["file"]).exists():
                n += self.cache.remove(m, t)
        if n:
            logger.info("prompt cache: session %s %s; %d entr%s parked on "
                        "disk", session, why, n, "y" if n == 1 else "ies")
        return {"saved": int((got or {}).get("saved", 0)),
                "bytes": int((got or {}).get("bytes", 0)), "freed": n,
                "in_memory": len(self.cache.of_session(session))}

    def _diverged(self, job: Job, prompt: list, hit: int) -> None:
        """A session's prompt that does not extend its own longest entry:
        where the two part, with a few tokens of each side as text -- a
        client resuming a session (a coordinator session) can see what it rendered
        differently (a turn's reasoning dropped, a header changed) instead
        of an unexplained re-prefill. Logged, and in usage."""
        mine = [tuple(t) for _, t in self.cache.of_session(job.session)] \
            if hasattr(self.cache, "of_session") else []
        if not mine:
            return
        def common(e):
            n = min(len(e), len(prompt))
            return next((i for i in range(n) if e[i] != prompt[i]), n)
        # an entry the prompt extends is a hit (its answer's own entry
        # parting at the re-rendered answer is normal): only when none is
        # does the prompt say something the session did not
        if any(common(e) == len(e) for e in mine):
            return
        e = max(mine, key=lambda e: (common(e), len(e)))
        at = common(e)
        tok = getattr(self.host, "tokenizer", None)

        def text(ts):
            try:
                return tok.decode(list(ts))[:120] if tok is not None else ""
            except Exception:  # a text view only; never the request's failure
                return ""
        job.diverged = {"entry_tokens": len(e), "at": at, "hit": int(hit),
                        "prompt_text": text(prompt[at:at + 32]),
                        "entry_text": text(e[at:at + 32])}
        logger.info("prompt cache: session %s's prompt (%d tokens) leaves "
                    "its %d-token entry at token %d (hit %d): prompt %r vs "
                    "entry %r", job.session, len(prompt), len(e), at, hit,
                    job.diverged["prompt_text"], job.diverged["entry_text"])

    def _read_back(self, job: Job, prompt: list) -> None:
        """Before the admission's hit: when an entry of this model on disk
        is a prefix of `prompt` and longer than the best memory hit, read
        it back into the prompt cache (here, on the scheduler thread), so
        fetch() serves it -- and usage.knurlogic.cache.disk says so, like a
        restored hit. Any on-disk entry, pinned or not. On a ring rank 0
        picks the file and a `read_back` op names it, ahead of the admit:
        every rank reads its own part (the ranks keep the same files: only
        rank 0 sweeps, and names what it deletes)."""
        if not self._disk_index or self._disk_key is None:
            return
        hit = self.cache.hit_length(self.host.model_key, prompt)
        best = None
        for t in self._disk_index:
            n = len(t)
            if n > hit + 1 and n <= len(prompt) and \
                    (best is None or n > len(best)) and \
                    tuple(prompt[:n]) == t:
                best = t
        if best is None:
            return
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        f = self._disk_index[best]["file"]
        got = prompt_disk.read(self._disk_key, [f])
        if not got:
            self._disk_index.pop(best, None)    # gone or corrupt: a miss
            return
        back = prompt_disk.insert(self.cache.lru, self.host.model_key, got)
        prompt_disk.adopt(self.cache.owners, self.cache.pinned, back,
                          shared=self.cache.shared)
        self._restored.update(back)
        if self.tensor is not None:
            self.tensor.journal.add("read_back", name=Path(f).name)

    def _restore_disk(self) -> None:
        """ModelHost.after_bind: what was saved for this model, into the
        new prompt cache, before the warm-up. On a ring a collective (every
        rank restores the same entries, or none)."""
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        try:
            key = prompt_disk.host_key(self.host)
        except Exception:  # a key that cannot be made is a miss (logged)
            logger.exception("prompt cache: no key for the loaded model")
            key = None
        self._restored = prompt_disk.restore(
            self.cache.lru, self.host.model_key, key,
            max_bytes=self.cache_bytes,
            link=self.tensor.link if self.tensor is not None else None)
        prompt_disk.adopt(self.cache.owners, self.cache.pinned,
                          self._restored, shared=self.cache.shared)
        self._disk_key = key
        self._disk_index = {}
        if key is not None and self.tensor is not None:
            # a ring: only what every rank agreed to and restored at this
            # load (their directories may differ from before), then what
            # they all write together -- a read_back of a file one rank
            # lacks would put the ranks out of step
            self._disk_index = {
                t: {"file": Path(v["file"]), "owner": v.get("owner")}
                for t, v in self._restored.items() if v.get("file")}
        elif key is not None:
            try:
                self._disk_index = prompt_disk.index(
                    prompt_disk.root() / prompt_disk.key_id(key))
            except OSError:
                logger.exception("prompt cache: indexing the saved entries")

    # ------------------------------------------ sessions' entries (commands)

    def _cmd_drop(self, c: Command) -> dict:
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        s = c.session
        if s is None:
            return self._drop_sessionless((c.result or {}).get("older_than_s"))
        mem = self.cache.drop(s) if self.cache is not None else 0
        disk = prompt_disk.drop_files(s)
        for t in [t for t, v in self._disk_index.items()
                  if (v.get("owner") or {}).get("session") == s]:
            del self._disk_index[t]
        for t in [t for t, v in self._restored.items()
                  if (v.get("owner") or {}).get("session") == s]:
            del self._restored[t]
        logger.info("prompt cache: session %s dropped (%d in memory, %d on "
                    "disk)", s, mem, disk)
        return {"session": s, "memory": mem, "disk": disk}

    def _drop_sessionless(self, older_than_s) -> dict:
        """The loaded model's session-less entries: on disk (by age when
        asked), and in memory when no age is given -- a live entry's age
        is not its file's."""
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        mem = 0
        if older_than_s is None and self.cache is not None:
            mem = self.cache.drop_sessionless()
        disk, gone = 0, set()
        if self._disk_key is not None:
            disk, gone = prompt_disk.drop_sessionless_files(
                prompt_disk.root() / prompt_disk.key_id(self._disk_key),
                older_than_s)
        for t in [t for t, v in self._disk_index.items()
                  if str(v.get("file")) in gone]:
            del self._disk_index[t]
        if self.tensor is not None and gone:
            # every rank deletes the same names: no rank keeps half an entry
            self.tensor.journal.add(
                "drop_files", names=sorted(Path(f).name for f in gone))
        logger.info("prompt cache: entries with no session dropped (%d in "
                    "memory, %d on disk)", mem, disk)
        return {"sessionless": True, "memory": mem, "disk": disk}

    def _cmd_park(self, c: Command) -> dict:
        got = self._park(c.session, why="parked on request")
        if self.tensor is not None:
            # every rank saves and frees the same entries (park_session);
            # a request on the session reads them back in step (read_back)
            self.tensor.journal.add("park_session", session=c.session)
        return {"session": c.session, **got}

    def _cmd_pin(self, c: Command) -> dict:
        n = self._pin(c.session, c.pinned)
        return {"session": c.session, "pinned": c.pinned, "entries": n}

    def _pin(self, session: str, pinned: bool) -> int:
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        n = self.cache.set_pinned(session, pinned) \
            if self.cache is not None else 0
        prompt_disk.set_pin(session, pinned)
        return n

    def _cmd_list(self, c: Command) -> dict:
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        key = self._disk_key
        out = []
        for _m, t, e, meta in (self.cache.live() if self.cache else []):
            meta = meta or {}
            ints = all(isinstance(x, int) for x in t)
            f = meta.get("file")
            out.append({
                "session": meta.get("session"), "role": meta.get("role"),
                "run": meta.get("run"), "tokens": len(t),
                "bytes": int(e.nbytes), "in_memory": True,
                "on_disk": bool(f) and Path(f).exists(),
                "saved_at": meta.get("saved_at"),
                "pinned": bool(meta.get("pinned")),
                "hash": prompt_disk.tokens_hash(t) if ints else None})
        return {"entries": out,
                "key_id": prompt_disk.key_id(key) if key else None,
                "model": Path(self.host.path or "").name or None}

    def _disk_hit(self, job: Job, prompt: list, used: int) -> None:
        """The entry fetch() handed `job` is one restored from disk: say so
        in its usage (usage.knurlogic.cache.disk). Once the session's next
        answer is cached, the entry it hits is that one, made in memory --
        a memory hit."""
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        src = prompt_disk.source_of(self.cache.lru, self.host.model_key,
                                    prompt, used)
        d = self._restored.get(src) if src is not None else None
        if d is not None:
            job.disk = {"tokens": min(int(d["tokens"]), used),
                        "read_ms": d["read_ms"]}
