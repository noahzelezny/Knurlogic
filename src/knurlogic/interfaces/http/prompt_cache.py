"""/v1/prompt-cache: save, drop, pin, park and list the prompt cache
(engine/prompt_cache). PromptCacheHandlers is a mixin of server.Handler."""

from __future__ import annotations

import json
import os

from . import openai as O


class PromptCacheHandlers:
    """Handler's /v1/prompt-cache routes (interfaces/http/server)."""

    def _save_prompt_cache(self, raw: bytes = b"") -> None:
        """POST /v1/prompt-cache/save [{"session"}]: the prompt cache to
        disk now (engine/prompt_cache/disk), between steps; on a ring every
        rank saves its part. With a session, only its newest (longest)
        entry, if not on disk yet: a client calls it right after its
        context compacts. The loopback operator only."""
        if not self._loopback():
            return self._json(403, {"error": {
                "message": "the prompt cache is saved from this machine "
                           "(loopback) only", "type": "permission_error"}})
        session = None
        if raw and raw.strip():
            try:
                body = json.loads(raw)
            except ValueError:
                return self._error(O.ApiError(400, "body must be JSON"))
            s = body.get("session") if isinstance(body, dict) else None
            if s is not None and not (isinstance(s, str) and s):
                return self._error(O.ApiError(400, '"session" is a string'))
            if s:
                from knurlogic.machine import ledger as L
                session = L.label(s, L.LABELS["X-Client-Session"][1])
        cmd = self.app.scheduler.save_prompt_cache(session) if session \
            else self.app.scheduler.save_prompt_cache()
        if not cmd.done.wait(600):
            return self._error(O.ApiError(504, "the save did not finish "
                                               "within 600 s"))
        if cmd.error:
            return self._error(O.ApiError(409, cmd.error))
        r = dict(cmd.result or {})
        r.pop("why", None)
        return self._json(200, {"object": "prompt_cache.save",
                                "session": session,
                                "model": os.path.basename(
                                    self.app.scheduler.host.path or "")
                                or None, **r})

    def _cache_command(self, cmd, what: str):
        """Wait out a prompt-cache command; its result, or None when an
        error was answered."""
        if not cmd.done.wait(600):
            self._error(O.ApiError(504, f"the {what} did not finish within "
                                        "600 s"))
            return None
        if cmd.error:
            self._error(O.ApiError(409, cmd.error))
            return None
        return dict(cmd.result or {})

    def _session_body(self, raw: bytes):
        """The body's "session" (and the whole body), or None when a 400
        was answered."""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        s = body.get("session") if isinstance(body, dict) else None
        if not isinstance(s, str) or not s:
            self._error(O.ApiError(400, 'the body names a "session"'))
            return None
        from knurlogic.machine import ledger as L
        return L.label(s, L.LABELS["X-Client-Session"][1]), body

    def _refused_remote(self, what: str) -> bool:
        if self._loopback():
            return False
        self._json(403, {"error": {
            "message": f"the prompt cache is {what} from this machine "
                       "(loopback) only", "type": "permission_error"}})
        return True

    def _drop_prompt_cache(self, raw: bytes) -> None:
        """POST /v1/prompt-cache/drop {"session"}: that session's entries
        out of memory (every rank's part on a ring) and its files off disk
        under every model's key. {"sessionless": true[, "older_than_s"]}:
        the loaded model's entries no session owns (calls that named none,
        the shared system-prompt copies) -- with an age, only its files
        unused that long. The loopback operator only."""
        if self._refused_remote("dropped"):
            return
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("sessionless") is True:
            age = body.get("older_than_s")
            if age is not None and (isinstance(age, bool) or
                                    not isinstance(age, (int, float))
                                    or age < 0):
                return self._error(O.ApiError(
                    400, '"older_than_s" is a number of seconds'))
            r = self._cache_command(
                self.app.scheduler.drop_sessionless(age), "drop")
            if r is not None:
                self._json(200, {"object": "prompt_cache.drop", **r})
            return
        got = self._session_body(raw)
        if got is None:
            return
        r = self._cache_command(
            self.app.scheduler.drop_prompt_cache(got[0]), "drop")
        if r is not None:
            self._json(200, {"object": "prompt_cache.drop", **r})

    def _pin_prompt_cache(self, raw: bytes) -> None:
        """POST /v1/prompt-cache/pin {"session", "pinned"}: a pinned
        session's files are never swept (TTL nor budget). The loopback
        operator only."""
        if self._refused_remote("pinned"):
            return
        got = self._session_body(raw)
        if got is None:
            return
        pinned = got[1].get("pinned", True)
        if not isinstance(pinned, bool):
            return self._error(O.ApiError(400, '"pinned" is true or false'))
        r = self._cache_command(
            self.app.scheduler.pin_prompt_cache(got[0], pinned), "pin")
        if r is not None:
            self._json(200, {"object": "prompt_cache.pin", **r})

    def _park_prompt_cache(self, raw: bytes) -> None:
        """POST /v1/prompt-cache/park {"session"}: that session's entries
        saved to disk (what is not there yet), then freed from memory; its
        next request reads them back. The loopback operator only."""
        if self._refused_remote("parked"):
            return
        got = self._session_body(raw)
        if got is None:
            return
        r = self._cache_command(
            self.app.scheduler.park_prompt_cache(got[0]), "park")
        if r is not None:
            self._json(200, {"object": "prompt_cache.park", **r})

    def _list_prompt_cache(self) -> None:
        """GET /v1/prompt-cache: every entry, in memory (this model's, read
        between steps) and on disk (every model's, from file headers). One
        in both shows once with both flags. The loopback operator only."""
        if self._refused_remote("read"):
            return
        from knurlogic.engine.prompt_cache import disk as prompt_disk
        mem = self._cache_command(self.app.scheduler.list_prompt_cache(),
                                  "listing")
        if mem is None:
            return
        kid, model = mem.get("key_id"), mem.get("model")
        rows, at = [], {}
        for e in mem.get("entries") or []:
            row = dict(e, model=model, key_id=kid)
            row["on_disk"] = False          # the disk side below says
            if e.get("hash"):
                at[(kid, e["hash"])] = row
            rows.append(row)
        for d in prompt_disk.list_disk():
            row = at.get((d["key_id"], d["hash"]))
            if row is not None:
                row["on_disk"] = True
                row["saved_at"] = row.get("saved_at") or d["saved_at"]
                row["pinned"] = bool(row["pinned"] or d["pinned"])
                continue
            rows.append({"session": d["session"], "role": d["role"],
                         "run": d["run"], "tokens": d["tokens"],
                         "bytes": d["bytes"], "in_memory": False,
                         "on_disk": True, "saved_at": d["saved_at"],
                         "pinned": d["pinned"], "hash": d["hash"],
                         "model": d["model"], "key_id": d["key_id"]})
        return self._json(200, {"object": "list", "data": rows})
