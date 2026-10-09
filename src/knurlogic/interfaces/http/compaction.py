"""The chat path through context management (context_management/): every
chat request goes through compaction.prepare, then the summary pass when
one is due (compaction.summarize, run as an in-process completion), then
the engine; a paused compaction warms its prompt into the prompt cache.
CompactingChat is a mixin of server.App.

Design: docs/design/compaction.md.
"""

from __future__ import annotations

import logging
import queue
import threading

from . import openai as O

logger = logging.getLogger(__name__)


class CompactingChat:
    """App's chat entry (interfaces/http/server): context management, then
    the engine."""

    def _generate(self, body: dict) -> dict:
        """One non-streaming completion, in-process (the summary pass).
        A template with no way to turn thinking off is asked again
        without the ask."""
        try:
            job, reply = self.submit(body, chat=True)
        except O.ApiError:
            if "reasoning_effort" not in body:
                raise
            body = {k: v for k, v in body.items() if k != "reasoning_effort"}
            job, reply = self.submit(body, chat=True)
        first = reply.first()
        if first[0] == "error":
            raise O.status_of(first[1])
        return reply.complete(first)

    def _warm(self, body: dict) -> None:
        """Prefill a compacted prompt so it is in the prompt cache before
        the client's next turn asks for it (pause_after_compaction: no
        continuation prefilled it). One token; nobody waits on it."""
        def run():
            try:
                job, reply = self.submit(dict(body, max_tokens=1,
                                              stream=False), chat=True)
                reply.complete(reply.first())
            # a daemon thread; the warm-up is best effort (logged)
            except Exception as e:
                logger.debug("warming the compacted prompt: %s", e)
        threading.Thread(target=run, daemon=True).start()

    def chat(self, body: dict):
        """A chat request through context management, then the engine:
        ("json", completion) or ("stream", SSE byte iterator); ApiError to
        refuse. A request that needs a summary pass and streams is
        answered 200 at once and kept alive while the summary is written;
        a later failure is an error event."""
        from knurlogic.context_management import compaction as C
        from knurlogic.context_management import context_edits as E

        from . import server as S
        if not isinstance(body, dict):
            raise O.ApiError(400, "the request body must be a JSON object")
        try:
            run, out, pending = C.prepare(
                body, count=self.count,
                window=self.window() if (body.get("context_management")
                                         or C.settings()["auto"]) else 0)
        except E.EditError as e:
            raise O.ApiError(400, str(e), param="context_management") from e
        except ValueError:
            # a template that cannot render this history is the engine's
            # to refuse, as it would without context management
            run, out, pending = dict(body), C.Outcome(), None
            run.pop("context_management", None)

        def extra():
            return {"compaction": out.compaction, "applied": out.applied,
                    "iteration": out.iteration}

        def pause_doc():
            import time
            import uuid
            self._warm(run2[0])
            return C.paused(out, id_=f"chatcmpl-{uuid.uuid4().hex}",
                            created=int(time.time()),
                            model=self.served().get("id", "")
                            or body.get("model") or "default",
                            context={"tokens": 0, "window": self.window()})

        run2 = [run]
        if pending is not None and not body.get("stream"):
            run2[0] = C.summarize(run, pending, out, self._generate)
            if out.pause:
                return "json", pause_doc()
            pending = None
        if pending is None:
            job, reply = self.submit(run2[0], chat=True, extra=extra())
            if reply.ctx["stream"]:
                try:
                    first = reply.first(timeout=S.PROGRESS_S
                                        if reply.ctx.get("progress")
                                        else S.QUEUED_KEEPALIVE_S)
                except queue.Empty:
                    return "stream", S.queued(job, reply, self.scheduler)
            else:
                first = reply.first()
            if first[0] == "error":
                raise O.status_of(first[1])
            if not reply.ctx["stream"]:
                return "json", reply.complete(first)

            def events():
                try:
                    yield from reply.events(first)
                finally:
                    job.cancel()         # abandoned mid-stream: free the row
            return "stream", events()

        def summarized():
            box = {}

            def work():
                try:
                    box["run"] = C.summarize(run, pending, out,
                                             self._generate)
                except BaseException as e:  # any end of the worker becomes the reply
                    box["error"] = e
            t = threading.Thread(target=work, daemon=True)
            t.start()
            while t.is_alive():
                t.join(5)
                if t.is_alive():
                    yield b": keepalive compaction\n\n"
            if "error" in box:
                yield O.sse_data(O.status_of(box["error"]).body())
                yield b"data: [DONE]\n\n"
                return
            run2[0] = box["run"]
            if out.pause:
                doc = pause_doc()
                msg = doc["choices"][0]["message"]
                base = {"id": doc["id"], "object": "chat.completion.chunk",
                        "created": doc["created"], "model": doc["model"]}
                yield O.sse_data(dict(base, choices=[{
                    "index": 0, "finish_reason": None,
                    "delta": {"role": "assistant",
                              "compaction": msg["compaction"]}}]))
                yield O.sse_data(dict(base, choices=[],
                                   context_management=doc[
                                       "context_management"]))
                yield O.sse_data(dict(base, choices=[{
                    "index": 0, "finish_reason": "compaction",
                    "delta": {}}]))
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield O.sse_data(dict(base, choices=[],
                                       usage=doc["usage"]))
                yield b"data: [DONE]\n\n"
                return
            try:
                job, reply = self.submit(run2[0], chat=True, extra=extra())
            except O.ApiError as e:
                yield O.sse_data(e.body())
                yield b"data: [DONE]\n\n"
                return
            try:
                first = reply.first()
                if first[0] == "error":
                    yield O.sse_data(O.status_of(first[1]).body())
                    yield b"data: [DONE]\n\n"
                    return
                yield from reply.events(first)
            finally:
                job.cancel()
        return "stream", summarized()
