"""ModelHost: the one model this process serves, and its state.

    empty -> loading -> ready -> unloading -> empty
                     -> failed (error kept, the next load may try again)

Requests wait for `ready`; they never race a load (mlx-lm answered HTTP
before its model was loaded, and early requests were mis-translated). A
load runs on the scheduler's thread -- the one that owns the MLX stream --
and binds, in order: the weights (knurlogic's VQ runtime for a verified
rung, the artifact's own loader otherwise), vision, the drafting head.

The host answers to the same attribute names as mlx-lm's ModelProvider
(`model`, `tokenizer`, `model_key`, `is_batchable`), and registers itself
as `state.SERVED["provider"]`, so the status, thinking, vision and drafting
code written against the old server reads it unchanged.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Optional

from knurlogic.engine.serve import state

logger = logging.getLogger(__name__)


class ModelHost:
    is_batchable = True        # every request goes through the batch engine

    def __init__(self, *, draft: bool = True,
                 executes_artifact_code: bool = False,
                 image_store_bytes: Optional[int] = None):
        self.draft = draft
        self.image_store_bytes = image_store_bytes
        self.executes_artifact_code = executes_artifact_code
        self.state = "empty"
        self.path: Optional[str] = None
        self.error = ""
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.loaded_at = 0.0
        self._ready = threading.Condition()

    # ---------------------------------------------------------- the machine

    def _set(self, st: str, error: str = "") -> None:
        with self._ready:
            self.state, self.error = st, error
            self._ready.notify_all()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until ready (True) or failed/empty (False) or timeout."""
        end = None if timeout is None else time.time() + timeout
        with self._ready:
            while self.state in ("loading", "unloading"):
                left = None if end is None else end - time.time()
                if left is not None and left <= 0:
                    return False
                self._ready.wait(left)
            return self.state == "ready"

    def expect(self, path: str) -> None:
        """Say a load of `path` is coming (from the HTTP thread, before the
        scheduler picks the command up), so a request arriving now waits
        for it instead of finding the host empty."""
        with self._ready:
            if self.state != "ready" or self.path != path:
                self.path = path
                self.state = "loading"
                state.SERVED["path"] = path

    # ------------------------------------------- scheduler thread only below

    def load(self, path: str, *, executes_artifact_code: bool = None) -> None:
        """`executes_artifact_code`: this artifact ships a runtime that WILL
        run (its own `model_file`) -- stated per load, since each artifact
        answers for itself."""
        path = str(path)
        if executes_artifact_code is not None:
            self.executes_artifact_code = bool(executes_artifact_code)
        if self.state == "ready" and self.path == path:
            return
        if self.model is not None:
            self.unload()
        self.path = path
        state.SERVED.update(path=path, draft=self.draft, provider=self)
        self._set("loading")
        try:
            from knurlogic.machine import loadlock
            with loadlock.model_load(path, "runtime.host"):
                self.model, self.tokenizer = self._weights(path)
            self.model_key = (path, None, None)
            self._bind_vision(path)
            self._bind_head(path)
        except BaseException as e:
            logger.exception("loading %s failed", path)
            self.model = self.tokenizer = self.model_key = None
            self._set("failed", f"{type(e).__name__}: {e}")
            if not isinstance(e, Exception):
                raise
            return
        self.loaded_at = time.time()
        self._set("ready")

    def unload(self) -> None:
        had = self.model is not None
        self._set("unloading")
        from knurlogic.engine.serve import vision
        vision.clear()
        state.DRAFT.update(head=None, spec=None, on=False,
                           why="model unloaded")
        self.model = self.tokenizer = self.model_key = None
        if had:
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        self._set("empty")

    def _weights(self, path: str):
        from knurlogic.engine.serve.load import load_unlocked
        return load_unlocked(path, self.executes_artifact_code)

    def _bind_vision(self, path: str) -> None:
        from knurlogic.engine.serve import vision
        try:
            vision.bind(path, self, store_bytes=self.image_store_bytes)
        except Exception as e:
            # A vision build that fails must not take the text model with
            # it; it is said on /status.json and images get a 400.
            logger.exception("vision did not bind")
            state.VISION.update(serve=None, model=self.model,
                                error=f"{type(e).__name__}: {e}")
            vision.set_spec(None)

    def _bind_head(self, path: str) -> None:
        state.DRAFT.update(head=None, spec=None, on=False)
        if not self.draft:
            state.DRAFT["why"] = "disabled (--no-draft)"
            return
        from knurlogic.engine.serve import drafting
        drafting.load_head(path)

    # --------------------------------------------------------------- status

    def status(self) -> dict:
        out = {"state": self.state, "model": self.path, "error": self.error}
        if self.state == "ready":
            try:
                import mlx.core as mx
                out["memory_bytes"] = int(mx.get_active_memory())
            except Exception:
                pass
        return out
