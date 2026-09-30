"""ModelHost: the one model this process serves, and its state.

    empty -> loading -> ready -> unloading -> empty
                     -> failed (error kept, the next load may try again)

Requests wait for `ready`; they never race a load. A load runs on the
scheduler's thread -- the one that owns the MLX stream -- and binds, in
order: the weights (knurlogic's VQ runtime for a verified rung, the
artifact's own loader otherwise), vision, the drafting head.

The host answers to mlx-lm's ModelProvider attribute names (`model`,
`tokenizer`, `model_key`, `is_batchable`) and registers itself as
`state.SERVED["provider"]`, so status, thinking, vision and drafting code
reads it unchanged.
"""

from __future__ import annotations

import gc
import logging
import threading
import time

from knurlogic.engine.serve import state

logger = logging.getLogger(__name__)


class ModelHost:
    is_batchable = True        # every request goes through the batch engine

    def __init__(self, *, draft: bool = True,
                 executes_artifact_code: bool = False,
                 image_store_bytes: int | None = None,
                 shard=None, vision: bool = True, load_wait_s: float = 0.0,
                 head_agree=None, kv_bits: int | None = None,
                 cross_chip: dict | None = None, tower: bool = True):
        """`shard(model)`: split the weights in place before they are read
        (a tensor ring: loaded lazily, split, then evaluated, so a rank
        never holds the whole model). `vision=False` binds no tower;
        `tower=False` binds the vision family without one (a follower rank
        of a split model: rank 0 encodes and ships the image rows).
        `load_wait_s`: how long to wait for the machine's load lock (ranks
        of one ring on one machine load one after another).
        `kv_bits`: store attention K/V at 8/6/4 bits (engine/kvquant.py);
        a model with no cache that can be is refused, not served bf16."""
        self.kv_bits = kv_bits
        #: engine/crosschip.resolve(...) for this job, None = off
        self.cross_chip = cross_chip
        self.shard = shard
        #: head_agree(bound: bool) -> bool, called after the head binds (or
        #: does not) on every load of a pipeline rank: rank 0's answer,
        #: told to every rank -- only rank 0 holds a head, and the others
        #: follow its drafting steps (engine/runtime/tensor.agree_head)
        self.head_agree = head_agree
        self.vision = vision
        self.tower = tower
        self.load_wait_s = load_wait_s
        self.draft = draft
        self.image_store_bytes = image_store_bytes
        self.executes_artifact_code = executes_artifact_code
        self.state = "empty"
        self.path: str | None = None
        self.error = ""
        self.model = None
        self.tokenizer = None
        self.model_key: tuple | None = None
        self.loaded_at = 0.0
        self._ready = threading.Condition()

    # ---------------------------------------------------------- the machine

    def _set(self, st: str, error: str = "") -> None:
        with self._ready:
            self.state, self.error = st, error
            self._ready.notify_all()

    def wait_ready(self, timeout: float | None = None) -> bool:
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
            # A ring's collectives (the pipeline split's dtype gather, the
            # head agreement) run OUTSIDE the machine's load lock: ranks of
            # one ring on one machine take it in turn, and a rank holding
            # it inside a collective waits forever on a sibling waiting for
            # the lock. Split lazily (no weights read) before the lock.
            t0 = time.monotonic()
            lazy = self._split_lazily(path) if self.shard else None
            with loadlock.model_load(path, "runtime.host",
                                     wait_s=self.load_wait_s):
                t1 = time.monotonic()
                self.model, self.tokenizer = self._weights(path, lazy)
                t2 = time.monotonic()
                self._quantize_kv()
                self._cross_chip()
                self.model_key = (path, None, None)
                self._bind_vision(path)
                self._bind_head(path)
                t3 = time.monotonic()
            # each step's time, so a slow load says which step was slow
            # (the read itself, the lock, or vision and the drafting head)
            gib = _active_gib()
            logger.info("loaded %s: weights %.1f GiB in %.1fs (%.2f GiB/s), "
                        "lock + split %.1fs, vision + head %.1fs", path, gib,
                        t2 - t1, gib / max(t2 - t1, 1e-3), t1 - t0, t3 - t2)
            if self.head_agree is not None:
                bound = bool(state.DRAFT.get("on"))
                if not self.head_agree(bound) and bound:
                    state.DRAFT.update(head=None, on=False,
                                       why="not every rank of the pipeline "
                                           "bound a drafting head")
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
            except (ImportError, AttributeError, RuntimeError):
                pass    # best effort: freeing Metal's cache is only an optimisation
        self._set("empty")

    def _split_lazily(self, path: str):
        """A ring's model, loaded lazily and split in place: nothing read
        yet, and the split's collectives done."""
        from knurlogic.engine.serve.load import load_unlocked
        model, tok = load_unlocked(path, self.executes_artifact_code,
                                   lazy=True)
        self.shard(model)
        return model, tok

    def _weights(self, path: str, lazy=None):
        """`lazy`: _split_lazily's (model, tokenizer), evaluated here."""
        from knurlogic.engine.serve.load import load_unlocked
        if lazy is None:
            return load_unlocked(path, self.executes_artifact_code)
        import mlx.core as mx
        model, tok = lazy
        mx.eval(model.parameters())
        return model, tok

    def _cross_chip(self) -> None:
        """Pad 9-31-row quantized matmuls to 32 (engine/crosschip.py) so
        a split across GPU architectures rounds identically everywhere.
        Process-wide and idempotent."""
        from knurlogic.engine import crosschip
        cc = dict(self.cross_chip or {"on": False, "why": "not set"})
        if cc.get("on"):
            crosschip.install()
            logger.info("cross-chip: %s", crosschip.describe(cc))
        cc["installed"] = crosschip.installed()
        state.SERVED["cross_chip"] = cc

    def _quantize_kv(self) -> None:
        state.SERVED["kv_bits"] = self.kv_bits
        if self.kv_bits is None:
            return
        from knurlogic.engine import kvquant
        n = kvquant.install(self.model, self.kv_bits)
        if n == 0:
            raise RuntimeError(
                f"KV cache at {self.kv_bits} bits: this model has no "
                f"attention cache engine/kvquant.py can store quantized "
                f"(its family's own cache classes); launch it at bf16")
        logger.info("KV cache: %d attention layers at %d bits", n,
                    self.kv_bits)

    def _bind_vision(self, path: str) -> None:
        from knurlogic.engine.serve import vision
        if not self.vision:
            state.VISION.update(serve=None, model=self.model,
                                error="vision is off for this instance")
            vision.set_spec(None)
            return
        try:
            vision.bind(path, self, store_bytes=self.image_store_bytes,
                        tower=self.tower)
        # a vision build must not take the text model with it (logged, on /status.json)
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
            state.DRAFT["why"] = "disabled (KNURLOGIC_MTP=off or --no-draft)"
            return
        from knurlogic.engine.serve import drafting
        drafting.load_head(path)

    # --------------------------------------------------------------- status

    def status(self) -> dict:
        out: dict = {"state": self.state, "model": self.path,
                     "error": self.error}
        if self.state == "ready":
            try:
                import mlx.core as mx
                out["memory_bytes"] = int(mx.get_active_memory())
            except (ImportError, AttributeError, RuntimeError):
                pass    # best effort: status still answers without the memory figure
        return out


def _active_gib() -> float:
    try:
        import mlx.core as mx
        return mx.get_active_memory() / (1 << 30)
    except (ImportError, AttributeError, RuntimeError):
        return 0.0
