"""Text embeddings: one embedding model, loaded at launch, dropped after it
has been idle and loaded again by the next request, so it costs
nothing while a big model on the same Mac needs the memory.

The recipe is Qwen3-Embedding's, the one Chuck's memory bench measured
with mlx_embeddings: the text's tokens (its tokenizer appends
<|endoftext|>), the final-normed hidden state of the last one, L2
normalized. ONE text per forward pass: with no padding, the last token is
the text's own. mlx-lm's qwen3 model is the same trunk with the same
weight names, so no embeddings library is needed: `model.model(ids)` is
the normed hidden state.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: How each supported model_type pools a text's hidden states. Qwen3-
#: Embedding takes the last token's (its model card's recipe).
POOLING = {"qwen3": "last"}

#: The longest text embedded, in tokens, its closing <|endoftext|>
#: included: the max_length Chuck's bench embedded with.
MAX_TOKENS = 512

#: How long the model stays loaded with no request, seconds. Reloading
#: ~320 MiB costs a second or two; holding it while a big model is short
#: of memory costs more.
IDLE_S = 600.0

#: How often the idle watch looks, seconds.
WATCH_S = 30.0


@dataclass(frozen=True)
class Embedded:
    """The vectors of a request's texts, in order, and the tokens read."""
    vectors: list
    tokens: int


def refusal(model_type: str) -> str | None:
    """Why a model of `model_type` cannot be served as an embedder, or None."""
    if model_type in POOLING:
        return None
    return (f"no embedding recipe for model_type {model_type!r}; "
            f"supported: {', '.join(sorted(POOLING))}")


def clip(ids: list, max_tokens: int = MAX_TOKENS) -> list:
    """`ids` cut to `max_tokens`, its last token (the tokenizer's closing
    <|endoftext|>) kept last: the truncation mlx_embeddings' tokenizer call
    makes, so a long text pools the same token it would there."""
    if len(ids) <= max_tokens:
        return list(ids)
    return [*ids[:max_tokens - 1], ids[-1]]


def _load(path: str):
    from knurlogic.engine.model import load
    return load(path)


class Embedder:
    """The embedding model at `path`: `embed` loads it when it is not held,
    `drop_if_idle` lets it go. One forward pass at a time (`_lock`)."""

    def __init__(self, path: str, *, idle_s: float = IDLE_S,
                 load: Callable[[str], tuple] = _load,
                 clock: Callable[[], float] = time.monotonic):
        self.path = path
        self.idle_s = idle_s
        self._load_fn = load
        self._clock = clock
        self._lock = threading.Lock()
        self.model: Any = None
        self.tokenizer: Any = None
        self.state = "empty"
        self.error = ""
        self.loaded_at = 0.0
        self.used_at = 0.0

    def status(self) -> dict:
        """{state, error}: empty (not held: the next request loads it),
        loading, ready or failed."""
        return {"state": self.state, "error": self.error}

    def _ensure(self) -> None:
        """Hold the model; the caller holds `_lock`."""
        if self.model is not None:
            return
        self.state, self.error = "loading", ""
        try:
            self.model, self.tokenizer = self._load_fn(self.path)
        except Exception as e:      # said in status and to the request
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"
            raise
        self.state, self.loaded_at = "ready", time.time()
        logger.info("embedding model loaded: %s", self.path)

    def warm(self) -> None:
        """Load now (a launch: the page sees it load, and a broken artifact
        fails at once); idle from now on. A failure is in status()."""
        with self._lock:
            try:
                self._ensure()
            except Exception:       # said in status(); the next request retries
                logger.exception("loading the embedding model failed")
                return
            self.used_at = self._clock()

    def embed(self, texts: list, max_tokens: int = MAX_TOKENS) -> Embedded:
        """Each text's unit vector, one forward pass per text."""
        with self._lock:
            self._ensure()
            vectors, tokens = [], 0
            for text in texts:
                ids = clip(self.tokenizer.encode(text), max_tokens)
                tokens += len(ids)
                vectors.append(self._vector(ids))
            self.used_at = self._clock()
            return Embedded(vectors, tokens)

    def _vector(self, ids: list) -> list:
        import mlx.core as mx
        hidden = self.model.model(mx.array([ids]))[0, -1].astype(mx.float32)
        unit = hidden / mx.maximum(mx.linalg.norm(hidden), 1e-12)
        return unit.tolist()

    def drop_if_idle(self) -> bool:
        """Let the model go when no request has used it for `idle_s`;
        whether it did. A request in flight holds `_lock`, so it is never
        dropped mid-pass."""
        with self._lock:
            if self.model is None or \
                    self._clock() - self.used_at < self.idle_s:
                return False
            self.model = self.tokenizer = None
            self.state = "empty"
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError, RuntimeError):
                pass    # best effort: freeing Metal's cache only helps
            logger.info("embedding model idle %.0f s: dropped", self.idle_s)
            return True

    def watch(self, stop: threading.Event, every_s: float = WATCH_S) -> None:
        """drop_if_idle every `every_s` until `stop` is set (a thread)."""
        while not stop.wait(every_s):
            self.drop_if_idle()
