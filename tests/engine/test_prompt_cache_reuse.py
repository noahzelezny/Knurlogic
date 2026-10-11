"""KNURLOGIC_PROMPT_CACHE_REUSE (the benchmark preset turns it off): off,
a request that would hit the prompt cache reads its prompt in full, and a
load restores nothing saved; on (the default), the same requests hit.
Saving is unchanged. Tiny fixtures in one process."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import memory as M  # noqa: E402


def test_the_setting_reads_on_unless_off():
    assert M.reuse_on({}) and M.reuse_on({M.REUSE_ENV: "on"})
    assert not M.reuse_on({M.REUSE_ENV: "off"})
    assert M.reuse_on({M.REUSE_ENV: "junk"})


def test_off_the_in_memory_cache_never_hits(monkeypatch):
    from mlx_lm.models.cache import KVCache
    monkeypatch.setenv(M.REUSE_ENV, "off")
    c = M.PromptCache(4)
    assert c.reuse is False
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 1, 5, 8)), mx.zeros((1, 1, 5, 8)))
    c.insert("m", [1, 2, 3, 4, 5], [kv], "assistant")
    assert c.hit_length("m", [1, 2, 3, 4, 5, 6]) == 0
    cache, rest = c.fetch("m", [1, 2, 3, 4, 5, 6])
    assert cache is None and rest == [1, 2, 3, 4, 5, 6]
    c.reuse = True
    assert c.hit_length("m", [1, 2, 3, 4, 5, 6]) == 5


@pytest.mark.parametrize("reuse", ["on", "off"])
def test_a_request_that_would_hit_does_not_when_reuse_is_off(
        tmp_path, monkeypatch, reuse):
    from test_batch_drafting import _tiny
    from test_prompt_disk import Host
    from test_scheduler import Tok, _collect, _job

    from knurlogic.engine.prompt_cache import location as LOC
    from knurlogic.engine.runtime.scheduler import Scheduler
    monkeypatch.setenv(D.ENV_ON, "on")
    monkeypatch.setattr(LOC, "fallback", lambda: tmp_path)
    monkeypatch.setattr(LOC, "root_for", lambda path: (tmp_path, "test"))
    model, _head, prompts = _tiny(512)
    s = Scheduler(Host(model, Tok(prompts)), prefill_step_size=16).start()
    path = "/nonexistent/tiny-reuse"
    p = prompts[2]
    # saved by a launch that reuses
    s.load(path).done.wait(60)
    _collect(s.submit(_job(p, max_tokens=4, session="r1")))
    assert s.save_prompt_cache().done.wait(60)
    s.unload().done.wait(60)
    assert D.list_disk(tmp_path)
    monkeypatch.setenv(M.REUSE_ENV, reuse)
    try:
        s.load(path).done.wait(60)
        # a saved entry would be restored at load and hit
        _, first = _collect(s.submit(_job(p + [5, 6, 7], max_tokens=4,
                                          session="r1")))
        # and an earlier request's prefix in memory would hit
        _, second = _collect(s.submit(_job(p + [5, 6, 7, 8, 9],
                                           max_tokens=4)))
        if reuse == "on":
            assert first["knurlogic"]["cache"]["disk"]["tokens"] > 0
            assert second["prompt_tokens_details"]["cached_tokens"] > 0
        else:
            assert first["prompt_tokens_details"]["cached_tokens"] == 0
            assert first["knurlogic"]["cache"]["disk"]["tokens"] == 0
            assert second["prompt_tokens_details"]["cached_tokens"] == 0
            # saving is unchanged
            assert s.save_prompt_cache().done.wait(60)
            assert D.list_disk(tmp_path)
    finally:
        s.unload().done.wait(60)
