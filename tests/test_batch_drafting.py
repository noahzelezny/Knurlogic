"""Batch drafting: the gate is token identity against mlx-lm's own
BatchGenerator, greedy, on a tiny random qwen3_5 with a random head.

A random head is rejected almost every step, so this is the rollback path at
its hardest; the small-vocab case forces drafting every step
(EXO_MTP_BATCH_MAX_ROWS) so accepts happen too. Three prompts of different
lengths, admitted one per call, so rows join a batch already decoding.

WHY NOT VOCAB 4. Measured: the verify forward is 2 tokens wide and the plain
one is 1, and through the recurrent kernels that alone moves logprobs by up
to 2e-2 in float32. At vocab 4 a row reached a top-2 margin of 8.6e-4 and
flipped -- a near-tie, not a logic fault (which would diverge at once, at
every vocab). A test that fails on numerics trains people to ignore it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mx = pytest.importorskip("mlx.core")


def _tiny(vocab):
    from knurlogic import register
    register.register("qwen3_5")
    from mlx_lm.models import qwen3_5 as arch
    from knurlogic.mtp.heads.qwen35 import MTPHeadQwen35

    mx.random.seed(0)
    tc = dict(model_type="qwen3_5", hidden_size=128, intermediate_size=256,
              num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=32, vocab_size=vocab,
              linear_num_value_heads=4, linear_num_key_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32,
              full_attention_interval=2, tie_word_embeddings=False)
    model = arch.Model(arch.ModelArgs(model_type="qwen3_5", text_config=tc))
    model.set_dtype(mx.float32)
    head = MTPHeadQwen35(model, arch, norm_shift=0.0)
    D = tc["hidden_size"]
    for n in ("norm_e", "norm_h", "norm_out"):
        setattr(head, n, head._norm(mx.ones((D,)), 0.0))
    head.fc = mx.random.normal((D, 2 * D)) * 0.05
    head.block.set_dtype(mx.float32)
    mx.eval(model.parameters(), head.block.parameters(), head.fc)
    prompts = [mx.random.randint(0, vocab, (n,)).tolist() for n in (37, 9, 70)]
    return model, head, prompts


def _run(gen, prompts, max_tokens, on_finish=None):
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    out, done = {u: [] for u in uids}, set()
    for _ in range(10_000):
        _, responses = gen.next()
        for r in responses:
            assert r.uid not in done, "a response after its finish"
            out[r.uid].append(r.token)
            if r.finish_reason:
                done.add(r.uid)
                if on_finish:
                    on_finish(r)
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids]


@pytest.mark.parametrize("vocab,always", [(512, False), (8, True)])
def test_drafting_batch_is_token_identical_to_mlx_lm(vocab, always,
                                                     monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from knurlogic.mtp.batch_generator import MTPBatchGenerator, trunk_offset

    if always:
        monkeypatch.setenv("EXO_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = _tiny(vocab)
    n_trunk = len(model.make_cache())

    def finished(r):
        # The prompt cache is keyed by all_tokens: it must be exactly what
        # the returned cache holds, and the head must ride beside the trunk
        # or a restore can never draft.
        assert trunk_offset(r.prompt_cache) == len(r.all_tokens)
        assert len(r.prompt_cache) == n_trunk + 1

    plain = _run(BatchGenerator(model, prefill_step_size=16), prompts, 60)
    stats = {}
    draft = _run(MTPBatchGenerator(model, head, stats=stats,
                                   prefill_step_size=16), prompts, 60, finished)
    assert stats["requests"] == 3          # the channel: it went through MTP
    assert stats["steps"] > 0
    if always:
        assert stats["accepted"] > 0       # the accept branch ran, not just reject
    assert draft == plain


def test_a_restored_prefix_keeps_drafting():
    """The entry handed back to the prompt cache, restored at its own length,
    admits WITH its head -- not the silent fresh prefill a misaligned head
    would cause."""
    from knurlogic.mtp.batch_generator import MTPBatchGenerator, split_pool_entry

    model, head, prompts = _tiny(512)
    got = {}
    _run(MTPBatchGenerator(model, head, prefill_step_size=16), prompts[:1], 12,
         on_finish=lambda r: got.update(entry=r.prompt_cache, toks=r.all_tokens))
    trunk, hc, hit = split_pool_entry(got["entry"], len(model.make_cache()),
                                      drafts=True, hit_len=len(got["toks"]))
    assert hit == len(got["toks"]) and hc is not None


def test_the_server_gets_the_drafting_generator_only_for_the_headed_model():
    """Installation is one name swap in the server module. It must decide per
    construction: the head's own model drafts; anything else (a switch to an
    artifact without a head) gets mlx-lm's generator untouched."""
    import types
    from mlx_lm.generate import BatchGenerator
    from knurlogic import engine
    from knurlogic.mtp.batch_generator import MTPBatchGenerator

    model, head, _ = _tiny(512)
    other, _, _ = _tiny(512)
    srv = types.SimpleNamespace(BatchGenerator=BatchGenerator,
                                _make_sampler=lambda args, tok: (lambda x: x))
    saved = dict(engine._DRAFT), dict(engine._SERVED)
    try:
        engine._DRAFT.update(head=head, on=True, batch_installed=False)
        engine._SERVED["provider"] = types.SimpleNamespace(model=model)
        engine._install_batch_drafting(srv)
        g = srv.BatchGenerator(model, prefill_step_size=16)
        assert isinstance(g, MTPBatchGenerator)
        g.close()
        g = srv.BatchGenerator(other, prefill_step_size=16)
        assert type(g) is BatchGenerator
        g.close()
        engine._DRAFT["on"] = False                      # --no-draft
        g = srv.BatchGenerator(model, prefill_step_size=16)
        assert type(g) is BatchGenerator
        g.close()
    finally:
        engine._DRAFT.clear(); engine._DRAFT.update(saved[0])
        engine._SERVED.clear(); engine._SERVED.update(saved[1])


def test_a_built_sampler_carries_the_parameters_verification_needs():
    """insert_segments only ever sees a BUILT sampler. Rejection sampling
    needs the temperature itself, so the sampler has to carry it -- or a
    temp-0.7 request would be verified as greedy."""
    import types
    from knurlogic.mtp.batch_generator import sampling_of, tag_samplers

    srv = types.SimpleNamespace(_make_sampler=lambda args, tok: (lambda x: x))
    tag_samplers(srv)
    s = types.SimpleNamespace(temperature=0.7, top_p=0.9, top_k=20, min_p=0.0,
                              xtc_probability=0.0, xtc_threshold=0.0)
    tok = types.SimpleNamespace(eos_token_id=2, encode=lambda t: [10])
    fn = srv._make_sampler(types.SimpleNamespace(sampling=s), tok)
    got = sampling_of(fn)
    assert got["temp"] == 0.7 and got["top_p"] == 0.9 and got["top_k"] == 20
    assert got["xtc_special_tokens"] == [2, 10]
