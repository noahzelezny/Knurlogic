"""Images in the batch engine: G10, G11 and the admission rules under them.

G10 (docs/design/vision.md) -- an image prompt with the drafting machinery
equals it without. In Phase A an image row never drafts, so on the image row
this compares plain decoding with plain decoding and passes by construction
: it is a SMOKE test, labelled as one. What it does pin is that
a head in the generator changes nothing for the image row while a text row
beside it still drafts.

G11 -- three rows, image and text mixed, admitted one per call into a batch
already decoding: each equals its solo run. Run twice: with the shared stub
(1D positions) and with a positions family (per-row rope_delta), which is
the batched-decode half of design risk 1.

Plus the two admission rules only `admit` enforces: prefill chunk edges
never cut an image block (D5), and a vision row keeps a trunk prefix hit
rather than dropping it to draft (G8 with a head).

Tiny fixtures only: the qwen3_5 of tests/test_batch_drafting.py (float32,
seed 0, vocab 512) and the shared StubFamily.
"""
import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402
from test_image_cache import (IMG, PosFamily, PosModel,  # noqa: E402
                              tiny_model)

from knurlogic.engine.vision import key as K  # noqa: E402


def _vision(fam):
    from knurlogic.engine.vision.request import VisionServe
    from knurlogic.engine.vision.store import ImageStore
    return VisionServe(fam, ImageStore(), ("tiny", None, None))


def _url(seed, w=16, h=12):
    b = fv.png_bytes(fv.tiny_image(w, h, seed))
    return "data:image/png;base64," + base64.b64encode(b).decode()


def _key(vis, text_ids, *seeds, tail=(7, 8, 9)):
    """ids with one image per seed after the text, then a short tail, as a
    cache key -- the way the tokenize wrap builds it (ensure pins each
    image; the generator releases at admit)."""
    refs = [vis.ensure(_url(s)) for s in seeds]
    ids = list(text_ids) + [IMG] * len(refs) + list(tail)
    return K.expand(K.expand_pads(ids, refs, IMG), refs, IMG)


def _run(gen, prompts, max_tokens=16):
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    out, done, fin = {u: [] for u in uids}, set(), {}
    for _ in range(10_000):
        _, responses = gen.next()
        for r in responses:
            assert r.uid not in done, "a response after its finish"
            out[r.uid].append(r.token)
            if r.finish_reason:
                done.add(r.uid)
                fin[r.uid] = r
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids], [fin[u] for u in uids]


def _gen(model, head, vis, **kw):
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    kw.setdefault("prefill_step_size", 4)
    return MTPBatchGenerator(model, head, vision=vis, **kw)


# --- G11 --------------------------------------------------------------------

def _mixed(vis):
    text = list(range(30, 52))
    return [_key(vis, text[:9], 1),                  # image row
            list(range(60, 97)),                     # text row
            _key(vis, text, 2, 3, tail=(5, 6))]      # two images


def test_g11_mixed_rows_each_equal_their_solo_run():
    model = tiny_model()
    fam = fv.StubFamily(IMG, 128)
    vis = _vision(fam)
    prompts = _mixed(vis)
    together, fins = _run(_gen(model, None, vis), prompts)
    solo = [_run(_gen(model, None, vis), [p])[0][0] for p in prompts]
    assert together == solo
    # the prompt cache is keyed by the KEY: sentinels kept, not ids
    for p, f in zip(prompts, fins):
        assert f.all_tokens[:len(p)] == p
    assert vis.pinned_count() == 0


def test_g11_per_row_rope_delta_in_a_batch():
    """With positions (a Qwen-shaped family), each row decodes at its own
    count + rope_delta. Mixed with a text row (no delta) and a two-image
    row (twice the delta): a batch that used one row's delta for all, or
    dropped it in decode, answers differently from the solo runs."""
    pm = PosModel(tiny_model())
    fam = PosFamily(IMG, 128, embed_fn=pm.embed_tokens)
    vis = _vision(fam)
    prompts = _mixed(vis)
    together, _ = _run(_gen(pm, None, vis), prompts)
    solo = [_run(_gen(pm, None, vis), [p])[0][0] for p in prompts]
    assert together == solo


# --- G10 (smoke, Phase A) ---------------------------------------------------

def test_g10_drafting_machinery_leaves_image_rows_alone(monkeypatch):
    from test_batch_drafting import _tiny
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")     # draft every step
    model, head, _ = _tiny(512)
    fam = fv.StubFamily(IMG, 128)
    vis = _vision(fam)
    img = _key(vis, range(30, 41), 4)
    text = list(range(60, 90))

    plain, _ = _run(_gen(model, None, vis), [img, text])
    img2 = _key(vis, range(30, 41), 4)                    # re-pin for run 2
    assert img2 == img
    stats = {}
    drafted, _ = _run(_gen(model, head, vis, stats=stats), [img, text])
    assert drafted == plain

    alone = {}
    _key(vis, range(30, 41), 4)
    _run(_gen(model, head, vis, stats=alone), [img])
    assert alone["requests"] == 1 and alone.get("steps", 0) == 0   # never drafts
    assert stats.get("steps", 0) > 0                   # the text row did
    assert vis.pinned_count() == 0


def test_a_vision_row_keeps_its_trunk_hit_with_a_head():
    """Image rows never seed the head, so their prompt-cache entries carry
    no head cache. Turn 2 must reuse the trunk KV anyway (not drafting)
    instead of throwing the image's KV away to draft -- G8 with a head."""
    from test_batch_drafting import _tiny
    model, head, _ = _tiny(512)
    vis = _vision(fv.StubFamily(IMG, 128))
    k1 = _key(vis, range(30, 41), 4)
    _, fins = _run(_gen(model, head, vis), [k1], max_tokens=6)
    entry, toks = fins[0].prompt_cache, fins[0].all_tokens
    assert len(entry) == len(model.make_cache())            # trunk only
    turn2 = list(toks) + [11, 12, 13]
    vis._pin(K.images_in(turn2))                             # tokenize's pin
    g = _gen(model, head, vis)
    g.insert_segments([[turn2[len(toks):]]], max_tokens=[4], caches=[entry],
                      all_tokens=[list(toks)])
    before = g._prompt_tokens_counter
    g.next()
    assert g._prompt_tokens_counter - before == 3            # suffix only
    assert g._batch.drafts == [False]
    g.close()


# --- D5: chunk edges never cut an image -------------------------------------

class Spy:
    """The tiny trunk, recording each forward's [start, end)."""

    def __init__(self, inner):
        self.inner, self.edges, self.pos = inner, [], 0

    def make_cache(self):
        return self.inner.make_cache()

    def __call__(self, inputs, cache=None, **kw):
        n = inputs.shape[1]
        self.edges.append((self.pos, self.pos + n))
        self.pos += n
        return self.inner(inputs, cache=cache, **kw)


@pytest.mark.parametrize("spans,n", [([(5, 12)], 20), ([(2, 4), (6, 17)], 20),
                                     ([(15, 20)], 20), ([(0, 9)], 12)])
def test_admit_snaps_chunks_to_image_blocks(spans, n):
    from knurlogic.engine.mtp.batch_loop import RowParams, admit
    spy = Spy(tiny_model())
    ids = mx.array(list(range(40, 40 + n)))
    params = RowParams(max_tokens=4, dist=None, processors=[], eos=set(),
                       drafts=False)
    admit(spy, None, lambda: None, ids, params, uid=0,
          make_draft_cache=lambda: None, prefill_step_size=4,
          chunk_boundaries=spans)
    assert spy.edges[0][0] == 0 and spy.edges[-1][1] == n
    for a, b in spy.edges:
        for s, e in spans:
            assert not (s < a < e), f"chunk {a, b} starts inside {s, e}"
            assert not (s < b < e), f"chunk {a, b} ends inside {s, e}"


def test_admit_feeds_embeddings_slice_by_slice():
    """Embeddings for ids[start:] arrive once; every chunk gets its slice.
    Same logits as the trunk embedding the ids itself."""
    from knurlogic.engine.mtp.batch_loop import RowParams, admit
    model = tiny_model()
    ids = mx.array(list(range(40, 63)))
    emb = model.language_model.model.embed_tokens(ids[None])
    params = RowParams(max_tokens=4, dist=None, processors=[], eos=set(),
                       drafts=False)
    kw = dict(uid=0, make_draft_cache=lambda: None, prefill_step_size=5)
    a = admit(model, None, lambda: None, ids, params, **kw)
    b = admit(model, None, lambda: None, ids, params, embeds=emb, **kw)
    assert mx.allclose(a.row_t1, b.row_t1, atol=1e-5).item()
