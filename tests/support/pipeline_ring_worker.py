"""One rank of tests/test_pipeline.py's two-process ring (not a test module).

    logits <family> <counts>   the tiny model split by layer runs
                               (counts: layers per rank, rank order) runs a
                               prefill + fixed decode tokens; rank 0 also
                               runs the unsplit model and writes both logits
    mtp <always>               the tiny qwen3_5 with a random MTP head in
                               rank 0's batch engine, rank 0 drafting and
                               sampling, the follower silenced and holding
                               no head (it runs the verify rows); rank 0 also
                               runs the unsplit engine and writes both token
                               streams and every rank's broadcast counts

Run with MLX_RANK and MLX_HOSTFILE set."""
import json
import os
import sys

import numpy as np


def build_glm():
    import mlx.core as mx
    from fixtures_vision_glm5 import glm5_tiny_config
    from test_vision_e2e import GLM_TEXT

    from knurlogic.engine import register
    register.register("glm5_next")
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    mx.random.seed(0)
    model = LanguageModel(TextConfig.from_dict(dict(
        glm5_tiny_config()["text_config"], **GLM_TEXT)))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def build_deepseek_v4():
    """The tiny random DeepSeek-V4 golden (float32), through mlx-lm's
    loader over knurlogic's vendored module."""
    from pathlib import Path

    from mlx_lm.utils import load_model

    from knurlogic.engine import register
    register.register("deepseek_v4")
    here = Path(__file__).parent / "goldens" / "deepseek_v4_tiny"
    return load_model(here)[0]


def build(family, seed=0, dtype="float32"):
    import importlib
    if family == "glm5_next":
        return build_glm()
    if family == "deepseek_v4":
        return build_deepseek_v4()

    import fixtures_vision_qwen as FQ
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from knurlogic.engine import register
    register.register(family, override=True)
    m = importlib.import_module(f"mlx_lm.models.{family}")
    # qwen4_exp's defaults are full size; its tiny config is the fixture's
    cfg = FQ.config(family) if family == "qwen4_exp" else \
        dict(FQ.TEXT[family], model_type=family)
    model = m.Model(m.ModelArgs.from_dict(cfg))
    shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
    w = FQ.init_weights(shapes, seed)
    model.update(tree_unflatten([(k, mx.array(v).astype(getattr(mx, dtype)))
                                 for k, v in w.items()]))
    mx.eval(model.parameters())
    return model


def run(model, ids, then):
    import mlx.core as mx
    cache = model.make_cache()

    def call(x):
        y = model(mx.array(x), cache=cache)
        return (y if isinstance(y, mx.array) else y.logits)[:, -1]
    out = [call([ids])]
    for t in then:
        out.append(call([[t]]))
    y = mx.concatenate(out).astype(mx.float32)
    mx.eval(y)
    return np.array(y)


def logits(link, out_path, family, counts):
    from knurlogic.engine.runtime import pipeline as PL
    ids = [5, 17, 3, 99, 42, 7, 64, 11, 23]
    then = [31, 104, 331, 32, 439, 214]
    if family == "deepseek_v4":                 # its vocabulary is 64
        ids, then = [t % 64 for t in ids], [t % 64 for t in then]
    bits = os.environ.get("KNURLOGIC_KV_BITS")

    def built():
        m = build(family)
        if bits:
            from knurlogic.engine import kvquant
            assert kvquant.install(m, kvquant.parse_bits(bits)) > 0
        return m
    whole = run(built(), ids, then) if link.rank == 0 else None
    model = built()
    info = PL.split(model, link.group, PL.bounds_of(counts))
    split = run(model, ids, then)
    link.barrier()
    if link.rank == 0:
        json.dump({"whole": whole.tolist(), "split": split.tolist(),
                   "info": info}, open(out_path, "w"))


def _tiny_with_head(vocab):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_batch_drafting import _tiny
    return _tiny(vocab)


def mtp(link, out_path, always):
    import mlx.core as mx

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import pipeline as PL
    if always:
        os.environ["KNURLOGIC_MTP_BATCH_MAX_ROWS"] = "8"
    vocab, max_tokens = (8, 40) if always else (512, 40)

    def drive(gen, prompts, coord=None):
        """Insert every prompt, step until all finish. With `coord`, rank
        0's next tokens reach the follower before each step (the step
        plan's `tokens`) and a flag says whether to go on."""
        uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts))
        out, done = {u: [] for u in uids}, set()
        steps = 0
        while True:
            go = len(done) < len(uids)
            if coord is not None:
                go = bool(coord._bcast([int(go)])[0])
            if not go:
                break
            b = gen._batch
            if coord is not None and len(b):
                t = coord._bcast([int(x) for x in b.t1.tolist()])
                b.t1 = mx.array(t, dtype=mx.int32)
            _, responses = gen.next()
            steps += 1
            for r in responses:
                out[r.uid].append(r.token)
                if r.finish_reason:
                    done.add(r.uid)
        gen.close()
        return [out[u] for u in uids], steps

    whole = None
    if link.rank == 0:
        model, head, prompts = _tiny_with_head(vocab)
        whole, _ = drive(MTPBatchGenerator(model, head, stats={},
                                           prefill_step_size=16), prompts)
    model, head, prompts = _tiny_with_head(vocab)
    PL.split(model, link.group, PL.bounds_of([1, 3]))
    stats = {}
    gen = MTPBatchGenerator(model, head if link.rank == 0 else None,
                            stats=stats, prefill_step_size=16)
    if link.rank > 0:
        PL.silence(gen)
    coord = PL.coordinate(gen, link.group, drafting=True)
    split, steps = drive(gen, prompts, coord)
    counts = mx.distributed.all_gather(
        mx.array([coord.calls["b0"], coord.calls["b1"], coord.calls["b2"],
                  steps, coord.calls["ba"]]), group=link.group,
        stream=mx.cpu).tolist()
    sent = mx.distributed.all_gather(
        mx.array([sum(sd.overlapped for sd in PL.sends_of(model))]),
        group=link.group, stream=mx.cpu).tolist()
    link.barrier()
    if link.rank == 0:
        json.dump({"whole": whole, "split": split,
                   "calls": [counts[:5], counts[5:]], "overlapped": sent,
                   "accepted": stats.get("accepted", 0),
                   "drafted": stats.get("steps", 0)}, open(out_path, "w"))


class FakeTok:
    """What control_machine reads of a tokenizer: token 2 ends a turn."""
    eos_token_ids = [2]

    def convert_ids_to_tokens(self, t):
        return f"<{t}>"


def _fail_on_rank_0(gen, fail):
    """Make rank 0 fail one row the follower does not: "nan" marks the
    second row's logits non-finite at the fourth decode step; "admit" makes
    the second admission raise after its forward ran (the follower's
    admission of it succeeded)."""
    if fail == "nan":
        real, n = gen._batch.step, [0]

        def step():
            out = real()
            n[0] += 1
            if n[0] == 4:
                victim = sorted(gen._batch.uids)[1]
                for rs in out:
                    if rs.uid == victim:
                        for em in rs.tokens:
                            em.finite = False
            return out
        gen._batch.step = step
    elif fail == "admit":
        real, n = gen._admit_one, [0]

        def admit():
            r = real()
            n[0] += 1
            if n[0] == 2:
                gen._batch.remove([r.uid])
                raise RuntimeError("injected admission failure")
            return r
        gen._admit_one = admit


def engine(link, out_path, fail="", split_kind="pipeline"):
    """The serving path: rank 0's TensorExecutor journals a step plan, the
    follower runs tensor.follow (split="pipeline"), MTP on both.

    `fail`: rank 0 alone fails one row (_fail_on_rank_0); the other rows
    stream to the end and the follower drops the row from the next plan.
    `split_kind="tensor"`: the tiny qwen3_5_moe sharded, no head."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.executor import Admission, LocalExecutor, Token
    from knurlogic.engine.runtime.request import control_machine
    tok = FakeTok()

    def admissions(prompts):
        out = []
        for p in prompts:
            sm, _ = control_machine(tok, "normal")
            out.append(Admission(
                segments=[p[:5], p[5:]], max_tokens=30,
                sampling={"seed": 7}, state_machine=sm,
                wire={"penalties": {}, "initial": "normal"}))
        return out

    def drain(ex, uids):
        toks = {u: [] for u in uids}
        done = set()
        for _ in range(10_000):
            for e in ex.step():
                if isinstance(e, Token):
                    toks[e.uid].append(e.token)
                if type(e).__name__ in ("Finished", "RowFailure"):
                    done.add(e.uid)
            if done >= set(uids):
                break
        return [toks[u] for u in uids]

    if split_kind == "tensor":
        return _tensor_engine(link, out_path, fail, admissions, drain, tok)
    whole = None
    if link.rank == 0 and not fail:
        model, head, prompts = _tiny_with_head(512)
        ex = LocalExecutor(MTPBatchGenerator(model, head, prefill_step_size=16,
                                             completion_batch_size=32))
        whole = drain(ex, [ex.insert(a) for a in admissions(prompts)])
        ex.close()
    model, head, prompts = _tiny_with_head(512)
    PL.split(model, link.group, PL.bounds_of([1, 3]))
    if link.rank > 0:
        T.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=32,
                 prefill_step_size=16, working_set=0,
                 split="pipeline", drafting=True)
        return
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=16,
                            completion_batch_size=32)
    PL.coordinate(gen, link.group)
    _fail_on_rank_0(gen, fail)
    ring = T.Ring(link, split="pipeline")
    ex = T.TensorExecutor(gen, ring, over=lambda: 0)
    split = drain(ex, [ex.insert(a) for a in admissions(prompts)])
    ring.stop()
    json.dump({"whole": whole, "split": split}, open(out_path, "w"))


def _tensor_engine(link, out_path, fail, admissions, drain, tok):
    from tensor_ring_worker import build

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import tensor as T
    prompts = [[5, 17, 3, 99, 42, 7, 64, 11], [23, 31, 104, 33, 9, 8, 7, 6],
               [1, 4, 9, 16, 25, 36, 49, 64]]
    model = build()
    T.shard(model, link.group)
    if link.rank > 0:
        T.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=32,
                 prefill_step_size=16, working_set=0, split="tensor")
        return
    gen = MTPBatchGenerator(model, None, stats={}, prefill_step_size=16,
                            completion_batch_size=32)
    _fail_on_rank_0(gen, fail)
    ring = T.Ring(link)
    ex = T.TensorExecutor(gen, ring, over=lambda: 0)
    split = drain(ex, [ex.insert(a) for a in admissions(prompts)])
    ring.stop()
    json.dump({"whole": None, "split": split}, open(out_path, "w"))


def hit(link, out_path):
    """A prompt-cache hit only the follower could use: rank 0 stores the
    first prompt's checkpoint (at 5 tokens) WITHOUT its head cache, as a
    non-drafting row would; the follower stores its own (which never has
    one). The second prompt shares those 5 tokens: rank 0's drafting row
    cannot use a headless entry and prefills from scratch, and the
    follower must too (Coord.ba) -- its own trie says 5. Both prompts'
    tokens are the unsplit executor's."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.executor import (
        Admission,
        Checkpoint,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.runtime.scheduler import PromptCache
    tok = FakeTok()
    key = ("tiny", None, None)

    def admission(p, cache=None, hit=0):
        sm, _ = control_machine(tok, "normal")
        rest = p[hit:]
        segs = [rest[:5 - hit], rest[5 - hit:]] if hit < 5 else [rest]
        return Admission(segments=[s for s in segs if s], max_tokens=20,
                         cache=cache, prefix=p[:hit], sampling={"seed": 7},
                         state_machine=sm,
                         wire={"penalties": {}, "initial": "normal"})

    def drain(ex, uid, on_ckpt=None):
        toks = []
        for _ in range(10_000):
            evs = ex.step()
            for e in evs:
                if isinstance(e, Token) and e.uid == uid:
                    toks.append(e.token)
                if isinstance(e, Checkpoint) and on_ckpt:
                    on_ckpt(e)
            if any(type(e).__name__ == "Finished" and e.uid == uid
                   for e in evs):
                return toks
        raise AssertionError("did not finish")

    model, head, prompts = _tiny_with_head(512)
    a = prompts[0][:12]
    b = a[:5] + [7, 8, 9, 10, 11, 12, 13]
    whole = None
    if link.rank == 0:
        ex = LocalExecutor(MTPBatchGenerator(model, head, prefill_step_size=4,
                                             completion_batch_size=8))
        whole = [drain(ex, ex.insert(admission(p))) for p in (a, b)]
        ex.close()
    model, head, _ = _tiny_with_head(512)
    PL.split(model, link.group, PL.bounds_of([1, 3]))
    if link.rank > 0:
        T.follow(model, tok, key, link, prompt_cache_size=4,
                 completion_batch_size=8, prefill_step_size=4,
                 working_set=0, split="pipeline", drafting=True)
        return
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=4,
                            completion_batch_size=8)
    PL.coordinate(gen, link.group)
    ring = T.Ring(link, split="pipeline")
    ex = T.TensorExecutor(gen, ring, over=lambda: 0)
    pc = T.JournalPromptCache(PromptCache(4), ring.journal)
    n_trunk = gen._n_trunk

    def store(e):
        # the trunk only: the entry a non-drafting row would have left
        pc.insert(key, e.tokens, e.cache[:n_trunk], "system",
                  origin=("checkpoint", e.uid))
    split = [drain(ex, ex.insert(admission(a)), store)]
    c, rest = pc.fetch(key, b)
    got = len(b) - len(rest)
    split.append(drain(ex, ex.insert(admission(b, c, got))))
    ring.stop()
    json.dump({"whole": whole, "split": split, "hit": got},
              open(out_path, "w"))


def image(link, out_path, split_kind="pipeline"):
    """A prompt with two images on a split tiny qwen3_5 (the vision
    goldens' model): rank 0 alone has the tower and the image store; the
    follower binds the family without one (MirrorVision) and gets each
    image's ref in the admit op and its rows in the admission. The first
    prompt stores a checkpoint just past the last image on both ranks; the
    second shares that prefix (images inside the hit) and goes on in text,
    so the follower's key must be rank 0's for its trie to hit, and its
    positions must come from the refs alone. Both prompts' tokens are the
    unsplit engine's."""
    sys.path.insert(0, os.path.dirname(__file__))
    from pathlib import Path

    import test_vision_qwen as tq

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.executor import (
        Admission,
        Checkpoint,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.runtime.scheduler import PromptCache
    from knurlogic.engine.vision import key as K
    from knurlogic.engine.vision import registry
    from knurlogic.engine.vision.request import MirrorVision, VisionServe
    from knurlogic.engine.vision.store import ImageStore
    tok = FakeTok()
    fam = "qwen3_5"
    mkey = ("tiny-vl", None, None)
    arrays, meta = tq._golden(fam)
    d = Path(out_path).parent / f"rung{link.rank}"
    d.mkdir(exist_ok=True)
    tq._model_dir(d, fam, meta, "hf")

    def admission(p, cache=None, hit=0, cut=None):
        sm, _ = control_machine(tok, "normal")
        rest = p[hit:]
        segs = [rest[:cut - hit], rest[cut - hit:]] if cut else [rest]
        return Admission(segments=[s for s in segs if s], max_tokens=10,
                         cache=cache, prefix=p[:hit], sampling={"seed": 7},
                         state_machine=sm,
                         wire={"penalties": {}, "initial": "normal"})

    def drain(ex, uid, on_ckpt=None):
        toks = []
        for _ in range(10_000):
            evs = ex.step()
            for e in evs:
                if isinstance(e, Token) and e.uid == uid:
                    toks.append(e.token)
                if isinstance(e, Checkpoint) and on_ckpt:
                    on_ckpt(e)
                if type(e).__name__ == "RowFailure":
                    raise e.error if hasattr(e, "error") else \
                        RuntimeError(repr(e))
            if any(type(e).__name__ == "Finished" and e.uid == uid
                   for e in evs):
                return toks
        raise AssertionError("did not finish")

    vs = key = None
    if link.rank == 0:
        f = registry.build(fam, str(d), None, meta["config"])
        f.load_weights(str(d))
        vs = VisionServe(f, ImageStore(), mkey)
        refs = []
        for i in (1, 2):
            pix, ref = f.preprocess(tq._image(arrays, i), f"sha{i}")
            vs.store.put(mkey, f.encode(pix, ref))
            refs.append(ref)
        key = tq._key(f, arrays["ids"], refs)
    else:
        vs = MirrorVision(registry.build(fam, str(d), None, meta["config"]))
        assert vs.family.tower is None
    whole = None
    if link.rank == 0:
        cut = K.image_spans(key)[-1].end + 1
        second = key[:cut] + [int(x) for x in arrays["t2_ids"][-12:]]
        ex = LocalExecutor(MTPBatchGenerator(
            tq._trunk(fam, meta), None, vision=vs, prefill_step_size=16,
            completion_batch_size=8))
        whole = [drain(ex, ex.insert(admission(p))) for p in (key, second)]
        ex.close()
    model = tq._trunk(fam, meta)
    if split_kind == "pipeline":
        PL.split(model, link.group, PL.bounds_of([1, 3]))
    else:
        T.shard(model, link.group)
    if link.rank > 0:
        T.follow(model, tok, mkey, link, prompt_cache_size=4,
                 completion_batch_size=8, prefill_step_size=16,
                 working_set=0, split=split_kind, vision=vs)
        return
    gen = MTPBatchGenerator(model, None, stats={}, vision=vs,
                            prefill_step_size=16, completion_batch_size=8)
    if split_kind == "pipeline":
        coord = PL.coordinate(gen, link.group)
    ring = T.Ring(link, split=split_kind)
    ex = T.TensorExecutor(gen, ring, over=lambda: 0)
    coord = gen._coord
    pc = T.JournalPromptCache(PromptCache(4), ring.journal)

    def store(e):
        pc.insert(mkey, e.tokens, e.cache, "user",
                  origin=("checkpoint", e.uid))
    split = [drain(ex, ex.insert(admission(key, cut=cut)), store)]
    c, rest = pc.fetch(mkey, second)
    got = len(second) - len(rest)
    split.append(drain(ex, ex.insert(admission(second, c, got))))
    ring.stop()
    json.dump({"whole": whole, "split": split, "hit": got, "cut": cut,
               "images": coord.calls["img"]}, open(out_path, "w"))


def main(argv):
    import faulthandler
    faulthandler.dump_traceback_later(float(os.environ.get(
        "PIPELINE_WORKER_DEADLINE", "150")), exit=True)
    from knurlogic.engine.runtime import tensor as T
    link = T.init("ring")
    mode, out_path = argv[0], argv[1]
    if mode == "engine":
        engine(link, out_path, *argv[2:])
    elif mode == "image":
        image(link, out_path, *argv[2:])
    elif mode == "hit":
        hit(link, out_path)
    elif mode == "logits":
        logits(link, out_path, argv[2], [int(x) for x in argv[3].split(",")])
    else:
        mtp(link, out_path, argv[2] == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
