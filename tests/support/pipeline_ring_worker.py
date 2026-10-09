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
    mtp <always> tensor        the same, the model sharded across both ranks
                               (the follower's logits real, not silenced);
                               also the same split with no head anywhere
    mtp <always> tensor qwen4_exp
                               the same on the tiny qwen4_exp and its head
    dspark <split> <counts> <seeded> <engine>
                               the tiny DeepSeek-V4 DSpark checkpoint, rank
                               0 alone holding the block head (drafts partly
                               right: _guess), split by layer runs `counts`
                               (pipeline) or sharded (tensor); rank 0 also
                               runs it unsplit and, sharded, the same split
                               with no head. `seeded`: two rows seeded, one
                               greedy. `engine`: through rank 0's
                               TensorExecutor and the follower's
                               follower.follow instead of a hand-driven loop
    ends <split> <loop>        rows that end inside a drafting step, on the
                               serving path (TensorExecutor / follower.follow),
                               the tiny DeepSeek-V4 split (`split`: tensor or
                               pipeline), rank 0 holding its 1-token MTP head
                               (`loop` mtp) or DSpark block head (dspark),
                               the regime timing-chosen: rows end on
                               max_tokens and on the end token, together and
                               alone; rank 0 also runs them unsplit, no head

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
    from knurlogic.engine.split import pipeline as PL
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


def _tiny_with_head(vocab, family="qwen3_5"):
    if family == "qwen4_exp":
        return _tiny_flash_next_with_head(vocab)
    sys.path.insert(0, os.path.dirname(__file__))
    from test_batch_drafting import _tiny
    return _tiny(vocab)


def _tiny_flash_next_with_head(vocab):
    """The tiny qwen4_exp (its fixture config at `vocab`, mlx's own random
    init, float32) and a random MTP head: the block's own init, random
    glue."""
    import fixtures_vision_qwen as FQ
    import mlx.core as mx

    from knurlogic.engine import register
    register.register("qwen4_exp", override=True)
    from mlx_lm.models import qwen4_exp as arch

    from knurlogic.engine.families.qwen.heads.qwen4_exp import MTPHead
    mx.random.seed(0)
    cfg = FQ.config("qwen4_exp")
    cfg["text_config"]["vocab_size"] = vocab
    model = arch.Model(arch.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.float32)
    head = MTPHead(model, arch)
    D, hc = head.D, head.hc
    head.norm_e = head._norm(D, mx.zeros((D,)))
    head.norm_h = head._norm(hc * D, mx.zeros((hc * D,)), group_size=D)
    head.fc = mx.random.normal((D, 2 * D)) * 0.05
    head.block.set_dtype(mx.float32)
    head.mixer.set_dtype(mx.float32)
    mx.eval(model.parameters(), head.block.parameters(),
            head.mixer.parameters(), head.fc)
    prompts = [mx.random.randint(0, vocab, (n,)).tolist() for n in (37, 9, 70)]
    return model, head, prompts


def mtp(link, out_path, always, split_kind="pipeline", family="qwen3_5"):
    import mlx.core as mx

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import tensor as T
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
        model, head, prompts = _tiny_with_head(vocab, family)
        whole, _ = drive(MTPBatchGenerator(model, head, stats={},
                                           prefill_step_size=16), prompts)
    tensor = split_kind == "tensor"
    baseline = None
    if tensor:
        # the same split with no head on either rank: what drafting must
        # not change
        model, _, prompts = _tiny_with_head(vocab, family)
        T.shard(model, link.group)
        gen = MTPBatchGenerator(model, None, stats={}, prefill_step_size=16)
        baseline, _ = drive(gen, prompts,
                            PL.coordinate(gen, link.group, drafting=False))
    model, head, prompts = _tiny_with_head(vocab, family)
    if tensor:
        T.shard(model, link.group)
    else:
        PL.split(model, link.group, PL.bounds_of([1, 3]))
    stats = {}
    gen = MTPBatchGenerator(model, head if link.rank == 0 else None,
                            stats=stats, prefill_step_size=16)
    if link.rank > 0:
        if tensor:
            gen.mirror_hidden()
        else:
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
        json.dump({"whole": whole, "split": split, "baseline": baseline,
                   "calls": [counts[:5], counts[5:]], "overlapped": sent,
                   "accepted": stats.get("accepted", 0),
                   "drafted": stats.get("steps", 0)}, open(out_path, "w"))


def _dspark_tensor_tiny():
    """(model, head) for a tensor split: the tiny DSpark config with
    random weights (as
    tensor_ring_worker.build_deepseek_v4 draws them), and the
    checkpoint's head moved onto it (its stages read the trunk's embedding
    and lm_head, which have the same shapes). Its indexer keeps every
    pool row: it runs whole on each rank on the all-summed hidden state,
    whose last bits differ from the unsplit run's, and its FP4-rounded
    scores (architecture edit 20) can then rank a near-tie the other way
    -- as the reference's would between tensor-parallel degrees."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "engine"))
    import test_deepseek_v4_dspark as D

    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    cfg = json.loads((D.G.TINY / "config.json").read_text())
    # (its experts 256 wide and wo_b's input 256: a rank's half is whole
    # 128-blocks, so both runs round the same blocks -- architecture edit
    # 21; tuning/resolve refuses a split that would cut one)
    model = M.Model(M.ModelArgs.from_dict(dict(cfg, index_topk=64)))
    rng = np.random.default_rng(0)
    w = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("tid2eid"):
            a = rng.integers(0, cfg["n_routed_experts"],
                             size=v.shape).astype(np.int32)
        elif "switch_mlp" in k and v.dtype == mx.uint8:   # E8M0 scales
            a = rng.integers(118, 124, size=v.shape).astype(np.uint8)
        elif "switch_mlp" in k:                           # packed mxfp4
            a = rng.integers(0, 2 ** 32, size=v.shape,
                             dtype=np.uint64).astype(np.uint32)
        elif k.endswith("norm.weight"):
            a = (1 + 0.1 * rng.standard_normal(v.shape)).astype(np.float32)
        else:
            a = (0.15 * rng.standard_normal(v.shape)).astype(np.float32)
        w.append((k, mx.array(a)))
    model.update(tree_unflatten(w))
    mx.eval(model.parameters())
    head, _ = D._head(D._load())
    head.model, head.core = model, model.model
    return model, head


def _no_rounding():
    """KNURLOGIC_TEST_NO_ROUNDING=1: architecture edits 20 and 21's
    activation rounding as identities, so a run differing from another only
    in summation order is the same token for token (the rounding turns a
    float32 ulp at an FP8 boundary into a different block value, and on a
    random tiny model a near-tie into another token)."""
    import mlx.core as mx

    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as A

    def nope(kv, rd, pe=None):
        if pe is None:
            return kv
        return mx.concatenate([kv[..., :kv.shape[-1] - rd],
                               pe.astype(kv.dtype)], axis=-1)

    def swiglu(gate, up, limit, w=None):
        h, u = gate.astype(mx.float32), up.astype(mx.float32)
        if limit and limit > 0:
            h, u = mx.minimum(h, limit), mx.clip(u, -limit, limit)
        h = h * mx.sigmoid(h) * u
        if w is not None:
            h = h * w.astype(mx.float32).reshape(*gate.shape[:-1], 1)
        return h.astype(gate.dtype)

    def norm(x, w, eps, keep=True, width=None):
        y = mx.fast.rms_norm(x if width is None else x[..., :width], w, eps)
        return (y, y) if keep else y
    A.fp8_act = A.fp4_simulate_rotated = A._indexer_q_qat = lambda x: x
    A.fp8_simulate = A.fp4_simulate = lambda x, block=0: x
    A.fp8_simulate_nope, A.swiglu_act, A.rms_norm_act = nope, swiglu, norm


class _Rounding:
    """KNURLOGIC_TEST_RECORD=1: every call of edits 20-21's rounding on
    this rank, as (step, site, input before rounding, output), float32
    (compile off, so each call is seen). `step` counts the driver's
    gen.next() calls; `on` gates recording to the runs compared."""

    def __init__(self):
        import mlx.core as mx

        from knurlogic.engine import register
        register.register("deepseek_v4")
        import mlx_lm.models.deepseek_v4 as A
        mx.disable_compile()
        self.calls, self.step, self.on, self.emits = [], 0, False, {}
        orig = {k: getattr(A, k) for k in (
            "fp8_act", "rms_norm_act", "swiglu_act", "fp8_simulate_nope",
            "fp4_simulate_rotated")}

        def keep(site, pre, post):
            if self.on:
                self.calls.append((self.step, site,
                                   np.array(pre.astype(mx.float32)),
                                   np.array(post.astype(mx.float32))))

        def fp8_act(x):
            y = orig["fp8_act"](x)
            keep("fp8_act", x, y)
            return y

        def rms_norm_act(x, w, eps, keep_=True, width=None, **kw):
            k = kw.get("keep", keep_)
            y, yq = orig["rms_norm_act"](x, w, eps, True, width)
            keep("rms_norm_act", y, yq)
            return (y, yq) if k else yq

        def swiglu_act(gate, up, limit, w=None):
            h, u = gate.astype(mx.float32), up.astype(mx.float32)
            if limit and limit > 0:
                h, u = mx.minimum(h, limit), mx.clip(u, -limit, limit)
            h = h * mx.sigmoid(h) * u
            if w is not None:
                h = h * w.astype(mx.float32).reshape(*gate.shape[:-1], 1)
            y = orig["swiglu_act"](gate, up, limit, w)
            keep("swiglu_act", h.astype(gate.dtype), y)
            return y

        def nope(kv, rd, pe=None):
            y = orig["fp8_simulate_nope"](kv, rd, pe)
            keep("fp8_simulate_nope", kv[..., :kv.shape[-1] - rd],
                 y[..., :kv.shape[-1] - rd])
            return y

        def rot(x):
            y = orig["fp4_simulate_rotated"](x)
            keep("fp4_simulate_rotated", x, y)
            return y
        A.fp8_act, A.rms_norm_act, A.swiglu_act = fp8_act, rms_norm_act, \
            swiglu_act
        A.fp8_simulate_nope = nope
        A.fp4_simulate_rotated = A._indexer_q_qat = rot

    def take(self):
        out = (self.calls, self.emits)
        self.calls, self.step, self.emits = [], 0, {}
        return out


def flip_proof(whole, split, whole_rec, split_rec, emits):
    """Where the streams part, why: walking both runs' rounding calls in
    order (a sharded input -- wo_b's, an expert's width -- is this rank's
    leading part of the whole's), the first call whose rounded output
    differs in some block (128 wide; 64 for the kv, 32 for FP4), and the
    largest difference of the inputs before rounding over every call up to
    and including it, relative to each input's largest magnitude. After
    that first flip the runs carry a real difference forward, so later
    inputs are not compared."""
    k = None
    for r, (a, b) in enumerate(zip(whole, split)):
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if n is not None:       # the step that emitted it (the same in both)
            k = emits[r][n] if k is None else min(k, emits[r][n])
    if k is None:
        return {"step": None}
    rel, calls = 0.0, 0
    for (s1, n1, p1, o1), (s2, n2, p2, o2) in zip(whole_rec, split_rec):
        assert s1 == s2 and n1 == n2, (s1, n1, s2, n2)
        w = p2.shape[-1]
        p1, o1 = p1[..., :w], o1[..., :w]
        assert p1.shape == p2.shape, (n1, p1.shape, p2.shape)
        rel = max(rel, float(np.abs(p1 - p2).max()
                             / max(np.abs(p1).max(), 1e-30)))
        calls += 1
        b = 32 if n1 == "fp4_simulate_rotated" else (
            64 if n1 == "fp8_simulate_nope" else 128)
        if w % b == 0:
            diff = (o1 != o2).reshape(*o1.shape[:-1], w // b, b).any(-1)
            if diff.any():
                return {"step": k, "flip_step": s1, "flip_site": n1,
                        "flip_blocks": int(diff.sum()), "rel": rel,
                        "calls": calls}
    return {"step": k, "flip_step": None, "rel": rel, "calls": calls}


def margins(model, prompts, outs, samp):
    """The reference run's decision margin at each of its tokens: the gap
    between the top two of what picks the token -- a greedy row's logits;
    a seeded row's scaled log-probabilities plus its key's Gumbel noise at
    that position (Keys, Gumbel-max) -- from one teacher-forced forward of
    the unsplit model; and whether that forward picks the reference's
    token there."""
    import mlx.core as mx

    from knurlogic.engine.mtp.sampling import Keys
    out = []
    for p, o, sp in zip(prompts, outs, samp):
        if not o:
            out.append(([], []))
            continue
        lg = model(mx.array([p + o]))[0, len(p) - 1:len(p) - 1 + len(o)]
        s = lg.astype(mx.float32)
        if sp:
            s = (s - mx.logsumexp(s, axis=-1, keepdims=True)) / sp["temp"]
            keys = Keys(sp["seed"])
            s = s + mx.concatenate([
                mx.random.gumbel(shape=(1, s.shape[-1]), key=keys.at(i))
                for i in range(len(o))])
        top = mx.sort(s, axis=-1)[:, -2:]
        out.append(((top[:, 1] - top[:, 0]).tolist(),
                    (mx.argmax(s, axis=-1) == mx.array(o)).tolist()))
    return out


def dspark(link, out_path, split_kind="pipeline", counts="1,3", seeded="",
           engine=""):
    import mlx.core as mx
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "engine"))
    import test_deepseek_v4_dspark as D

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import tensor as T
    os.environ["KNURLOGIC_MTP_BATCH_MAX_ROWS"] = "8"      # draft always
    if os.environ.get("KNURLOGIC_TEST_NO_ROUNDING"):
        _no_rounding()
    rec = _Rounding() if os.environ.get("KNURLOGIC_TEST_RECORD") else None
    recs = {}
    G = D.G
    prompts = [G.PROMPT, G.PROMPT[:3], G.PROMPT[2:9] + G.DECODE]
    samp = ([{"temp": 0.8, "seed": 1234}, None, {"temp": 0.8, "seed": 99}]
            if seeded else [None] * 3)
    max_tokens = 24
    tensor = split_kind == "tensor"

    def load():
        if tensor:
            return _dspark_tensor_tiny()
        # float32 (its bf16 norms and projections), so a stage's stream
        # crosses ranks unrounded and the split is the unsplit model's
        # arithmetic exactly
        model = D._load()
        model.set_dtype(mx.float32)
        return model, None

    def bind(model, head):
        if head is None:
            head, _ = D._head(model)
        return head

    def record(head, sink):
        """Keep every main hidden state the head takes (the target layers'
        outputs, HC-meaned): the drafts are _guess's, so the tokens alone
        would not show a wrong capture."""
        real = head.advance

        def advance(main_h, cache):
            sink.append(np.array(main_h.astype(mx.float32)))
            return real(main_h, cache)
        head.advance = advance
    seen_whole, seen_split = [], []

    def drive(gen, coord=None):
        uids = gen.insert_segments(
            segments=[[p] for p in prompts], max_tokens=[max_tokens] * 3,
            caches=[None] * 3, all_tokens=[[] for _ in prompts],
            samplers=samp)
        out, done = {u: [] for u in uids}, set()
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
            _, rs = gen.next()
            for r in rs:
                if rec is not None:
                    rec.emits.setdefault(uids.index(r.uid), []).append(
                        rec.step)
                out[r.uid].append(r.token)
                if r.finish_reason is not None:
                    done.add(r.uid)
            if rec is not None:
                rec.step += 1
        gen.close()
        return [out[u] for u in uids]

    def cut(model):
        if tensor:
            T.shard(model, link.group)
        else:
            PL.split(model, link.group,
                     PL.bounds_of([int(c) for c in counts.split(",")]))

    whole = whole_draft = baseline = margin = None
    if link.rank == 0:
        model, head = load()
        if rec is not None:
            rec.take()
            rec.on = True
        whole = drive(MTPBatchGenerator(model, None, prefill_step_size=4))
        if rec is not None:
            rec.on, recs["whole"] = False, rec.take()
        margin = margins(model, prompts, whole, samp)
        head = bind(model, head)
        D._guess(head, [p + o for p, o in zip(prompts, whole)])
        record(head, seen_whole)
        whole_draft = drive(MTPBatchGenerator(model, head,
                                              prefill_step_size=4))
    if tensor:
        # the same split with no head on either rank
        model, _ = load()
        cut(model)
        gen = MTPBatchGenerator(model, None, prefill_step_size=4)
        if rec is not None:
            rec.take()
            rec.on = True
        baseline = drive(gen, PL.coordinate(gen, link.group, drafting=False))
        if rec is not None:
            rec.on, recs["split"] = False, rec.take()
            if link.rank == 0:
                json.dump({"whole": whole, "baseline": baseline,
                           "proof": flip_proof(whole, baseline,
                                               recs["whole"][0],
                                               recs["split"][0],
                                               recs["whole"][1])},
                          open(out_path, "w"))
            link.barrier()
            return
    model, head = load()
    cut(model)
    if link.rank == 0:
        head = bind(model, head)
        D._guess(head, [p + o for p, o in zip(prompts, whole)])
        record(head, seen_split)
    else:
        head = None
    K, outs = G.K, list(model.args.dspark_target_layer_ids)
    if engine:
        return _dspark_engine(link, out_path, model, head, prompts, samp,
                              max_tokens, split_kind, K, outs, load,
                              {"whole": whole, "whole_draft": whole_draft,
                               "baseline": baseline, "margin": margin})
    stats = {}
    gen = MTPBatchGenerator(model, head, stats=stats, prefill_step_size=4)
    if link.rank > 0:
        gen.follow_block(K, outs)
        if tensor:
            gen.mirror_hidden()
        else:
            PL.silence(gen)
    coord = PL.coordinate(gen, link.group, drafting=True)
    split = drive(gen, coord)
    counts_ = mx.distributed.all_gather(
        mx.array([coord.calls[k] for k in ("b0", "b1", "b2", "ba")]),
        group=link.group, stream=mx.cpu).tolist()
    link.barrier()
    if link.rank == 0:
        json.dump({"whole": whole, "whole_draft": whole_draft,
                   "baseline": baseline, "split": split, "margin": margin,
                   "calls": [counts_[:4], counts_[4:]],
                   "accepted": stats.get("accepted", 0),
                   "drafted": stats.get("steps", 0),
                   "hidden": [len(seen_whole), len(seen_split),
                              max((float(np.abs(a - b).max()) if a.shape ==
                                   b.shape else float("inf"))
                                  for a, b in zip(seen_whole, seen_split))]},
                  open(out_path, "w"))


def _dspark_engine(link, out_path, model, head, prompts, samp, max_tokens,
                   split_kind, K, outs, load, ref):
    """The serving path for `dspark ... engine`: rank 0's TensorExecutor
    and the follower's follower.follow with rank 0's block size and target
    layers (as agree_head tells it), against rank 0's unsplit executor
    with no head (FakeTok's end token on both). `keyed`: every Finished
    entry is at exactly its key's length."""
    from knurlogic.engine.mtp.batch_generator import (
        MTPBatchGenerator,
        trunk_offset,
    )
    from knurlogic.engine.runtime.executor import (
        Admission,
        Finished,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import ring as split_ring
    tok = FakeTok()
    if link.rank > 0:
        split_follower.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=8,
                 prefill_step_size=4, working_set=0, split=split_kind,
                 drafting=True, block=K, outputs=tuple(outs))
        return

    def drain(ex, n_trunk):
        uids = []
        for p, sp in zip(prompts, samp):
            sm, _ = control_machine(tok, "normal")
            uids.append(ex.insert(Admission(
                segments=[p], max_tokens=max_tokens, sampling=sp or {},
                state_machine=sm,
                wire={"penalties": {}, "initial": "normal"})))
        toks, done, keyed = {u: [] for u in uids}, set(), []
        for _ in range(10_000):
            for e in ex.step():
                if isinstance(e, Token):
                    toks[e.uid].append(e.token)
                if isinstance(e, Finished):
                    done.add(e.uid)
                    keyed.append(trunk_offset(e.cache[:n_trunk])
                                 == len(e.tokens))
            if done >= set(uids):
                break
        return [toks[u] for u in uids], keyed

    plain, _ = load()
    ex = LocalExecutor(MTPBatchGenerator(plain, None, prefill_step_size=4,
                                         completion_batch_size=8))
    served, _ = drain(ex, len(plain.make_cache()))
    ex.close()
    ref = dict(ref, served_margin=margins(plain, prompts, served, samp))
    stats = {}
    gen = MTPBatchGenerator(model, head, stats=stats, prefill_step_size=4,
                            completion_batch_size=8)
    PL.coordinate(gen, link.group)
    ring = split_ring.Ring(link, split=split_kind)
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    split, keyed = drain(ex, gen._n_trunk)
    ring.stop()
    json.dump(dict(ref, served=served, split=split, keyed=keyed,
                   accepted=stats.get("accepted", 0),
                   drafted=stats.get("steps", 0)), open(out_path, "w"))


def _v4_mtp_tiny(tmp):
    """(model, head): the tiny DeepSeek-V4 with 64-wide experts (shards on
    a tensor split) and a random 1-token MTP head packed beside it, whose
    drafts are steered to "the last token again" (the tiny model's greedy
    loops accept some)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from tensor_ring_worker import build_deepseek_v4

    from knurlogic.engine.families.deepseek.heads.deepseek_v4 import MTPHead
    from knurlogic.engine.mtp import registry
    model = build_deepseek_v4()
    arch = sys.modules[type(model.model).__module__]
    mx.random.seed(7)
    h = MTPHead(model, arch)
    h.m.update(tree_unflatten([
        (k, mx.ones(v.shape) if k.endswith("norm.weight")
         else 0.15 * mx.random.normal(v.shape))
        for k, v in tree_flatten(h.m.parameters())]))
    nn.quantize(h.m, class_predicate=lambda p, m: (
        "switch_mlp" in p and hasattr(m, "to_quantized")
        and {"group_size": 32, "bits": 4, "mode": "mxfp4"}))
    path = os.path.join(tmp, "mtp-head-mxfp4.safetensors")
    h.save(path)
    head, _ = registry.load_head(model, sidecar=path, family="deepseek_v4")
    real = head.draft_logits

    def draft_logits(hh, ids, cache=None):
        out = real(hh, ids, cache)
        return out + 100.0 * (mx.arange(out.shape[-1])
                              == ids[:, :, None]).astype(out.dtype)
    head.draft_logits = draft_logits
    return model, head


def ends(link, out_path, split_kind="tensor", loop="mtp"):
    """Rows ending inside a drafting step on the serving path. Every rank
    first finds the end token the same way (the unsplit plain run's 9th
    token of the last prompt, whose cap is 12), so the follower's control
    machine is rank
    0's. Then, through rank 0's TensorExecutor and the follower's
    follower.follow: eight rows in one batch ending on max_tokens 3, 6..12 or
    the end token, then each of them alone. Rank 0 counts the steps that
    ended a row before their last position (`_ends` hit inside a drafting
    step) and the regimes taken."""
    import tempfile

    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import ring as split_ring

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "goldens"))
    import build_deepseek_v4 as G

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime.executor import (
        Admission,
        Finished,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import tensor as T
    tmp = tempfile.mkdtemp()

    def load():
        if loop == "dspark":
            model, head = _dspark_tensor_tiny()
            return model, head
        return _v4_mtp_tiny(tmp)

    # past the sliding window (8): a headless batch extending a cache
    # shorter than its window fails the row (DeepseekV4Cache.extend)
    base = 2 * (G.PROMPT + G.DECODE)
    prompts = [base[i:i + 9 + i % 4] for i in range(8)]
    # the first row's first block step commits 5 under DSpark (_guess):
    # a cap of 3 ends it inside
    caps = [3] + list(range(6, 13))
    seeds = [split_ring.assign_seed({}) for _ in prompts]   # as a ring admits

    class Tok:
        eos_token_ids = [1 << 20]                  # none, until found

        def convert_ids_to_tokens(self, t):
            return f"<{t}>"
    tok = Tok()

    def drain(ex, rows):
        uids = []
        for p, cap, sp in rows:
            sm, _ = control_machine(tok, "normal")
            uids.append(ex.insert(Admission(
                segments=[p], max_tokens=cap, sampling=dict(sp),
                state_machine=sm,
                wire={"penalties": {}, "initial": "normal"})))
        toks, done = {u: [] for u in uids}, set()
        for _ in range(10_000):
            for e in ex.step():
                if isinstance(e, Token):
                    toks[e.uid].append(e.token)
                if isinstance(e, Finished):
                    done.add(e.uid)
            if done >= set(uids):
                break
        return [toks[u] for u in uids]

    def phases(ex):
        rows = list(zip(prompts, caps, seeds))
        return [drain(ex, rows)] + [drain(ex, [r]) for r in rows]

    # the end token, found alike on every rank
    plain, _ = load()
    ex = LocalExecutor(MTPBatchGenerator(plain, None, prefill_step_size=4))
    end = drain(ex, [(prompts[-1], 16, {})])[0][8]
    tok = Tok()                    # control_machine caches per tokenizer
    tok.eos_token_ids = [end]
    ex.close()
    served = None
    if link.rank == 0:
        ex = LocalExecutor(MTPBatchGenerator(plain, None, prefill_step_size=4,
                                             completion_batch_size=8))
        served = phases(ex)
        # each row's own tokens past its cap: what _guess steers toward
        longer = [drain(ex, [(p, 16, sp)])[0]
                  for p, sp in zip(prompts, seeds)]
        ex.close()
    del plain
    model, head = load()
    if split_kind == "tensor":
        T.shard(model, link.group)
    else:
        PL.split(model, link.group, PL.bounds_of([1, 3]))
    block = head.block_size if loop == "dspark" else 0
    outs = (tuple(model.args.dspark_target_layer_ids) if block else ())
    if link.rank > 0:
        split_follower.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=8,
                 prefill_step_size=4, working_set=0, split=split_kind,
                 drafting=True, block=block, outputs=outs)
        return
    if block:
        # blocks partly right (the rows alone, unsplit): steps commit
        # several tokens, so a row can end before a block's last
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "engine"))
        import test_deepseek_v4_dspark as D
        D._guess(head, [p + o for p, o in zip(prompts, longer)])
    stats = {}
    gen = MTPBatchGenerator(model, head, stats=stats, prefill_step_size=4,
                            completion_batch_size=8)
    PL.coordinate(gen, link.group)
    b = gen._batch
    regimes, inside = [], [0]
    real_note, real_ends = b._note_regime, b._ends

    def note(drafting, rows):
        regimes.append(bool(drafting))
        return real_note(drafting, rows)

    def ends_(i, toks):
        j = real_ends(i, toks)
        if (regimes and regimes[-1] and j is not None
                and (j < len(toks) - 1 if block else True)):
            inside[0] += 1
        return j
    b._note_regime, b._ends = note, ends_
    ring = split_ring.Ring(link, split=split_kind)
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    split = phases(ex)
    ring.stop()
    json.dump({"served": served, "split": split, "inside": inside[0],
               "drafting": sum(regimes),
               "plain": len(regimes) - sum(regimes),
               "eos": tok.eos_token_ids[0], "caps": caps,
               "drafted": stats.get("steps", 0),
               "accepted": stats.get("accepted", 0)}, open(out_path, "w"))


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
    elif fail == "cancel":
        # rank 0's client goes away during the second admission's prefill:
        # its hook asks to stop at the first chunk boundary (Coord.stop)
        real, n = gen._admit_one, [0]

        def admit():
            n[0] += 1
            if n[0] == 2:
                uid = gen._unprocessed_sequences[0][0]
                gen.prefill_hooks[uid] = lambda u, done, total: True
            return real()
        gen._admit_one = admit
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


def engine(link, out_path, fail="", split_kind="pipeline", drafting=""):
    """The serving path: rank 0's TensorExecutor journals a step plan, the
    follower runs follower.follow (split="pipeline"), MTP on both.

    `fail`: rank 0 alone fails one row (_fail_on_rank_0); the other rows
    stream to the end and the follower drops the row from the next plan.
    `split_kind="tensor"`: the tiny qwen3_5_moe sharded, no head; with
    `drafting`, the tiny qwen3_5 sharded, rank 0 holding the head."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime.executor import Admission, LocalExecutor, Token
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import ring as split_ring
    tok = FakeTok()

    def admissions(prompts):
        out = []
        if fail == "cancel":
            # several prefill chunks (16 tokens) a prompt, so the stop lands
            # mid-prefill
            prompts = [list(p) * 6 for p in prompts]
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
        return _tensor_engine(link, out_path, fail, admissions, drain, tok,
                              bool(drafting))
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
        split_follower.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=32,
                 prefill_step_size=16, working_set=0,
                 split="pipeline", drafting=True)
        return
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=16,
                            completion_batch_size=32)
    PL.coordinate(gen, link.group)
    _fail_on_rank_0(gen, fail)
    ring = split_ring.Ring(link, split="pipeline")
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    split = drain(ex, [ex.insert(a) for a in admissions(prompts)])
    ring.stop()
    json.dump({"whole": whole, "split": split}, open(out_path, "w"))


def _tensor_engine(link, out_path, fail, admissions, drain, tok,
                   drafting=False):
    from tensor_ring_worker import build

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime.executor import LocalExecutor
    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import ring as split_ring
    from knurlogic.engine.split import tensor as T
    prompts = [[5, 17, 3, 99, 42, 7, 64, 11], [23, 31, 104, 33, 9, 8, 7, 6],
               [1, 4, 9, 16, 25, 36, 49, 64]]
    whole, head = None, None
    if drafting:
        if link.rank == 0 and not fail:
            model, head, prompts = _tiny_with_head(512)
            ex = LocalExecutor(MTPBatchGenerator(
                model, head, prefill_step_size=16, completion_batch_size=32))
            whole = drain(ex, [ex.insert(a) for a in admissions(prompts)])
            ex.close()
        model, head, prompts = _tiny_with_head(512)
    else:
        model = build()
    T.shard(model, link.group)
    if link.rank > 0:
        split_follower.follow(model, tok, ("tiny", None, None), link,
                 prompt_cache_size=4, completion_batch_size=32,
                 prefill_step_size=16, working_set=0, split="tensor",
                 drafting=drafting)
        return
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=16,
                            completion_batch_size=32)
    if drafting:
        PL.coordinate(gen, link.group)          # as the scheduler does
    _fail_on_rank_0(gen, fail)
    ring = split_ring.Ring(link)
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    split = drain(ex, [ex.insert(a) for a in admissions(prompts)])
    ring.stop()
    json.dump({"whole": whole, "split": split}, open(out_path, "w"))


def hit(link, out_path, split_kind="pipeline"):
    """A prompt-cache hit only the follower could use: rank 0 stores the
    first prompt's checkpoint (at 5 tokens) WITHOUT its head cache, as a
    non-drafting row would; the follower stores its own (which never has
    one). The second prompt shares those 5 tokens: rank 0's drafting row
    cannot use a headless entry and prefills from scratch, and the
    follower must too (Coord.ba) -- its own trie says 5. Both prompts'
    tokens are the unsplit executor's."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.prompt_cache.memory import PromptCache
    from knurlogic.engine.prompt_cache.ring import JournalPromptCache
    from knurlogic.engine.runtime.executor import (
        Admission,
        Checkpoint,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import ring as split_ring
    from knurlogic.engine.split import tensor as T
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
    if split_kind == "tensor":
        T.shard(model, link.group)
    else:
        PL.split(model, link.group, PL.bounds_of([1, 3]))
    if link.rank > 0:
        split_follower.follow(model, tok, key, link, prompt_cache_size=4,
                 completion_batch_size=8, prefill_step_size=4,
                 working_set=0, split=split_kind, drafting=True)
        return
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=4,
                            completion_batch_size=8)
    PL.coordinate(gen, link.group)
    ring = split_ring.Ring(link, split=split_kind)
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    pc = JournalPromptCache(PromptCache(4), ring.journal)
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
    from knurlogic.engine.split import follower as split_follower
    from knurlogic.engine.split import ring as split_ring
    sys.path.insert(0, os.path.dirname(__file__))
    from pathlib import Path

    import test_vision_qwen as tq

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.prompt_cache.memory import PromptCache
    from knurlogic.engine.prompt_cache.ring import JournalPromptCache
    from knurlogic.engine.runtime.executor import (
        Admission,
        Checkpoint,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    from knurlogic.engine.split import pipeline as PL
    from knurlogic.engine.split import tensor as T
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
        split_follower.follow(model, tok, mkey, link, prompt_cache_size=4,
                 completion_batch_size=8, prefill_step_size=16,
                 working_set=0, split=split_kind, vision=vs)
        return
    gen = MTPBatchGenerator(model, None, stats={}, vision=vs,
                            prefill_step_size=16, completion_batch_size=8)
    if split_kind == "pipeline":
        coord = PL.coordinate(gen, link.group)
    ring = split_ring.Ring(link, split=split_kind)
    ex = split_ring.TensorExecutor(gen, ring, over=lambda: 0)
    coord = gen._coord
    pc = JournalPromptCache(PromptCache(4), ring.journal)

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

    from knurlogic.engine.split import link as split_link
    faulthandler.dump_traceback_later(float(os.environ.get(
        "PIPELINE_WORKER_DEADLINE", "150")), exit=True)
    link = split_link.init("ring")
    mode, out_path = argv[0], argv[1]
    if mode == "engine":
        engine(link, out_path, *argv[2:])
    elif mode == "image":
        image(link, out_path, *argv[2:])
    elif mode == "hit":
        hit(link, out_path, *argv[2:])
    elif mode == "dspark":
        dspark(link, out_path, *argv[2:])
    elif mode == "ends":
        ends(link, out_path, *argv[2:])
    elif mode == "logits":
        logits(link, out_path, argv[2], [int(x) for x in argv[3].split(",")])
    else:
        mtp(link, out_path, argv[2] == "1", *argv[3:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
