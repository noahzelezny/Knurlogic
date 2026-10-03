"""Vision end to end: every REAL family through the REAL serve path.

The serve path's own tests use the shared StubFamily only
(positions -> (None, 0), no extras). The real families disagree with it on
the join -- what `embed` returns, what `positions` means in decode, what the
trunk calls its keywords, what a forward returns. This file runs each tiny
family through the same objects a served request uses: knurlogic's own
server -- the OpenAI surface, the scheduler's tokenize (vision's
engine/vision/request.py inside it), the real `LRUPromptCache`, and
`MTPBatchGenerator` admit and decode (test_image_cache.Harness).

  E1  turn 1 (image + text): greedy tokens == the family's own model-level
      reference forward (embed + positions -> trunk, one shot, the path
      each family's own tests use) for the same input.
  E2  turn 2 (text only), WARM from turn 1's prompt-cache entry == COLD
      (fresh server): tokens identical, first-token logprobs close. G7b
      through the real path; for Qwen it fails if rope_delta is lost on the
      text suffix or in decode.
  E3  the tower ran exactly once across both turns (counted from OUTSIDE,
      by a proxy around the family's tower module).
  E4  two rows in one batch -- one with an image, one text only -- each ==
      its solo run (per-row rope_delta).
  E5  gemma with prefill_step_size=16 and an image block straddling token
      16: tokens identical to the one-shot reference (chunk snapping).
  E6  a seeded image request takes the batch path and answers as the
      reference does.

Tiny random models only (float32, seed 0, vocab 512). No real model, no
server socket, no exo.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402
from test_image_cache import (  # noqa: E402,F401
    Harness,
    cached,
    data_url,
    first_top,
    server,
)

from knurlogic.engine.vision import key as K  # noqa: E402

VOCAB = 512
EOS = 500
#: every id from 500 up is a special of some family (image, framing, eos):
#: a random model must not type one, or turn 2 would re-tokenize an image
#: pad the client never sent.
BANNED = list(range(500, VOCAB))
BAN = {str(i): -1e9 for i in BANNED}
MAX_TOKENS = 12


# --- tokenizer ----------------------------------------------------------------

class SpecTok:
    """Lossless byte tokenizer with a table of special strings, each ONE id
    (the P0 contract: a family's placeholder text tokenizes to exactly one
    image_token_id plus framing). Ids 0-255 are bytes; any other id is a
    private-use char, so a reply re-encodes to exactly the ids generated
    and turn 2's key extends turn 1's stored entry. The chat template never
    rewrites history (test_image_cache.ByteTok says why)."""

    chat_template = "role: content, one line each"
    clean_up_tokenization_spaces = False
    eos_token_id = EOS

    def __init__(self, specials):
        self.specials = dict(specials)
        self.by_id = {v: k for k, v in self.specials.items()}

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        out, i = [], 0
        names = sorted(self.specials, key=len, reverse=True)
        while i < len(text):
            for s in names:
                if text.startswith(s, i):
                    out.append(self.specials[s])
                    i += len(s)
                    break
            else:
                o = ord(text[i])
                if o < 256:
                    out.append(o)
                elif 0xE000 <= o < 0xE000 + VOCAB:
                    out.append(o - 0xE000)
                else:
                    out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, **kw):
        s = []
        for t in ids:
            t = int(t)
            s.append(self.by_id.get(t) or ("" if t == EOS else chr(t)
                                           if t < 256 else chr(0xE000 + t)))
        return "".join(s)

    def convert_ids_to_tokens(self, t):
        if isinstance(t, (list, tuple)):
            return [self.decode([x]) for x in t]
        return self.decode([t])

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tools=None, tokenize=True, **kw):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant: "
        return self.encode(text) if tokenize else text


def wrap(tok):
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    return TokenizerWrapper(tok, eos_token_ids=[EOS])


# --- the families ---------------------------------------------------------------

class Rig:
    """One tiny family, ready to serve: the trunk the server holds, a
    factory for a FRESH family (its own tower, same weights), the tokenizer,
    and the trunk's calling convention for the reference forward."""

    def __init__(self, name, model, make_family, specials, img, *,
                 call=None, img_size=(32, 24)):
        self.name, self.model = name, model
        self.make_family = make_family
        self.specials = specials
        self.img = img
        self.img_size = img_size
        self._call = call

    def tok(self):
        return wrap(SpecTok(self.specials))

    def url(self, seed):
        return data_url(seed, *self.img_size)

    def call(self, ids, cache, **kw):
        if self._call is not None:
            return self._call(self.model, ids, cache, **kw)
        return self.model(ids, cache=cache, **kw)


def _qwen_rig(fam, tmp):
    import fixtures_vision_qwen as fq
    import test_vision_qwen as tq
    _, meta = tq._golden(fam)
    model = tq._trunk(fam, meta)
    d = tmp / fam
    d.mkdir()
    tq._model_dir(d, fam, meta, "hf")

    def make():
        from knurlogic.engine.vision import registry
        f = registry.build(fam, str(d), model, meta["config"])
        f.load_weights(str(d))
        return f
    t = fq.ids(fam)
    specials = {"<|vision_start|>": t["vision_start_token_id"],
                "<|vision_end|>": t["vision_end_token_id"],
                "<|image_pad|>": t["image_token_id"],
                "<|video_pad|>": t["video_token_id"]}
    return Rig(fam, model, make, specials, t["image_token_id"])


#: gemma: 6 layers, not the scaler's 35 (fixtures_vision leaves
#: num_hidden_layers unscaled for gemma4). Keeps the
#: real structure: sliding + full layers, KV-shared tail layers, PLE.
GEMMA_TEXT = dict(num_hidden_layers=6, num_kv_shared_layers=2,
                  layer_types=["sliding_attention", "sliding_attention",
                               "full_attention"] * 2)


def _gemma_rig(tmp):
    from fixtures_vision_gemma4 import tiny_text_model
    cfg = fv.tiny_config("gemma4")
    model, _ = tiny_text_model(dict(cfg["text_config"], **GEMMA_TEXT))
    mx.eval(model.parameters())

    def make():
        from knurlogic.engine.families.gemma4.vision import build
        mx.random.seed(0)
        f = build("/nonexistent", model, cfg)
        mx.eval(f.vision_tower.parameters(), f.embed_vision.parameters())
        return f
    t = fv.tiny_ids("gemma4")
    # The released tokenizer's strings (gemma e4b tokenizer.json): the
    # template emits "<|image|>"; "<image_soft_token>" is not a token there.
    specials = {"<|image|>": t["image_token_id"],
                "<|image>": t["boi_token_id"],
                "<image|>": t["eoi_token_id"]}
    # 144 x 144 -> 9 x 9 patches -> 9 tokens after the 3x3 pool
    return Rig("gemma4", model, make, specials, t["image_token_id"],
               img_size=(144, 144))


#: glm5_next's text config at fixture size. The scaler gives only hidden
#: and vocab (fixtures_vision.REAL), so every other field would be the
#: RELEASED default (45 layers, 288 experts: billions of parameters). Same
#: structure: linear-attention layers, one sparse-attention (MLA + indexer)
#: layer, dense and MoE MLPs, hyper-connections.
GLM_TEXT = dict(num_hidden_layers=4,
                layer_types=["linear_attention"] * 3
                + ["deepseek_sparse_attention"],
                indexer_types=["full"] * 4,
                mlp_layer_types=["dense", "dense", "dense", "sparse"],
                intermediate_size=256, moe_intermediate_size=64,
                n_routed_experts=4, num_experts_per_tok=2,
                num_attention_heads=4, num_key_value_heads=4,
                kv_lora_rank=32, q_lora_rank=64, qk_nope_head_dim=32,
                v_head_dim=32, qk_head_dim=32, index_head_dim=32,
                index_n_heads=4, linear_head_dim=32, linear_num_heads=4,
                first_k_dense_replace=3, pad_token_id=0,
                # required by mlx-vlm 0.6.17's TextConfig, the version the
                # released GLM rungs are built on; real values at tiny size
                n_shared_experts=1, routed_scaling_factor=2.5,
                qk_rope_head_dim=0, max_position_embeddings=4096,
                rms_norm_eps=1e-5, index_topk=64,
                linear_attn_config=dict(num_heads=4, head_dim=32,
                                        gate_lower_bound=-5.0,
                                        short_conv_kernel_size=4,
                                        kda_layers=[0, 1, 2],
                                        full_attn_layers=[3]))


def _glm_call(model, ids, cache, input_embeddings=None, **kw):
    """GLM's own convention: `inputs_embeds`, and a LanguageModelOutput."""
    return model(ids, cache=cache, inputs_embeds=input_embeddings,
                 **kw).logits


def _glm_rig(tmp):
    from fixtures_vision_glm5 import glm5_family, glm5_tiny_config

    from knurlogic.engine import register
    register.register("glm5_next")
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    cfg = glm5_tiny_config()
    mx.random.seed(0)
    model = LanguageModel(TextConfig.from_dict(dict(cfg["text_config"],
                                                    **GLM_TEXT)))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())

    def make():
        f = glm5_family(cfg)
        mx.eval(f.tower_model.parameters())
        return f
    t = fv.tiny_ids("glm5_next")
    specials = {"<|image|>": t["image_token_id"],
                "<|begin_of_image|>": t["image_start_token_id"],
                "<|end_of_image|>": t["image_end_token_id"]}
    # 56 x 56 -> 4 x 4 patches of 14 -> 2 x 2 merged tokens
    return Rig("glm5_next", model, make, specials, t["image_token_id"],
               call=_glm_call, img_size=(56, 56))


def _deepseek_rig(tmp):
    """DeepSeek-V4-Flash-Vision-Exp at test_vision_deepseek's tiny size, on
    this file's vocab: image tokens are ids 512..516, past the vocab."""
    import test_vision_deepseek as td

    from knurlogic.engine.families.deepseek.vision import DeepseekVisionFamily
    M = td._arch()
    cfg = dict(td.TINY, vocab_size=VOCAB)
    model = M.Model(M.ModelArgs.from_dict(cfg))
    td._random(model, np.random.default_rng(0))
    mx.eval(model.parameters())
    image_id = 501

    def make():
        f = DeepseekVisionFamily(cfg, image_id)
        td._random(f._build_tower(), np.random.default_rng(1))
        core = model.model
        f.set_rows({k: getattr(core, k) for k in
                    ("image_start", "image_pad", "image_newline",
                     "image_end")})
        mx.eval(f.tower.parameters(), f.rows)
        return f
    return Rig("deepseek_v4", model, make,
               {"<｜deepseek_image｜>": image_id}, image_id)


FAMILIES = ("qwen3_5", "qwen3_5_moe", "qwen4_exp", "gemma4", "glm5_next",
            "deepseek_v4")
_RIGS = {}


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    """Built once per family per module (the trunks are the slow part);
    every test gets a FRESH family object from `make_family`."""
    def get(name):
        if name not in _RIGS:
            tmp = tmp_path_factory.mktemp("e2e")
            _RIGS[name] = (_gemma_rig(tmp) if name == "gemma4"
                           else _glm_rig(tmp) if name == "glm5_next"
                           else _deepseek_rig(tmp) if name == "deepseek_v4"
                           else _qwen_rig(name, tmp))
        return _RIGS[name]
    return get


# --- counting the tower from outside -------------------------------------------

class Counted:
    """A proxy for a tower module: every call counted, everything else
    (weights, dtype, submodules) passed through."""

    def __init__(self, inner):
        self.__dict__["_inner"] = inner
        self.__dict__["calls"] = 0

    def __call__(self, *a, **k):
        self.__dict__["calls"] += 1
        return self._inner(*a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def counted(fam):
    for attr in ("tower", "vision_tower", "tower_model"):
        if getattr(fam, attr, None) is not None:
            c = Counted(getattr(fam, attr))
            setattr(fam, attr, c)
            return c
    raise AssertionError(f"no tower on {type(fam).__name__}")


# --- the reference: the family's own model-level forward -----------------------

def user(text, *images):
    parts = [{"type": "text", "text": text}]
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return {"role": "user", "content": parts}


def reference(rig, fam, messages, n=MAX_TOKENS):
    """Greedy tokens and first-token logprobs for `messages`, computed the
    way the family packages were tested: the prompt tokenized with each
    image part replaced by the family's placeholder, pads expanded, ONE
    forward of embed + positions through the trunk, then decode (Qwen on
    rope_delta). Its own store; nothing from the serve path."""
    from knurlogic.engine.vision import images
    tok = SpecTok(rig.specials)
    feats, refs_by = {}, {}
    text, refs = "", []
    # as the serve path joins a message's parts: by the template
    # (runtime/prompt.part_separator; this file's template joins with "")
    from knurlogic.engine.runtime.prompt import part_separator
    sep = part_separator(tok)
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            s = ""
            for i, p in enumerate(c):
                if i:
                    s += sep
                if p["type"] == "text":
                    s += p["text"]
                else:
                    img, sha = images.load(p["image_url"]["url"])
                    pix, ref = fam.preprocess(img, sha)
                    enc = fam.encode(pix, ref)
                    feats[(sha, ref.proc_hash)] = enc
                    refs_by[(sha, ref.proc_hash)] = ref
                    refs.append(ref)
                    s += fam.placeholder_text(ref)
            c = s
        text += f"{m['role']}: {c}\n"
    text += "assistant: "
    iid = fam.spec.image_token_id
    ids = tok.encode(text)
    key = K.expand(K.expand_pads(ids, refs, iid), refs, iid)
    if hasattr(fam, "frame_key"):           # DeepSeek-V4's alignment pads
        key, _ = fam.frame_key(key, [key])
    emb = dict(fam.embed(rig.model, key, 0,
                         lambda s, p: feats[(s, p)]))
    pos, delta = fam.positions(key, lambda s, p: refs_by[(s, p)])
    kw = dict(emb)
    dkw = {}
    if pos is not None:
        kw["position_ids"] = pos
        dkw["rope_delta"] = delta
    cache = rig.model.make_cache()
    x = mx.array(K.to_ids(key, iid))[None]
    lg = rig.call(x, cache, **kw)[0, -1]
    out, first = [], None
    ban = mx.array(BANNED)
    for i in range(n):
        lg = lg.astype(mx.float32)
        if first is None:
            # the response's logprobs are of the RAW logits (before
            # logit_bias), as batch_generator computes them
            first = np.array(lg - mx.logsumexp(lg))
        lg[ban] = -1e9
        t = int(mx.argmax(lg).item())
        out.append(t)
        if i + 1 < n:
            lg = rig.call(mx.array([[t]]), cache, **dkw)[0, -1]
    return key, out, first


def said(rig, resp):
    """The COMMITTED token ids of a response, re-encoded from its text
    (lossless, SpecTok). Not the logprobs entries' `id`: mlx-lm fills that
    from the top of the RAW logprobs (server._format_top_logprobs), before
    logit_bias -- with a banned id on top it names a token never sampled."""
    return SpecTok(rig.specials).encode(
        resp["choices"][0]["message"].get("content") or "")


def chat(h, messages, **kw):
    body = dict(messages=messages, max_tokens=MAX_TOKENS, temperature=0.0,
                logprobs=True, top_logprobs=5, logit_bias=BAN)
    body.update(kw)
    status, _, payload = h.post(body)
    assert status == 200, payload
    return json.loads(payload)


def harness(server, rig, fam, **cli):
    return Harness(server, rig.model, rig.tok(), family=fam, **cli)


def close_tops(a, b, atol):
    assert a.keys() == b.keys(), (a, b)
    for t in a:
        assert abs(a[t] - b[t]) < atol, (t, a[t], b[t])


Q1 = "look "
Q2 = "and what else is there?"


def two_turns(h, url):
    m1 = [user(Q1, url)]
    r1 = chat(h, m1)
    reply = r1["choices"][0]["message"].get("content", "")
    m2 = m1 + [{"role": "assistant", "content": reply}, user(Q2)]
    return m1, r1, m2, chat(h, m2)


# --- E1-E3 ----------------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_e1_e2_e3_two_turns_through_the_serve_path(server, rigs, name):
    rig = rigs(name)
    fam = rig.make_family()
    tower = counted(fam)
    url = rig.url(11)
    h = harness(server, rig, fam)
    m1 = [user(Q1, url)]
    r1 = chat(h, m1)
    before = h.prefilled
    m2 = m1 + [{"role": "assistant",
                "content": r1["choices"][0]["message"].get("content", "")},
               user(Q2)]
    r2 = chat(h, m2)
    prefilled2 = h.prefilled - before

    # E3: one tower run across both turns (turn 2 resends the image: a
    # store hit, and its KV is inside the cached prefix anyway)
    assert tower.calls == 1
    assert h.vision.pinned_count() == 0
    # ...and the cached conversation HOLDS its image: the prompt cache's
    # entries pin it in the store, so byte pressure cannot evict it while a
    # later turn could still need it. Through the
    # real install path, not the unit test's.
    from knurlogic.engine.vision import cachehook
    assert cachehook.pinned_entries(h.cache) >= 1

    # E1: turn 1 == the family's own one-shot forward
    ref_fam = rig.make_family()
    key1, want1, lp1 = reference(rig, ref_fam, m1)
    assert K.has_image(key1)
    assert r1["usage"]["prompt_tokens"] == len(key1)
    assert said(rig, r1) == want1, (said(rig, r1), want1)
    got = first_top(r1)
    close_tops(got, {t: float(lp1[t]) for t in got}, 1e-3)

    # E2: turn 2 warm (turn 1 restored from the prompt cache) == cold
    turn1 = r1["usage"]["prompt_tokens"] + r1["usage"]["completion_tokens"]
    assert cached(r2) == turn1
    assert prefilled2 == r2["usage"]["prompt_tokens"] - turn1  # USED
    cold = harness(server, rig, rig.make_family())
    c2 = chat(cold, m2)
    assert cached(c2) == 0
    assert said(rig, r2) == said(rig, c2), (said(rig, r2), said(rig, c2))
    close_tops(first_top(r2), first_top(c2), 1e-4)
    # and both == the reference for turn 2
    _, want2, _ = reference(rig, ref_fam, m2)
    assert said(rig, c2) == want2

    # E1 again with short chunks: 6-token prefill steps cut a Qwen/GLM
    # image mid-span (positions and embeds sliced per chunk), make gemma
    # snap, and give every trunk 2-8 token forwards (GLM's short-block
    # projection path, _vendor/quantized_verifier.py)
    short = harness(server, rig, rig.make_family(), prefill_step_size=6)
    assert said(rig, chat(short, m1)) == want1


# --- E4: two rows, one batch ----------------------------------------------------

def post_together(h, bodies):
    """Every body queued BEFORE the scheduler starts, so they are admitted
    into one batch: the first builds it, the rest join it."""
    replies = [h.submit(b) for b in bodies]
    h.sched.start()
    return [h.result(r) for r in replies]


def _body(messages):
    return dict(messages=messages, max_tokens=MAX_TOKENS, temperature=0.0,
                logprobs=True, top_logprobs=5, logit_bias=BAN)


@pytest.mark.parametrize("name", FAMILIES)
def test_e4_image_row_and_text_row_in_one_batch(server, rigs, name,
                                                 monkeypatch):
    """Per-row positions in one decode: the image row (Qwen: MRoPE, a
    rope_delta) and a text row of a different length, in one batch, each
    == its solo run. The batch width is read off MTPBatch.step, so a run
    that served them one after the other cannot pass."""
    from knurlogic.engine.mtp import batch_loop
    rig = rigs(name)
    img_msgs = [user("what is in this picture? ", rig.url(21))]
    txt_msgs = [user("tell me a story about a long river and a boat")]

    widths = []
    real_step = batch_loop.MTPBatch.step

    def step(self):
        widths.append(len(self))
        return real_step(self)
    monkeypatch.setattr(batch_loop.MTPBatch, "step", step)

    h = Harness(server, rig.model, rig.tok(), family=rig.make_family(),
                start=False)
    together = post_together(h, [_body(img_msgs), _body(txt_msgs)])
    assert max(widths) == 2                     # both rows in one step
    widths.clear()

    alone = []
    for m in (img_msgs, txt_msgs):
        s = harness(server, rig, rig.make_family())
        alone.append(chat(s, m))
    assert max(widths) == 1
    for t, a in zip(together, alone):
        assert said(rig, t) == said(rig, a), (said(rig, t), said(rig, a))
        close_tops(first_top(t), first_top(a), 1e-3)


# --- E5: gemma's chunk snapping ---------------------------------------------------

@pytest.mark.parametrize("side", [144, 240])
def test_e5_gemma_chunk_edges_never_split_an_image(server, rigs, monkeypatch,
                                                   side):
    """prefill_step_size=16 with the image block straddling token 16 (144px:
    9 tokens at 11..20) and with a block longer than a whole step (240px:
    25 tokens at 11..36). gemma attends bidirectionally inside a block, so
    a chunk edge inside it would compute the first half blind to the
    second. Tokens == the one-shot reference; every forward's span is
    recorded and none cuts the block."""
    rig = rigs("gemma4")
    fam = rig.make_family()
    url = data_url(31, side, side)
    msgs = [user(Q1, url)]
    key, want, _ = reference(rig, rig.make_family(), msgs)
    (span,) = K.image_spans(key)
    assert span.start < 16 < span.end          # a 16-step chunk WOULD cut it

    edges, pos = [], [0]
    from knurlogic.engine.mtp import batch_generator as bg
    real = bg.logits_trunk

    def spy(model):
        t = real(model)

        def call(inputs, cache=None, **kw):
            if pos[0] < len(key):              # prefill forwards only
                n = inputs.shape[1]
                edges.append((pos[0], pos[0] + n))
                pos[0] += n
            return t(inputs, cache=cache, **kw)

        class Spy:
            def __call__(self, *a, **k):
                return call(*a, **k)

            def __getattr__(self, name):
                return getattr(t, name)
        return Spy()
    monkeypatch.setattr(bg, "logits_trunk", spy)

    h = harness(server, rig, fam, prefill_step_size=16)
    r = chat(h, msgs)
    assert edges[0][0] == 0 and edges[-1][1] == len(key)
    assert len(edges) > 1                      # it really was chunked
    for a, b in edges:
        assert not (span.start < a < span.end), (a, b, span)
        assert not (span.start < b < span.end), (a, b, span)
    assert said(rig, r) == want


# --- E6: a seed does not take an image off the batch path ---------------------------

def test_e6_seeded_image_request_takes_the_batch_path(server, rigs):
    """A seeded image request comes through the batch engine like any
    other (mlx-lm sent seeds down a sequential path that could not read a
    key) -- and answers as the reference does."""
    from knurlogic.engine.serve import state
    rig = rigs("qwen3_5")
    msgs = [user(Q1, rig.url(41))]
    h = harness(server, rig, rig.make_family())
    r = chat(h, msgs, seed=3)
    assert state.VISION_STATS.get("requests") == 1
    _, want, _ = reference(rig, rig.make_family(), msgs)
    assert said(rig, r) == want


# --- the trunks vision edited are pinned as they are ---------------------------------

def test_every_vision_trunk_pin_matches_its_file():
    """The vision work edits vendored trunks (MRoPE; the image-block
    mask). A trunk whose file no longer matches PINS.json reads DRIFTED in
    `doctor`; the qwen3_5_moe_text pin test alone would not catch
    gemma4_text drifting."""
    from knurlogic.engine import arch
    if not arch.PINNED_SHA256:
        pytest.skip("no pins on this checkout")
    for mt in ("qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text",
               "gemma4_text"):
        for row in arch.check(mt):
            if row.module in arch.PINNED_SHA256 and row.vendored:
                assert row.state == "OK", f"{row.module} is {row.state}"


# --- the cache report ------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_usage_reports_what_the_cache_actually_did(server, rigs, name):
    """usage.knurlogic.cache comes from the engine, not the trie: on turn 2
    the image sits in the reused span, cached_tokens is what was USED, and
    prompt = used + prefilled. Turn 1 used nothing."""
    rig = rigs(name)
    h = harness(server, rig, rig.make_family())
    m1, r1, m2, r2 = two_turns(h, rig.url(11))

    c1 = r1["usage"]["knurlogic"]["cache"]
    assert c1["used"] == 0 and c1["via"] == "none"
    assert c1["images"] == {"total": 1, "in_cached_span": 0, "prefilled": 1,
                            "encoded": 1}

    u2 = r2["usage"]
    c2 = u2["knurlogic"]["cache"]
    assert c2["used"] > 0 and c2["discarded"] == 0
    assert u2["prompt_tokens_details"]["cached_tokens"] == c2["used"]
    assert c2["used"] + c2["prefilled"] == u2["prompt_tokens"]
    # turn 2 resends the image: a store hit, no tower run
    assert c2["images"] == {"total": 1, "in_cached_span": 1, "prefilled": 0,
                            "encoded": 0}


def test_cache_report_says_when_an_offered_prefix_was_discarded():
    """The reason the report exists: the trie offers, the engine refuses
    (a drafting row with no aligned head), and cached_tokens must not claim
    the offer. Shown on the usage the request stage builds."""
    from knurlogic.engine.runtime.request import Request
    r = Request(types.SimpleNamespace(reset=lambda: None), sequences={},
                prompt_tokens=50)
    usage = r.usage({"offered": 40, "used": 0, "discarded": 40,
                     "prefilled": 50, "via": "none", "images": {},
                     "checkpoints_stored": 0})
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert usage["knurlogic"]["cache"]["discarded"] == 40


# --- encode-twice: the tower is deterministic -----------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_encoding_the_same_image_twice_is_bit_identical(rigs, name):
    """A cached image's features stand in for a fresh encode only if the
    tower is deterministic: preprocess + encode the same picture twice,
    from scratch, and the features must be equal bit for bit (design v2,
    review #7, kept as a cheap invariant)."""
    import hashlib

    from PIL import Image
    rig = rigs(name)
    fam = rig.make_family()
    img = Image.new("RGB", rig.img_size, (200, 40, 90))
    for x in range(0, rig.img_size[0], 7):
        img.putpixel((x, x % rig.img_size[1]), (0, 255, 0))
    sha = hashlib.sha256(img.tobytes()).hexdigest()
    a = fam.encode(*fam.preprocess(img, sha))
    b = fam.encode(*fam.preprocess(img.copy(), sha))
    mx.eval(a.feats, b.feats)
    assert a.feats.shape == b.feats.shape
    assert mx.array_equal(a.feats, b.feats).item()


def test_encode_twice_can_fail(rigs):
    """The comparison sees a one-pixel change."""
    from PIL import Image
    rig = rigs(FAMILIES[0])
    fam = rig.make_family()
    img = Image.new("RGB", rig.img_size, (200, 40, 90))
    other = img.copy()
    other.putpixel((0, 0), (0, 0, 0))
    a = fam.encode(*fam.preprocess(img, "a"))
    b = fam.encode(*fam.preprocess(other, "b"))
    assert not mx.array_equal(a.feats, b.feats).item()
