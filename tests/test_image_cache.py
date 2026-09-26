"""Images as real context, end to end through knurlogic's own server.

Gates G5-G9 (docs/design/vision.md): a request body goes through the
OpenAI surface (interfaces/http/openai.py), is tokenized on the scheduler's
thread (vision included), meets the real prompt cache (mlx-lm's
LRUPromptCache behind the scheduler's wrapper) and comes back as the
response. Sentinel keys reaching code that assumes ints (prefix
arithmetic, the segment trim, `insert_cache`, usage counts) is run for
real rather than argued about. (Until 2026-09-25 this drove mlx-lm's own
server objects; the gates are the same.)

Fixtures, and why each is shaped the way it is:

* a tiny random qwen3_5 (float32, seed 0, vocab 512): a HYBRID trunk, so its
  recurrent caches take only exact-prefix hits (design risk 4) -- the
  hardest case for G8, and the real target family.
* `ByteTok`: a template that does NOT rewrite history (critique 3: Qwen's
  real template drops earlier thinking, which would fail G8 for template
  reasons). Byte ids 0-255, every other id a private-use char, so a reply
  re-encodes to exactly the ids generated and turn 2's key extends turn 1's
  stored entry. `<image>` is the one image token (P0 contract: the
  placeholder is a single special id). The image and eos ids are banned by
  logit_bias so a random model cannot type them.
* P0's `StubFamily` (identity tower, pixel-dependent features), never a
  real family package (design P4). Tower calls are counted by wrapping
  `fam.tower` from OUTSIDE (critique 4).
"""
import base64
import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402

IMG = 500          # the stub's image token id
EOS = 501
VOCAB = 512
BAN = {str(IMG): -1e9, str(EOS): -1e9}


# --- fixtures ---------------------------------------------------------------

class ByteTok:
    """The raw tokenizer (mlx-lm wraps it in its own TokenizerWrapper, as it
    does a Hugging Face one): lossless both ways, one image token."""

    chat_template = "role: content, one line each"
    clean_up_tokenization_spaces = False
    eos_token_id = EOS

    def get_vocab(self):
        return {}                        # no thinking, no tool tokens

    def encode(self, text, add_special_tokens=True):
        out = []
        for i, part in enumerate(text.split("<image>")):
            if i:
                out.append(IMG)
            for ch in part:
                o = ord(ch)
                if o < 256:
                    out.append(o)
                elif 0xE000 <= o < 0xE000 + VOCAB:
                    out.append(o - 0xE000)
                else:
                    out.extend(ch.encode("utf-8"))
        return out

    def decode(self, ids, **kw):
        s = []
        for t in ids:
            t = int(t)
            s.append("<image>" if t == IMG else "" if t == EOS
                     else chr(t) if t < 256 else chr(0xE000 + t))
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


def byte_tok():
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    return TokenizerWrapper(ByteTok(), eos_token_ids=[EOS])


def tiny_model():
    from knurlogic.engine import register
    register.register("qwen3_5")
    from mlx_lm.models import qwen3_5 as arch

    mx.random.seed(0)
    tc = dict(model_type="qwen3_5", hidden_size=128, intermediate_size=256,
              num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=32, vocab_size=VOCAB,
              linear_num_value_heads=4, linear_num_key_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32,
              full_attention_interval=2, tie_word_embeddings=False)
    model = arch.Model(arch.ModelArgs(model_type="qwen3_5", text_config=tc))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def stub_family(**kw):
    return fv.StubFamily(IMG, 128, **kw)


def counted(fam):
    """Wrap the stub's tower from outside; returns the call counter."""
    n = {"calls": 0}
    real = fam.tower

    def tower(pixel_values):
        n["calls"] += 1
        return real(pixel_values)
    fam.tower = tower
    return n


def data_url(seed, w=16, h=12):
    b = fv.png_bytes(fv.tiny_image(w, h, seed))
    return "data:image/png;base64," + base64.b64encode(b).decode()


def user(text, *images):
    parts = [{"type": "text", "text": text}]
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return {"role": "user", "content": parts}


class Provider:
    """The fields vision.bind reads off a loaded model's host."""

    def __init__(self, model, tok):
        self.model, self.tokenizer = model, tok
        self.model_key = ("tiny", None, None)


class Host(Provider):
    """ModelHost without the loading: always ready with the tiny model."""
    state = "ready"
    error = ""
    path = "/models/tiny"
    loaded_at = 0.0

    def expect(self, path):
        pass

    def status(self):
        return {"state": "ready", "model": self.path, "error": ""}


@pytest.fixture
def server(monkeypatch):
    """The serve package's module state, pristine for each test."""
    from knurlogic.engine.serve import state
    monkeypatch.setattr(state, "VISION",
                        {"serve": None, "model": None, "error": ""})
    monkeypatch.setattr(state, "VISION_STATS", {})
    monkeypatch.setattr(state, "DRAFT", dict(state.DRAFT, head=None,
                                             on=False, batch_installed=False))
    monkeypatch.setattr(state, "SERVED", {"path": None, "provider": None})
    yield None
    from knurlogic.engine.vision import set_served_vision
    set_served_vision(None)


def materialize(obj, depth=12, seen=None):
    """Evaluate every array reachable from obj (modules, attributes) here:
    MLX streams are per thread, and a lazy array made on this thread cannot
    be evaluated on the scheduler's. A real load builds everything on the
    scheduler thread; a test builds its family on this one."""
    import mlx.nn as nn
    seen = set() if seen is None else seen
    if id(obj) in seen or depth < 0:
        return
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        mx.eval(obj)
    elif isinstance(obj, nn.Module):
        # a module keeps private arrays (`_rope`, buffers) as dict entries,
        # which parameters() leaves out
        for v in list(obj.values()) + list(vars(obj).values()):
            materialize(v, depth - 1, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            materialize(v, depth - 1, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            materialize(v, depth - 1, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for v in vars(obj).values():
            materialize(v, depth - 1, seen)


class Harness:
    """One scheduler (its own prompt cache and image store) and the OpenAI
    surface in-process. The scheduler thread is never stopped: MLX arrays
    it made inside the shared tiny model must not be freed after it ends
    (measured: segfault), and a test process may keep idle threads."""

    def __init__(self, srv, model, tok=None, family=None, *, store=None,
                 start=True, prefill_step_size=16):
        from knurlogic.engine.runtime.scheduler import Scheduler
        from knurlogic.engine.serve import state
        from knurlogic.interfaces.http import scout
        from knurlogic.interfaces.http.server import App
        self.host = Host(model, tok or byte_tok())
        state.SERVED["provider"] = self.host
        self.vision = None
        materialize(model)
        if family is not None:
            materialize(family)
            from knurlogic.engine.vision import set_served_vision
            from knurlogic.engine.vision.request import VisionServe
            from knurlogic.engine.vision.store import ImageStore
            self.vision = VisionServe(family, store if store is not None else ImageStore(),
                                      self.host.model_key)
            state.VISION.update(serve=self.vision, model=model)
            set_served_vision(family.spec)
        self.sched = Scheduler(self.host, prefill_step_size=prefill_step_size)
        if start:
            self.sched.start()
        self.app = App(self.sched, served=lambda: {"id": "tiny"})
        self.app.translate = None

    @property
    def cache(self):
        """The prompt cache's LRU (what cachehook pins through)."""
        return self.sched.cache.lru

    def submit(self, body):
        """Queue a request; its Reply. Several before start() share a batch."""
        job, reply = self.app.submit(body, chat=True)
        return reply

    @staticmethod
    def result(reply) -> dict:
        from knurlogic.interfaces.http import openai as O
        first = reply.first(timeout=120)
        if first[0] == "error":
            raise O._status_of(first[1])
        return reply.complete(first)

    @property
    def prefilled(self) -> int:
        """Prompt tokens the engine actually prefilled, so far."""
        ex = self.sched._ex
        return ex.gen._prompt_tokens_counter if ex is not None else 0

    def post(self, body, path="/v1/chat/completions"):
        """(status, [], body bytes)."""
        from knurlogic.interfaces.http import openai as O
        try:
            job, reply = self.app.submit(body, chat=True)
            first = reply.first(timeout=120)
            if first[0] == "error":
                raise O._status_of(first[1])
            out = reply.complete(first)
        except O.ApiError as e:
            return e.status, [], json.dumps(e.body()).encode()
        return 200, [], json.dumps(out).encode()

    def chat(self, messages, **kw):
        body = dict(messages=messages, max_tokens=12, temperature=0.0,
                    logprobs=True, top_logprobs=5, logit_bias=BAN)
        body.update(kw)
        status, _, payload = self.post(body)
        assert status == 200, payload
        return json.loads(payload)

    def close(self):
        pass


def tokens_of(resp):
    return [c["id"] for c in resp["choices"][0]["logprobs"]["content"]]


def first_top(resp):
    """{token id: logprob} of the first generated token's top 5: the
    last-prefill logits, as the response carries them."""
    tops = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {t["id"]: t["logprob"] for t in tops}


def cached(resp):
    return resp["usage"].get("prompt_tokens_details", {}).get(
        "cached_tokens", 0)


def normalize(payload):
    d = json.loads(payload)
    d["id"] = "ID"
    return json.dumps(d, sort_keys=True).encode()


@pytest.fixture(scope="module")
def model():
    return tiny_model()


# --- G5: text is unchanged ----------------------------------------------------

TEXT = [{"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello there, tiny model"}]


def test_g5_text_on_a_vision_model_matches_a_text_model(server, model):
    """With a vision family served, a text-only request must answer as it
    does with none: same text and tokens, logprobs to float tolerance."""
    body = dict(messages=TEXT, max_tokens=10, temperature=0.0, logprobs=True)
    main = Harness(server, model)
    try:
        want = json.loads(main.post(body)[2])
    finally:
        main.close()
    ours = Harness(server, model, family=stub_family())
    try:
        got = json.loads(ours.post(body)[2])
        # the engine that answered was the one serving the vision family
        assert ours.sched._ex.gen._vision is ours.vision
    finally:
        ours.close()
    assert tokens_of(got) == tokens_of(want)
    assert got["choices"][0]["message"] == want["choices"][0]["message"]
    for a, b in zip(got["choices"][0]["logprobs"]["content"],
                    want["choices"][0]["logprobs"]["content"]):
        assert abs(a["logprob"] - b["logprob"]) < 1e-4


def test_images_to_a_text_model_are_a_400(server, model):
    """D3: `_post` refuses and does nothing else. Nothing reaches the
    generator thread (mlx-lm's own answer would be a 404 from
    process_message_content's ValueError)."""
    h = Harness(server, model)
    try:
        status, _, payload = h.post(dict(messages=[user("hi", data_url(0))],
                                         max_tokens=4))
        assert status == 400
        assert b"no vision" in payload
        # text still serves
        assert h.post(dict(messages=TEXT, max_tokens=2))[0] == 200
    finally:
        h.close()


# --- G6-G9: the image cache ---------------------------------------------------

def _two_turns(h, img_url, q2="and what else?"):
    m1 = [user("what is this? ", img_url)]
    r1 = h.chat(m1)
    reply = r1["choices"][0]["message"].get("content", "")
    m2 = m1 + [{"role": "assistant", "content": reply}, user(q2)]
    r2 = h.chat(m2)
    return m1, r1, m2, r2


def test_g6_tower_runs_once_across_two_turns(server, model):
    fam = stub_family()
    n = counted(fam)
    h = Harness(server, model, family=fam)
    try:
        _two_turns(h, data_url(1))
        assert n["calls"] == 1
        assert h.vision.pinned_count() == 0      # every pin released
    finally:
        h.close()


def test_g7_g8_warm_turn_two_equals_cold_and_reuses_the_image(server, model):
    """G7: turn 2 through the prompt cache and the image store == turn 2
    cold (fresh server): tokens identical, last-prefill logits atol 1e-4.
    G8: turn 2's cached prefix covers ALL of turn 1 -- prompt and reply --
    so the image is never prefilled twice. Asserted on mlx-lm's own
    `cached_tokens`, i.e. its prefix arithmetic on the key."""
    fam = stub_family()
    n = counted(fam)
    warm = Harness(server, model, family=fam)
    try:
        m1 = [user("what is this? ", data_url(2))]
        r1 = warm.chat(m1)
        before = warm.prefilled
        m2 = m1 + [{"role": "assistant",
                    "content": r1["choices"][0]["message"].get("content", "")},
                   user("and what else?")]
        r2 = warm.chat(m2)
        prefilled = warm.prefilled - before
    finally:
        warm.close()
    turn1 = r1["usage"]["prompt_tokens"] + r1["usage"]["completion_tokens"]
    assert cached(r2) == turn1                       # G8: the trie's hit ...
    assert prefilled == r2["usage"]["prompt_tokens"] - turn1   # ... was USED
    assert n["calls"] == 1

    cold = Harness(server, model, family=stub_family())
    try:
        c2 = cold.chat(m2)
    finally:
        cold.close()
    assert cached(c2) == 0
    assert tokens_of(r2) == tokens_of(c2)            # G7
    a, b = first_top(r2), first_top(c2)
    assert a.keys() == b.keys()
    for t in a:
        assert abs(a[t] - b[t]) < 1e-4


def test_g9_a_different_image_of_the_same_size_misses(server, model):
    """The false-hit gate: same size, so the same token ids before the key
    -- only the sentinels tell them apart. B after A must not reuse A's KV
    past the image start, and must answer as B does cold."""
    fam = stub_family()
    h = Harness(server, model, family=fam)
    try:
        ra = h.chat([user("describe: ", data_url(3))])
        rb = h.chat([user("describe: ", data_url(4))])
    finally:
        h.close()
    start = len(ByteTok().encode("user: describe: "))
    assert cached(rb) <= start
    cold = Harness(server, model, family=stub_family())
    try:
        cb = cold.chat([user("describe: ", data_url(4))])
    finally:
        cold.close()
    assert tokens_of(rb) == tokens_of(cb)
    assert tokens_of(rb) != tokens_of(ra)


# --- G7b: positions from the whole key (D4) -----------------------------------

class PosFamily(fv.StubFamily):
    """The stub with Qwen-shaped positions: every token after an image runs
    DELTA positions ahead of its index, [3, 1, L] ids, and rope_delta. Pure
    in the key, as the contract says."""

    DELTA = 7

    def positions(self, key, refs):
        from knurlogic.engine.vision.key import image_spans
        spans = image_spans(key)
        if not spans:
            return None, 0
        shift = [0] * len(key)
        d = 0
        for s in spans:
            for j in range(s.end, len(key)):
                shift[j] += self.DELTA
            d += self.DELTA
        p = mx.array([j + shift[j] for j in range(len(key))])
        return mx.broadcast_to(p[None, None], (3, 1, len(key))), d


class PosModel:
    """The tiny trunk behind a position-dependent input term, so a wrong
    position changes the answer. With no `position_ids` it uses what a
    trunk would -- the cache offset -- which is exactly what a row gets if
    D4 is missing (positions only while the new span has an image)."""

    def __init__(self, inner, seed=1):
        self.inner = inner
        mx.random.seed(seed)
        self.E = mx.random.normal((2048, 128)) * 0.5
        mx.eval(self.E)

    @property
    def embed_tokens(self):
        return self.inner.language_model.model.embed_tokens

    def make_cache(self):
        return self.inner.make_cache()

    def __call__(self, inputs, cache=None, input_embeddings=None,
                 position_ids=None):
        if input_embeddings is None:
            input_embeddings = self.embed_tokens(inputs)
        L = inputs.shape[1]
        if position_ids is None:
            off = next(c.offset for c in cache if hasattr(c, "offset"))
            off = off if isinstance(off, mx.array) else mx.array([off])
            pos = off[:, None] + mx.arange(L)[None]
        else:
            pos = position_ids[0]
        pos = mx.broadcast_to(pos, (inputs.shape[0], L))
        return self.inner(inputs, cache=cache,
                          input_embeddings=input_embeddings + self.E[pos])


def pos_family(pm):
    return PosFamily(IMG, 128, embed_fn=pm.embed_tokens)


def test_g7b_text_turn_after_an_image_warm_equals_cold(server, model):
    """Turn 1 has the image; turn 2 is TEXT ONLY and warm. Its positions
    must still carry the image's rope_delta -- prefill and every decode
    step. Fails if positions are computed only when the new span holds an
    image (critique B1)."""
    pm = PosModel(model)
    warm = Harness(server, pm, family=pos_family(pm))
    try:
        m1, r1, m2, r2 = _two_turns(warm, data_url(5), q2="more words here")
    finally:
        warm.close()
    assert cached(r2) > 0                     # it really was warm
    cold = Harness(server, pm, family=pos_family(pm))
    try:
        c2 = cold.chat(m2)
    finally:
        cold.close()
    assert tokens_of(r2) == tokens_of(c2)
    a, b = first_top(r2), first_top(c2)
    assert a.keys() == b.keys()
    for t in a:
        assert abs(a[t] - b[t]) < 1e-4


# --- the load path --------------------------------------------------------------

def test_load_builds_the_family_through_the_registry(server, model, tmp_path,
                                                     monkeypatch):
    """bind_vision reads config.json, builds through registry.build (here
    pointed at the P0 stub builder, exactly as a real family is), loads the
    tower and publishes the spec; clear_vision takes it all back."""
    from knurlogic.engine import serve
    from knurlogic.engine.serve import state
    from knurlogic.engine.vision import registry, served_vision

    monkeypatch.setitem(registry.FAMILIES, "qwen3_5",
                        "fixtures_vision:stub_build")
    cfg = dict(model_type="qwen3_5", image_token_id=IMG,
               vision_config={"patch_size": 4},
               text_config={"hidden_size": 128})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    prov = Provider(model, byte_tok())
    v = serve.bind_vision(str(tmp_path), prov)
    assert v is not None and state.VISION["serve"] is v
    assert served_vision() is v.spec and serve.served_vision() is v.spec
    assert v.spec.image_token_id == IMG
    assert serve.vision_status()["on"]

    serve.clear_vision()
    assert served_vision() is None and state.VISION["serve"] is None

    cfg.pop("vision_config")                      # a text-only artifact
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert serve.bind_vision(str(tmp_path), prov) is None
    assert served_vision() is None


def test_images_that_overflow_the_store_together_are_a_413(server, model):
    """No count limit: the bound is the store's memory. Every image of a
    request must be resident at admission, so a request whose images do not
    fit together is refused with the numbers -- and its pins are released."""
    from knurlogic.engine.vision.store import ImageStore
    fam = stub_family()
    probe = Harness(server, model, family=fam)
    probe.chat([user("x ", data_url(60))])
    one = probe.vision.store.nbytes               # one image's features
    small = ImageStore(max_bytes=int(one * 2.5))  # two fit, three do not
    h = Harness(server, model, family=stub_family(), store=small)
    assert h.post(dict(messages=[user("two ", data_url(61), data_url(62))],
                       max_tokens=2, logit_bias=BAN))[0] == 200
    code, _, body = h.post(dict(messages=[user(
        "three ", data_url(63), data_url(64), data_url(65))], max_tokens=2,
        logit_bias=BAN))
    err = json.loads(body)["error"]
    assert code == 413 and err["code"] == "image_too_large"
    assert "MiB" in err["message"] and "--image-store-gib" in err["message"]
    assert h.vision.pinned_count() == 0


def test_a_failure_between_tokenize_and_admit_releases_the_images(
        server, model, monkeypatch):
    """Tokenize pins the request's images; if admission fails before the
    engine owns the row, the pins must go (the scheduler's admit guard)."""
    fam = stub_family()
    h = Harness(server, model, family=fam)
    real = h.sched.cache.fetch

    def boom(*a, **k):
        raise RuntimeError("the prompt cache fell over")
    monkeypatch.setattr(h.sched.cache, "fetch", boom)
    code, _, body = h.post(dict(messages=[user("x ", data_url(70))],
                                max_tokens=2, logit_bias=BAN))
    assert code == 500 and b"fell over" in body
    assert h.vision.pinned_count() == 0
    monkeypatch.setattr(h.sched.cache, "fetch", real)


def test_a_burst_of_image_requests_interleaves_with_decoding(server, model,
                                                             monkeypatch):
    """Images are encoded at tokenize, on the scheduler thread; at most one
    image request is admitted per tick, so a decoding row gets a step
    between encodes instead of waiting for the whole burst."""
    from knurlogic.engine.mtp import batch_loop
    h = Harness(server, model, family=stub_family(), start=False)
    order = []
    real_insert = h.sched._insert
    real_step = batch_loop.MTPBatch.step

    def insert(job):
        order.append("admit")
        return real_insert(job)

    def step(self):
        order.append("step")
        return real_step(self)
    monkeypatch.setattr(h.sched, "_insert", insert)
    monkeypatch.setattr(batch_loop.MTPBatch, "step", step)
    replies = [h.submit(dict(messages=[user("x ", data_url(80 + i))],
                             max_tokens=4, logit_bias=BAN))
               for i in range(3)]
    h.sched.start()
    for r in replies:
        h.result(r)
    admits = [i for i, x in enumerate(order) if x == "admit"]
    assert len(admits) == 3
    # a step ran between each pair of admissions
    assert all("step" in order[a:b] for a, b in zip(admits, admits[1:]))
