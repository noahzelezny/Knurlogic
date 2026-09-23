"""Images as real context, end to end through mlx-lm's OWN server objects.

Gates G5-G9 (docs/design/vision.md): a request goes in as an HTTP body
through `APIHandler.do_POST`, is tokenized on the real `ResponseGenerator`
thread, meets the real `LRUPromptCache` and comes back as the response
bytes. That is critique risk 2 -- sentinel keys reaching mlx-lm code that
assumes ints (stats, prefix arithmetic, the segment trim, `insert_cache`,
usage counts) -- run for real rather than argued about. Tuple sentinels
survived it unchanged; the negative-int fallback was not needed.

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
    """mlx-lm's ModelProvider without the loading: the fields the server
    reads, and a `load` that hands back the tiny model."""

    def __init__(self, model, tok, **cli):
        self.model, self.tokenizer = model, tok
        self.draft_model = None
        self.model_key = ("tiny", None, None)
        self.is_batchable = True
        base = dict(chat_template_args={}, prompt_cache_bytes=None,
                    decode_concurrency=32, prompt_concurrency=8,
                    prefill_step_size=16, max_tokens=12, num_draft_tokens=3,
                    temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
                    allowed_origins=["*"])
        base.update(cli)
        self.cli_args = types.SimpleNamespace(**base)

    def load_default(self):
        pass

    def load(self, *a, **k):
        return self.model, self.tokenizer


@pytest.fixture
def server(monkeypatch):
    """Snapshots every name the vision install touches and the seam's
    module state, so each test starts from a pristine mlx-lm server."""
    from mlx_lm import server as srv
    from knurlogic.engine import seam

    for owner, name in seam.VISION_WRAPS:
        cls = getattr(srv, owner)
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(srv, "BatchGenerator", srv.BatchGenerator)
    monkeypatch.setattr(srv, "_make_sampler", srv._make_sampler)
    monkeypatch.setattr(seam, "_VISION",
                        {"serve": None, "model": None, "error": ""})
    monkeypatch.setattr(seam, "_VISION_STATS", {})
    monkeypatch.setattr(seam, "_DRAFT", dict(seam._DRAFT, head=None,
                                             on=False, batch_installed=False))
    monkeypatch.setattr(seam, "_SERVED", {"path": None, "provider": None})
    yield srv
    from knurlogic.engine.vision import set_served_vision
    set_served_vision(None)


class _Parked:
    """Stands in for the generator Thread mlx-lm's ResponseGenerator starts
    in its constructor: the loop is run by `Harness.post` instead.

    WHY. Measured on mlx 0.31.2: once a thread that ran a model forward
    EXITS, the process segfaults shortly after (a plain matmul thread does
    not; the tiny qwen3_5 forward does). A real server's generator thread
    never exits, so it never meets this -- but a test that stops one does.
    So the generation loop -- mlx-lm's own `_generate`, unchanged -- runs
    on the MAIN thread, and the HTTP handler, which does no mlx work, runs
    on the thread that comes and goes."""

    def __init__(self, target=None, **kw):
        pass

    def start(self):
        pass

    def join(self, *a):
        pass


class Harness:
    """One ResponseGenerator (mlx-lm's generation loop, its real
    LRUPromptCache) and a handler that speaks HTTP into a buffer."""

    def __init__(self, srv, model, tok=None, family=None, *, store=None,
                 **cli):
        from knurlogic.engine import seam
        self.srv = srv
        self.provider = Provider(model, tok or byte_tok(), **cli)
        seam._SERVED["provider"] = self.provider
        self.vision = None
        if family is not None:
            from knurlogic.engine.vision import set_served_vision
            from knurlogic.engine.vision.request import VisionServe
            from knurlogic.engine.vision.store import ImageStore
            self.vision = VisionServe(family, store or ImageStore(),
                                      self.provider.model_key)
            seam._VISION.update(serve=self.vision, model=model)
            set_served_vision(family.spec)
        self.cache = srv.LRUPromptCache()
        # Every batch generator the server builds, so a gate can read what
        # it actually PREFILLED -- not only what the trie claimed to hit.
        self.gens = []
        make = srv.BatchGenerator

        def recorded(*a, **k):
            g = make(*a, **k)
            self.gens.append(g)
            return g
        srv.BatchGenerator = recorded
        real_thread = srv.Thread
        srv.Thread = _Parked
        try:
            self.rg = srv.ResponseGenerator(self.provider, self.cache)
        finally:
            srv.Thread = real_thread

    def post(self, body, path="/v1/chat/completions"):
        """(status, header lines without Date/Server, body bytes)."""
        srv = self.srv

        class H(srv.APIHandler):
            def log_message(self, *a):
                pass

        raw = json.dumps(body).encode()
        h = H.__new__(H)
        h.created = 0
        h.system_fingerprint = "fp"
        h.response_generator = self.rg
        h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
        h.headers = {"Content-Length": str(len(raw))}
        h.path, h.command = path, "POST"
        h.request_version, h.requestline = "HTTP/1.1", "POST " + path
        h.client_address = ("test", 0)
        h.close_connection = True
        self._serve(h.do_POST)
        head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        keep = [x for x in lines[1:]
                if not x.lower().startswith((b"date:", b"server:"))]
        return status, keep, payload

    def chat(self, messages, **kw):
        body = dict(messages=messages, max_tokens=12, temperature=0.0,
                    logprobs=True, top_logprobs=5, logit_bias=BAN)
        body.update(kw)
        status, _, payload = self.post(body)
        assert status == 200, payload
        return json.loads(payload)

    def _serve(self, handle):
        """Run `handle` (HTTP side) on a worker thread while mlx-lm's
        generation loop runs here, until the handler has written its
        response. The batch generator lives for one call; the prompt cache
        and the image store -- the state G6-G9 are about -- live on."""
        import threading
        err = []

        def http():
            try:
                handle()
            except BaseException as e:            # surfaced below
                err.append(e)
            finally:
                self.rg._stop = True

        t = threading.Thread(target=http, daemon=True)
        self.rg._stop = False
        t.start()
        self.rg._generate()
        t.join()
        if err:
            raise err[0]

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


def install(srv):
    from knurlogic.engine import seam
    seam.install_vision(srv)


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


def test_g5_no_image_is_byte_identical_to_main(server, model):
    """The golden for G5: with the vision wraps installed and no vision
    family served, a text request's HTTP output -- status, headers, body
    with token logprobs -- is byte-for-byte what untouched mlx-lm (main's
    path for a model with no head) returns. Only the uuid in `id` differs
    between any two requests and is masked."""
    body = dict(messages=TEXT, max_tokens=10, temperature=0.0, logprobs=True,
                top_logprobs=3, seed=None)
    main = Harness(server, model)
    try:
        want = main.post(body)
        # A seed sends mlx-lm down its SEQUENTIAL path; sampled, so a
        # request rerouted to the batch engine (which ignores the seed)
        # would answer differently.
        want_seeded = main.post(dict(body, seed=3, temperature=0.9))
    finally:
        main.close()
    install(server)
    ours = Harness(server, model)
    try:
        got = ours.post(body)
        got_seeded = ours.post(dict(body, seed=3, temperature=0.9))
    finally:
        ours.close()
    assert want[0] == got[0] == 200
    assert want[1] == got[1]
    assert normalize(want[2]) == normalize(got[2])
    assert normalize(want_seeded[2]) == normalize(got_seeded[2])


def test_g5_text_on_a_vision_model_matches_main(server, model):
    """With a vision family served, every batch is MTPBatchGenerator (D5),
    including a text-only request's. Its answer must be main's: same text
    and tokens; logprobs to float tolerance, since the batch engines chunk
    the prefill differently."""
    body = dict(messages=TEXT, max_tokens=10, temperature=0.0, logprobs=True)
    main = Harness(server, model)
    try:
        want = json.loads(main.post(body)[2])
    finally:
        main.close()
    install(server)
    ours = Harness(server, model, family=stub_family())
    try:
        got = json.loads(ours.post(body)[2])
        from knurlogic.engine import seam
        assert seam._VISION_STATS.get("requests") == 1   # it went through ours
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
    install(server)
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
    install(server)
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
    install(server)
    fam = stub_family()
    n = counted(fam)
    warm = Harness(server, model, family=fam)
    try:
        m1, r1, m2, r2 = _two_turns(warm, data_url(2))
    finally:
        warm.close()
    turn1 = r1["usage"]["prompt_tokens"] + r1["usage"]["completion_tokens"]
    assert cached(r2) == turn1                       # G8: the trie's hit ...
    prefilled = warm.gens[-1]._prompt_tokens_counter
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
    install(server)
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
    install(server)
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
    from knurlogic.engine import seam
    from knurlogic.engine.vision import registry, served_vision

    monkeypatch.setitem(registry.FAMILIES, "qwen3_5",
                        "fixtures_vision:stub_build")
    cfg = dict(model_type="qwen3_5", image_token_id=IMG,
               vision_config={"patch_size": 4},
               text_config={"hidden_size": 128})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    prov = Provider(model, byte_tok())
    v = seam.bind_vision(str(tmp_path), prov)
    assert v is not None and seam._VISION["serve"] is v
    assert served_vision() is v.spec and seam.served_vision() is v.spec
    assert v.spec.image_token_id == IMG
    assert seam.vision_status()["on"]

    seam.clear_vision()
    assert served_vision() is None and seam._VISION["serve"] is None

    cfg.pop("vision_config")                      # a text-only artifact
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert seam.bind_vision(str(tmp_path), prov) is None
    assert served_vision() is None


def test_every_wrapped_method_is_resolved_by_name(server):
    """Design D2: a pinned mlx-lm that renamed a wrapped method fails the
    install, loudly, rather than serving images through a wrap that never
    runs."""
    from knurlogic.engine import seam
    for owner, name in seam.VISION_WRAPS:
        assert callable(getattr(getattr(server, owner), name))
    fake = types.SimpleNamespace(
        APIHandler=server.APIHandler,
        ResponseGenerator=type("RG", (), {"generate": lambda s: 0,
                                          "_is_batchable": lambda s: 0}))
    with pytest.raises(AssertionError, match="_tokenize"):
        seam.install_vision(fake)
