"""Fetching a saved entry's missing pieces from a peer
(engine/prompt_cache/transfer.py, interfaces/page/prompt_cache.piece_route):
two machines that do not share storage, each with its own cache root; an
entry saved under one split restores under another (or on one machine),
bit for bit, its missing layers fetched through the peer's page route. A
corrupt or wrong-key piece is refused and the entry is a miss; the route
refuses names that are not its formats, and clients that are not peers."""
import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")

from test_prompt_cache_layers import (  # noqa: E402
    LINK,
    _gather_as,
    _key,
    _lru_of,
    _model,
    _parts,
)
from test_prompt_disk import _forced  # noqa: E402

from knurlogic.engine.prompt_cache import disk as D  # noqa: E402
from knurlogic.engine.prompt_cache import layers as LY  # noqa: E402
from knurlogic.engine.prompt_cache import transfer as TR  # noqa: E402
from knurlogic.interfaces.page import prompt_cache as PC  # noqa: E402


class _Handler:
    """What piece_route touches of a request handler."""

    def __init__(self, ip="127.0.0.1", local="127.0.0.1", origin=None):
        self.client_address = (ip, 5)
        self.headers = {} if origin is None else {"Origin": origin}
        self.connection = NS(getsockname=lambda: (local, 8899))
        self.wfile = io.BytesIO()
        self.code = None
        self.close_connection = False

    def send_response(self, code):
        self.code = code

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def doc(self):
        return json.loads(self.wfile.getvalue())


def _route(path, query, roots, **kw):
    h = _Handler(**kw)
    PC.piece_route(h, path, {k: [str(v)] for k, v in query.items()},
                   roots=roots, manual=())
    return h


class _Peer:
    """A peer's page serving `roots` through the real route, as the
    fetcher would reach it; counts what it was asked."""

    def __init__(self, roots, spoil=None):
        self.roots = [Path(r) for r in roots]
        self.calls = []
        self.spoil = spoil            # (piece, bytes -> bytes)

    def peers(self):
        self.calls.append("peers")
        return ["http://peer"]

    def listing(self, peer, key_id):
        self.calls.append("listing")
        h = _route(PC.ENTRIES_PATH, {"key": key_id}, self.roots)
        assert h.code == 200
        return h.doc()["entries"]

    def get(self, peer, key_id, entry, piece, dest, size):
        self.calls.append(("get", piece))
        h = _route(PC.PIECE_PATH, {"key": key_id, "entry": entry,
                                   "layer": piece}, self.roots)
        if h.code != 200:
            raise OSError(f"{h.code}")
        body = h.wfile.getvalue()
        if self.spoil and self.spoil[0] == piece:
            body = self.spoil[1](body)
        Path(dest).write_bytes(body)
        return len(body)


def _save_apart(monkeypatch, roots, key, parts, tokens, cache):
    """Save as the ranks of `parts` would, each on its own machine (rank r
    writes into roots[r]), then the commit: every rank's entry.json."""
    stores = [LY.Store(roots[p.rank], key, p) for p in parts]
    entries = None
    for st in stores:
        p = st.part
        mine = list(cache[p.start:p.start + p.n]) + \
            (list(cache[p.total:]) if p.extras else [])
        got = st.save(_lru_of(tokens, mine), commit=len(parts) == 1,
                      sweep_after=False)
        if p.rank == 0:
            entries = got["pending"]
    if len(parts) > 1:
        _gather_as(monkeypatch, [[1 if st.ready(e) else 0 for e in entries]
                                 for st in stores])
        for st in stores:
            assert LY.commit(LINK, st, entries)
    return stores


def _restore_apart(monkeypatch, roots, key, parts, peers):
    """Each rank's restore on its own machine, its fetcher the peer page
    serving the other machines' roots, then the vote."""
    gots = []
    for p in parts:
        st = LY.Store(roots[p.rank], key, p)
        monkeypatch.setattr(TR, "_FETCHER", [peers[p.rank]])
        gots.append(LY.restore(8, st, sweep_first=False))
    monkeypatch.setattr(TR, "_FETCHER", [None])
    if len(parts) > 1:
        _gather_as(monkeypatch, [LY.vote_row(g) for g in gots])
        gots = [LY.agree(LINK, g) for g in gots]
    return gots


def _whole(gots, parts):
    out = []
    for p, g in sorted(zip(parts, gots), key=lambda x: x[0].start):
        out += list(g[0][2])[:p.n]
    return out


@pytest.fixture
def machines(tmp_path):
    roots = [tmp_path / n / "Prompt cache" for n in ("a", "b", "c")]
    for r in roots:
        r.mkdir(parents=True)
    return roots


@pytest.mark.parametrize("load,where", [
    ([4], [0]),            # the split's rank-0 machine alone
    ([4], [2]),            # a machine that held none of it (entry.json too)
    ([3, 1], [0, 1]),      # another cut over the same two machines
])
def test_a_split_entry_restores_from_the_peers_pieces_bit_for_bit(
        monkeypatch, machines, load, where):
    """Saved on a [1, 3] pipeline, each rank's layers in its own machine's
    root; loaded under another layout, each rank fetches what it lacks
    from the other machines' pages: the next tokens' logits equal the
    original's, bit for bit."""
    model = _model("qwen3_5_moe")
    mx.random.seed(5)
    prompt = (mx.random.randint(0, 60, (40,)) % 60).tolist()
    toks = [7, 3, 11, 5]
    cache, cont = _forced(model, prompt, toks)
    keep = copy.deepcopy(cache)
    key = _key()
    _save_apart(monkeypatch, machines[:2], key, _parts([1, 3]), prompt,
                cache)
    h = D.tokens_hash(prompt)
    held = [sorted(f.name for f in (r / D.key_id(key) / h).iterdir())
            for r in machines[:2]]
    assert held == [["L003.safetensors", "entry.json"],
                    ["L000.safetensors", "L001.safetensors",
                     "L002.safetensors", "entry.json"]]
    parts = _parts(load)
    roots = [machines[w] for w in where]
    peers = [_Peer([r for r in machines if r != mine]) for mine in roots]
    gots = _restore_apart(monkeypatch, roots, key, parts, peers)
    assert all(len(g) == 1 and g[0][1] == prompt for g in gots)
    assert mx.array_equal(cont(keep), cont(_whole(gots, parts)))
    # fetched files are ordinary local files now: a second load asks for
    # nothing
    again = [_Peer([r for r in machines if r != mine]) for mine in roots]
    _restore_apart(monkeypatch, roots, key, parts, again)
    assert not any(isinstance(c, tuple) for p in again for c in p.calls)


def _tiny(n_layers=4, length=12, first=200):
    from mlx_lm.models.cache import KVCache
    caches = []
    for _ in range(n_layers):
        c = KVCache()
        k = mx.random.normal((1, 2, length, 32))
        c.update_and_fetch(k, k)
        caches.append(c)
    return [first] + list(range(length - 1)), caches


def _split_tiny(monkeypatch, machines, key, first=200):
    toks, caches = _tiny(first=first)
    _save_apart(monkeypatch, machines[:2], key, _parts([2, 2]), toks, caches)
    return toks, caches


@pytest.mark.parametrize("spoil", ["garbage", "short", "other_key"])
def test_a_corrupt_or_wrong_key_piece_is_refused_and_a_miss(
        monkeypatch, machines, spoil):
    key = _key()
    toks, _ = _split_tiny(monkeypatch, machines, key)
    h = D.tokens_hash(toks)
    if spoil == "other_key":
        # a layer file of the same tokens under another model's key
        other = _key(kv_bits=4)
        _split_tiny(monkeypatch, machines, other)
        src = machines[1] / D.key_id(other) / h / "L000.safetensors"
        dst = machines[1] / D.key_id(key) / h / "L000.safetensors"
        dst.write_bytes(src.read_bytes())
        peer = _Peer([machines[1]])
    else:
        fn = (lambda b: b"\0" * len(b)) if spoil == "garbage" else \
            (lambda b: b[:-7])
        peer = _Peer([machines[1]], spoil=(0, fn))
    st = LY.Store(machines[0], key, LY.Part(0, 4, 4))
    got = TR.fill(st, 8, fetch=peer)
    assert got["rejected"] == 1 and got["fetched"] == 0
    assert not (st.dir / h / "L000.safetensors").exists()
    assert not [f for f in (st.dir / h).iterdir()
                if f.name.startswith(".tmp-")]
    monkeypatch.setattr(TR, "_FETCHER", [peer])
    assert LY.restore(8, st, sweep_first=False) == []


def test_an_entry_json_of_another_model_is_refused(monkeypatch, machines):
    """A machine with nothing of the entry fetches entry.json first and
    checks it names this model and these tokens before any layer."""
    key = _key()
    toks, _ = _split_tiny(monkeypatch, machines, key)
    h = D.tokens_hash(toks)

    def lie(b):
        m = json.loads(b)
        m["tokens"] = m["tokens"][:-1] + [999]
        return json.dumps(m).encode().ljust(len(b))
    peer = _Peer(machines[:2], spoil=("entry", lie))
    st = LY.Store(machines[2], key, LY.Part(0, 4, 4))
    got = TR.fill(st, 8, fetch=peer)
    assert got["rejected"] == 1
    assert not (st.dir / h / LY.ENTRY).exists()
    assert not any(isinstance(c, tuple) and c[1] != "entry"
                   for c in peer.calls)


def test_no_fetch_when_every_file_is_here(monkeypatch, machines):
    """Shared storage (or an entry this machine wrote whole): the restore
    asks for no piece, and a read-back of a whole entry asks nobody."""
    key = _key()
    toks, caches = _tiny()
    st = LY.Store(machines[0], key, LY.Part(0, 4, 4))
    st.save(_lru_of(toks, caches), sweep_after=False)
    peer = _Peer([machines[0]])
    monkeypatch.setattr(TR, "_FETCHER", [peer])
    assert len(LY.restore(8, st, sweep_first=False)) == 1
    assert not any(isinstance(c, tuple) for c in peer.calls)
    peer.calls.clear()
    TR.fill(st, dirs=[st.dir / D.tokens_hash(toks)])
    assert peer.calls == []


def test_a_read_back_fetches_what_this_machine_lacks(monkeypatch, machines):
    """One machine after a split's save: the entry is in its read-back
    index (a fetcher installed), and reading it back fetches the layers
    the other machine wrote; the arrays are the saved ones."""
    key = _key()
    toks, caches = _split_tiny(monkeypatch, machines, key)
    st = LY.Store(machines[0], key, LY.Part(0, 4, 4))
    assert st.index() == {}                       # no fetcher: not readable
    peer = _Peer([machines[1]])
    monkeypatch.setattr(TR, "_FETCHER", [peer])
    d = st.index()[tuple(toks)]["file"]
    TR.fill(st, dirs=[d])
    (_, back, got, *_), = st.read([d])
    assert back == toks
    assert all(mx.array_equal(a.keys, b.keys) for a, b in zip(caches, got))


def test_a_fetch_slower_than_recomputing_is_skipped(monkeypatch, machines):
    key = _key()
    _split_tiny(monkeypatch, machines, key)
    st = LY.Store(machines[0], key, LY.Part(0, 4, 4))
    peer = _Peer([machines[1]])
    monkeypatch.setitem(TR.RATES, "http://peer", 1.0)       # 1 byte/s
    got = TR.fill(st, 8, fetch=peer, prefill_tps=1000.0)
    assert got["not_worth"] == 1 and not any(isinstance(c, tuple)
                                             for c in peer.calls)
    monkeypatch.setitem(TR.RATES, "http://peer", 1e12)
    got = TR.fill(st, 8, fetch=peer, prefill_tps=1000.0)
    assert got["fetched"] == 1
    assert TR.RATES["http://peer"] != 1e12             # measured


def test_a_fill_stops_at_its_time_budget(monkeypatch, machines):
    # a slow peer must not hold a ring's ranks at the restore vote
    key = _key()
    _split_tiny(monkeypatch, machines, key)
    st = LY.Store(machines[0], key, LY.Part(0, 4, 4))
    got = TR.fill(st, 8, fetch=_Peer([machines[1]]), budget_s=-1.0)
    assert got["out_of_time"] == 1 and got["fetched"] == 0


# --- the route ------------------------------------------------------------

def test_the_piece_route_refuses_names_that_are_not_its_formats(
        monkeypatch, machines, tmp_path):
    key = _key()
    toks, _ = _split_tiny(monkeypatch, machines, key)
    kid, h = D.key_id(key), D.tokens_hash(toks)
    roots = [machines[0]]
    ok = _route(PC.PIECE_PATH, {"key": kid, "entry": h, "layer": 2}, roots)
    assert ok.code == 200 and ok.wfile.getvalue() == \
        (machines[0] / kid / h / "L002.safetensors").read_bytes()
    (tmp_path / "secret").write_text("no")
    for q in [{"key": "../" + kid, "entry": h, "layer": 2},
              {"key": kid, "entry": "../" + h, "layer": 2},
              {"key": kid, "entry": h, "layer": "../../secret"},
              {"key": kid, "entry": h, "layer": "/etc/passwd"},
              {"key": kid, "entry": h, "layer": "2/../2"},
              {"key": kid, "entry": h, "layer": "-1"},
              {"key": kid, "entry": "..", "layer": "entry"},
              {"key": kid.upper(), "entry": h, "layer": 2}]:
        got = _route(PC.PIECE_PATH, q, roots)
        assert got.code in (400, 404), q
        assert got.wfile.getvalue() != b"no"
    # a symbolic link planted in an entry is not followed
    link = machines[0] / kid / h / "L007.safetensors"
    link.symlink_to(tmp_path / "secret")
    assert _route(PC.PIECE_PATH, {"key": kid, "entry": h, "layer": 7},
                  roots).code == 404
    assert TR.listing(kid, roots)[h]["layers"] == [2, 3]
    # an entry without its entry.json (being written) is not served
    (machines[0] / kid / h / LY.ENTRY).unlink()
    assert _route(PC.PIECE_PATH, {"key": kid, "entry": h, "layer": 2},
                  roots).code == 404
    assert TR.listing(kid, roots) == {}


def test_the_piece_routes_refuse_a_client_that_is_not_a_peer(
        monkeypatch, machines):
    key = _key()
    toks, _ = _split_tiny(monkeypatch, machines, key)
    q = {"key": D.key_id(key), "entry": D.tokens_hash(toks), "layer": 0}
    roots = [machines[1]]
    # a machine on the network, arriving on a non-Thunderbolt address
    far = _route(PC.PIECE_PATH, q, roots, ip="192.0.2.7", local="192.0.2.1")
    assert far.code == 403 and far.close_connection
    assert _route(PC.ENTRIES_PATH, {"key": q["key"]}, roots,
                  ip="192.0.2.7", local="192.0.2.1").code == 403
    # a web page (an Origin header), even on loopback
    assert _route(PC.PIECE_PATH, q, roots,
                  origin="http://example.test").code == 403
    # the peer list: this machine's own servers only
    monkeypatch.setattr(PC.peers, "peer_pages", lambda: {"http://x:1"})
    assert _route(PC.PEERS_PATH, {}, roots, ip="192.0.2.7",
                  local="127.0.0.1").code == 403
    mine = _route(PC.PEERS_PATH, {}, roots)
    assert mine.code == 200 and mine.doc() == {"pages": ["http://x:1"]}
    assert _route(PC.PIECE_PATH, q, roots).code == 200


def test_a_model_server_started_by_a_page_gets_a_fetcher(monkeypatch):
    monkeypatch.setattr(TR, "_FETCHER", [None])
    assert not PC.install_fetcher({})
    assert TR.fetcher() is None
    assert PC.install_fetcher({PC.PAGE_PORT_ENV: "8899"})
    assert TR.fetcher().page == "http://127.0.0.1:8899"
