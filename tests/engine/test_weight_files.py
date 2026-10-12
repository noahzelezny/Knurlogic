"""engine/model/weight_files.py: the one opener of an artifact's tensor
files -- as mx.load without a peer, fetched from a fake peer page with one,
deferred and recorded for a split's fill, and falling back to the file."""
import mlx.core as mx
import pytest

from knurlogic.engine.model import weight_files as WF


@pytest.fixture
def art(tmp_path):
    """An artifact folder with one shard of known tensors, several dtypes."""
    d = tmp_path / "Tiny"
    d.mkdir()
    t = {"a.weight": mx.random.normal((16, 8)).astype(mx.bfloat16),
         "b.weight": mx.arange(40, dtype=mx.int32).reshape(5, 8),
         "c.scales": mx.random.normal((3,)),
         "d.empty": mx.zeros((0, 4))}
    mx.save_safetensors(str(d / "model.safetensors"), t,
                        metadata={"format": "mlx", "k": "v"})
    return d, t


class Page:
    """A fake peer page: serves byte ranges of the artifact's files."""

    def __init__(self, root, fail=False):
        self.root, self.fail, self.calls = root, fail, []

    def __call__(self, file, offset, length, into):
        assert "/" not in file
        if self.fail:
            raise ConnectionError("the peer went away")
        self.calls.append((file, offset, length))
        with open(self.root / file, "rb") as f:
            f.seek(offset)
            into[:] = f.read(length)


@pytest.fixture
def peer():
    def use(root, **kw):
        page = Page(root, **kw)
        WF.use_peer(WF.Peer(read=page, source="holder", root=root))
        return page
    yield use
    WF.use_peer(None)


def same(got, want):
    assert set(got) == set(want)
    for k in want:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape
        assert mx.array_equal(got[k], want[k]).item(), k


def test_names_and_metadata_come_from_the_header(art):
    d, t = art
    f = d / "model.safetensors"
    assert sorted(WF.names(f)) == sorted(t)
    assert WF.metadata(f) == {"format": "mlx", "k": "v"}


def test_without_a_peer_it_is_mx_load(art):
    d, t = art
    same(WF.load(d / "model.safetensors"), t)
    got = WF.load(d / "model.safetensors", keys=["b.weight"])
    same(got, {"b.weight": t["b.weight"]})


def test_with_a_peer_a_family_open_is_fetched_now(art, peer):
    d, t = art
    page = peer(d)
    same(WF.load(d / "model.safetensors"), t)
    assert page.calls                         # over the link, not the file


def test_only_the_keys_asked_for_are_fetched(art, peer):
    d, t = art
    page = peer(d)
    got = WF.load(d / "model.safetensors", keys=["b.weight"])
    same(got, {"b.weight": t["b.weight"]})
    head = WF.header(d / "model.safetensors")
    assert sum(n for _, _, n in page.calls) == head["b.weight"][1]


def test_a_failed_fetch_reads_the_file(art, peer):
    d, t = art
    peer(d, fail=True)
    same(WF.load(d / "model.safetensors"), t)


def test_a_file_outside_the_artifact_is_read_from_disk(art, peer, tmp_path):
    d, t = art
    page = peer(tmp_path / "Other")
    same(WF.load(d / "model.safetensors"), t)
    assert not page.calls


def test_a_deferred_open_is_lazy_and_recorded(art, peer):
    d, t = art
    page = peer(d)
    got = WF.load(d / "model.safetensors", deferred=True)
    assert not page.calls
    assert WF.origin(got["a.weight"]) == (str(d / "model.safetensors"),
                                          "a.weight")
    assert WF.origin(mx.zeros((1,))) is None
    WF.forget()
    assert WF.origin(got["a.weight"]) is None


def test_ranges_merge_neighbours_and_split_large_ones(monkeypatch):
    monkeypatch.setattr(WF, "PIECE", 64)
    data = bytes(range(256)) * 4
    calls = []

    def read(file, off, n, into):
        calls.append((file, off, n))
        into[:] = data[off:off + n]
    items = [("f", 0, 10), ("f", 10, 20), ("f", 100, 200), ("f", 30, 5)]
    seen = {}
    got = WF.ranges(items, read, lambda i, v: seen.update({i: bytes(v)}))
    assert got == 235
    for i, (_, off, n) in enumerate(items):
        assert seen[i] == data[off:off + n]
    assert ("f", 0, 35) in calls             # three neighbours, one request
    assert all(n <= 64 for _, _, n in calls)  # 200 bytes in 64-byte pieces


def test_a_stopped_fetch_raises(monkeypatch):
    with pytest.raises(InterruptedError):
        WF.ranges([("f", 0, 4)], lambda *a: None, lambda i, v: None,
                  stop=lambda: True)


def test_ctrl_c_is_never_swallowed(art, peer, monkeypatch):
    d, _ = art
    peer(d)

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(WF, "fetch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        WF.load(d / "model.safetensors")


def test_no_artifact_file_is_opened_anywhere_else():
    """The onboarding rule (docs/design/new-model.md): an artifact's
    tensor files are opened through weight_files only. The prompt cache
    reads its own files, not an artifact's."""
    from pathlib import Path
    src = Path(WF.__file__).resolve().parents[2]
    bad = []
    for p in src.rglob("*.py"):
        rel = p.relative_to(src).as_posix()
        if rel in ("engine/model/weight_files.py",) or \
                rel.startswith("engine/prompt_cache/"):
            continue
        text = p.read_text()
        if "mx.load(" in text.replace("`mx.load(", ""):
            bad.append(rel)
    assert not bad, f"open artifact files through weight_files: {bad}"


class Fuses:
    """A model whose sanitize glues two file tensors into one, drops one,
    and passes one through -- popping from the dict it is given."""

    def __init__(self):
        self.calls = 0

    def sanitize(self, w):
        self.calls += 1
        if "a.weight" not in w:
            raise KeyError("a.weight")
        w["ab.weight"] = mx.concatenate([w.pop("a.weight"),
                                         w.pop("b.weight").astype(
                                             mx.bfloat16)], axis=0)
        w.pop("d.empty")
        return w


def test_sanitize_records_what_it_consumed_and_made(art, peer):
    d, t = art
    page = peer(d)
    f = str(d / "model.safetensors")
    out = WF.sanitize(Fuses(), WF.load(f, deferred=True))
    assert WF.made(out["ab.weight"]) == "ab.weight"
    assert WF.made(out["c.scales"]) is None          # passed through
    assert WF.origin(out["c.scales"]) == (f, "c.scales")
    made, got = WF.remake()
    assert mx.array_equal(made["ab.weight"], out["ab.weight"]).item()
    head = WF.header(f)
    want = sum(head[k][1] for k in ("a.weight", "b.weight", "d.empty"))
    assert got == want == sum(n for _, _, n in page.calls)
    WF.forget()
    assert WF.made(out["ab.weight"]) is None


def test_without_a_peer_sanitize_records_nothing(art):
    d, _ = art
    out = WF.sanitize(Fuses(), WF.load(d / "model.safetensors"))
    assert WF.made(out["ab.weight"]) is None


def test_remake_refuses_inputs_over_the_limit(art, peer, monkeypatch):
    d, _ = art
    page = peer(d)
    WF.sanitize(Fuses(), WF.load(d / "model.safetensors", deferred=True))
    monkeypatch.setattr(WF, "REMAKE_LIMIT", 64)
    with pytest.raises(ValueError, match="over the limit"):
        WF.remake()
    assert not page.calls
