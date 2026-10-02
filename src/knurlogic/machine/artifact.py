"""Read what an artifact declares about itself.

AUTHORITY RULE: the artifact's own
config.json is the record of what shipped. Never characterize an artifact
from a card, a ledger, or an experiment entry -- read the config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30


@dataclass
class Artifact:
    path: Path
    model_type: str
    model_file: str | None          # bundled VQ runtime, e.g. "model.py"
    bytes_on_disk: int
    hidden_size: int | None
    moe_intermediate_size: int | None
    vq_modules: dict = field(default_factory=dict, repr=False)
    vq_other: dict = field(default_factory=dict, repr=False)
    #: `knobs` from config.json -- what the artifact says about its own
    #: controls. Authoritative over anything scanned out of the runtime.
    knobs: dict = field(default_factory=dict, repr=False)
    #: The whole config, for questions the typed fields do not cover.
    raw_config: dict = field(default_factory=dict, repr=False)

    @property
    def gib(self) -> float:
        return self.bytes_on_disk / GIB

    @property
    def is_vq(self) -> bool:
        """MoE artifacts declare `vq_modules`; DENSE ones declare `vq_linear`
        / `vq_embed` instead. Keying on vq_modules alone called every dense
        rung "not a VQ artifact" and skips its kernel settings."""
        return bool(self.vq_modules or self.vq_other)

    @property
    def geometries(self) -> dict:
        """{(d, K): module_count} -- what the kernels will actually dispatch."""
        out: dict = {}
        for m in self.vq_modules.values():
            if not isinstance(m, dict):
                continue
            d, K = m.get("d") or m.get("dim"), m.get("K") or m.get("k")
            if d and K:
                key = (int(d), int(K))
                out[key] = out.get(key, 0) + 1
        return out

    def runtime_source(self) -> str:
        """The bundled runtime's text, or empty when it ships none."""
        if not self.model_file:
            return ""
        f = self.path / self.model_file
        try:
            return f.read_text() if f.is_file() else ""
        except (OSError, ValueError):
            return ""

    @property
    def has_mtp(self) -> bool:
        """Does this artifact ship a multi-token-prediction head?

        Declared in config.json (`mtp`, `mtp_num_hidden_layers`), and it is
        weights that were downloaded: 40 of the 54 artifacts on this machine
        declare one, 3925 GiB of them.
        """
        blob = json.dumps(self.raw_config)
        return '"mtp"' in blob or "mtp_num_hidden_layers" in blob

    def chat_template(self) -> str:
        """The template text, from wherever this artifact keeps it.

        Newer exports put it in `chat_template.jinja` and leave
        `tokenizer_config.json`'s field empty; older ones do the opposite.
        Reading only one of the two answers "no template" for half the
        artifacts on this machine.
        """
        f = self.path / "chat_template.jinja"
        if f.is_file():
            try:
                return f.read_text()
            except OSError:
                return ""
        cfg = self.path / "tokenizer_config.json"
        if cfg.is_file():
            try:
                return json.loads(cfg.read_text()).get("chat_template") or ""
            except (OSError, ValueError, AttributeError):
                return ""
        return ""

    def declared_knobs(self) -> dict:
        """Knobs the artifact DECLARES, from its own config.json.

        The authority rule, applied to the control surface: the artifact is
        the record of what shipped, so if it declares its knobs -- name,
        default, range, one line of documentation -- that beats anything
        knurlogic guesses by scanning the runtime for `os.environ`. Kernel
        work belongs with whoever packs the kernels; this is the hand-off
        point, and it is empty until a packer writes to it.

            "knobs": {"VQ_D8_ROWS_TG": {"default": "8",
                                        "values": [4, 8, 16],
                                        "doc": "rows per threadgroup"}}
        """
        return self.knobs if isinstance(self.knobs, dict) else {}

    def knobs_read(self) -> list:
        """Every environment variable the bundled runtime reads.

        The honest size of the surface. Knurlogic has a measured answer for a
        fraction of it, and printing its own list as though it were the whole
        environment is a quieter version of the same overclaiming this
        package objects to everywhere else.
        """
        import re
        src = self.runtime_source()
        if not src:
            return []
        return sorted(set(re.findall(
            r'environ(?:\.get)?\(?\s*\[?["\']([A-Z][A-Z0-9_]{3,})["\']',
            src)))

    def reads_knob(self, name: str) -> bool | None:
        """Does the bundled runtime read this environment variable?

        None means "no bundled runtime to ask" -- not "no". The distinction
        matters: a stock artifact is served by the engine, which has its own
        answer, and reporting that as 'has no effect' would be a guess.

        This exists because of a real miss. Knurlogic emitted
        VQLAB_PREFILL_CHUNK for every artifact, and not one of the 37 bundled
        runtimes on this machine reads it -- a resolved setting that does
        nothing, which is the exact failure this package was written to
        prevent, committed by the package.
        """
        src = self.runtime_source()
        if not src:
            return None
        return name in src

    @classmethod
    def load(cls, path) -> Artifact:
        p = Path(path)
        cfg_path = p / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"no config.json in {p}")
        cfg = json.loads(cfg_path.read_text())
        # Multimodal configs nest the language model; single-modal ones do not.
        tc = cfg.get("text_config", cfg)
        total = sum(f.stat().st_size for f in p.iterdir()
                    if f.suffix == ".safetensors" and f.is_file())
        return cls(
            path=p,
            # The full model's type (e.g. qwen3_5), not the nested TEXT
            # config's (qwen3_5_text) -- a vision model loaded with its
            # tower still IS the full type; the text config's own spelling
            # is a fallback only for a config that never nests one.
            model_type=cfg.get("model_type") or tc.get("model_type") or "unknown",
            model_file=cfg.get("model_file"),
            bytes_on_disk=total,
            hidden_size=tc.get("hidden_size"),
            moe_intermediate_size=tc.get("moe_intermediate_size"),
            vq_modules=cfg.get("vq_modules") or {},
            vq_other={k: cfg[k] for k in ("vq_linear", "vq_embed")
                      if cfg.get(k)},
            knobs=cfg.get("knobs") or {},
            raw_config=cfg,
        )


def context_length(path) -> int:
    """The model's context window from its config.json --
    max_position_embeddings, or text_config's for a multimodal wrapper --
    0 when the config does not say. The page offers max_tokens up to it."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return 0
    for c in (cfg, cfg.get("text_config") if isinstance(cfg, dict) else None):
        v = c.get("max_position_embeddings") if isinstance(c, dict) else None
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return 0


#: generation_config.json key -> the sampler's name for it
_SAMPLING_KEYS = (("temperature", "temp"), ("top_p", "top_p"),
                  ("top_k", "top_k"), ("min_p", "min_p"))


def sampling_defaults(path) -> dict:
    """The sampling a model's makers recommend (its generation_config.json),
    as the sampler names it: {} when there is none, or when it says
    do_sample false (greedy is then what they meant)."""
    try:
        g = json.loads((Path(path) / "generation_config.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(g, dict) or g.get("do_sample") is False:
        return {}
    out = {}
    for key, name in _SAMPLING_KEYS:
        v = g.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[name] = v
    return out


# --- identity: the same weights on another machine ---------------------------
# A peer is asked to load an artifact by WHAT it is, never by where it sits:
# a path from another machine means nothing here, and a path taken from a
# request is a way to point a loader at an arbitrary directory. The identity
# is cheap on purpose -- no full hash of the weights: sha256 over config.json,
# the safetensors index (when there is one), per shard its name, its size,
# its safetensors header and three sampled windows of its tensor data
# (start, middle, end), and every *.py the artifact ships (model.py), whole.
# Two artifacts with the same config and layout whose weights differ (a re-quantised
# expert set, a different pin) disagree; the
# same artifact read on another Mac -- over SMB, say -- agrees, because
# nothing in it is an mtime or a path. Shards that are symlinks are read
# through to the files they name.
_IDENT: dict = {}
#: bytes read from each sampled window of a shard's tensor data
_SAMPLE = 64 * 1024
#: a safetensors header larger than this is not read whole (hashing stops)
_MAX_HEADER = 64 * 1024 * 1024


# The cache persists in <cache_dir>/identities.json (machine/disk_cache),
# keyed by path and the same stamp, so a page or serve start does not
# rehash every shard.
from knurlogic.machine.disk_cache import DiskCache  # noqa: E402

_DISK = DiskCache("identities.json", valid=lambda v: isinstance(v, str))


class AmbiguousIdentity(ValueError):
    """Kept for callers that catch it: resolve_identity no longer raises it.
    Two paths with one identity are the same weights, and one is chosen."""


def _shards(p: Path) -> list:
    return sorted((f for f in p.iterdir()
                   if f.suffix == ".safetensors" and f.is_file()),
                  key=lambda f: f.name)


def _shard_digest(f: Path, size: int) -> bytes:
    import hashlib
    import struct
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        head = fh.read(8)
        start = 0
        if len(head) == 8:
            (n,) = struct.unpack("<Q", head)
            if 0 < n <= _MAX_HEADER and 8 + n <= size:
                h.update(fh.read(n))
                start = 8 + n
        data = size - start
        for off in sorted({start, start + max(0, data // 2 - _SAMPLE // 2),
                           max(start, size - _SAMPLE)}):
            fh.seek(off)
            h.update(struct.pack("<Q", off) + fh.read(_SAMPLE))
    return h.digest()


def identity(path) -> str:
    """16 hex of sha256(config.json + model.safetensors.index.json + each
    shard's name, size, header and sampled data + each shipped *.py's name
    and text); "" when there is no
    config.json. Cached per path until config.json, the index, any shard
    or any *.py changes (size, mtime or ctime -- this machine's own view,
    used only for the cache; ctime because cp -p and rsync -a keep mtime
    and a rebuilt shard often keeps its size)."""
    import hashlib
    p = Path(path)
    cfg = p / "config.json"
    try:
        st = cfg.stat()
        shards = _shards(p)
        stats = [(f.name, f.stat()) for f in shards]
    except OSError:
        return ""
    try:
        code = sorted((f for f in p.iterdir()
                       if f.suffix == ".py" and f.is_file()),
                      key=lambda f: f.name)
        cstats = [(f.name, f.stat()) for f in code]
    except OSError:
        return ""
    try:
        ist = (p / "model.safetensors.index.json").stat()
        istamp = (ist.st_size, ist.st_mtime_ns, ist.st_ctime_ns)
    except OSError:
        istamp = None
    key = str(p)
    stamp = ((st.st_mtime_ns, st.st_size, st.st_ctime_ns), istamp,
             tuple((n, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                   for n, s in stats),
             tuple((n, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                   for n, s in cstats))
    hit = _IDENT.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    jstamp = json.loads(json.dumps(stamp))
    dhit = _DISK.get(key, jstamp)
    if dhit is not None:
        _IDENT[key] = (stamp, dhit)
        return dhit
    h = hashlib.sha256()
    try:
        h.update(cfg.read_bytes())
        idx = p / "model.safetensors.index.json"
        h.update(b"\0index\0")
        if idx.is_file():
            h.update(idx.read_bytes())
        for f, (name, s) in zip(shards, stats):
            h.update(b"\0shard\0" + name.encode() + b"\0"
                     + str(s.st_size).encode() + b"\0")
            h.update(_shard_digest(f, s.st_size))
        # the runtime it ships (model.py and any other *.py): two builds
        # whose weights agree and whose model.py differs -- a vqlab A/B
        # pin -- are different artifacts. Hashed whole, by name and
        # content only, so two Macs agree.
        for f in code:
            h.update(b"\0code\0" + f.name.encode() + b"\0"
                     + f.read_bytes())
    except OSError:
        return ""
    out = h.hexdigest()[:16]
    _IDENT[key] = (stamp, out)
    _DISK.put(key, jstamp, out)
    return out


_MOUNTS: list = []


def _network_mounts() -> list:
    """Mount points of network filesystems (smbfs, nfs, afpfs, webdav),
    longest first; read once from `mount`."""
    if _MOUNTS:
        return _MOUNTS[0]
    import re
    import subprocess
    out = []
    try:
        text = subprocess.run(["mount"], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        text = ""
    for line in text.splitlines():
        m = re.match(r".+? on (.+) \(([a-z0-9]+)[,)]", line)
        if m and m.group(2) in ("smbfs", "nfs", "afpfs", "webdav", "cifs",
                                "ftp"):
            out.append(m.group(1))
    out.sort(key=len, reverse=True)
    _MOUNTS.append(out)
    return out


def on_network(path) -> bool:
    """Is `path` (read through symlinks) on a network mount -- the shared
    SMB copy rather than this Mac's own disk?"""
    try:
        real = str(Path(path).resolve())
    except (OSError, RuntimeError):
        real = str(path)
    return any(real == m or real.startswith(m.rstrip("/") + "/")
               for m in _network_mounts())


def resolve_identity(ident: str | None, paths=None, name: str = "") -> str | None:
    """The local artifact directory with this identity, or None. `paths`
    defaults to every artifact in this machine's model stores
    (machine/discover.py); nothing outside them is ever considered.

    Several matches are the SAME weights (that is what one identity means),
    so one is chosen, deterministically, and logged: the one called `name`
    (a directory name, never a path) when the requester named one, else
    one on this Mac's own disk over one on a network mount, else the first
    by path."""
    if not isinstance(ident, str) or not ident or len(ident) > 64:
        return None
    name = Path(str(name or "")).name
    hits: list = []
    if paths is None:
        from knurlogic.machine import discover
        if name:
            # the requester knows the name: look there first, confirm the
            # identity, and scan every store only when that misses
            hits = sorted({str(f.path) for f in discover.find_named(name)
                           if identity(f.path) == ident})
        if not hits:
            paths = [f.path for f in discover.find()]
    if not hits:
        hits = sorted({str(p) for p in paths if identity(p) == ident})
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    named = [h for h in hits if name and Path(h).name == name]
    local = [h for h in (named or hits) if not on_network(h)]
    pool = local or named or hits
    # no copy of the name asked for: the plain name over a prefixed copy
    # of it (`X` over `sp190--X`), so every machine tends to the same name
    plain = [h for h in pool if not any(
        Path(h).name.endswith("--" + Path(o).name) for o in pool if o != h)]
    pick = (plain or pool)[0]
    import logging
    why = ("the one named" if named else "on this Mac's disk, the plain name"
           if local else "the plain name, first by path")
    logging.getLogger(__name__).info(
        "identity %s: %d copies of the same weights (%s); using %s (%s)",
        ident, len(hits), ", ".join(hits), pick, why)
    return pick


def prefer_shared(ident: str, name: str, paths=None) -> tuple:
    """(identity, alert) for a cluster job: the identity every rank should
    load. A Mac holding its own copy of a model beside the shared (network)
    copy, with the two DIFFERENT (another identity under the same name),
    would split the ring across two builds: the shared copy wins, and the
    alert says so. Identical copies, or no shared copy: (ident, "")."""
    name = Path(str(name or "")).name
    if not name:
        return ident, ""
    if paths is None:
        from knurlogic.machine import discover
        paths = [f.path for f in discover.find()]
    same = [p for p in paths if Path(str(p)).name == name]
    shared: dict = {}
    for p in same:
        if on_network(p):
            i = identity(p)
            if i:
                shared.setdefault(i, str(p))
    if not shared or ident in shared or len(shared) != 1:
        return ident, ""
    (other, where), = shared.items()
    mine = next((str(p) for p in same if identity(p) == ident), "")
    return other, (f"the local copy of {name}{f' ({mine})' if mine else ''}"
                   f" differs from the shared copy ({where}): identity "
                   f"{ident} here, {other} shared. Every rank loads the "
                   f"shared copy so all run the same weights; remove or "
                   f"update the local copy to silence this.")
