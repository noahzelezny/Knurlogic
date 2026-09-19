"""Read what an artifact declares about itself.

AUTHORITY RULE (inherited from vqlab AGENTS.md): the artifact's own
config.json is the record of what shipped. Never characterize an artifact
from a card, a ledger, or an experiment entry -- read the config.
"""

from __future__ import annotations

import json
import os
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

    @property
    def gib(self) -> float:
        return self.bytes_on_disk / GIB

    @property
    def is_vq(self) -> bool:
        """MoE artifacts declare `vq_modules`; DENSE ones declare `vq_linear`
        / `vq_embed` instead. Keying on vq_modules alone called every dense
        rung "not a VQ artifact" and skipped its kernel settings -- caught
        2026-09-18 when `serve` printed exactly that for a VQ 27B."""
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
        except Exception:
            return ""

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
    def load(cls, path) -> "Artifact":
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
            model_type=tc.get("model_type") or cfg.get("model_type") or "unknown",
            model_file=cfg.get("model_file"),
            bytes_on_disk=total,
            hidden_size=tc.get("hidden_size"),
            moe_intermediate_size=tc.get("moe_intermediate_size"),
            vq_modules=cfg.get("vq_modules") or {},
            vq_other={k: cfg[k] for k in ("vq_linear", "vq_embed")
                      if cfg.get(k)},
        )
