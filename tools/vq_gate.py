"""The identity gate: knurlogic's VQ runtime against a rung's PUBLISHED bundle.

    python tools/vq_gate.py fetch <dir> [repo ...]     # model.py + config.json
    python tools/vq_gate.py knobs <dir> [--write] [--markdown]
    python tools/vq_gate.py gate <artifact> [--bundle <dir>] [--record]

`fetch` downloads ONLY the two small text files per repo (`hf download
<repo> model.py config.json`), never weights. `knobs` reads every flag
default out of each published model.py and regenerates
src/knurlogic/engine/vq/rungs.json (and the table in
docs/design/vq-rung-knobs.md): the record is the shipped artifact, never a
local copy, which may have drifted from the Hub.

`gate` is the identity gate, run by hand on real rungs, one at a time,
behind the model-load lock. Same artifact weights, same short prompt,
two runtimes: the rung's PUBLISHED bundled model.py, and knurlogic's
(vendored vqlab 42df84f + the rung's knobs from rungs.json). PASS means
logits over the whole prompt within atol 1e-5 AND 40 greedy tokens
identical. `--record` then marks the rung verified in rungs.json -- the
only thing that lets knurlogic serve it on its own runtime.

WHY EACH SIDE IS ITS OWN PROCESS. Both runtimes read their flags ONCE at
import, into module globals, and both size the mlx buffer cache at import.
Two sides in one process would share whatever the first import froze and
whatever memory the first model left; a gate that can pass by sharing state
is not a gate. Each side also runs with every VQ_*/VQLAB_* variable removed
from its environment, so each reads its OWN defaults -- the bundle its baked
text, knurlogic its knobs -- which is exactly the claim under test.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knurlogic.engine.vq import rungs as R  # noqa: E402  (stdlib only)

RUNGS_JSON = R.RUNGS_JSON
PROMPT = ("Explain, in two sentences, why the sky appears blue during the "
          "day and red at sunset.")
N_TOKENS = 40
ATOL = 1e-5
RUNTIME_FILES = ("vq_switch.py", "vq_dense.py")


# --- knobs: published defaults -> rungs.json --------------------------------

def _head_defaults() -> dict:
    vq = ROOT / "src/knurlogic/engine/vq"
    return R.flag_defaults("\n\n".join((vq / f).read_text()
                                       for f in RUNTIME_FILES))


def _dense_only_flags() -> list:
    vq = ROOT / "src/knurlogic/engine/vq"
    sw = R.flag_defaults((vq / "vq_switch.py").read_text())
    return [f for f in R.flag_defaults((vq / "vq_dense.py").read_text())
            if f not in sw]


def rung_entry(model_py: str, cfg: dict, head: dict,
               old: dict | None = None) -> dict:
    """One rungs.json row from a published bundle's text and config.

    knobs = the flags HEAD reads whose published default differs from
    HEAD's. For an arc6-era bundle the two numerics flags do not exist; the
    arc6 arithmetic IS bf16-I/O off, so they are set "0" and listed as
    inferred -- and running such a rung on HEAD is a runtime change that
    only this gate may bless."""
    pub = R.flag_defaults(model_py)
    gen = R.generation(pub)
    knobs, inferred = {}, []
    for flag, dflt in sorted(head.items()):
        if flag in pub:
            if pub[flag] != dflt:
                knobs[flag] = pub[flag]
        elif flag in ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO"):
            if dflt != "0":
                knobs[flag] = "0"
            inferred.append(flag)
    dense = bool(cfg.get("vq_linear") or cfg.get("vq_embed"))
    # vq_dense.py's flags are absent from every MoE bundle by construction
    # (MoE bundles do not carry that file); that is not a runtime difference
    dense_only = set(_dense_only_flags()) if not dense else set()
    absent = sorted(f for f in head if f not in pub and f not in inferred
                    and f not in dense_only)
    row = {
        "generation": gen,
        "model_type": cfg.get("model_type"),
        "dense": dense,
        "published_model_py": {
            "lines": model_py.count("\n"),
            "sha256": hashlib.sha256(model_py.encode()).hexdigest()},
        "published_config_sha256":
            hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "published_defaults": dict(sorted(pub.items())),
        "knobs": knobs,
        "inferred_knobs": inferred,
        "flags_absent_in_published": absent,
        "runtime_change": gen == "arc6-no-flags",
        "verified": False,
        "gate": None,
    }
    if old and old.get("published_model_py") == row["published_model_py"]:
        # the bundle did not change: a recorded gate still describes it
        row["verified"] = old.get("verified", False)
        row["gate"] = old.get("gate")
    return row


def cmd_knobs(a) -> int:
    hub = Path(a.dir)
    head = _head_defaults()
    old = json.loads(RUNGS_JSON.read_text()) if RUNGS_JSON.is_file() else {}
    rows = {}
    for d in sorted(hub.iterdir()):
        mp, cp = d / "model.py", d / "config.json"
        if not (mp.is_file() and cp.is_file()):
            continue
        repo = R.repo_of(d) or d.name
        rows[repo] = rung_entry(mp.read_text(), json.loads(cp.read_text()),
                                head, old.get("rungs", {}).get(repo))
    doc = {
        "_about": ("Per released rung: what its PUBLISHED model.py ships. "
                   "Generated by tools/vq_gate.py knobs from `hf download "
                   "<repo> model.py config.json`; do not edit by hand except "
                   "via `vq_gate.py gate --record`."),
        "runtime": {"vqlab_commit": "42df84f",
                    "files": {f: hashlib.sha256(
                        (ROOT / "src/knurlogic/engine/vq" / f).read_bytes()
                    ).hexdigest() for f in RUNTIME_FILES},
                    "head_numerics": {f: head.get(f) for f in
                                      ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO",
                                       "VQ_DENSE_SS")}},
        "harvested": a.date or _dt.date.today().isoformat(),
        "rungs": rows,
    }
    text = json.dumps(doc, indent=1, sort_keys=False) + "\n"
    if a.write:
        RUNGS_JSON.write_text(text)
        R.reload()
        print(f"wrote {RUNGS_JSON} ({len(rows)} rungs)")
    if a.markdown:
        print(markdown(doc))
    if not (a.write or a.markdown):
        print(text)
    return 0


def markdown(doc: dict) -> str:
    lines = ["| repo | generation | model.py lines | sha256 (12) | "
             "GEMMSEG_BF16IO | DECODE_BF16IO | DENSE_SS | knobs on HEAD | "
             "verified |",
             "|---|---|---|---|---|---|---|---|---|"]
    for repo, r in doc["rungs"].items():
        p = r["published_defaults"]
        kn = ", ".join(f"{k}={v}" + ("*" if k in r["inferred_knobs"] else "")
                       for k, v in r["knobs"].items()) or "none"
        lines.append(
            f"| {repo.split('/', 1)[1]} | {r['generation']} | "
            f"{r['published_model_py']['lines']} | "
            f"`{r['published_model_py']['sha256'][:12]}` | "
            f"{p.get('VQ_GEMMSEG_BF16IO', '--')} | "
            f"{p.get('VQ_DECODE_BF16IO', '--')} | "
            f"{p.get('VQ_DENSE_SS', '--')} | {kn} | "
            f"{'yes' if r['verified'] else 'no'} |")
    return "\n".join(lines)


def cmd_fetch(a) -> int:
    repos = a.repos or sorted(json.loads(RUNGS_JSON.read_text())["rungs"])
    bad = 0
    for repo in repos:
        out = Path(a.dir) / repo.replace("/", "--", 1)
        p = subprocess.run([_hf(), "download", repo, "model.py", "config.json",
                            "--local-dir", str(out)],
                           capture_output=True, text=True)
        print(f"{'ok  ' if p.returncode == 0 else 'FAIL'} {repo}")
        bad += p.returncode != 0
    return 1 if bad else 0


# --- gate --------------------------------------------------------------------

def _hf() -> str:
    """The `hf` beside this interpreter (a venv not activated has it only
    there), else whatever PATH finds."""
    here = Path(sys.executable).parent / "hf"
    return str(here) if here.is_file() else "hf"


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("VQ_", "VQLAB_", "KNURLOGIC_CACHE"))}


def side(which: str, artifact: str, bundle: str, out: str,
         prompt: str, n: int) -> None:
    """One runtime, in this (child) process: prompt logits + greedy tokens."""
    import importlib.util
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, load_tokenizer

    art = Path(artifact)
    # Both sides get knurlogic's vendored architecture modules, as serving
    # does: the claim under test is the VQ runtime, and a published model.py
    # imports its arch from mlx_lm (qwen4_exp exists only in the fork and in
    # knurlogic's copy, not in stock mlx-lm).
    from knurlogic.engine import arch as _arch, register as _register
    from knurlogic.machine.artifact import Artifact
    _mods = _arch.modules_for_artifact(Artifact.load(art))
    if _mods:
        _register.register(*_mods)
    if which == "bundle":
        def classes(config):
            spec = importlib.util.spec_from_file_location(
                "custom_model", Path(bundle) / "model.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.Model, mod.ModelArgs
        model, config = load_model(art, model_config={"model_file": None},
                                   get_model_classes=classes)
    else:
        from knurlogic.engine.vq import runtime
        model, config = runtime.load_model(art)
    tok = load_tokenizer(art, eos_token_ids=config.get("eos_token_id"))
    ids = mx.array(tok.encode(prompt))[None]
    cache = make_prompt_cache(model)
    def fwd(x):
        # An mlx-vlm-shaped bundle (GLM) returns an output object.
        out = model(x, cache=cache)
        return getattr(out, "logits", out)

    logits = fwd(ids)
    prompt_logits = logits.astype(mx.float32)
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    toks = []
    for _ in range(n):
        toks.append(int(nxt.item()))
        logits = fwd(nxt[:, None])
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(prompt_logits)
    np.savez(out, logits=np.array(prompt_logits), tokens=np.array(toks))


def compare(a_npz, b_npz, atol: float = ATOL) -> dict:
    import numpy as np
    a, b = np.load(a_npz), np.load(b_npz)
    same_shape = a["logits"].shape == b["logits"].shape
    diff = (float(np.max(np.abs(a["logits"] - b["logits"])))
            if same_shape else float("inf"))
    ta, tb = a["tokens"].tolist(), b["tokens"].tolist()
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    ok = same_shape and diff <= atol and ta == tb
    return {"pass": ok, "max_abs_logit_diff": diff, "atol": atol,
            "tokens_equal": ta == tb, "first_token_divergence": first,
            "n_tokens": len(ta)}


def cmd_gate(a) -> int:
    art = Path(a.artifact).expanduser()
    repo = R.repo_of(art)
    row = R.rung(art)
    if row is None:
        print(f"{art.name}: not in rungs.json -- run `knobs` first")
        return 2
    with tempfile.TemporaryDirectory() as td:
        bundle = Path(a.bundle) if a.bundle else Path(td) / "bundle"
        if not a.bundle:
            p = subprocess.run([_hf(), "download", repo, "model.py",
                                "config.json", "--local-dir", str(bundle)],
                               capture_output=True, text=True)
            if p.returncode:
                print(p.stderr)
                return 2
        got = hashlib.sha256((bundle / "model.py").read_bytes()).hexdigest()
        want = row["published_model_py"]["sha256"]
        if got != want:
            print(f"published model.py is {got[:12]}, rungs.json says "
                  f"{want[:12]}: the Hub changed. Re-run `knobs` first.")
            return 2
        # the bundle's shim reads config.json next to itself; the weights'
        # own config must be the one published, or two things are compared
        if json.loads((bundle / "config.json").read_text()) != \
                json.loads((art / "config.json").read_text()):
            print("WARNING: local config.json differs from the published "
                  "one; the gate runs the local weights with the published "
                  "runtime.")
        from knurlogic.machine import loadlock
        outs = {}
        try:
            with loadlock.model_load(str(art), purpose="vq identity gate", wait_s=a.wait):
                for which in ("bundle", "knurlogic"):
                    outs[which] = str(Path(td) / f"{which}.npz")
                    p = subprocess.run(
                        [sys.executable, __file__, "_side", which, str(art),
                         str(bundle), outs[which], "--n", str(a.n)],
                        env=_clean_env())
                    if p.returncode:
                        print(f"{which} side failed ({p.returncode})")
                        return 1
        except loadlock.Busy as e:
            print(e)
            return loadlock.EXIT_BUSY
        v = compare(outs["bundle"], outs["knurlogic"])
    v.update(repo=repo, bundle_sha256=want,
             date=_dt.datetime.now().isoformat(timespec="seconds"),
             knobs=row["knobs"])
    print(json.dumps(v, indent=1))
    if a.record:
        record(repo, v)
    return 0 if v["pass"] else 1


def record(repo: str, verdict: dict) -> None:
    """Write the verdict into rungs.json. A FAIL is recorded too: a rung
    that fails stays on its bundle and is listed, not silently served."""
    doc = json.loads(RUNGS_JSON.read_text())
    row = doc["rungs"][repo]
    row["verified"] = bool(verdict["pass"])
    row["gate"] = verdict
    RUNGS_JSON.write_text(json.dumps(doc, indent=1) + "\n")
    R.reload()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download each repo's model.py + "
                       "config.json (no weights)")
    f.add_argument("dir", help="where to put one folder per repo")
    f.add_argument("repos", nargs="*",
                   help="Hub repos (default: every repo in rungs.json)")
    k = sub.add_parser("knobs", help="regenerate rungs.json from fetched "
                       "bundles")
    k.add_argument("dir", help="the folder `fetch` wrote")
    k.add_argument("--write", action="store_true",
                   help="write rungs.json (default: print only)")
    k.add_argument("--markdown", action="store_true",
                   help="print the table for docs/design/vq-rung-knobs.md")
    k.add_argument("--date", default=None,
                   help="the harvest date recorded (default: today)")
    g = sub.add_parser("gate", help="compare knurlogic's runtime with the "
                       "published model.py on one artifact")
    g.add_argument("artifact", help="path to the model folder")
    g.add_argument("--bundle", default=None,
                   help="dir with the PUBLISHED model.py + config.json "
                        "(default: hf download into a temp dir)")
    g.add_argument("--record", action="store_true",
                   help="mark the rung verified in rungs.json on PASS")
    g.add_argument("--n", type=int, default=N_TOKENS,
                   help="greedy tokens compared")
    g.add_argument("--wait", type=float, default=0.0,
                   help="seconds to wait for the model-load lock")
    s = sub.add_parser("_side")
    for x in ("which", "artifact", "bundle", "out"):
        s.add_argument(x)
    s.add_argument("--n", type=int, default=N_TOKENS)
    s.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args(argv)
    if a.cmd == "_side":
        side(a.which, a.artifact, a.bundle, a.out, a.prompt, a.n)
        return 0
    return {"fetch": cmd_fetch, "knobs": cmd_knobs, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
