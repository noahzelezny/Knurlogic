"""GatedDeltaNet's compiled glue (qwen arch "knurlogic vendored edit 9"):
the conv + silu + q/k/v split + q/k norm before the scan and the gated
output norm after it run under mx.compile (qwen3_5 unless KNURLOGIC_GDN_COMPILE=off;
qwen4_exp only when it is on).
The same ops in the same order, so the logits are bit-for-bit the plain
path's: single machine (batch 1 and 2, float32 and bfloat16, prefill and
decode), and through a 2-rank tensor and pipeline split."""

import importlib
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import tensor_ring_worker as W
from backends import bf16_gather

IDS = [[5, 17, 3, 99, 42, 7, 64, 11, 23], [8, 2, 77, 13, 5, 61, 40, 9, 30]]
THEN = [[31, 104, 331, 32, 439, 214], [7, 12, 250, 3, 88, 19]]


def _arch(family):
    return importlib.import_module(f"mlx_lm.models.{family}")


def _run(model, rows):
    cache = model.make_cache()
    out = [model(mx.array([IDS[r] for r in range(rows)]), cache=cache)[:, -1]]
    for i in range(len(THEN[0])):
        step = mx.array([[THEN[r][i]] for r in range(rows)])
        out.append(model(step, cache=cache)[:, -1])
    y = mx.stack(out).astype(mx.float32)
    mx.eval(y)
    return np.array(y)


@bf16_gather
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe", "qwen4_exp"])
def test_compiled_glue_is_the_plain_path_bit_for_bit(monkeypatch, family,
                                                     rows, dtype):
    model = W.build(family, dtype=dtype)
    # qwen3_5_moe runs qwen3_5's GatedDeltaNet
    arch = _arch("qwen4_exp" if family == "qwen4_exp" else "qwen3_5")
    monkeypatch.setenv("KNURLOGIC_GDN_COMPILE", "off")
    plain = _run(model, rows)
    arch._gdn_pre_compiled.cache_clear()
    monkeypatch.setenv("KNURLOGIC_GDN_COMPILE", "on")
    compiled = _run(model, rows)
    assert arch._gdn_pre_compiled.cache_info().currsize > 0  # it ran
    assert np.array_equal(plain, compiled), np.abs(plain - compiled).max()


# unset: qwen3_5 compiles, qwen4_exp does not (it measured slower there)
@pytest.mark.parametrize("value,q35,q4", [("", True, False),
                                          ("on", True, True),
                                          ("off", False, False),
                                          ("0", False, False)])
def test_the_switch(monkeypatch, value, q35, q4):
    monkeypatch.setenv("KNURLOGIC_GDN_COMPILE", value)
    for family, on in (("qwen3_5", q35), ("qwen4_exp", q4)):
        W.build(family)  # registers the vendored arch
        assert _arch(family)._gdn_compile_on() is on


def _two_ranks(tmp_path, worker, args, gdn):
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p
    hosts = tmp_path / f"hosts-{gdn}.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{free_port()}"],
                                 [f"127.0.0.1:{free_port()}"]]))
    here = Path(__file__).resolve().parents[1] / "support"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts), KNURLOGIC_GDN_COMPILE=gdn,
               PYTHONPATH=os.pathsep.join(
                   [str(here.parents[1] / "src"), str(here)] + sys.path))
    env.pop("KNURLOGIC_MTP_BATCH_MAX_ROWS", None)
    env.pop("KNURLOGIC_KV_BITS", None)
    procs = [subprocess.Popen(
        [sys.executable, str(here / worker), *args(gdn)],
        env=dict(env, MLX_RANK=str(r)), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT) for r in range(2)]
    try:
        logs = [p.communicate(timeout=180)[0].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert all(p.returncode == 0 for p in procs), "\n".join(logs)


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen4_exp"])
def test_a_two_rank_split_is_the_same_compiled_or_not(tmp_path, split,
                                                      family):
    """The tensor split runs the compiled fns on a rank's half of the heads;
    the pipeline split on a rank's own layers."""
    def out(gdn):
        return tmp_path / f"{split}-{gdn}.json"
    if split == "tensor":
        worker = "tensor_ring_worker.py"

        def args(gdn):
            return [str(out(gdn)), family]
    else:
        worker = "pipeline_ring_worker.py"

        def args(gdn):
            return ["logits", str(out(gdn)), family, "1,3"]
    got = {}
    for gdn in ("off", "on"):
        _two_ranks(tmp_path, worker, args, gdn)
        got[gdn] = np.array(json.loads(out(gdn).read_text())["split"])
    assert np.array_equal(got["off"], got["on"]), \
        np.abs(got["off"] - got["on"]).max()
