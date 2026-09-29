"""End to end on a real model: decode tok/s at bf16, 8-bit dequantize+sdpa
(KNURLOGIC_KV_KERNEL=off) and 8-bit through engine/kvattn's kernel, modes
interleaved, plus a needle check (exact answer) per mode.

    python e2e_kernel.py MODEL 6000,16000 [reps] [decode]
"""
import os
import sys
import time

import mlx.core as mx

from knurlogic.engine import kvattn, kvquant
from knurlogic.engine.serve.load import load

P = sys.argv[1]
Ns = [int(x) for x in sys.argv[2].split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DEC = int(sys.argv[4]) if len(sys.argv) > 4 else 200
MODES = ("bf16", "q8old", "q8k")

model, tok = load(P)
orig = model.make_cache
words = open("/usr/share/dict/words").read().split()


def setmode(m):
    model.make_cache = orig
    if m == "bf16":
        return
    os.environ[kvattn.ENV] = "on" if m == "q8k" else "off"
    kvquant.install(model, 8)
    assert kvquant.KERNEL == (m == "q8k")


def prefill(ids, cache, C=512):
    N = ids.shape[1]
    for i in range(0, N - 1, C):
        model(ids[:, i:min(i + C, N - 1)], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    return model(ids[:, N - 1:], cache=cache)[:, -1].argmax(-1)


def speed(m, N):
    setmode(m)
    cache = model.make_cache()
    ids = mx.array(tok.encode(" ".join(words[1000:1000 + N]))[:N])[None]
    y = prefill(ids, cache)
    mx.eval(y)
    t = time.perf_counter()
    for _ in range(DEC):
        y = model(y[None], cache=cache)[:, -1].argmax(-1)
        mx.eval(y)
    return DEC / (time.perf_counter() - t)


def needle(m, N, code="48213"):
    setmode(m)
    hay = words[5000:5000 + int(N * 0.55)]
    hay.insert(len(hay) // 2, f". The secret code is {code}. ")
    msg = [{"role": "user", "content": " ".join(hay) +
            "\n\nWhat is the secret code? Answer with the number only."}]
    s = tok.apply_chat_template(msg, add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    ids = mx.array(tok.encode(s))[None]
    cache = model.make_cache()
    y = prefill(ids, cache)
    out = [y.item()]
    for _ in range(12):
        y = model(y[None], cache=cache)[:, -1].argmax(-1)
        out.append(y.item())
    text = tok.decode(out)
    return ids.shape[1], code in text, text.strip()


for m in MODES:
    speed(m, 1024)                                 # warm up / compile
for N in Ns:
    res = {m: [] for m in MODES}
    for rep in range(REPS):
        for m in MODES:
            res[m].append(speed(m, N))
            mx.clear_cache()
    for m in MODES:
        r = sorted(res[m])
        print(f"N={N} {m:6s} decode tok/s median {r[len(r) // 2]:.2f} "
              f"all {[round(x, 2) for x in res[m]]}", flush=True)
for N in Ns:
    for m in MODES:
        print(f"needle N={N} {m:6s}", needle(m, N), flush=True)
