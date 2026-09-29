import sys, time, mlx.core as mx
from knurlogic.engine.serve.load import load
from knurlogic.engine import kvquant
import variants
Q = kvquant.QuantKVCache
P = "/Volumes/Models/Models/TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw"
model, tok = load(P); orig = model.make_cache
mods = sys.argv[1].split(","); Ns = [int(x) for x in sys.argv[2].split(",")]
text = open("/usr/share/dict/words").read().split()
def run(mode, N, C=512, dec=200):
    model.make_cache = orig
    if mode != "bf16":
        kvquant.QuantKVCache = Q if mode == "q8" else variants.MAKERS[mode]
        kvquant.install(model, 8)
    cache = model.make_cache()
    ids = mx.array(tok.encode(" ".join(text[1000:1000+N]))[:N])[None]
    t0 = time.perf_counter()
    for i in range(0, N - 1, C):
        j = min(i + C, N - 1)
        model(ids[:, i:j], cache=cache); mx.eval([c.state for c in cache]); mx.clear_cache()
    y = model(ids[:, N-1:], cache=cache)[:, -1].argmax(-1); mx.eval(y)
    t1 = time.perf_counter(); out = [y.item()]
    for _ in range(dec):
        y = model(y[None], cache=cache)[:, -1].argmax(-1); mx.eval(y); out.append(y.item())
    return t1 - t0, time.perf_counter() - t1, out
for N in Ns:
    res = {m: [] for m in mods}
    for m in mods: run(m, 1024, dec=5)
    for rep in range(3):
        for m in mods: res[m].append(run(m, N))
    ref = res[mods[0]][0][2]
    for m in mods:
        r = res[m]; o = r[0][2]
        same = next((i for i, (a, b) in enumerate(zip(ref, o)) if a != b), len(o))
        print(f"N={N} {m:10s} prefill {[round(a,2) for a,b,_ in r]} decode200 {[round(b,2) for a,b,_ in r]} match-first {same}", flush=True)
