"""Does the MTP head produce DIFFERENT logits from the main head?

One arm, one channel, no speed claim. The question is only whether the
sidecar head is a working second predictor or an expensive no-op.

THE CONTROL MATTERS MORE THAN THE ARM. A comparison that cannot detect
SAMENESS proves nothing about difference, so the same comparison is run on
the main head against itself first. If that does not report identical, the
instrument is broken and the real answer is meaningless.

Reads only. Loads a model, loads a sidecar, compares two vectors, exits.
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ART = Path(sys.argv[1])
SIDECAR = ART / "mtp-head-q6.safetensors"
PROMPT = "The capital of France is"

t0 = time.time()
print(f"artifact  {ART.name}")
print(f"sidecar   {SIDECAR.name}  "
      f"{SIDECAR.stat().st_size / (1 << 30):.2f} GiB")

# mlx-lm 0.32.0 put model_file execution behind trust_remote_code; 0.31.x
# runs it unconditionally. Ask the signature rather than guess.
import inspect
kw = {}
if "trust_remote_code" in inspect.signature(load).parameters:
    kw["tokenizer_config"] = {"trust_remote_code": True}
model, tok = load(str(ART), **kw)
print(f"loaded    {time.time() - t0:.0f}s  {type(model).__name__}")

from exo.worker.engines.mlx.mtp import capture, registry
spec = registry.resolve(model)
arch = spec.arch_module(model)
print(f"family    {spec.name}  capture={spec.capture}  head={spec.head}")

head = spec.head_cls().from_sidecar(model, arch, str(SIDECAR))
print(f"head      loaded, fa_idx={getattr(head, 'fa_idx', '?')}")

ids = mx.array([tok.encode(PROMPT)])
core = getattr(model, "language_model", model).model

from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
with capture.capture_input(core, spec.capture) as get_h:
    logits = model(ids, cache=cache)
    h_row = get_h()
mx.eval(logits, h_row)

main = logits[0, -1].astype(mx.float32)
nxt = mx.argmax(main).reshape(1, 1)
print(f"prompt    {PROMPT!r} -> next token "
      f"{tok.decode([int(nxt.item())])!r}")

# Exactly as loop.py builds it: the head's own if it has one, else the
# registry's. NOT wrapped in a list -- the block takes one cache object, and
# passing [cache] got `'list' object has no attribute 'offset'`.
dcache = (head.make_draft_cache()
          if hasattr(head, "make_draft_cache")
          else spec.make_draft_cache(arch))
draft = head.draft_logits(h_row[:, -1:, :], nxt, dcache)[0, -1].astype(mx.float32)
mx.eval(draft)


def compare(a, b, label):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    same_arg = int(mx.argmax(a32).item()) == int(mx.argmax(b32).item())
    max_abs = float(mx.max(mx.abs(a32 - b32)).item())
    cos = float((mx.sum(a32 * b32) /
                 (mx.sqrt(mx.sum(a32 * a32)) * mx.sqrt(mx.sum(b32 * b32)))).item())
    top = lambda v: [tok.decode([int(i)]) for i in
                     mx.argsort(-v)[:5].tolist()]
    print(f"\n{label}")
    print(f"  argmax same   {same_arg}")
    print(f"  max |a-b|     {max_abs:.4f}")
    print(f"  cosine        {cos:.6f}")
    print(f"  top-5 a       {top(a32)}")
    print(f"  top-5 b       {top(b32)}")
    return max_abs, cos


print("\n" + "=" * 62)
print("CONTROL -- the main head against itself. If this is not identical,")
print("the comparison is broken and the arm below means nothing.")
m0, c0 = compare(main, main, "control: main vs main")

print("\n" + "=" * 62)
print("ARM -- the drafting head against the main head, same position.")
m1, c1 = compare(main, draft, "arm: main vs mtp draft")

print("\n" + "=" * 62)
ok_control = m0 == 0.0 and abs(c0 - 1.0) < 1e-5
differs = m1 > 1e-3
print(f"control identical : {ok_control}")
print(f"heads differ      : {differs}")
print(json.dumps({"control_max_abs": m0, "control_cos": c0,
                  "arm_max_abs": m1, "arm_cos": c1,
                  "control_ok": ok_control, "differs": differs}))
