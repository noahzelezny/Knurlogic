# The engine

## src/knurlogic/engine/__init__.py

The engine folder is laid out as:

  model/           the served model: load, memory, the live knobs, thinking
                   translation, the vision family bound to the loaded
                   model (see model/__init__.py; drafting is mtp/binding.py)
  runtime/         knurlogic's own server's engine half: the model host,
                   the scheduler, the executor, the prompt and request
                   stages (docs/design/server.md; HTTP is interfaces/http)
  split/           one model across ranks: tensor and pipeline splits
  mtp/             multi-token-prediction drafting, sequential and batched.
                   Its front door (`knurlogic.engine.mtp`) is stdlib only, so
                   asking whether an artifact has a head costs no mlx import
  families/        one folder per model family, everything knurlogic knows
                   about it: MANIFEST (architectures, model_type spellings,
                   heads, prefill widths), architecture/ (vendored model
                   code + PROVENANCE, pins, licenses), vision/, heads/.
                   Adding a family = one folder + one line in
                   families/__init__.py
  vision/          images as context, family-agnostic: contracts, the cache
                   key, the image store, the request path
  arch.py          which architecture a model_type needs, and whether it is
                   present (the maps are built from the family manifests)
  register.py      puts vendored architectures in front of installed ones
  vendor.py        takes an architecture file under version control
  smoke.py         generates a token and proves where the code came from

## src/knurlogic/engine/arch.py

A VQ artifact ships its own kernels: config.json names a bundled
`model.py` (the VQ runtime), so that half is correct as downloaded. What it
does not ship is the architecture model file -- qwen4_exp, qwen3_5,
gemma4_text, glm5_next -- which is a graft into mlx_lm/models/. Those are
unversioned, unpinned, and in practice they drift.

Compared across two environments on different mlx-lm versions (0.32.0 and
0.31.9) -- a grafted file inherits its install's version, so nothing
answers "which arithmetic is this":

    qwen4_exp    1136 vs 1138 lines   cosmetic predicate-arity shim, safe
    qwen3_5       574 vs  535 lines   QK-norm rewrite: algebraically identical,
                                      but 1.2e-02 max rel in bf16; and
                                      PipelineMixin present in only one
    gemma4_text   688 vs  675 lines   different parameter set loaded
    glm5_next    MISSING in both      3 artifacts load in neither env

Comparing a number measured in one env against the other on the qwen3_5
family (11 artifacts, since qwen3_5_moe subclasses it) is a one-harness
violation. The fix is to treat these files as versioned source and verify
them, not to graft them by hand.

## src/knurlogic/engine/register.py

Registering a module object in `sys.modules` before mlx-lm's first
`import_module` matters more than the convenience. Across one machine's two
environments, three of four grafted architecture files differed and a
fourth was absent from both -- and the environments were on different
mlx-lm versions (0.32.0 and 0.31.9), which explains most of the
difference. A file grafted into someone else's install inherits that
install's version, so "which arithmetic am I running" has no answer. A file
vendored inside a versioned package does.

Reversible by construction: `unregister()` drops the entries, and an env
that never imported knurlogic is byte-identical to one that did. Pinning
the file does not pin the library it calls into, so the vendored file is
still validated against a known mlx-lm.
