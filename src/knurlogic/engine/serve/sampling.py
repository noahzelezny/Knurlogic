"""Sampling that honours the request's seed.

mlx-lm compiles its categorical sampler with the random state as an
input (`mx.compile(inputs=mx.random.state, ...)`). MLX's random state is
PER THREAD, and the server samples on its generation thread -- where the
compiled function reads a state that neither `mx.random.seed` nor its own
draws update. Measured with no model (mlx 0.31.2, mlx-lm 0.31.3): in a
worker thread the compiled sampler returns the same token for seeds 1-4;
through `knurlogic serve`, gemma e4b at temperature 1.5 answered the same
word for every seed.

The plain call reads the calling thread's state, so seeds work there and
draws advance. `make_sampler` and the top-p/min-p/top-k chains look
`categorical_sampling` up by name at call time, so replacing that one
function fixes every sampler the server builds. Costs one uncompiled
elementwise multiply per token.
"""

from __future__ import annotations


def categorical_sampling(logits, temp):
    import mlx.core as mx
    return mx.random.categorical(logits * (1 / temp))


def install() -> None:
    from mlx_lm import sample_utils
    if getattr(sample_utils.categorical_sampling, "_knurlogic", False):
        return
    categorical_sampling._knurlogic = True
    sample_utils.categorical_sampling = categorical_sampling
