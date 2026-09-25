"""engine/serve/sampling: a seed set on the generation thread reaches the
sampler. Without the fix, mlx-lm's compiled sampler returns one token
forever in a worker thread."""
import threading

import pytest

mx = pytest.importorskip("mlx.core")


def _draws(results):
    from mlx_lm.sample_utils import make_sampler
    logits = mx.log(mx.ones((1, 50)) / 50)
    for s in (1, 2, 1):
        mx.random.seed(s)
        sm = make_sampler(1.5)
        results.append(tuple(sm(logits).item() for _ in range(4)))


def test_seeds_work_on_a_worker_thread_once_installed():
    from knurlogic.engine.serve import sampling
    sampling.install()
    out = []
    t = threading.Thread(target=_draws, args=(out,))
    t.start()
    t.join()
    one, two, one_again = out
    assert one == one_again           # the same seed repeats
    assert one != two                 # a different seed differs
    assert len(set(one)) > 1          # draws advance within a request
