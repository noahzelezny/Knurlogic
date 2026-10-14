"""engine/gated_delta: the scan on every back end (docs/design/os-agnostic.md).
The CPU path is mlx-lm's arithmetic bit for bit with its memory bounded;
Metal keeps the family's kernel; the CUDA kernel runs only on CUDA."""
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.gated_delta import gated_delta_ops  # noqa: E402

from knurlogic.engine import backend, gated_delta  # noqa: E402

T = 2 * gated_delta.CHUNK + 5


def _inputs(T=T, Hk=2, Hv=4, D=32, vectorized=False, masked=False):
    mx.random.seed(0)
    q, k = (mx.random.normal((1, T, Hk, D)) * 0.3 for _ in range(2))
    v = mx.random.normal((1, T, Hv, D))
    g = mx.random.uniform(0.8, 1.0, (1, T, Hv) + ((D,) if vectorized else ()))
    beta = mx.random.uniform(shape=(1, T, Hv))
    state = mx.zeros((1, Hv, D, D))
    mask = mx.arange(T)[None] >= 3 if masked else None
    return q, k, v, g, beta, state, mask


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_the_bounded_ops_are_mlx_lms_bit_for_bit(vectorized, masked):
    with mx.stream(mx.cpu):
        args = _inputs(vectorized=vectorized, masked=masked)
        y, s = gated_delta.ops(*args)
        y0, s0 = gated_delta_ops(*args)
        assert mx.array_equal(y, y0).item() and mx.array_equal(s, s0).item()


def test_the_cpu_scan_keeps_a_chunk_of_states_not_the_prompts():
    with mx.stream(mx.cpu):
        args = _inputs(T=1024, Hk=4, Hv=8, D=128)
        mx.eval(*[a for a in args if a is not None])
        mx.reset_peak_memory()
        mx.eval(*gated_delta.ops(*args))
        bounded = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.eval(*gated_delta_ops(*args))
        whole = mx.get_peak_memory()
    # one state is 8 x 128 x 128 float32 = 0.5 MiB: 1024 of them unbounded
    assert bounded < whole / 4


def test_off_the_gpu_and_without_the_kernel_it_is_the_ops(monkeypatch):
    called = []
    monkeypatch.setattr(gated_delta, "ops",
                        lambda *a: called.append(1) or ("ops",))
    args = _inputs(T=4)
    with mx.stream(mx.cpu):
        gated_delta.recurrence(*args[:6])
    gated_delta.recurrence(*args[:6], use_kernel=False)
    assert len(called) == 2


@pytest.mark.skipif(not backend.metal(), reason="Metal's kernel")
def test_metal_runs_the_familys_kernel():
    args = _inputs(T=8)
    y, s = gated_delta.recurrence(*args[:6])
    y0, s0 = gated_delta.gated_delta_kernel(*args[:6])
    assert mx.array_equal(y, y0).item() and mx.array_equal(s, s0).item()
    seen = []
    gated_delta.recurrence(*args[:6], metal=lambda *a: seen.append(1) or a)
    assert seen == [1]


@pytest.mark.skipif(backend.name() != "cuda", reason="the CUDA kernel")
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_the_cuda_kernel_matches_the_ops(vectorized, masked):
    args = _inputs(D=128, vectorized=vectorized, masked=masked)
    y, s = gated_delta.recurrence(*args)
    with mx.stream(mx.cpu):
        y0, s0 = gated_delta_ops(*args)
    assert mx.allclose(y, y0, atol=1e-4).item()
    assert mx.allclose(s, s0, atol=1e-4).item()
