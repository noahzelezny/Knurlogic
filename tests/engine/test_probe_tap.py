"""engine/probe/tap: the wanted layer's output is kept, per position, and
the class is left as it was when the tap goes."""
import mlx.core as mx
import numpy as np
from mlx import nn

from knurlogic.engine.probe import store
from knurlogic.engine.probe.tap import Tap, default_layers


class Layer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, x):
        return x + self.k


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Layer(float(i)) for i in range(4)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Core()

    def __call__(self, x):
        return self.model(x)


def test_a_layer_is_read_at_each_position_and_untapped_after():
    m = Model()
    before = Layer.__call__
    tap = Tap(m, [1, 2])
    m(mx.zeros((2, 3, 5)))                  # B=2, width 3, D=5
    a = np.array(tap.at(1, 0))
    assert a.shape == (2, 5) and np.allclose(a, 0 + 1)        # after 0, 1
    assert np.allclose(np.array(tap.at(2, -1)), 0 + 1 + 2)
    assert np.array(tap.states(0)).shape == (2, 2, 5)
    tap.remove()
    assert Layer.__call__ is before


def test_the_score_is_the_probe_on_the_state():
    m = Model()
    tap = Tap(m, [3])
    p = store.Probe(base="b", layer=3, mean=np.zeros(5, np.float32),
                    scale=np.ones(5, np.float32),
                    w=np.full(5, 0.1, np.float32), b=-1.0)
    m(mx.zeros((1, 1, 5)))
    s = np.array(tap.score(p, 0))
    assert np.allclose(s, p.score(np.full((1, 5), 6.0)), atol=1e-5)
    tap.remove()


def test_onboarding_reads_between_half_and_two_thirds_depth():
    assert default_layers(64) == [32, 38, 42]
