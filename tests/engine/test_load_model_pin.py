"""engine/model/load_model.py is mlx-lm's load_model with its weight-file
open and its sanitize call going through weight_files (and its bitnet
import made absolute): it must be redone when the installed mlx-lm's
changes."""
import hashlib
import inspect

import mlx_lm
import mlx_lm.utils as upstream

from knurlogic.engine.model import load_model as ours


def test_the_copy_matches_the_installed_mlx_lm():
    src = inspect.getsource(upstream.load_model)
    got = hashlib.sha256(src.encode()).hexdigest()
    assert got == ours.UPSTREAM_SHA256, (
        f"mlx-lm {mlx_lm.__version__}'s load_model changed since the copy "
        f"in engine/model/load_model.py (made from {ours.UPSTREAM_VERSION}): "
        f"copy it again, keeping the two changes (the weight files opened "
        f"by open_weight_file, sanitize called through weight_files)")


def test_the_changes_are_the_file_open_and_the_sanitize():
    up = inspect.getsource(upstream.load_model).splitlines()
    me = inspect.getsource(ours.load_model).splitlines()
    diff = [(a, b) for a, b in zip(up, me) if a != b]
    assert len(up) == len(me)
    assert [b.strip() for _, b in diff] == [
        "weights.update(open_weight_file(wf, deferred=True))",
        "weights = sanitize(model, weights)",
        "from mlx_lm.models.bitlinear_layers import bitnet_quantize"]
