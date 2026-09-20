"""MTP: the difference between declaring a head and having one.

The bug these tests pin is a claim knurlogic USED to print: that 40 artifacts
had downloaded MTP weights which the architecture discards at load. Measured
against all 54 artifacts on the machine, 39 of those 40 ship no head tensors
at all -- the converter dropped them, not the loader. A person told "you
downloaded them and they will not run" would go looking for a flag that
cannot exist.
"""
import json
import struct

from knurlogic import mtp
from knurlogic.artifact import Artifact


def _safetensors(path, keys, metadata=None):
    """A real safetensors file: length-prefixed JSON header, then the data.

    Written rather than mocked, because the thing under test IS the header
    parse -- a fake that returns a dict would test nothing.
    """
    hdr = {k: {"dtype": "F32", "shape": [1], "data_offsets": [i, i + 4]}
           for i, k in enumerate(keys)}
    if metadata:
        hdr["__metadata__"] = metadata
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob
                     + b"\0" * (len(keys) + 4))
    return path


def _artifact(d, config):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    _safetensors(d / "model.safetensors", ["model.embed_tokens.weight"])
    return Artifact.load(str(d))


def test_a_community_rung_declaring_a_head_is_not_a_defect(tmp_path):
    """The `mtp` key is inherited from the upstream config and no publisher
    ships the weights. Measured here: 11 of 11 built heads sit beside a VQ
    artifact and not one community rung has one. Reporting this as something
    missing sends someone looking for a fix that does not exist."""
    a = _artifact(tmp_path / "m", {"model_type": "qwen4_exp", "mtp": {}})
    s = mtp.status(a)
    assert s.state == mtp.DECLARED and s.is_vq is False
    assert "nothing to do" in s.render()
    assert "vqlab" not in s.render()


def test_a_vq_artifact_without_a_head_points_at_the_thing_that_builds_them(
        tmp_path):
    """vqlab builds models, knurlogic runs them. A missing head on an
    artifact vqlab built is a packing step that did not happen -- still not
    knurlogic's to do, but worth naming."""
    a = _artifact(tmp_path / "m", {"model_type": "qwen4_exp", "mtp": {},
                                   "vq_modules": {"a": {"d": 2, "K": 256}}})
    s = mtp.status(a)
    assert s.state == mtp.DECLARED and s.is_vq is True
    assert "vqlab" in s.render()


def test_graft_weights_in_the_trunk_are_graftable(tmp_path):
    d = tmp_path / "m"
    a = _artifact(d, {"model_type": "qwen4_exp", "mtp": {}})
    _safetensors(d / "model-00002.safetensors",
                 ["mtp.fc.weight", "mtp.layers.0.mlp.gate.weight"])
    s = mtp.status(a)
    assert s.state == mtp.GRAFTABLE
    assert s.graft_tensors == 2


def test_a_built_head_is_found_outside_the_model_glob(tmp_path):
    """The sidecar is named to stay OUT of `model*.safetensors` so it costs
    nothing until asked for. An index-only scan misses all eleven real ones."""
    d = tmp_path / "m"
    a = _artifact(d, {"model_type": "qwen4_exp"})
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight":
                                   "model.safetensors"}}))
    _safetensors(d / "mtp-head-q6.safetensors",
                 ["block.self_attn.q_proj.weight", "fc.weight",
                  "mixer.weight", "norm_e.weight", "norm_h.weight"],
                 {"vqlab_mtp": json.dumps({"bits": 6, "group_size": 32})})
    s = mtp.status(a)
    assert s.state == mtp.BUILT
    assert s.head.bits == 6
    assert s.head.family == "qwen4_exp"
    assert "PRESENT and built" in s.render()


def test_family_comes_from_the_module_tree_not_only_the_label(tmp_path):
    """The qwen4_exp packs predate the `family` metadata field entirely, and
    the qwen3_5 head is a different tree (norm_out, no mixer). The layout is
    what the head IS; the label is a string someone wrote."""
    d = tmp_path / "m"
    _artifact(d, {"model_type": "qwen3_5"})
    _safetensors(d / "mtp-head-q6.safetensors",
                 ["block.x", "fc.weight", "norm_e.weight", "norm_h.weight",
                  "norm_out.weight"], {"vqlab_mtp": "{}"})
    assert mtp.find_head(d).family == "qwen3_5"

    e = tmp_path / "g"
    _artifact(e, {"model_type": "glm5_next"})
    _safetensors(e / "mtp-head-q6.safetensors",
                 ["eh_proj.weight", "enorm.weight", "hnorm.weight",
                  "final_norm.weight", "self_attn.q_proj.weight"],
                 {"vqlab_mtp": json.dumps({"family": "glm5_next"})})
    assert mtp.find_head(e).family == "glm5_next"


def test_no_head_and_no_declaration_says_nothing(tmp_path):
    a = _artifact(tmp_path / "m", {"model_type": "qwen3_5"})
    s = mtp.status(a)
    assert s.state == mtp.NONE
    assert s.render() == ""
