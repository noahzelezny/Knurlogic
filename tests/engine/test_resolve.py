"""The resolver's job is to be the last word on every knob. Pin that."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import measured, numerics
from knurlogic.tuning.fit import decode_chunk_for
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def _art(**kw):
    base = dict(path=Path("/nonexistent/art"), model_type="qwen4_exp_text",
                model_file="model.py", bytes_on_disk=70 * GIB,
                hidden_size=2560, moe_intermediate_size=640,
                vq_modules={"m": {"d": 4, "K": 2048}})
    base.update(kw)
    return Artifact(**base)


def test_unknown_budget_keeps_defaults():
    r = resolve(_art(), 0)
    assert r.env["VQ_DECODE_CHUNK"] == str(measured.DECODE_CHUNK_DEFAULT)
    assert r.env["KNURLOGIC_PREFILL_CHUNK"] == str(measured.PREFILL_CHUNK_DEFAULT)


def test_does_not_fit_goes_tightest_not_default():
    """The bug this pins: negative headroom must not fall into the
    'unknown budget' branch and hand back the roomy default."""
    r = resolve(_art(), 48 * GIB)
    assert r.env["VQ_DECODE_CHUNK"] == str(measured.DECODE_CHUNK_MIN)
    assert r.warnings, "an artifact that does not fit must say so"


def test_chunk_never_exceeds_default():
    """Smaller is also faster (128 -> 32 is 1.37x), so headroom never buys
    a larger chunk."""
    assert decode_chunk_for(10_000 * GIB) == measured.DECODE_CHUNK_DEFAULT




def test_profile_selects_numerics_flags():
    assert all(resolve(_art(), 96 * GIB, "v1.5").env[f] == "0"
               for f in numerics.NUMERICS_FLAGS)
    assert all(resolve(_art(), 96 * GIB, "v2").env[f] == "1"
               for f in numerics.NUMERICS_FLAGS)


def test_vq_without_model_file_is_a_warning():
    r = resolve(_art(model_file=None), 96 * GIB)
    assert any("model_file" in w for w in r.warnings)


def test_non_vq_artifact_gets_only_the_generic_knobs():
    """The value proposition is 'it runs', not 'it runs VQ'. A stock affine
    artifact still needs the prefill and cache knobs; it has no dense-expert
    decode buffer, so VQ_DECODE_CHUNK would be cargo cult."""
    r = resolve(_art(vq_modules={}), 96 * GIB)
    assert "VQ_DECODE_CHUNK" not in r.env
    # the prompt chunk is still set -- its width is read from the room
    # (test_tuning.py covers which width)
    assert int(r.env["KNURLOGIC_PREFILL_CHUNK"]) in (
        measured.PREFILL_CHUNK_DEFAULT, *measured.PREFILL_CHUNK_LADDER)
    assert r.env["KNURLOGIC_CACHE_LIMIT_GB"] == str(measured.CACHE_LIMIT_GB_DEFAULT)
    assert "VQ_CACHE_LIMIT_GB" not in r.env


def test_vendored_architecture_wins_over_site_packages():
    """Vendoring is only meaningful if the vendored copy is the one used."""
    from knurlogic.engine import arch
    from knurlogic.engine.register import source_for
    for row in arch.check("qwen4_exp_text"):
        if source_for(row.module)[0] is not None:
            assert row.vendored, f"{row.module} should resolve to the vendored copy"


def test_moe_pulls_in_its_base_architecture():
    """One drifted base reaches 11 artifacts through the subclass."""
    from knurlogic.engine import arch
    assert arch.required_modules("qwen3_5_moe_text") == ["qwen3_5_moe", "qwen3_5"]


def test_registering_a_subclass_pulls_its_base_first():
    """A vendored subclass must never land on a site-packages base: that
    silently mixes two versions of the arithmetic, which is the exact failure
    this package exists to end. It passed once only by alphabetical luck."""
    from knurlogic.engine.register import _with_dependencies
    order = _with_dependencies(["qwen3_5_moe"])
    assert order.index("qwen3_5") < order.index("qwen3_5_moe")


def test_pins_are_loaded_and_make_doctor_say_ok():
    """A pin is only written after a model generated a token with clean
    provenance, so an 'ok' from doctor means 'it ran', not 'it imports'."""
    from knurlogic.engine import arch
    if not arch.PINNED_SHA256:
        return  # nothing validated on this checkout yet
    for row in arch.check("qwen3_5_moe_text"):
        if row.module in arch.PINNED_SHA256 and row.vendored:
            assert row.state == "OK", f"{row.module} is {row.state}"


def test_package_architectures_are_found_and_hosted_correctly():
    """glm5_next is a PACKAGE (vendored with its import closure), not a flat
    file; it registers under mlx_lm's name like every other architecture
    (mlx-vlm is not needed)."""
    from knurlogic.engine import arch
    from knurlogic.engine.register import available, source_for
    if "glm5_next" not in available():
        return
    src, is_pkg = source_for("glm5_next")
    assert is_pkg, "glm5_next must vendor as a package"
    assert arch.host_for("glm5_next") == "mlx_lm"
    assert arch.host_for("qwen4_exp") == "mlx_lm"


#: Directories that ARE engine code, and are allowed to import one. Each is
#: here for a stated reason, not because a glob happened to miss it.
ENGINE_SIDE = {
    # THE FOLDER IS THE RULE. engine/ holds serving (serve/), the drafting code, the
    # vendored architectures and the tools that act on them -- every line
    # that is arithmetic on an mlx model or edits mlx-lm's namespace. Nothing
    # outside it may import mlx. It used to be a list of files and folders,
    # each exempted with a reason; now that the layout says it, the list is
    # one entry.
    "engine",
}


def test_mlx_lives_behind_the_engine_seam():
    """The point of engine/ is that it is the ONLY folder importing an
    engine. If mlx names leak back into the other modules, swapping the
    engine stops being a one-folder change and this test is the tripwire.

    It used to glob `src/knurlogic/*.py`, so every subpackage was exempt by
    accident. It walks the tree now, and a directory is exempt only by being
    named in ENGINE_SIDE with a reason.
    """
    import ast
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "knurlogic"
    offenders = {}
    for f in sorted(src.rglob("*.py")):
        rel = f.relative_to(src)
        if rel.parts[0] in ENGINE_SIDE:
            continue
        # Parsed, not grepped: a regex over the text matched prose -- "exo
        # imports them from mlx-lm directly" inside a string -- and the
        # answer to a false alarm must never be rewording the sentence.
        # ast sees import statements, including ones nested in functions,
        # which is exactly where a lazy import would hide.
        hits = []
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                hits += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                hits.append(node.module or "")
        hits = [h.split(".")[0] for h in hits
                if re.match(r"mlx\w*$", h.split(".")[0])]
        if hits:
            offenders[str(rel)] = sorted(set(hits))
    assert not offenders, f"mlx imported outside the seam: {offenders}"


def test_asking_what_an_artifact_has_does_not_load_an_engine():
    """`doctor`, `discover` and the page all ask whether an artifact ships a
    drafting head. That is a question about a file -- a safetensors header is
    a length prefix and a JSON blob -- and none of them should pay for mlx to
    answer it. The drafting code lives in the same package and imports mlx on
    every line, so the split has to be enforced rather than intended."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import knurlogic.engine.mtp as m; m.find_head; "
         "print(any(k == 'mlx' or k.startswith('mlx.') or "
         "k.startswith('mlx_') for k in sys.modules))"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing knurlogic.engine.mtp pulled in an engine: " + out.stdout)


def test_dense_vq_artifacts_are_recognised_as_vq():
    """Dense rungs declare vq_linear/vq_embed, not vq_modules. Keying on
    vq_modules alone made `serve` announce "not a VQ artifact" for a VQ 27B
    and skip every kernel setting."""
    dense = _art(vq_modules={}, vq_other={"vq_linear": {"some": "module"}})
    assert dense.is_vq
    r = resolve(dense, 96 * GIB)
    assert r.env.get("VQ_DECODE_CHUNK")
    assert not resolve(_art(vq_modules={}), 96 * GIB).env.get(
        "VQ_DECODE_CHUNK")


# --- the spike is a FAMILY effect, not a box effect -------------------------
# Measured across boxes and families: the same prefill spike on M4 and on M3,
# and DeepSeek V4 far more dramatic than Qwen3.5. A resolver keyed on the box
# alone cannot express that. The transient is chunk * out * in * 2, and
# out/in are the model's -- gate_up is [2 * moe_intermediate_size,
# hidden_size] -- so the formula predicts exactly the observation.

def _family(hidden, moe_inter, size_gib=70):
    return _art(hidden_size=hidden, moe_intermediate_size=moe_inter,
                bytes_on_disk=size_gib * GIB)


def test_same_box_different_family_resolves_differently():
    """The observation, encoded: on ONE box, a big-expert family must get a
    tighter chunk than a small-expert one. Same headroom, same everything
    else -- only the model's shape differs."""
    box = 80 * GIB
    big = resolve(_family(7168, 2048), box)      # deepseek-shaped experts
    small = resolve(_family(2560, 640), box)     # qwen-shaped experts
    assert int(big.env["VQ_DECODE_CHUNK"]) < int(small.env["VQ_DECODE_CHUNK"]), (
        "the family with larger experts holds a larger dense transient per "
        "unit of chunk, so it is the one that has to be resolved down")


def test_same_artifact_two_boxes_same_headroom_resolves_the_same():
    """The other half of the observation: the box is not the discriminator.
    An M3 and an M4 with the same usable working set get the same answer --
    nothing in the transient formula refers to the machine."""
    a = _family(7168, 2048, size_gib=60)
    assert (resolve(a, 96 * GIB).env == resolve(a, 96 * GIB).env)


def test_a_config_without_an_expert_shape_says_it_assumed_one():
    """A dense or unusual config declares no MoE shape. Falling back is fine;
    falling back silently is not -- the number would look measured."""
    r = resolve(_art(hidden_size=None, moe_intermediate_size=None), 80 * GIB)
    assert any("ASSUMED" in n or "assumed" in n.lower() for n in r.notes)


def test_the_model_shape_may_tighten_but_not_loosen_yet():
    """Sizing from the model loosens the knob for small-expert families. That
    direction has not been measured, and being wrong there is an OOM -- so it
    is refused, and the refusal is said out loud rather than hidden."""
    from knurlogic.tuning.fit import decode_chunk_for, expert_transient_bytes_per_unit
    small = _family(2560, 640)
    headroom = 2 * GIB
    per, _ = expert_transient_bytes_per_unit(small)
    would = decode_chunk_for(headroom, bytes_per_unit=per)
    frozen = decode_chunk_for(headroom)
    assert would > frozen, "fixture must exercise the loosening direction"

    r = resolve(small, small.bytes_on_disk + headroom)
    assert int(r.env["VQ_DECODE_CHUNK"]) == frozen
    assert any("NOT taken" in n for n in r.notes)


def test_a_rank_is_judged_on_its_share_not_the_whole_artifact():
    """Each rank of a 397B pipeline split printed '114.9 GiB to hold
    against a 84.0 GiB working set -- it does not fit this box' though its
    share fit: the warning must be about what THIS box holds."""
    a = _art(bytes_on_disk=115 * GIB)
    assert any("does not fit" in w for w in resolve(a, 84 * GIB).warnings)
    r = resolve(a, 84 * GIB, holds_bytes=50 * GIB)
    assert not any("does not fit" in w for w in r.warnings)
    r = resolve(a, 40 * GIB, holds_bytes=50 * GIB)
    assert any("50.0 GiB to hold" in w for w in r.warnings)


def test_a_pipeline_ranks_share_is_its_layers_plus_what_every_rank_holds():
    from knurlogic.interfaces.serve import pipeline_share_bytes
    per = [GIB] * 8
    # rank 0 holds the LAST counts[0] layers, rank 1 the first counts[1]
    assert pipeline_share_bytes(per, GIB, 0, 2, [6, 2]) == 7 * GIB
    assert pipeline_share_bytes(per, GIB, 1, 2, [6, 2]) == 3 * GIB
    assert pipeline_share_bytes(per, GIB, 1, 2, None) == 5 * GIB
    # the MTP head is rank 0's alone
    assert pipeline_share_bytes(per, GIB, 0, 2, [6, 2], leader=2 * GIB) \
        == 9 * GIB
    assert pipeline_share_bytes(per, GIB, 1, 2, [6, 2], leader=2 * GIB) \
        == 3 * GIB
