"""The real Vision-Exp tokenizer: knurlogic's prompt (its template, the
image placeholders, the part joining) tokenizes to the ids DeepSeek's own
encoder gives, chat and every thinking effort. Skipped without the
artifact on this Mac (KNURLOGIC_DSV4_VISION names another copy)."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: the converted Vision-Exp artifact's folder
ART = Path(os.environ.get("KNURLOGIC_DSV4_VISION") or "/nonexistent")
FIX = ROOT / "tests/support/fixtures_deepseek_v4_vision"
IMAGE_ID = 129264       # <｜deepseek_image｜>

pytestmark = pytest.mark.skipif(not (ART / "tokenizer.json").is_file(),
                                reason="set KNURLOGIC_DSV4_VISION to the "
                                "converted Vision-Exp artifact")


def _encoder():
    spec = importlib.util.spec_from_file_location(
        "encoding_dsv4_vision_tok", FIX / "encoding_dsv4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mode,effort", [("chat", None), ("thinking", None),
                                         ("thinking", "high"),
                                         ("thinking", "max")])
def test_prompt_ids_equal_deepseeks_encoder(mode, effort):
    from mlx_lm.utils import load_tokenizer

    from knurlogic.engine import templates
    from knurlogic.engine.runtime.prompt import flatten
    from knurlogic.engine.vision.request import with_placeholders
    enc = _encoder()
    msgs = json.loads((FIX / "example_vl_harmony.json").read_text())[0][
        "messages"]
    tok = load_tokenizer(ART)
    assert templates.install(tok) == "deepseek_v4_vision"
    want = tok._tokenizer.encode(
        enc.encode_messages(copy.deepcopy(msgs), thinking_mode=mode,
                            reasoning_effort=effort),
        add_special_tokens=False)
    kw = {"thinking_mode": mode}
    if effort:
        kw["reasoning_effort"] = effort
    flat = flatten(with_placeholders(copy.deepcopy(msgs),
                                     [enc.IMAGE_PLACEHOLDER] * 2), tok)
    ids = tok.apply_chat_template(flat, add_generation_prompt=True,
                                  tokenize=True, **kw)
    ids = ids["input_ids"] if isinstance(ids, dict) else list(ids)
    assert ids == want
    assert sum(i == IMAGE_ID for i in ids) == 2
