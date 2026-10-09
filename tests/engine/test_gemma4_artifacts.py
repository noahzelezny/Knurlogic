"""What the released gemma 4 artifacts' own configs give knurlogic: the
stop tokens, the tool-call parser and the sampling defaults are read from
the files, not assumed. The small json/jinja files of two released
artifacts (no weights, no tokenizer.json) are copied under
tests/support/artifacts/:

  gemma-4-e4b-it-8bit             mlx-community/gemma-4-e4b-it-8bit
  gemma-4-26b-a4b-it-VQ-6.2bpw    TheDrainFlorist/gemma-4-26b-a4b-it-VQ-6.2bpw
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS = sorted((ROOT / "tests/support/artifacts").glob("gemma-4-*"))


def test_both_artifacts_are_here():
    assert [a.name for a in ARTIFACTS] == [
        "gemma-4-26b-a4b-it-VQ-6.2bpw", "gemma-4-e4b-it-8bit"]


@pytest.mark.parametrize("art", ARTIFACTS, ids=lambda a: a.name)
def test_stop_tokens_are_the_generation_configs(art):
    """<eos> (1), <turn|> (106) and 50: mlx-lm's load_config reads
    generation_config.json's eos_token_id over config.json's, and load()
    hands it to the TokenizerWrapper as eos_token_ids -- the set
    engine/runtime/request.py stops on."""
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_config
    gen = json.loads((art / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [1, 106, 50]
    assert load_config(art)["eos_token_id"] == [1, 106, 50]


@pytest.mark.parametrize("art", ARTIFACTS, ids=lambda a: a.name)
def test_the_tool_parser_is_inferred_as_gemma4(art):
    """The template's <|tool_call> ... <tool_call|> names mlx-lm's gemma4
    parser (mlx_lm.tokenizer_utils._infer_tool_parser, through
    engine/model/load.tool_support); a tokenizer_config.json
    tool_parser_type, which mlx-lm reads before inferring (the 26B VQ
    artifact sets one), names the same parser."""
    pytest.importorskip("mlx_lm")
    from knurlogic.engine.model.load import tool_support
    tc = json.loads((art / "tokenizer_config.json").read_text())
    assert tc.get("tool_parser_type", "gemma4") == "gemma4"
    template = (art / "chat_template.jinja").read_text()
    assert tool_support(template)["parser"] == "gemma4"
    parser = importlib.import_module("mlx_lm.tool_parsers.gemma4")
    assert parser.tool_call_start in template
    assert parser.tool_call_end in template


@pytest.mark.parametrize("art", ARTIFACTS, ids=lambda a: a.name)
def test_sampling_defaults_are_the_generation_configs(art):
    """temperature 1.0, top_k 64, top_p 0.95 (Google's published set), and
    no non-thinking set for gemma4."""
    from knurlogic.machine.artifact import sampling_defaults
    assert sampling_defaults(art) == {"temp": 1.0, "top_k": 64,
                                      "top_p": 0.95}
