"""engine/runtime/tool_calls.recover: arguments mlx-lm's qwen3_coder parser
drops (no </parameter>, a JSON body) are read back from the call's text."""
import logging

from mlx_lm.tool_parsers import qwen3_coder

from knurlogic.engine.runtime import tool_calls

TOOLS = [{"type": "function", "function": {
    "name": "answer", "parameters": {
        "type": "object", "required": ["path"],
        "properties": {"path": {"type": "string"},
                       "lines": {"type": "integer"}}}}}]


def _parse(text):
    return tool_calls.recover(text, qwen3_coder.parse_tool_call(text, TOOLS),
                              TOOLS)


def test_a_well_formed_call_is_untouched():
    t = "<function=answer>\n<parameter=path>\nengine/x.py\n</parameter>\n</function>"
    assert _parse(t)["arguments"] == {"path": "engine/x.py"}


def test_a_missing_close_parameter_tag_keeps_the_value():
    t = ("<function=answer>\n<parameter=path>\nengine/x.py\n"
         "<parameter=lines>\n40\n</function>")
    assert qwen3_coder.parse_tool_call(t, TOOLS)["arguments"] == {}
    assert _parse(t)["arguments"] == {"path": "engine/x.py", "lines": 40}


def test_a_json_body_is_read():
    t = '<function=answer>\n{"path": "engine/x.py"}\n</function>'
    assert _parse(t)["arguments"] == {"path": "engine/x.py"}


ONE = [{"type": "function", "function": {
    "name": "answer", "parameters": {
        "type": "object", "required": ["path"],
        "properties": {"path": {"type": "string"}}}}}]


def test_a_bare_value_goes_to_the_only_parameter():
    # what Qwen3.6 wrote, from the server log
    t = "\n<function=answer>\nengine/prompt_cache/disk.py\n</parameter>\n</function>\n"
    parsed = qwen3_coder.parse_tool_call(t, ONE)
    assert parsed["arguments"] == {}
    assert tool_calls.recover(t, parsed, ONE)["arguments"] == \
        {"path": "engine/prompt_cache/disk.py"}
    # with two parameters a bare value is not guessed
    assert _parse("<function=answer>\nengine/x.py\n</function>")["arguments"] == {}


def test_a_call_with_nothing_to_recover_is_logged(caplog):
    t = "<function=answer>\n</function>"
    with caplog.at_level(logging.WARNING):
        assert _parse(t)["arguments"] == {}
    assert "requires ['path']" in caplog.text and "<function=answer>" in caplog.text
