from knurlogic.interfaces.http.messages import to_openai


def test_a_system_message_inside_messages_joins_the_leading_one():
    # Claude Code sends one mid-way; Qwen's template refuses it anywhere but first
    out = to_openai({"system": "S", "messages": [
        {"role": "user", "content": "u"},
        {"role": "system", "content": [{"type": "text", "text": "late"}]}]})
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user"]
    assert out["messages"][0]["content"] == "S\n\nlate"


def test_a_late_system_message_alone_becomes_the_first():
    out = to_openai({"messages": [{"role": "user", "content": "u"},
                                  {"role": "system", "content": "late"}]})
    assert out["messages"][0] == {"role": "system", "content": "late"}
