"""machine/allowance.py: the most memory knurlogic may use on this machine,
remembered in a file, lowering the load budget and the scheduler's working
set; the page server's /allowance.json; and the room a fit leaves to talk
in (tuning/resolve.context_room)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.machine import allowance, wired  # noqa: E402

GIB = 1 << 30


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_the_file_round_trips_and_clears(home):
    assert allowance.get() == 0
    assert allowance.set(80 * GIB) == 80 * GIB
    assert allowance.path() == home / "knurlogic" / "allowance.json"
    assert json.loads(allowance.path().read_text()) == {
        "allowance_bytes": 80 * GIB}
    assert allowance.get() == 80 * GIB
    allowance.set(0)
    assert not allowance.path().exists() and allowance.get() == 0
    with pytest.raises(ValueError):
        allowance.set(-1)


def test_a_damaged_file_is_no_allowance(home):
    allowance.path().write_text("{not json")
    assert allowance.get() == 0


def test_cap_only_lowers(home):
    assert allowance.cap(100 * GIB) == 100 * GIB      # none set
    allowance.set(60 * GIB)
    assert allowance.cap(100 * GIB) == 60 * GIB
    assert allowance.cap(40 * GIB) == 40 * GIB
    assert allowance.cap(0) == 60 * GIB               # the only limit known


def test_load_budget_is_capped_and_says_so(home, monkeypatch):
    from knurlogic.machine import loaded
    monkeypatch.setattr(wired, "detected_working_set_bytes", lambda: 100 * GIB)
    monkeypatch.setattr(loaded, "available_memory",
                        lambda: {"available_bytes": 90 * GIB})
    b = wired.load_budget()
    assert b["bytes"] == 90 * GIB and b["allowance_bytes"] == 0
    allowance.set(50 * GIB)
    b = wired.load_budget()
    assert b["bytes"] == 50 * GIB
    assert b["limited_by"] == "the knurlogic allowance"
    allowance.set(95 * GIB)           # above what is free: free still wins
    assert wired.load_budget()["limited_by"] == "memory available now"


def test_the_page_route_reads_and_sets_this_machine(home, monkeypatch):
    from knurlogic.interfaces import web
    monkeypatch.setattr(wired, "detected_working_set_bytes", lambda: 100 * GIB)
    monkeypatch.setattr(wired, "advise", lambda b: {"total_bytes": 128 * GIB})
    assert web.allowance_doc()["allowance_gib"] == 0
    out = web.set_allowance(b'{"gib": 64}')
    assert out["allowance_gib"] == 64 and out["effective_gib"] == 64
    assert allowance.get() == 64 * GIB
    assert "error" in web.set_allowance(b'{"gib": 500}')
    assert "error" in web.set_allowance(b'{"gib": "x"}')
    assert allowance.get() == 64 * GIB    # a refusal changes nothing
    web.set_allowance(b'{"gib": 0}')
    assert allowance.get() == 0


def test_the_scheduler_is_handed_the_working_set():
    from knurlogic.interfaces.http import scheduler_options
    assert scheduler_options({})["working_set_bytes"] is None
    assert scheduler_options({"working_set_bytes": 7})["working_set_bytes"] == 7


# --- the room a fit leaves ---------------------------------------------------

GLM = {"num_hidden_layers": 78, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
       "index_head_dim": 128, "layer_types": ["full_attention"] * 78,
       "max_position_embeddings": 1 << 20}


def test_room_is_working_set_less_weights_less_the_step_margin():
    from knurlogic.tuning.resolve import context_room, step_margin
    assert step_margin(40 * GIB) == 4 * GIB           # the 4 GiB floor
    assert step_margin(120 * GIB) == 6 * GIB          # 5%
    r = context_room(120 * GIB, 108 * GIB, GLM)
    assert r["left_bytes"] == 6 * GIB and r["margin_bytes"] == 6 * GIB
    assert r["tokens"] == 6 * GIB // r["kv_bytes_per_token"]
    assert r["small"]                   # far under a fifth of 1M tokens
    assert r["text"].startswith("leaves 6 GiB, about ")
    roomy = context_room(120 * GIB, 20 * GIB, GLM)
    assert not roomy["small"] and roomy["tokens"] > (1 << 20) // 5


def test_room_never_goes_negative_and_says_when_kv_is_unknown():
    from knurlogic.tuning.resolve import context_room
    r = context_room(10 * GIB, 9 * GIB, {})
    assert r["left_bytes"] == 0 and r["tokens"] == 0 and r["small"]
    assert r["fits"] and not context_room(10 * GIB, 11 * GIB, {})["fits"]
    assert "not known" in r["text"]
