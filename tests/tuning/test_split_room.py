"""tuning/fit.split_room: the room a model split across machines leaves to
talk in -- each rank's budget less its share and step margin, KV per token
by the layers (pipeline) or heads (tensor) it holds, the context the first
rank to fill allows; and cluster/launch.placed_room over a placement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.tuning import fit  # noqa: E402

GIB = 1 << 30
# a hybrid: every 4th of 8 layers is full attention, 2 KV heads x 128 dims
HYBRID = {"num_hidden_layers": 8, "full_attention_interval": 4,
          "num_attention_heads": 16, "num_key_value_heads": 2,
          "head_dim": 128, "max_position_embeddings": 262144}


def test_kv_layer_weights_follow_kv_bytes_per_token():
    assert fit.kv_layer_weights(HYBRID) == [0, 0, 0, 1, 0, 0, 0, 1]
    typed = {"num_hidden_layers": 3,
             "layer_types": ["sliding_attention", "full_attention",
                             "full_attention"]}
    assert fit.kv_layer_weights(typed) == [0, 1, 1]
    assert fit.kv_layer_weights({"num_hidden_layers": 2}) == [1, 1]


def test_pipeline_rank_pays_for_its_own_attention_layers_only():
    per, _ = fit.kv_bytes_per_token(HYBRID)
    ranks = [{"name": "a", "budget_bytes": 100 * GIB,
              "holds_bytes": 50 * GIB, "bounds": (4, 8)},
             {"name": "b", "budget_bytes": 100 * GIB,
              "holds_bytes": 50 * GIB, "bounds": (0, 4)}]
    r = fit.split_room(ranks, HYBRID, "pipeline")
    assert r["fits"] and r["split"] == "pipeline"
    a, b = r["ranks"]
    assert a["kv_bytes_per_token"] == b["kv_bytes_per_token"] == per // 2
    assert a["left_bytes"] == 100 * GIB - 50 * GIB - fit.step_margin(100 * GIB)
    assert r["tokens"] == a["left_bytes"] // (per // 2)
    assert r["text"].startswith("leaves 90 GiB across 2 machines, about ")


def test_the_rank_that_fills_first_sets_the_context():
    ranks = [{"name": "a", "budget_bytes": 100 * GIB,
              "holds_bytes": 90 * GIB},
             {"name": "b", "budget_bytes": 100 * GIB,
              "holds_bytes": 40 * GIB}]
    r = fit.split_room(ranks, HYBRID, "tensor")
    a, b = r["ranks"]
    assert a["kv_bytes_per_token"] == fit.kv_bytes_per_token(HYBRID)[0] // 2
    assert r["tokens"] == a["tokens"] < b["tokens"]
    over = fit.split_room([{"name": "a", "budget_bytes": 10 * GIB,
                            "holds_bytes": 9 * GIB},
                           {"name": "b", "budget_bytes": 100 * GIB,
                            "holds_bytes": 1 * GIB}], HYBRID, "tensor")
    assert not over["fits"] and over["tokens"] == 0


def test_placed_room_reads_a_placement():
    from knurlogic.cluster import launch
    plan = {"split": "pipeline", "shares": [
        {"rank": 0, "machine": "a", "bytes": 50 * GIB, "bounds": [4, 8]},
        {"rank": 1, "machine": "b", "bytes": 50 * GIB, "bounds": [0, 4]}]}
    infos = [{"name": "a", "working_set_bytes": 100 * GIB,
              "available_bytes": 80 * GIB},
             {"name": "b", "working_set_bytes": 100 * GIB}]
    r = launch.placed_room(plan, infos, HYBRID)
    assert [x["budget_bytes"] for x in r["ranks"]] == [80 * GIB, 100 * GIB]
    assert r["tokens"] == r["ranks"][0]["tokens"]
