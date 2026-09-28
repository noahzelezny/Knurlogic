"""The prompt cache's size (interfaces/serve.prompt_cache_policy): four
agents on a hybrid model lost their entries to a server-wide cap of 10
(2026-09-27). One machine: sized by bytes against its memory. A ring:
count-based, scaled with the concurrent agents, and said."""
from knurlogic.interfaces.serve import (GIB, PROMPT_CACHE_AGENTS_MAX,
                                        PROMPT_CACHE_PER_AGENT,
                                        prompt_cache_policy)


def test_one_machine_is_sized_by_bytes_against_the_headroom():
    n, b, why = prompt_cache_policy({"decode_concurrency": 32}, 1,
                                    100 * GIB, 60 * GIB)
    assert b == 20 * GIB
    assert n >= 4 * PROMPT_CACHE_PER_AGENT       # bytes decide, not a count
    assert "20.0 GiB" in why and "sized by memory" in why


def test_one_machine_explicit_bytes_and_size_win():
    n, b, why = prompt_cache_policy({"prompt_cache_bytes": 3 * GIB,
                                     "prompt_cache_size": 7}, 1,
                                    100 * GIB, 10 * GIB)
    assert (n, b) == (7, 3 * GIB) and "--prompt-cache-gib" in why


def test_one_machine_without_a_known_working_set_counts_per_agent():
    n, b, why = prompt_cache_policy({"decode_concurrency": 4}, 1, 0, GIB)
    assert b is None and n == 4 * PROMPT_CACHE_PER_AGENT
    assert "4 concurrent agents" in why


def test_a_ring_is_count_based_and_scales_with_the_agents():
    n, b, why = prompt_cache_policy({"decode_concurrency": 32,
                                     "prompt_cache_bytes": GIB}, 2,
                                    100 * GIB, 50 * GIB)
    assert b is None                              # never bytes on a ring
    assert n == PROMPT_CACHE_PER_AGENT * PROMPT_CACHE_AGENTS_MAX
    assert "per agent" in why and "count-based" in why
    n4, _, _ = prompt_cache_policy({"decode_concurrency": 4}, 2, 0, 0)
    assert n4 == 4 * PROMPT_CACHE_PER_AGENT > 10  # four agents keep theirs


def test_a_ring_takes_an_explicit_count_as_given():
    n, b, why = prompt_cache_policy({"prompt_cache_size": 12}, 2, 0, 0)
    assert (n, b) == (12, None) and "--prompt-cache-size" in why


def test_every_rank_gets_the_same_count_whatever_its_memory():
    s = {"decode_concurrency": 16}
    assert prompt_cache_policy(s, 2, 128 * GIB, 60 * GIB)[0] == \
        prompt_cache_policy(s, 2, 96 * GIB, 80 * GIB)[0]
