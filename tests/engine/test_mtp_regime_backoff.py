"""The draft/plain controller backs off re-trying a regime that keeps losing."""
from knurlogic.engine.mtp import batch_loop as bl


def _batch():
    b = bl.MTPBatch(None, object(), lambda: None, copy_caches=False,
                    draft_max_rows=None)
    b.draft_max_rows = None
    return b


def _run(b, n, draft_s, plain_s):
    drafted = 0
    for _ in range(n):
        d = b.drafting_pays(1)
        drafted += d
        b._record_cost(1, d, draft_s if d else plain_s, 1)
    return drafted


def test_a_much_dearer_draft_is_rarely_retried():
    b = _batch()
    drafted = _run(b, 20000, 0.72, 0.055)       # measured GLM numbers
    # old rule: 6 draft steps every ~96 -> ~1200; now a handful of probes
    assert drafted <= 6 * 6


def test_a_winner_flip_resets_the_backoff():
    b = _batch()
    _run(b, 3000, 0.10, 0.05)                   # plain wins, backoff grows
    assert b._backoff[1][0] > 1
    _run(b, 20000, 0.02, 0.05)                  # drafting gets cheap
    assert b._backoff[1][0] == 1 or b.drafting_pays(1)


def test_a_close_race_still_rechecks_at_the_base_rate():
    b = _batch()
    drafted = _run(b, 400, 0.05, 0.05)
    assert drafted > 6
