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


def test_a_seeded_row_drafts_every_step_whatever_the_timing():
    from knurlogic.engine.mtp.sampling import Keys
    b = _batch()
    b.params = [bl.RowParams(max_tokens=1, dist=None, processors=[], eos=set(),
                             keys=Keys(1234))]
    b.drafts = [True]
    assert _run(b, 400, 0.72, 0.055) == 400     # plain is far cheaper
    b.drafts = [False]                          # a row the head can't draft
    assert _run(b, 400, 0.72, 0.055) < 400


def test_a_rings_own_seed_does_not_pin_drafting():
    """A split seeds every row so its ranks draw alike (tensor.assign_seed);
    that seed is not the client's and must not force drafting -- it did,
    so a split drafted (and DSpark verified all K) whatever the timing."""
    from knurlogic.engine.mtp.sampling import Keys
    from knurlogic.engine.runtime.tensor import RING_SEED, assign_seed
    s = assign_seed({"temp": 0.0})
    assert s[RING_SEED] is True and assign_seed({"seed": 7}).get(RING_SEED) is None
    b = _batch()
    b.params = [bl.RowParams(max_tokens=1, dist=None, processors=[], eos=set(),
                             keys=Keys(1234, pins=False))]
    b.drafts = [True]
    assert _run(b, 400, 0.72, 0.055) < 400      # plain is far cheaper


def test_a_rings_own_seed_does_not_pin_the_verify_width(monkeypatch):
    from knurlogic.engine.mtp import block_loop as BL
    from knurlogic.engine.mtp.sampling import Keys
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    b = BL.BlockBatch(None, None, lambda: None, copy_caches=True,
                      block_size=5)
    b.params = [bl.RowParams(max_tokens=1, dist=None, processors=[], eos=set(),
                             keys=Keys(1234, pins=False))]
    b.drafts = [True]
    for k in range(1, 6):
        b._vcost[(1, k)] = (0.060 + 0.010 * k, 99)
        b._vwarm.add((1, k))
    b._vacc[1] = [0.8, 0.4, 0.2, 0.1, 0.05]
    assert b._width(1, None) < 5
    b.params[0] = bl.RowParams(max_tokens=1, dist=None, processors=[],
                               eos=set(), keys=Keys(1234))
    assert b._width(1, None) == 5               # the client's seed pins K
