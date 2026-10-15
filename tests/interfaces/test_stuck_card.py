"""A rank that stayed inside a step rather than leave (its GPU may be hung)
shows on the page's card, and in the JSON agents read, as stuck with the
machine to restart -- not loaded, not stopping forever."""
from knurlogic.cluster import jobs as J
from knurlogic.cluster import launch
from knurlogic.interfaces.page import loads, running
from knurlogic.machine import identity

REASON = "box-a's GPU may be hung -- restart box-a if it does not leave"


def _jobs(monkeypatch, stopping):
    rec = {"job": "j1", "rank": 0, "pid": 5, "port": 8080, "artifact": "/m/A",
           "machines": ["box-a", "box-b"], **({"stopping": True}
                                              if stopping else {})}
    monkeypatch.setattr(identity, "identity", lambda: {"name": "box-a"})
    monkeypatch.setattr(J, "by_job", lambda reg=None: {"j1": [rec]})
    monkeypatch.setattr(J, "phase_of", lambda job, recs, read=None: "stuck")
    monkeypatch.setattr(launch, "ENDED", {})
    monkeypatch.setattr(launch, "SPECS", {})
    (doc,) = launch.jobs_document()
    return doc


def test_a_live_job_with_a_stuck_rank_is_stuck(monkeypatch):
    doc = _jobs(monkeypatch, stopping=False)
    assert doc["phase"] == "stuck" and doc["stuck"] == REASON


def test_a_stopped_job_whose_rank_was_spared_is_stuck_not_stopping(
        monkeypatch):
    doc = _jobs(monkeypatch, stopping=True)
    assert doc["phase"] == "stuck" and doc["stuck"] == REASON


def test_the_card_and_running_json_say_stuck(monkeypatch):
    job = {"job": "j1", "phase": "stuck", "stuck": REASON, "port": 8080,
           "machines": ["box-a", "box-b"], "leader": "box-a"}
    monkeypatch.setattr(launch, "jobs_document", lambda: [dict(job)])
    monkeypatch.setattr(J, "registry", dict)
    from knurlogic.cluster import recovery
    monkeypatch.setattr(recovery, "for_job", lambda j: None)
    monkeypatch.setattr(recovery, "not_serving", list)
    doc = loads.with_jobs({"resident": [
        {"runtime": "knurlogic", "name": "A", "state": "loaded",
         "where": "http://127.0.0.1:8080"}]})
    (row,) = doc["resident"]
    assert row["state"] == "stuck" and row["detail"] == REASON
    monkeypatch.setattr(identity, "identity", lambda: {"name": "box-a"})
    (m,) = running.models_across(doc, "box-a")
    assert m["state"] == "stuck" and m["stuck"] == REASON


def test_a_stuck_job_without_a_row_is_stuck_in_running_json():
    doc = {"jobs": [{"job": "j1", "phase": "stuck", "stuck": REASON,
                     "artifact": "A", "machines": ["box-a"]}]}
    (m,) = running.models_across(doc, "box-a")
    assert m["state"] == "stuck" and m["stuck"] == REASON
