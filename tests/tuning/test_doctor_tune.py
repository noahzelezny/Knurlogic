"""`doctor --tune` takes the same presets `serve --tune` does."""
import pytest

from knurlogic.interfaces import doctor
from knurlogic.tuning import settings as S


@pytest.mark.parametrize("preset", S.PRESETS)
def test_doctor_accepts_every_preset(monkeypatch, preset):
    seen = {}
    monkeypatch.setattr(doctor, "run",
                        lambda *a, **k: seen.setdefault("tune", a[-1]) and 0)
    doctor.main(["some-artifact", "--tune", preset])
    assert seen["tune"] == preset


def test_every_preset_resolves():
    assert set(S.PRESETS) <= set(S.TUNE_PROFILES)
