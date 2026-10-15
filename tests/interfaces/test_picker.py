"""interfaces/page/assets/views/picker.js: what greys the Launch button."""
from knurlogic.interfaces import documents


def test_launch_is_greyed_when_the_split_preview_cannot_place_it():
    """The split preview places a model as Launch would (loads.room_across);
    what it refuses, or finds too big for the memory free now, greys Launch
    with that reason rather than only printing it."""
    js = (documents.ASSETS / "views" / "picker.js").read_text()
    gate = js[js.index("function launchBlock"):js.index("// --- one peer picked")]
    assert "sr.error" in gate and "sr.fits===false" in gate
