"""An element the page hides by id (`$('x').hidden=...`) is hidden: a class
rule that sets `display` outranks the browser's own [hidden] rule, so its
class needs one of its own. `.optrow{display:flex}` kept Dynamic MTP on the
load panel with MTP switched off."""
import re
from pathlib import Path

ASSETS = (Path(__file__).resolve().parents[2]
          / "src" / "knurlogic" / "interfaces" / "page" / "assets")


def _css() -> str:
    return (ASSETS / "page.css").read_text()


def _classes_of(id_: str, html: str) -> list:
    m = re.search(rf'<[^>]*\bid="{re.escape(id_)}"[^>]*>', html)
    if not m:
        return []
    c = re.search(r'\bclass="([^"]*)"', m.group(0))
    return c.group(1).split() if c else []


def _displayed(cls: str, css: str) -> bool:
    """A rule for exactly `.cls` sets display to something visible."""
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        sels = [s.strip() for s in sel.split(",")]
        if f".{cls}" in sels and re.search(r"display\s*:\s*(?!none)", body):
            return True
    return False


def _hidden_rule(cls: str, css: str) -> bool:
    return bool(re.search(rf"\.{re.escape(cls)}\[hidden\]\s*\{{[^}}]*"
                          r"display\s*:\s*none", css)) or \
        bool(re.search(r"(^|[\s,}])\[hidden\]\s*\{[^}]*display\s*:\s*none",
                       css))


def test_every_id_the_page_hides_has_a_hidden_rule():
    html = (ASSETS / "index.html").read_text()
    css = _css()
    js = "\n".join(p.read_text() for p in ASSETS.rglob("*.js")
                   if "vendor" not in p.parts)
    ids = set(re.findall(r"\$\('([\w-]+)'\)\.hidden\s*=", js))
    assert "mtpdynrow" in ids
    bad = [(i, c) for i in sorted(ids) for c in _classes_of(i, html)
           if _displayed(c, css) and not _hidden_rule(c, css)]
    assert not bad, f"hidden by id, shown by its class's display: {bad}"
