"""The layer rules (docs/architecture.md), enforced on the source.

Every .py under src/knurlogic is parsed with `ast`; imports anywhere in a
file count, top level or inside a function, so a lazy import does not hide
a violation:

  1. only engine/ imports mlx, mlx_lm or mlx_vlm;
  2. only interfaces/ creates an HTTP server (http.server, socketserver);
  3. engine/, machine/, tuning/ and context_management/ never import
     knurlogic.interfaces.

Nothing is exempt today. A new exception is a design decision: make it
here, with the reason, not by loosening a rule silently.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "knurlogic"

MLX = {"mlx", "mlx_lm", "mlx_vlm"}
HTTP_SERVER = {"http.server", "socketserver"}
BELOW_INTERFACES = {"engine", "machine", "tuning", "context_management"}


def _package(path):
    """The top-level knurlogic subpackage a file is in ('' for the
    package's own modules)."""
    rel = path.relative_to(SRC).parts
    return rel[0] if len(rel) > 1 else ""


def _imports(path):
    """(lineno, absolute dotted name) of every import in the file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = ["knurlogic"] + list(path.relative_to(SRC).parent.parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            yield node.lineno, mod
            # `from knurlogic import interfaces` names the package in the alias
            for a in node.names:
                yield node.lineno, f"{mod}.{a.name}"


def _violations(rule):
    out = []
    for path in sorted(SRC.rglob("*.py")):
        pkg = _package(path)
        for lineno, name in _imports(path):
            if rule(pkg, name):
                out.append(f"{path.relative_to(SRC.parent.parent)}:{lineno}"
                           f" imports {name}")
    return out


def test_the_scan_sees_the_source():
    files = list(SRC.rglob("*.py"))
    assert len(files) > 100
    assert any(n.startswith("mlx") for p in SRC.glob("engine/*.py")
               for _, n in _imports(p)), "the scan must see engine's mlx imports"


def test_only_engine_imports_mlx():
    bad = _violations(lambda pkg, n: pkg != "engine"
                      and n.split(".")[0] in MLX)
    assert not bad, "mlx outside engine/:\n" + "\n".join(bad)


def test_only_interfaces_creates_an_http_server():
    bad = _violations(lambda pkg, n: pkg != "interfaces"
                      and n in HTTP_SERVER)
    assert not bad, "an HTTP server outside interfaces/:\n" + "\n".join(bad)


def test_lower_layers_do_not_import_interfaces():
    bad = _violations(lambda pkg, n: pkg in BELOW_INTERFACES
                      and (n == "knurlogic.interfaces"
                           or n.startswith("knurlogic.interfaces.")))
    assert not bad, ("engine/machine/tuning/context_management importing "
                     "interfaces:\n" + "\n".join(bad))
