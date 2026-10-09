"""The page wires cluster/launch and cluster/recovery when it builds its
handler; importing it, or using the MCP tools, does not."""
import subprocess
import sys

_MCP_ALONE = r"""
import sys
from knurlogic.cluster import launch, recovery
from knurlogic.interfaces.mcp import inspection
before = (launch.status_fn, recovery.load_fn)
assert isinstance(inspection.state(), dict)
assert isinstance(inspection.ready(), dict)
assert "knurlogic.interfaces.page.server" not in sys.modules
assert (launch.status_fn, recovery.load_fn) == before
assert launch.status_fn is launch._no_status
assert recovery.load_fn is recovery._no_load
import knurlogic.interfaces.page.server as server
assert (launch.status_fn, recovery.load_fn) == before, "import wired"
server.make_handler({})
assert launch.status_fn is not launch._no_status
assert recovery.load_fn is not recovery._no_load
print("ok")
"""


def test_mcp_tools_never_need_the_page_and_the_page_wires_on_start():
    r = subprocess.run([sys.executable, "-c", _MCP_ALONE], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")
