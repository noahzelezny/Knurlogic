"""One builder for the `knurlogic serve` command line: the argv the page /
MCP spawn and a cluster rank build is byte-identical to the lists they
built by hand before."""
import json
import sys

from knurlogic.cluster.launch import rank_argv
from knurlogic.machine.servers import serve_argv


def _old_spawn(path, port, tune, sets, draft):
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--port", str(port), "--tune", tune]
    for k, v in sorted((sets or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if not draft:
        cmd.append("--no-draft")
    return cmd


def _new_spawn(path, port, tune, sets, draft):
    cmd = serve_argv(path, "--port", str(port), "--tune", tune)
    for k, v in sorted((sets or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if not draft:
        cmd.append("--no-draft")
    return cmd


def _old_rank(path, spec, files):
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--rank", str(spec["rank"]), "--world", str(spec["world"]),
           "--split", spec["split"], "--link", spec["link"],
           "--job", spec["job"],
           "--prefill-chunk", str(spec["prefill_chunk"]),
           "--prefill-why", str(spec.get("prefill_why") or ""),
           "--working-set-gib", f"{float(spec.get('working_set_gib') or 0):.3f}",
           "--tune", spec.get("tune") or "default"]
    if spec.get("port"):
        cmd += ["--port", str(int(spec["port"]))]
    if spec.get("serve_hosts") and spec["rank"] == 0:
        cmd += ["--host", ",".join(spec["serve_hosts"])]
    if spec["link"] == "ring":
        cmd += ["--hosts", ",".join(spec["hosts"])]
    else:
        cmd += ["--ibv-devices", files["ibv"],
                "--coordinator", spec["coordinator"]]
    if spec.get("layers"):
        cmd += ["--layers", ",".join(str(int(x)) for x in spec["layers"])]
    if spec.get("bandwidth_gbs"):
        cmd += ["--bandwidth-gbs", str(float(spec["bandwidth_gbs"]))]
    for k, v in sorted((spec.get("sets") or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if spec.get("chips"):
        cmd += ["--ring-chips", json.dumps(spec["chips"])]
    return cmd


def test_spawn_argv_is_unchanged():
    for args in [("/m/a", 8080, "default", None, True),
                 ("/m/b c", 8093, "fast", {"Z": 1, "A": "x"}, False)]:
        assert _new_spawn(*args) == _old_spawn(*args)


def test_rank_argv_is_unchanged():
    base = {"rank": 0, "world": 2, "split": "tensor", "job": "ab12",
            "prefill_chunk": 512}
    specs = [
        {**base, "link": "ring", "hosts": ["192.0.2.1", "192.0.2.2"]},
        {**base, "rank": 1, "link": "jaccl", "coordinator": "192.0.2.1:5000",
         "port": 8090, "serve_hosts": ["192.0.2.1"], "tune": "fast",
         "prefill_why": "fits", "working_set_gib": 41.5, "layers": [30, 31],
         "bandwidth_gbs": 9.5, "sets": {"B": 2, "A": 1},
         "chips": ["M3 Ultra", "M4 Max"]},
        {**base, "link": "ring", "hosts": ["a", "b"], "port": 8080,
         "serve_hosts": ["192.0.2.1", "127.0.0.1"]},
    ]
    files = {"ibv": "/tmp/ibv.json"}
    for spec in specs:
        assert rank_argv("/m/x", spec, files) == _old_rank("/m/x", spec, files)
