"""The machine readouts off macOS (docs/design/os-agnostic.md), run on any
machine by faking the platform and the tools it would ask."""
import subprocess
from collections import namedtuple
from types import SimpleNamespace

import pytest

Times = namedtuple("Times", "user idle")


def _smi(monkeypatch, mod, stdout, code=0):
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=code, stdout=stdout))


def test_nvidia_reads_the_first_gpu(monkeypatch):
    from knurlogic.machine import nvidia
    _smi(monkeypatch, nvidia, "87, 40960\n12, 0\n")
    assert nvidia.query("utilization.gpu", "memory.used") == ["87", "40960"]


def test_nvidia_without_a_driver_is_none(monkeypatch):
    from knurlogic.machine import nvidia
    monkeypatch.setattr(nvidia.subprocess, "run", lambda *a, **k:
                        (_ for _ in ()).throw(FileNotFoundError()))
    assert nvidia.query("name") is None
    _smi(monkeypatch, nvidia, "", code=9)
    assert nvidia.query("name") is None


def test_unified_memory_has_no_gpu_bytes(monkeypatch):
    from knurlogic.machine import metrics, nvidia
    monkeypatch.setattr(metrics.sys, "platform", "linux")
    _smi(monkeypatch, nvidia, "55, [N/A]\n")
    assert metrics._gpu() == (55, None)
    _smi(monkeypatch, nvidia, "55, 1024\n")
    assert metrics._gpu() == (55, 1 << 30)


def test_linux_has_no_pressure_or_thermal_state(monkeypatch):
    from knurlogic.machine import metrics
    monkeypatch.setattr(metrics.sys, "platform", "linux")
    assert metrics._vm_pressure() is None and metrics._thermal() is None


def test_linux_temperature_is_the_hottest_sensor(monkeypatch):
    from knurlogic.machine import metrics
    monkeypatch.setattr(metrics.sys, "platform", "linux")
    t = SimpleNamespace
    monkeypatch.setattr(metrics.psutil, "sensors_temperatures", lambda: {
        "k10temp": [t(current=61.25)], "nvme": [t(current=40.0),
                                                t(current=0.0)]}, raising=False)
    assert metrics._temp_c() == 61.2
    monkeypatch.setattr(metrics.psutil, "sensors_temperatures", lambda: {},
                        raising=False)
    assert metrics._temp_c() is None


def test_cpu_share_from_tick_counters(monkeypatch):
    from knurlogic.machine import metrics
    ticks = iter([Times(user=10, idle=90), Times(user=40, idle=110)])
    monkeypatch.setattr(metrics.psutil, "cpu_times", lambda: next(ticks))
    monkeypatch.setattr(metrics, "_last_ticks", None)
    assert metrics._cpu_pct() is None          # nothing to diff yet
    assert metrics._cpu_pct() == 60.0          # 30 busy of 50


def test_linux_id_is_the_machine_id(monkeypatch, tmp_path):
    from knurlogic.machine import identity
    f = tmp_path / "machine-id"
    f.write_text("0123abcd\n")
    monkeypatch.setattr(identity.sys, "platform", "linux")
    monkeypatch.setattr(identity, "MACHINE_ID_FILES",
                        (str(tmp_path / "absent"), str(f)))
    monkeypatch.setattr(identity, "_ID", {})
    got = identity.identity()
    assert got["id_source"] == "machine-id"
    assert got["name"] == identity.socket.gethostname().removesuffix(".local")
    monkeypatch.setattr(identity, "MACHINE_ID_FILES", ())
    monkeypatch.setattr(identity, "_ID", {})
    assert identity.identity()["id_source"] == "hostname"


def test_linux_machine_names_the_gpu(monkeypatch, tmp_path):
    from knurlogic.machine import identity, nvidia
    product = tmp_path / "product_name"
    product.write_text("DGX Spark\n")
    monkeypatch.setattr(identity, "DMI_PRODUCT", str(product))
    monkeypatch.setattr(identity.sys, "platform", "linux")
    monkeypatch.setattr(identity, "_CHIP", {})
    _smi(monkeypatch, nvidia, "NVIDIA GB10\n")
    got = identity._other_machine()
    assert got["model"] == "DGX Spark" and got["chip"] == "GB10"
    assert got["kind"] == ""
    monkeypatch.setattr(identity, "_CHIP", {})
    _smi(monkeypatch, nvidia, "", code=9)
    assert identity._other_machine()["chip"] == identity._processor()


def test_linux_has_no_wired_limit(monkeypatch):
    from knurlogic.machine.memory import wired
    monkeypatch.setattr(wired.sys, "platform", "linux")
    w = wired.read()
    assert w.total_bytes > 0 and not w.known


def test_linux_memory_is_the_kernels_estimate(monkeypatch):
    import psutil

    from knurlogic.machine.memory import footprint
    monkeypatch.setattr("sys.platform", "linux")
    vm = SimpleNamespace(total=100, available=30, free=10, cached=15)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)
    got = footprint.available_memory()
    assert got["available_bytes"] == 30 and got["used_bytes"] == 70


@pytest.mark.parametrize("err", [subprocess.TimeoutExpired("x", 1), OSError()])
def test_nvidia_failures_read_none(monkeypatch, err):
    from knurlogic.machine import nvidia
    monkeypatch.setattr(nvidia.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(err))
    assert nvidia.query("name") is None
