"""Memory pressure as macOS reports it, read by the scheduler's memory
guard (engine/runtime/memory_guard.py) to warn, never to refuse:
macOS's own pressure level and this process's bytes in the compressor.
"""

from __future__ import annotations


def system_pressure_level() -> int:
    """macOS's own memory pressure: kern.memorystatus_vm_pressure_level,
    1 normal, 2 warn, 4 critical. 1 where it cannot be read."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"))
        v = ctypes.c_int(0)
        n = ctypes.c_size_t(ctypes.sizeof(v))
        if lib.sysctlbyname(b"kern.memorystatus_vm_pressure_level",
                            ctypes.byref(v), ctypes.byref(n), None, 0):
            return 1
        return int(v.value) or 1
    except (OSError, AttributeError, ValueError, TypeError):
        return 1


def own_compressed_bytes() -> int:
    """Bytes of this process the macOS compressor holds (compressed in RAM
    or swapped out with its segment): task_info(TASK_VM_INFO).compressed.
    0 where it cannot be read."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System"))
        buf = (ctypes.c_uint64 * 64)()
        cnt = ctypes.c_uint32(ctypes.sizeof(buf) // 4)
        task = ctypes.c_uint32.in_dll(lib, "mach_task_self_")
        if lib.task_info(task, 22, ctypes.byref(buf), ctypes.byref(cnt)):
            return 0
        # task_vm_info: virtual_size, region_count+page_size, resident,
        # resident_peak, device, device_peak, internal, internal_peak,
        # external, external_peak, reusable, reusable_peak, purgeable x3,
        # compressed (u64 index 15)
        return int(buf[15])
    except (OSError, AttributeError, ValueError, TypeError):
        return 0
