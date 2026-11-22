"""Windows physical-core discovery and reversible affinity for this process only."""
from __future__ import annotations

import ctypes
import os
import struct


def _kernel():
    if os.name != "nt":
        raise RuntimeError("registration CPU partition requires Windows; disable registration_affinity on this platform")
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.GetCurrentProcess.restype = ctypes.c_void_p
    lib.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    lib.GetProcessAffinityMask.restype = ctypes.c_int
    lib.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.SetProcessAffinityMask.restype = ctypes.c_int
    lib.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    lib.GetLogicalProcessorInformationEx.restype = ctypes.c_int
    return lib


def current_cpus():
    kernel = _kernel()
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return [bit for bit in range(ctypes.sizeof(process)*8) if process.value & (1 << bit)]


def set_cpus(cpus):
    if not cpus or any(type(cpu) is not int or cpu < 0 or cpu >= ctypes.sizeof(ctypes.c_size_t)*8 for cpu in cpus):
        raise ValueError("CPU affinity must contain valid logical processor IDs")
    kernel = _kernel()
    mask = sum(1 << cpu for cpu in set(cpus))
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
    if current_cpus() != sorted(set(cpus)):
        raise RuntimeError("CPU affinity verification failed")


def parse_cores(raw, pointer_size):
    """Decode GLPIEX RelationProcessorCore, including all SMT siblings."""
    if pointer_size not in (4, 8):
        raise ValueError("unsupported native pointer size")
    cores, offset = [], 0
    while offset < len(raw):
        if len(raw)-offset < 8:
            raise ValueError("truncated processor topology header")
        relation, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset+size > len(raw):
            raise ValueError("invalid processor topology record length")
        if relation == 0:
            if size < 32:
                raise ValueError("truncated processor relationship")
            efficiency = raw[offset+9]
            groups = struct.unpack_from("<H", raw, offset+30)[0]
            stride = pointer_size+8
            if groups != 1 or size < 32+groups*stride:
                raise RuntimeError("CPU partition currently requires a single processor group")
            mask = int.from_bytes(raw[offset+32:offset+32+pointer_size], "little")
            group = struct.unpack_from("<H", raw, offset+32+pointer_size)[0]
            if group != 0:
                raise RuntimeError("CPU partition currently supports processor group 0 only")
            cpus = [bit for bit in range(pointer_size*8) if mask & (1 << bit)]
            if not cpus:
                raise ValueError("empty physical-core affinity mask")
            cores.append(dict(logical_cpus=cpus, efficiency_class=efficiency))
        offset += size
    if not cores or len({cpu for core in cores for cpu in core["logical_cpus"]}) != sum(len(c["logical_cpus"]) for c in cores):
        raise ValueError("empty or overlapping physical-core topology")
    return cores


def physical_cores():
    kernel = _kernel()
    length = ctypes.c_ulong()
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
    if ctypes.get_last_error() != 122 or not length.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(length.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    return parse_cores(buffer.raw[:length.value], ctypes.sizeof(ctypes.c_size_t))


def choose_partition(cores, allowed, count, *, reserved_cpus=()):
    allowed = set(allowed)
    available = [dict(core, logical_cpus=sorted(set(core["logical_cpus"]) & allowed)) for core in cores]
    available = [core for core in available if core["logical_cpus"]]
    available.sort(key=lambda core: (-core["efficiency_class"], core["logical_cpus"][0]))
    reserved = set(reserved_cpus)
    if not reserved.issubset(allowed):
        raise ValueError("reserved main-process CPUs must belong to the current allowed CPU set")
    selectable = [core for core in available if not reserved.intersection(core["logical_cpus"])]
    if type(count) is not int or count < 1 or count >= len(available) or count > len(selectable):
        raise ValueError(f"registration_affinity.physical_cores must reserve at least one main-process physical core; available={len(available)}")
    selected = selectable[:count]
    registration = {cpu for core in selected for cpu in core["logical_cpus"]}
    return dict(original_cpus=sorted(allowed), main_cpus=sorted(allowed-registration),
                registration_cores=selected, registration_cpus=sorted(registration),
                reserved_main_cpus=sorted(reserved))
