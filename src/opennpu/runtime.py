"""
OpenNPU Runtime — raw DRM ioctl execution for RK3588 NPU.
Integrates with the MLIR dialect and the PJRT plugin.
"""
import os, struct, fcntl, mmap, json, time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# IOCTL constants (discovered via public DRM interface)────────
DRM_DEV = "/dev/dri/card1"
IOCTL = {
    "MEM_CREATE":  0xC0306442,
    "MEM_MAP":     0xC0106443,
    "MEM_SYNC":    0xC0206445,
    "MEM_DESTROY": 0xC0106444,  # IOWR nr=0x04, size=16
    "SUBMIT":      0xC0686441,
    "ACTION":      0xC0086440,
    "BO_REGISTER": 0xC008640a,
    "PRIME_FD":    0xC00C642D,
}
CREATE_FLAGS = 0x040b
TASK_SZ = 40
SUBMIT_SZ = 104
REGC_PER_TASK = 640  # 80 entries x 8 bytes

# Original BO DMA addresses from the SmolVLM capture capture.
# These match the fresh-reboot IOVA allocator output. On subsequent runs the
# allocator produces DIFFERENT addresses, so we patch these to actual BO DMAs
# before submitting. See _patch_replay_dmas() below.
_ORIG_DMAS  = [0xfffe1000, 0xffab3000, 0xf4802000, 0xf4682000, 0xf4502000]
_ORIG_SIZES = [126976,    5431296,    187371520,   1572864,    1572864]


# ── DRM version helpers (matching real runtime) ─────────────────

def _drm_version(fd: int):
    buf = bytearray(64)
    fcntl.ioctl(fd, 0xC0406400, buf, True)

def _drm_get_unique(fd: int):
    buf = bytearray(16)
    fcntl.ioctl(fd, 0xC0106401, buf, True)


# ── Low-level ioctl wrappers ─────────────────────────────────────

def gem_flink(fd: int, handle: int) -> int:
    buf = bytearray(8)
    struct.pack_into("<I", buf, 0, handle)
    struct.pack_into("<I", buf, 4, 0)
    fcntl.ioctl(fd, IOCTL["BO_REGISTER"], buf, True)
    return struct.unpack_from("<I", buf, 4)[0]

def prime_handle_to_fd(fd: int, handle: int) -> int:
    buf = bytearray(12)
    struct.pack_into("<i", buf, 0, handle)
    struct.pack_into("<i", buf, 4, 0)
    struct.pack_into("<I", buf, 8, 0)
    fcntl.ioctl(fd, IOCTL["PRIME_FD"], buf, True)
    return struct.unpack_from("<i", buf, 4)[0]

def open_npu() -> int:
    return os.open(DRM_DEV, os.O_RDWR)

def action(fd: int, flags: int, value: int = 0) -> Tuple[int, int]:
    buf = bytearray(8)
    struct.pack_into("<I", buf, 0, flags)
    struct.pack_into("<I", buf, 4, value)
    fcntl.ioctl(fd, IOCTL["ACTION"], buf, True)
    return (struct.unpack_from("<I", buf, 0)[0], struct.unpack_from("<I", buf, 4)[0])

def create_bo(fd: int, size: int, flags: int = CREATE_FLAGS) -> Tuple[int, int, int, int]:
    buf = bytearray(48)
    struct.pack_into("<I", buf, 0, 0)
    struct.pack_into("<I", buf, 4, flags)
    struct.pack_into("<Q", buf, 8, size)
    struct.pack_into("<Q", buf, 16, 0)
    struct.pack_into("<Q", buf, 24, 0)
    struct.pack_into("<Q", buf, 32, 0)
    struct.pack_into("<i", buf, 40, 0)
    struct.pack_into("<I", buf, 44, 1)
    fcntl.ioctl(fd, IOCTL["MEM_CREATE"], buf, True)
    return (struct.unpack_from("<I", buf, 0)[0],
            struct.unpack_from("<Q", buf, 8)[0],
            struct.unpack_from("<Q", buf, 16)[0],
            struct.unpack_from("<Q", buf, 24)[0])

def map_bo(fd: int, handle: int, size: int) -> Tuple[int, mmap.mmap]:
    buf = bytearray(16)
    struct.pack_into("<I", buf, 0, handle)
    struct.pack_into("<I", buf, 4, 0)
    struct.pack_into("<Q", buf, 8, 0)
    fcntl.ioctl(fd, IOCTL["MEM_MAP"], buf, True)
    off = struct.unpack_from("<Q", buf, 8)[0]
    return off, mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=off)

def register_bo(fd: int, handle: int) -> bool:
    """Register a BO with the NPU via 0xC008640a (hijacked SYNCOBJ_DESTROY)."""
    buf = bytearray(8)
    struct.pack_into("<I", buf, 0, handle)
    struct.pack_into("<I", buf, 4, 0)
    try:
        fcntl.ioctl(fd, IOCTL["BO_REGISTER"], buf, True)
        return True
    except OSError:
        return False

def sync_bo(fd: int, obj_addr: int, size: int, direction: int = 3):
    buf = bytearray(32)
    struct.pack_into("<I", buf, 0, direction)
    struct.pack_into("<I", buf, 4, 0)
    struct.pack_into("<Q", buf, 8, obj_addr)
    struct.pack_into("<Q", buf, 16, 0)
    struct.pack_into("<Q", buf, 24, size)
    fcntl.ioctl(fd, IOCTL["MEM_SYNC"], buf, True)

def submit(fd: int, task_obj_addr: int, n_tasks: int,
           task_start: int = 0, timeout: int = 6000, flags: int = 5,
           task_base_addr: int = 0, core_mask: int = 1,
           subcore_tasks: Optional[List[Tuple[int, int]]] = None) -> bytearray:
    """Submit tasks to NPU core(s). Returns the submit struct (kernel-updated).

    flags: RKNPU_JOB_PC=1 | RKNPU_JOB_NONBLOCK=2 | RKNPU_JOB_PINGPONG=4
           | RKNPU_JOB_FENCE_IN=8 | RKNPU_JOB_FENCE_OUT=0x10
    Default `flags=5` = PC | PINGPONG — standard NPU init sequence.

    core_mask: bitmask of cores (CORE0=1, CORE1=2, CORE2=4). Default 1=CORE0.
      The kernel dispatches one rknpu_job_subcore_commit per set bit and waits
      on the lowest-numbered core (see rknpu_wait_core_index). Multi-core runs
      use core_mask=7 and distribute tasks via subcore_tasks.

    `subcore_tasks` is a list of (start, num) pairs, one per subcore (max 5).
    If None, all tasks are routed to subcore 0 (standard NPU
    pattern for single-core workloads).
    """
    if subcore_tasks is None:
        # All tasks on subcore 0; subcores 1 and 2 idle
        subcore_tasks = [(task_start, n_tasks), (0, 0), (0, 0), (0, 0), (0, 0)]
    sub = bytearray(SUBMIT_SZ)
    struct.pack_into("<I", sub, 0, flags)
    struct.pack_into("<I", sub, 4, timeout)
    struct.pack_into("<I", sub, 8, task_start)
    struct.pack_into("<I", sub, 12, n_tasks)
    struct.pack_into("<I", sub, 16, n_tasks)  # task_counter = task_number
    struct.pack_into("<Q", sub, 24, task_obj_addr)
    struct.pack_into("<Q", sub, 40, task_base_addr)  # PC_DMA_BASE_ADDR (0 = default)
    struct.pack_into("<I", sub, 56, core_mask)   # core_mask
    struct.pack_into("<i", sub, 60, -1)       # fence_fd = -1 (no fence)
    for i, (start, num) in enumerate(subcore_tasks[:5]):
        struct.pack_into("<II", sub, 64 + i * 8, start, num)
    fcntl.ioctl(fd, IOCTL["SUBMIT"], sub, True)
    return sub


# ── Buffer ───────────────────────────────────────────────────────

class NPUBuffer:
    """NPU-accessible buffer (BO) with CPU mapping. Full creation sequence."""
    def __init__(self, fd: int, size: int, flags: int = CREATE_FLAGS):
        self.fd = fd
        self.handle, self.ret_size, self.obj_addr, self.dma_addr = create_bo(fd, size, flags)
        self.size = self.ret_size
        self._mm: Optional[mmap.mmap] = None
        gem_flink(fd, self.handle)
        self._prime_fd = prime_handle_to_fd(fd, self.handle)
        self._ensure_mapped()
        # BIDIR sync (direction=3) to establish IOMMU mapping — matches NPU hardware behavior
        sync_bo(fd, self.obj_addr, self.size, 3)
    
    def _ensure_mapped(self):
        if self._mm is not None: return
        _, self._mm = map_bo(self.fd, self.handle, self.size)
    
    def write(self, data: bytes, offset: int = 0):
        self._ensure_mapped()
        self._mm[offset:offset + len(data)] = data
    
    def read(self, size: int = -1, offset: int = 0) -> bytes:
        self._ensure_mapped()
        if size < 0: size = self.size - offset
        return bytes(self._mm[offset:offset + size])
    
    def to_host(self, dtype=np.float32, shape=None):
        self._ensure_mapped()
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize if shape else self.size
        return np.frombuffer(self._mm[:nbytes], dtype=dtype).reshape(shape) if shape else bytes(self._mm[:nbytes])
    
    def sync_to_device(self, size: int = 0):
        sync_bo(self.fd, self.obj_addr, size or self.size, 1)
    
    def sync_from_device(self, size: int = 0):
        sync_bo(self.fd, self.obj_addr, size or self.size, 2)
    
    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        # Close the prime dma-buf fd — but NEVER close the NPU device fd.
        # In some ssh/non-interactive contexts, stdin (fd 0) is already
        # closed before we start, so the NPU fd may be 0. The PRIME_FD
        # ioctl might also allocate fd 0 for a prime_fd if the NPU fd
        # was temporarily closed. Guard against closing the NPU fd.
        if getattr(self, "_prime_fd", None) is not None:
            npu_fd = getattr(self, "fd", None)
            if self._prime_fd != npu_fd:
                try:
                    os.close(self._prime_fd)
                except OSError:
                    pass
            self._prime_fd = None
        # Free the GEM handle via MEM_DESTROY so the IOMMU mapping is released.
        # struct rknpu_mem_destroy { u32 handle; u32 reserved; u64 obj_addr; } = 16 bytes
        if getattr(self, "handle", None) is not None and getattr(self, "fd", None) is not None:
            try:
                buf = bytearray(16)
                struct.pack_into("<I", buf, 0, self.handle)
                struct.pack_into("<I", buf, 4, 0)
                struct.pack_into("<Q", buf, 8, self.obj_addr)
                fcntl.ioctl(self.fd, IOCTL["MEM_DESTROY"], buf, True)
            except OSError:
                pass
            self.handle = None


# ── NPU Runtime (standalone, no proprietary libraries) ──────────────────────

class NPURuntime:
    """Raw DRM ioctl runtime for RK3588 NPU."""
    
    def __init__(self):
        self.fd = open_npu()
        self._init_npu()
    
    def _init_npu(self):
        """Full NPU init — matches real proprietary libraries init_runtime exactly."""
        action(self.fd, 0, 0xFFFFFFFF)   # GET_HW_VERSION
        action(self.fd, 1, 0)            # GET_DRV_VERSION
        action(self.fd, 19, 0xFFFFFFED)  # SET_PROC_NICE(-19)
        action(self.fd, 1, 0)            # GET_DRV_VERSION
        action(self.fd, 18, 0)           # GET_IOMMU_EN
    
    def allocate_buffer(self, size: int, flags: int = 0x0403) -> NPUBuffer:
        buf = NPUBuffer(self.fd, size, flags)
        action(self.fd, 18, 0)
        return buf

    def close(self):
        os.close(self.fd)
    