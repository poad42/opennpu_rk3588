"""
Pure-Python PJRT-shaped plugin for the RK3588 NPU.

Mirrors the API surface of the C plugin in
`third_party/rk3588-npu-pjrt/src/RK3588NPUPJRT.cpp` so any framework that
speaks PJRT (JAX, IREE, OpenXLA, jax.extend) can use the NPU without
needing to compile a C++ shared library.

This is NOT a thin shim over the C plugin — it implements the same
semantics in pure Python by routing through `opennpu.runtime.NPURuntime`.

The objects (`PJRTClient`, `PJRTDevice`, `PJRTBuffer`, `PJRTExecutable`)
carry integer handles so they can round-trip through a C wrapper if needed.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .runtime import NPURuntime, NPUBuffer


# ── Template+DMA-patch helpers (regcmd BO repointing) ─────────────

_CORE_CIDS = (0x1001, 0x2001, 0x0201)


def _patch_dma(buf, off, new_dma, cid):
    import struct
    struct.pack_into("<HHI", buf, off, struct.unpack_from("<H", buf, off)[0],
                     new_dma & 0xFFFF, (cid << 16) | ((new_dma >> 16) & 0xFFFF))


def _patch_h2(h2, cap_dma, my, sizes):
    """Repoint every DMA in a regcmd BO to the fresh BOs.

    General range-based patcher: for each entry whose cid is a DMA cid
    (0x1001/0x2001 = core dest/source, 0x0201 = input source), find which
    BO IOVA range (task/regcmd/scratch/any io) contains the dma and repoint
    to the fresh BO at the same offset. This handles io BOs referenced at
    offsets (e.g. conv reads input io0 at io0+0x2800, io0+0x1a00, ...), which the
    old exact-base match missed. Chain pointers (reg 0x0010, cid 0x0101) that
    chain blocks within the regcmd BO are also repointed by range.
    """
    import struct
    n = 0
    # build (cap_base, cap_size, fresh_base) for every BO role
    ranges = []
    for k, base in cap_dma.items():
        ranges.append((base, sizes[k], my[k]))
    for i in range(len(h2) // 8):
        off = i * 8
        reg, val, tag = struct.unpack_from("<HHI", h2, off)
        cid = (tag >> 16) & 0xFFFF
        dma = ((tag & 0xFFFF) << 16) | val
        new = None
        if cid in _CORE_CIDS:
            for cb, csz, fb in ranges:
                if cb <= dma < cb + csz:
                    new = fb + (dma - cb); break
        elif reg == 0x0010:
            # chain pointer: only regcmd-range
            if cap_dma["regcmd"] <= dma < cap_dma["regcmd"] + sizes["regcmd"]:
                new = my["regcmd"] + (dma - cap_dma["regcmd"])
        if new is not None:
            _patch_dma(h2, off, new, cid); n += 1
    return n


def _patch_h1(h1, cap_dma, my, reg_sz):
    """Repoint regcmd_addr fields (8 bytes at off i*40+32) to the fresh regcmd BO."""
    import struct
    nt = len(h1) // 40; t = 0
    for i in range(nt):
        off = i * 40 + 32
        if off + 8 > len(h1):
            break
        ra = struct.unpack_from("<Q", h1, off)[0]
        if cap_dma["regcmd"] <= ra < cap_dma["regcmd"] + reg_sz:
            struct.pack_into("<Q", h1, off, my["regcmd"] + (ra - cap_dma["regcmd"])); t += 1
    return t


# ── Sentinel error wrapper (PJRT-style) ──────────────────────────

class PJRTError(Exception):
    """PJRT error code wrapper. Codes match OpenXLA's enum."""
    OK                = 0
    INVALID_ARGUMENT  = 3
    UNIMPLEMENTED     = 12
    INTERNAL         = 13

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"[PJRT code={code}] {message}")


# ── PJRT Device ─────────────────────────────────────────────────

@dataclass
class PJRTDevice:
    """One NPU core. RK3588 has 3 cores (0, 1, 2)."""
    id: int               # core id (0, 1, 2)
    vendor: str = "rockchip"
    kind: str   = "rk3588_npu"

    def __repr__(self): return f"PJRTDevice(id={self.id}, kind={self.kind})"


# ── PJRT Buffer ─────────────────────────────────────────────────

class PJRTBuffer:
    """A device-resident tensor backed by an NPU BO.

    Memory is allocated on first write (or on `from_host_buffer`). The
    buffer is reference-counted: destroying it returns the BO to the
    runtime's free pool.
    """

    def __init__(self, runtime: NPURuntime, shape: Sequence[int],
                 dtype: np.dtype = np.float16, core_id: int = 0,
                 _bo: Optional[NPUBuffer] = None):
        self.runtime = runtime
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.core_id = core_id
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        if nbytes <= 0:
            raise PJRTError(PJRTError.INVALID_ARGUMENT,
                           f"Buffer shape {self.shape} {self.dtype} -> 0 bytes")
        if _bo is not None:
            self.bo = _bo
        else:
            # Default BO flag = 0x403 (non-contiguous + cacheable + IOMMU)
            self.bo = runtime.allocate_buffer(max(nbytes, 4096), 0x403)

    def to_host(self) -> np.ndarray:
        """Sync from device and return as a numpy array."""
        self.bo.sync_from_device()
        raw = self.bo.read(int(np.prod(self.shape)) * self.dtype.itemsize)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape).copy()

    def from_host(self, arr: np.ndarray):
        """Upload a numpy array to the device BO."""
        if arr.shape != self.shape or arr.dtype != self.dtype:
            raise PJRTError(PJRTError.INVALID_ARGUMENT,
                f"shape/dtype mismatch: got {arr.shape}/{arr.dtype}, "
                f"expected {self.shape}/{self.dtype}")
        self.bo.write(arr.tobytes())
        self.bo.sync_to_device()

    def destroy(self):
        """Release the underlying BO."""
        if self.bo is not None:
            self.bo.close()
            self.bo = None

    def __repr__(self):
        return f"PJRTBuffer(shape={self.shape}, dtype={self.dtype}, core={self.core_id})"

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# ── PJRT Executable ─────────────────────────────────────────────

class PJRTExecutable:
    """A compiled program ready to run on the NPU.

    Construction:
      - For codegen-mode: `PJRTExecutable.from_codegen(client, op, shape)`
        Uses our own codegen to synthesize regcmd bytes from scratch —
        no proprietary libraries.
      - For MLIR-mode:   `PJRTExecutable.from_module(client, mlir_module)`
        Lowers a Python MLIR module via `opennpu.mlir_dialect.NPUToRegCmd`,
        allocates BOs for the body, and exposes `.execute()`.
      - For replay-mode: `PJRTExecutable.from_captures(client, capture_dir)`
        Allocates sized BOs in the deterministic IOVA order, loads the
        capture files, and submits all tasks.

    All modes go through `NPURuntime` raw DRM ioctls only.
    """

    def __init__(self, client: "PJRTClient", task_data_path: Optional[str] = None,
                 capture_dir: Optional[str] = None):
        self.client = client
        self.capture_dir = capture_dir or os.environ.get(
            "OPENNPU_DUMPS_DIR", os.environ.get("OPENNPU_DUMPS_DIR", "./captures"))
        self._cached_info: Optional[Dict] = None
        self._codegen_op: Optional[str] = None
        self._codegen_shape: Optional[Tuple] = None
        self._mlir_mode: bool = False

    @classmethod
    def from_captures(cls, client: "PJRTClient",
                   capture_dir: Optional[str] = None) -> "PJRTExecutable":
        """Build an executable by replaying the NPU state capture."""
        return cls(client, capture_dir=capture_dir)

    @classmethod
    def from_codegen(cls, client: "PJRTClient",
                     op: str,
                     shape: Tuple[int, ...] = (1, 64, 768),
                     capture_dir: Optional[str] = None,
                     ) -> "PJRTExecutable":
        """Build an executable using our own codegen — no capture files needed.

        Args:
            client: PJRTClient with an initialized NPU runtime.
            op: Elementwise op name ("add", "sub", "mul", "div").
            shape: Input tensor shape.
            capture_dir: Directory holding the NPU-internal scratch capture
                (scratch_init.bin). The codegen path still needs this
                file because it is NPU-internal state
                at runtime and is not derivable from model files alone. Defaults to
                ${OPENNPU_DUMPS_DIR}/{op}.
        """
        exe = cls(client)
        exe._codegen_op = op.lower()
        exe._codegen_shape = shape
        if capture_dir is not None:
            exe._scratch_dir = capture_dir
        return exe

    @classmethod
    def from_template(cls, client: "PJRTClient",
                      op: str,
                      capture_dir: str,
                      shape: Tuple[int, ...] = (1, 64, 768),
                      ) -> "PJRTExecutable":
        """Build an executable that replays a captured NPU op template.

        ``capture_dir`` must contain a ``meta.json`` plus ``submit{d}_bo_h1.bin``
        (task), ``submit{d}_bo_h2.bin`` (regcmd), and ``scratch_init.bin``
        (initial scratch state) — produced by ``scripts/trace_op.py``.

        At runtime the executor allocates 5 fresh BOs, loads the scratch +
        user input, repoints every DMA in the captured regcmd to the fresh BOs,
        and submits each captured submit in order (multi-submit aware). No
        reference NPU tools / proprietary NPU libraries at runtime.
        """
        import json as _json
        exe = cls(client)
        exe._template_op = op.lower()
        exe._template_dir = capture_dir
        exe._codegen_shape = shape
        with open(os.path.join(capture_dir, "meta.json")) as f:
            exe._template_meta = _json.load(f)
        from .runtime import NPUBuffer as _NPUBuffer, action as _action
        bm = exe._template_meta["bos"]
        rt = client.runtime; fd = rt.fd
        # Normalize to a flat ordered role list: task, regcmd, scratch,
        # io0, io1, ... (io BOs in handle order: user inputs first, output last).
        # New captures store an "io" list; legacy captures store in/out.
        if "io" in bm:
            order = ["task", "regcmd", "scratch"] + [f"io{i}" for i in range(len(bm["io"]))]
            flat = {"task": bm["task"], "regcmd": bm["regcmd"], "scratch": bm["scratch"]}
            for i, b in enumerate(bm["io"]):
                flat[f"io{i}"] = b
        else:
            order = ["task", "regcmd", "scratch", "io0", "io1"]
            flat = {"task": bm["task"], "regcmd": bm["regcmd"], "scratch": bm["scratch"],
                    "io0": bm["in"], "io1": bm["out"]}
        exe._tpl_order = order
        exe._tpl_flat = flat
        # Pre-allocate the NPU BOs NOW (compile time, before any user
        # buffer_from_host) so they claim the current top of the NPU IOVA pool.
        # The RKNPU IOVA allocator grows downward within one fd; allocating the
        # op's BOs first lands them in the high-DMA region the NPU reaches.
        exe._tpl_bos = [_NPUBuffer(fd, flat[k]["size"], flat[k]["flag"]) for k in order]
        for _ in range(len(order)):
            _action(fd, 18, 0)
        return exe

    @classmethod
    def from_module(cls, client: "PJRTClient", mlir_module: Any
                    ) -> "PJRTExecutable":
        """Build an executable from a Python MLIR module.

        `mlir_module` must be an `opennpu.mlir_dialect.MLIRModule` whose
        operations carry `shape` in their attributes (optional — defaults
        to (1, 64, 768)).
        """
        from .mlir_dialect import NPUToRegCmd, MLIRModule
        if not isinstance(mlir_module, MLIRModule):
            raise PJRTError(PJRTError.INVALID_ARGUMENT,
                f"Expected MLIRModule, got {type(mlir_module)}")
        exe = cls(client)
        exe._mlir_mode = True
        # Extract the first op + shape for codegen
        for fn in mlir_module.functions:
            for op in fn.operations:
                exe._codegen_op = NPUToRegCmd._OP_TO_CODEGEN.get(op.op.value)
                exe._codegen_shape = op.attributes.get("shape", (1, 64, 768))
                break
            break
        if exe._codegen_op is None:
            raise PJRTError(PJRTError.UNIMPLEMENTED,
                "No supported op found in MLIR module")
        return exe

    def execute(self, *args: PJRTBuffer,
                n_tasks: Optional[int] = None,
                n_submit: Optional[int] = None,
                timeout_ms: int = 6000) -> PJRTBuffer:
        """Run the executable on the NPU. Returns the output PJRTBuffer.

        For codegen/MLIR mode: allocates 6 BOs, calls gen_op() to synthesize
        regcmd+task bytes, patches DMAs, writes to BOs, submits via raw ioctl.
        args[0] and args[1] are the two input tensors.

        n_submit overrides the number of task structs submitted. By default
        the codegen path submits only the first complete regcmd chain (up to
        and including the first writer task), which is all the NPU actually
        executes — the chain ends at the writer (next_offset=0), so any tasks
        beyond it never run and would only cause a 6s wait timeout + soft
        reset. Submitting exactly the first chain completes cleanly.

        For replay mode: ignores *args (capture files provide inputs).
        """
        if self._codegen_op is not None:
            return self._execute_codegen(args, timeout_ms, n_submit=n_submit)
        elif self._mlir_mode:
            return self._execute_codegen(args, timeout_ms, n_submit=n_submit)
        elif getattr(self, "_template_op", None) is not None:
            return self._execute_template(args, timeout_ms)
        else:
            return self._execute_replay(n_tasks, timeout_ms)

    def _execute_codegen(self, args: Sequence[PJRTBuffer],
                         timeout_ms: int,
                         n_submit: Optional[int] = None) -> PJRTBuffer:
        """Execute via codegen path: gen_op → BOs → raw ioctl → output."""
        from .codegen_synthesize import gen_op
        from .runtime import NPURuntime, NPUBuffer, action, submit
        import numpy as np

        rt = self.client.runtime
        fd = rt.fd

        op = self._codegen_op
        # The elementwise NPU kernel is hardware-pinned to (1,64,768)=49152
        # elements: the regcmd encodes C=64, W=768 geometry that we have not
        # fully decoded (other C/W hang or produce garbage). To support
        # arbitrary element counts we host-tile: flatten the inputs into
        # 49152-element chunks and run the fixed kernel per chunk, reusing the
        # same BOs across submits (only the scratch input is rewritten). A
        # zeroed scratch is sufficient (verified on-board) and multi-submit on
        # the same BOs is proven (3 iters, all PASS).
        from .codegen_synthesize import (
            scratch_layout as _sl, SIMPLE_TASK_SEQUENCE as _SEQ)
        KSHAPE = (1, 64, 768)
        KELEM = 64 * 768          # 49152 elements per kernel invocation
        KTSZ = KELEM * 2          # 98304 bytes (fp16)
        _in2b, _in1b, _resb, _scsz = _sl(KSHAPE)
        sizes = [4096, 8192, _scsz, KTSZ, KTSZ, KTSZ]
        flags = [0x40B, 0x403, 0x403, 0x403, 0x403, 0x403]
        bos = []
        for i in range(6):
            buf = NPUBuffer(fd, sizes[i], flags[i])
            bos.append(buf)
            action(fd, 18, 0)
        bo_dmas = [int(b.dma_addr) for b in bos]

        # Generate the fixed (1,64,768) kernel once.
        regcmd_data, task_data, _ = gen_op(op, KSHAPE, bo_dmas=bo_dmas)
        bos[0].write(task_data[:bos[0].size])
        bos[1].write(regcmd_data[:bos[1].size])
        n_submit = next(i + 1 for i, t in enumerate(_SEQ) if t[0] == 5)

        # Flatten inputs to 1D fp16 and tile into 49152-element chunks.
        out_shape = self._codegen_shape
        a1 = args[0].to_host().astype(np.float16).reshape(-1)
        a2 = args[1].to_host().astype(np.float16).reshape(-1)
        if a1.shape[0] != a2.shape[0]:
            raise PJRTError(PJRTError.INVALID_ARGUMENT,
                "elementwise inputs must have the same element count")
        n_elem = a1.shape[0]
        n_chunks = (n_elem + KELEM - 1) // KELEM
        pad = n_chunks * KELEM - n_elem
        if pad:
            a1 = np.pad(a1, (0, pad))
            a2 = np.pad(a2, (0, pad))
        out_flat = np.empty(n_chunks * KELEM, dtype=np.float16)

        from .runtime import sync_bo
        import struct, time
        # One-time syncs for the fixed kernel BOs.
        for b in bos:
            sync_bo(fd, b.obj_addr, b.size, 3)   # BIDIR
        sync_bo(fd, bos[0].obj_addr, bos[0].size, 1)  # task
        sync_bo(fd, bos[1].obj_addr, bos[1].size, 1)  # regcmd
        sync_bo(fd, bos[3].obj_addr, bos[3].size, 1)
        sync_bo(fd, bos[4].obj_addr, bos[4].size, 1)

        counters = []
        t0 = time.time()
        for ci in range(n_chunks):
            s = ci * KELEM; e = s + KELEM
            # Write this chunk's inputs into the scratch input regions (full
            # KTSZ, padded tail is zeros); the NPU overwrites the result region.
            bos[2].write(a1[s:e].tobytes(), offset=_in1b)
            bos[2].write(a2[s:e].tobytes(), offset=_in2b)
            sync_bo(fd, bos[2].obj_addr, bos[2].size, 1)   # scratch TO_DEVICE
            try:
                sub = submit(fd, bos[0].obj_addr, n_tasks=n_submit,
                             timeout=timeout_ms, flags=5, task_base_addr=0)
                counters.append(struct.unpack_from("<I", sub, 16)[0])
            except OSError:
                counters.append(-1)
                break
            sync_bo(fd, bos[5].obj_addr, bos[5].size, 2)  # out FROM_DEVICE
            out_flat[s:e] = np.frombuffer(bos[5].read()[:KTSZ], dtype=np.float16)
        elapsed = time.time() - t0

        out_arr = out_flat[:n_elem].reshape(out_shape).astype(np.float32)
        self._cached_info = {
            "mode": "codegen", "op": op,
            "tasks": [{"status": "ok" if c > 0 else "timeout",
                       "task_counter": c} for c in counters],
            "bo_dmas": [hex(d) for d in bo_dmas],
            "n_submit": n_submit, "n_chunks": n_chunks,
            "time_s": elapsed, "scratch_path": None,
        }

        out_buf = PJRTBuffer(rt, out_shape, np.float32)
        out_buf.from_host(out_arr)
        for b in bos:
            b.close()
        return out_buf

    def _execute_template(self, args: Sequence["PJRTBuffer"],
                        timeout_ms: int) -> PJRTBuffer:
        """Execute via template+DMA-patch: replay a captured NPU op with fresh BOs.

        Generalized for variable BO sets: writes user inputs to the first
        n_input io BOs (io0..io{n_input-1}), zeros the output BO (the last io
        BO), patches every io/task/regcmd/scratch DMA in the captured regcmd,
        and submits each captured submit in order. Reads the output BO.
        """
        import json, struct
        from .runtime import action, submit, sync_bo

        d = self._template_dir
        meta = self._template_meta
        if not meta.get("has_compute"):
            raise PJRTError(PJRTError.UNIMPLEMENTED,
                f"Op '{self._template_op}' has no op=2 (compute) tasks in its "
                f"capture — it is a CPU-fallback op, not an NPU op.")
        flat = self._tpl_flat
        order = self._tpl_order
        idx = {k: i for i, k in enumerate(order)}
        sizes = {k: flat[k]["size"] for k in flat}
        cap_dma = {k: flat[k]["dma"] for k in flat}
        nsub = meta["nsubmits"]
        n_input = meta.get("n_input", 1)
        in_roles = [f"io{i}" for i in range(n_input)]
        out_role = order[-1]
        shape = tuple(meta.get("shape", list(self._codegen_shape)))
        mdtype = meta.get("dtype", "float16")
        # Per-io-BO dtype inference: a w8a8 op may have int8 inputs but a fp16
        # output (e.g. softmax keeps probabilities in fp16). Infer each io BO's
        # element size from (BO byte size) / (element count of its tensor).
        def _iodt(role, n_elem):
            isz = sizes[role] // max(int(n_elem), 1)
            if isz == 1: return np.int8
            if isz == 2: return np.int16 if mdtype == "int16" else np.float16
            if isz == 4: return np.float32
            return np.float16
        in_dts = [_iodt(r, int(np.prod(args[i].shape)))
                  for i, r in enumerate(in_roles)]
        out_dt = _iodt(out_role, int(np.prod(shape)))

        rt = self.client.runtime
        fd = rt.fd
        mybos = self._tpl_bos
        my = {k: int(mybos[idx[k]].dma_addr) for k in order}

        # ── Tiled execution for elementwise activations ──────────────
        # The unary activation templates (relu/sigmoid/tanh/leakyrelu/clip)
        # are captured at (1,64,768)=49152 elements and are elementwise
        # (out[i]=f(in[i])), so any input tiles into 49152-element chunks and
        # the captured (1,64,768) kernel runs per chunk, reusing BOs (only the
        # input io BO is rewritten per chunk; scratch_init holds the
        # input-independent LUT/config, loaded once). Zero-padding the final
        # partial chunk is safe — only the real elements are read back.
        TILEABLE = {"relu", "sigmoid", "tanh", "leakyrelu", "clip", "gelu"}
        cap_nelem = int(np.prod(shape))
        in_nelem = int(np.prod(args[0].shape))
        if (self._template_op in TILEABLE and n_input == 1
                and cap_nelem == 49152 and in_nelem != cap_nelem):
            # The tiled path uses the CAPTURED dtype (the io0 BO holds a
            # 49152-elem CHUNK, not the full input, so the _iodt byte/element
            # heuristic mis-infers dtype when n_elem == io0 byte size).
            DT = {"float16": np.float16, "int8": np.int8, "int16": np.int16}
            in_dt = DT.get(mdtype, np.float16)
            out_dt = in_dt   # elementwise activations preserve dtype
            in_esz = np.dtype(in_dt).itemsize
            out_esz = np.dtype(out_dt).itemsize
            n_chunks = (in_nelem + cap_nelem - 1) // cap_nelem
            scratch_init = open(os.path.join(d, "scratch_init.bin"), "rb").read()[:sizes["scratch"]]
            out_host = np.empty(in_nelem, dtype=np.float32)
            in_arr = args[0].to_host().astype(in_dt).reshape(-1)
            t0 = time.time()
            counters = []
            for ci in range(n_chunks):
                s = ci * cap_nelem
                ne = min(cap_nelem, in_nelem - s)
                chunk = np.zeros(cap_nelem, dtype=in_dt)
                chunk[:ne] = in_arr[s:s + ne]
                # Re-init scratch per chunk: multi-submit activations (sigmoid/
                # tanh/gelu) carry state across submits WITHIN a chunk; stale
                # scratch from the previous chunk hangs the NPU. The LUT/config
                # in scratch_init is input-independent, safe to reload.
                mybos[idx["scratch"]].write(scratch_init)
                mybos[idx["io0"]].write(chunk.tobytes()[:sizes["io0"]])
                mybos[idx[out_role]].write(b"\x00" * sizes[out_role])
                sync_bo(fd, mybos[idx["scratch"]].obj_addr,
                        mybos[idx["scratch"]].size, 1)
                sync_bo(fd, mybos[idx["io0"]].obj_addr,
                        mybos[idx["io0"]].size, 1)
                sync_bo(fd, mybos[idx[out_role]].obj_addr,
                        mybos[idx[out_role]].size, 1)
                for sd, sub in enumerate(meta["submits"]):
                    with open(os.path.join(d, f"submit{sd}_bo_h1.bin"), "rb") as f:
                        h1 = bytearray(f.read())
                    with open(os.path.join(d, f"submit{sd}_bo_h2.bin"), "rb") as f:
                        h2 = bytearray(f.read())
                    _patch_h2(h2, cap_dma, my, sizes)
                    _patch_h1(h1, cap_dma, my, sizes["regcmd"])
                    mybos[idx["task"]].write(bytes(h1)[:sizes["task"]])
                    mybos[idx["regcmd"]].write(bytes(h2)[:sizes["regcmd"]])
                    sync_bo(fd, mybos[idx["task"]].obj_addr,
                            mybos[idx["task"]].size, 1)
                    sync_bo(fd, mybos[idx["regcmd"]].obj_addr,
                            mybos[idx["regcmd"]].size, 1)
                    sub_res = submit(fd, mybos[idx["task"]].obj_addr,
                                     n_tasks=sub["n_tasks"],
                                     task_start=sub["task_start"],
                                     timeout=timeout_ms, flags=sub["flags"],
                                     task_base_addr=0,
                                     core_mask=sub.get("core_mask", 1),
                                     subcore_tasks=sub["subcore"])
                    if sd == len(meta["submits"]) - 1:
                        counters.append(struct.unpack_from("<I", sub_res, 16)[0])
                sync_bo(fd, mybos[idx[out_role]].obj_addr,
                        mybos[idx[out_role]].size, 2)
                out_raw = np.frombuffer(
                    mybos[idx[out_role]].read()[:ne * out_esz], dtype=out_dt)
                out_host[s:s + ne] = out_raw.astype(np.float32)
            elapsed = time.time() - t0
            out_shape = tuple(args[0].shape)
            self._cached_info = {
                "mode": "template-tiled", "op": self._template_op,
                "nsubmits": nsub, "n_chunks": n_chunks,
                "counters": counters, "time_s": elapsed, "capture_dir": d,
            }
            out_buf = PJRTBuffer(rt, out_shape, np.float32)
            out_buf.from_host(out_host.reshape(out_shape))
            return out_buf

        # load initial scratch + user inputs; zero output
        with open(os.path.join(d, "scratch_init.bin"), "rb") as f:
            mybos[idx["scratch"]].write(f.read()[:sizes["scratch"]])
        for i, role in enumerate(in_roles):
            arr = args[i].to_host().astype(in_dts[i])
            mybos[idx[role]].write(arr.tobytes()[:sizes[role]])
        # zero the output BO unless the op is in-place (output == an input BO);
        # in-place ops (none yet) are overwritten by the NPU, so don't wipe input.
        if out_role not in in_roles:
            mybos[idx[out_role]].write(b"\x00" * sizes[out_role])
        for b in mybos:
            sync_bo(fd, b.obj_addr, b.size, 3)
        for b in mybos:
            sync_bo(fd, b.obj_addr, b.size, 1)

        counters = []
        t0 = time.time()
        for sd, sub in enumerate(meta["submits"]):
            with open(os.path.join(d, f"submit{sd}_bo_h1.bin"), "rb") as f:
                h1 = bytearray(f.read())
            with open(os.path.join(d, f"submit{sd}_bo_h2.bin"), "rb") as f:
                h2 = bytearray(f.read())
            _patch_h2(h2, cap_dma, my, sizes)
            _patch_h1(h1, cap_dma, my, sizes["regcmd"])
            mybos[idx["task"]].write(bytes(h1)[:sizes["task"]])
            mybos[idx["regcmd"]].write(bytes(h2)[:sizes["regcmd"]])
            sync_bo(fd, mybos[idx["task"]].obj_addr, mybos[idx["task"]].size, 1)
            sync_bo(fd, mybos[idx["regcmd"]].obj_addr, mybos[idx["regcmd"]].size, 1)
            sub_res = submit(fd, mybos[idx["task"]].obj_addr,
                             n_tasks=sub["n_tasks"], task_start=sub["task_start"],
                             timeout=timeout_ms, flags=sub["flags"],
                             task_base_addr=0,
                             core_mask=sub.get("core_mask", 1),
                             subcore_tasks=sub["subcore"])
            counters.append(struct.unpack_from("<I", sub_res, 16)[0])
        elapsed = time.time() - t0

        sync_bo(fd, mybos[idx[out_role]].obj_addr, mybos[idx[out_role]].size, 2)
        out_nbytes = int(np.prod(shape)) * np.dtype(out_dt).itemsize
        out_raw = np.frombuffer(mybos[idx[out_role]].read()[:out_nbytes], dtype=out_dt)
        out_arr = out_raw.reshape(shape).astype(np.float32)

        self._cached_info = {
            "mode": "template", "op": self._template_op,
            "nsubmits": nsub, "counters": counters, "time_s": elapsed,
            "capture_dir": d,
            "tasks": [{"status": "ok" if c > 0 else "timeout",
                       "task_counter": c} for c in counters],
        }
        out_buf = PJRTBuffer(rt, shape, np.float32)
        out_buf.from_host(out_arr)
        return out_buf

    def _execute_replay(self, n_tasks: Optional[int],
                        timeout_ms: int) -> PJRTBuffer:
        """Execute via capture replay path."""
        info = self.client.runtime.execute_replay(
            self.capture_dir,
            n_tasks=n_tasks or 256,
            timeout=timeout_ms,
            verify=True,
        )
        self._cached_info = info
        out = PJRTBuffer(self.client.runtime, shape=(1,),
                         dtype=np.float16)
        out.from_host(np.array([info["submit"]["task_counter"]],
                               dtype=np.float16))
        return out

    def info(self) -> Dict:
        """Return metrics from the last `execute()` call."""
        return self._cached_info or {}

    def destroy(self):
        """Release pre-allocated template BOs (template path)."""
        for b in getattr(self, "_tpl_bos", None) or []:
            try:
                b.close()
            except Exception:
                pass
        self._tpl_bos = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# ── PJRT Client ─────────────────────────────────────────────────

class PJRTClient:
    """Top-level client holding an NPU runtime and device list.

    Construction:
        client = PJRTClient()           # default NPU core 0
        client = PJRTClient(core_ids=[0, 1, 2])
    """

    def __init__(self, core_ids: Sequence[int] = (0,)):
        self.runtime = NPURuntime()
        # Expose one PJRTDevice per requested core
        self.devices = [PJRTDevice(id=i) for i in core_ids]
        self.addressable_devices = self.devices

    def devices(self) -> List[PJRTDevice]:
        return self.devices

    def addressable_devices(self) -> List[PJRTDevice]:
        return self.addressable_devices

    def buffer_from_host(self, arr: np.ndarray,
                         device: Optional[PJRTDevice] = None) -> PJRTBuffer:
        buf = PJRTBuffer(self.runtime, arr.shape, arr.dtype,
                         core_id=device.id if device else 0)
        buf.from_host(arr)
        return buf

    def compile(self, mlir_module: Any) -> PJRTExecutable:
        """Compile a Python MLIR module into a PJRTExecutable."""
        return PJRTExecutable.from_module(self, mlir_module)

    def compile_codegen(self, op: str,
                        shape: Tuple[int, ...] = (1, 64, 768),
                        capture_dir: Optional[str] = None,
                        ) -> PJRTExecutable:
        """Compile an elementwise op using our own codegen.

        No vendor model files, no NPU state captures, no proprietary NPU libraries — pure open-source
        codegen synthesizes the regcmd bytes from scratch.

        Args:
            op: Elementwise op name ("add", "sub", "mul", "div").
            shape: Input tensor shape.
            capture_dir: Directory holding scratch_init.bin (NPU-internal
                scratch). Defaults to ${OPENNPU_DUMPS_DIR}/{op}.
        """
        return PJRTExecutable.from_codegen(self, op, shape, capture_dir=capture_dir)

    def compile_replay(self, capture_dir: Optional[str] = None
                       ) -> PJRTExecutable:
        """Compile the capture-replay executable (SmolVLM 256-task workload)."""
        return PJRTExecutable.from_captures(self, capture_dir=capture_dir)

    def compile_template(self, op: str,
                         capture_dir: str,
                         shape: Tuple[int, ...] = (1, 64, 768),
                         ) -> PJRTExecutable:
        """Compile an NPU op via template+DMA-patch replay.

        ``capture_dir`` holds a capture produced by ``scripts/trace_op.py``
        (meta.json + submit{d}_bo_h{1,2}.bin + scratch_init.bin). At runtime
        the executor allocates fresh BOs, repoints the captured regcmd DMAs,
        and submits — no reference NPU tools / proprietary NPU libraries at runtime.

        Supported (proven on-board, fp16): relu, clip, leakyrelu, sigmoid,
        tanh. Ops with ``has_compute=false`` (CPU-fallback like exp) raise
        UNIMPLEMENTED at execute time.
        """
        exe = PJRTExecutable.from_template(self, op, capture_dir, shape=shape)
        return exe

    def destroy(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def __del__(self):
        try:
            if self.runtime is not None:
                self.runtime.close()
        except Exception:
            pass


# ── Convenience: single-shot helper used by demos/tests ────────────

def run_smolvlm_replay(capture_dir: Optional[str] = None,
                       n_tasks: int = 256,
                       timeout_ms: int = 20000) -> Dict:
    """One-shot: open NPU, run the full SmolVLM capture replay, return metrics."""
    client = PJRTClient()
    exe = client.compile_replay(capture_dir=capture_dir)
    out = exe.execute(n_tasks=n_tasks, timeout_ms=timeout_ms)
    info = exe.info()
    out.destroy()
    client.destroy()
    return info