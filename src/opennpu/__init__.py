"""
OpenNPU — Open-source compiler + runtime for the RK3588 NPU.

Executes ONNX / JAX graphs on the Rockchip RK3588 NPU via raw DRM ioctls.
No proprietary libraries at runtime. No proprietary toolkits at runtime.

Components:
  - runtime:         Raw DRM ioctl runtime (NPURuntime, NPUBuffer)
  - codegen_synthesize: ONNX op + shape → regcmd byte synthesis
  - pjrt:            Pure-Python PJRT client (for ONNX/PyTorch runners)
  - jax_npu:         JAX custom op (npu_cached_mm)
  - onnx_runner:     ONNX → NPU/CPU dispatch
  - pjrt_c:          PJRT C plugin for JAX (libpjrt_npu.so)
"""

VERSION = "1.0.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import NPURuntime, NPUBuffer
    from .codegen_synthesize import gen_add, gen_elementwise, gen_op

def __getattr__(name):
    if name in {"NPURuntime", "NPUBuffer"}:
        from . import runtime
        return getattr(runtime, name)
    if name in {"gen_add", "gen_elementwise", "gen_op"}:
        from . import codegen_synthesize
        return getattr(codegen_synthesize, name)
    if name == "NPUDevice":
            return NPUDevice
    raise AttributeError(f"module 'opennpu' has no attribute {name!r}")

__all__ = ["VERSION", "NPURuntime", "NPUBuffer",
           "gen_add", "gen_elementwise", "gen_op"]