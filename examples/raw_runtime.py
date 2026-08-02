#!/usr/bin/env python3
"""Example: Raw NPU execution via Python runtime (no JAX, no plugin).

Demonstrates the pure-Python runtime that talks directly to the NPU
via raw DRM ioctls.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opennpu.runtime import NPURuntime, NPUBuffer, submit

# Open the NPU device
rt = NPURuntime()
print(f"NPU opened: fd={rt.fd}")

# Allocate an input buffer (fp16, [1, 64, 768])
buf = rt.allocate_buffer(98304)  # 64 × 768 × 2 bytes
x = np.random.randn(1, 64, 768).astype(np.float16)
buf.write(x.tobytes())
buf.sync_to_device()
print(f"Buffer allocated: dma=0x{buf.dma_addr:x}, size={buf.size}")

# The full execution path (codegen + submit) is in the PJRT C plugin.
# This example shows the low-level buffer API.
# For actual op execution, use:
#   - JAX path: examples/jax_add.py
#   - C path: examples/gpt2_generate.py
#   - ONNX path: examples/onnx_inference.py

buf.close()
rt.close()
print("Done")