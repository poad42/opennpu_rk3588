#!/usr/bin/env python3
"""Example: Matmul on the NPU via JAX custom op (cached weights).

Requires pre-loaded weights in $NPU_WEIGHTS_DIR (w000.bin, w001.bin, ...).
See scripts/extract_gpt2_w.py for weight extraction.
"""
import os
import sys

plugin_path = os.environ.get("NPU_PLUGIN_LIB", os.path.join(os.path.dirname(__file__), "..", "src", "opennpu", "pjrt_c", "libpjrt_npu.so"))
os.environ["JAX_PLATFORMS"] = "npu,cpu"
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"npu:{plugin_path}"

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opennpu.jax_npu import npu_cached_mm

print(f"Devices: {jax.devices()}")

x = jnp.ones((1, 64, 768), dtype=jnp.float16)

@jax.jit
def matmul(x):
    return npu_cached_mm(x, w_idx=0)

result = matmul(x)
print(f"Result shape: {result.shape}")