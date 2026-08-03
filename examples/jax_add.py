#!/usr/bin/env python3
"""Example: Elementwise add on the NPU via JAX."""
import os
import sys

# Point JAX at our plugin
plugin_path = os.environ.get("NPU_PLUGIN_LIB", os.path.join(os.path.dirname(__file__), "..", "src", "opennpu", "pjrt_c", "libpjrt_npu.so"))
os.environ["JAX_PLATFORMS"] = "npu,cpu"
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"npu:{plugin_path}"

import jax
import jax.numpy as jnp

print(f"Devices: {jax.devices()}")

x = jnp.ones((1, 64, 768), dtype=jnp.float16)
y = jnp.ones((1, 64, 768), dtype=jnp.float16) * 2

@jax.jit
def add(a, b):
    return a + b

result = add(x, y)
print(f"Result shape: {result.shape}")
print(f"Result[0,0,:5]: {result[0,0,:5]}")  # should be 3.0