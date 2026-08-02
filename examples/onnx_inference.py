#!/usr/bin/env python3
"""Example: ONNX model inference on the NPU.

Runs an ONNX model with supported ops on the NPU, falling back to CPU
for unsupported ops.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opennpu.onnx_runner import ONNXRunner

# Create a simple ONNX model (add + relu)
# In practice, export from PyTorch: torch.onnx.export(model, ...)
# or load a pre-existing .onnx file

# Example with a pre-existing ONNX file:
onnx_path = os.environ.get("ONNX_MODEL_PATH", "model.onnx")
if not os.path.exists(onnx_path):
    print(f"Set ONNX_MODEL_PATH to your .onnx file")
    print(f"Supported ops: Add, Sub, Mul, Div, Relu, Sigmoid, Tanh,")
    print(f"  LeakyReLU, MatMul, Softmax, LayerNorm, GELU, Concat, Transpose")
    sys.exit(0)

runner = ONNXRunner(onnx_path)
x = np.random.randn(1, 64, 768).astype(np.float32)
outputs = runner.run({"input": x})
for name, out in outputs.items():
    print(f"  {name}: shape={out.shape}, mean={out.mean():.4f}")