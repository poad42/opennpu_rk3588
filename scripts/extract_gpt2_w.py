#!/usr/bin/env python3
"""Extract GPT-2 weights for NPU inference.

Thin wrapper around opennpu.lm.extract_gpt2_weights.
Saves f32 and f16 weight files for use with NPUModel.

Usage:
  python3 scripts/extract_gpt2_w.py [model_name] [output_dir]
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opennpu.lm import extract_gpt2_weights

model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
f32 = os.path.join(out_dir, f"{model_name}_w_hybrid.npz")
f16 = os.path.join(out_dir, f"{model_name}_w16_hybrid.npz")
print(f"Extracting {model_name} weights...")
W, W16 = extract_gpt2_weights(model_name, out_f32=f32, out_f16=f16)
print(f"Saved: {f32} ({os.path.getsize(f32)//1024}KB), {f16} ({os.path.getsize(f16)//1024}KB)")
