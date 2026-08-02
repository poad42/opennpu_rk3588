#!/usr/bin/env python3
"""Example: GPT-2 text generation on the RK3588 NPU.

Single entry point for GPT-2 124M generation. All 84 matmuls per token
run on the NPU via CNA descriptor (raw DRM ioctls). KV cache eliminates
recomputation — 28 ms/token (36 tok/s), 1.6x faster than prior art.

No librknnrt at runtime. No RKNN toolkit at runtime. Pure open stack.

Prerequisites:
  1. Build: cd src/opennpu/pjrt_c && make full
  2. Extract weights: python3 scripts/extract_gpt2_w.py
  3. pip install torch transformers numpy  (for --framework torch)
     pip install jax jaxlib numpy  (for --framework jax)

Run on the SBC:
  NPU_PLUGIN_LIB=... python3 examples/gpt2_generate.py --prompt "The future of AI is"
  NPU_PLUGIN_LIB=... python3 examples/gpt2_generate.py --framework torch --prompt "Hello world"
"""
import os, sys, time, ctypes, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_WAIT_POLICY", "active")

PLUGIN_PATH = os.environ.get("NPU_PLUGIN_LIB",
    os.path.join(os.path.dirname(__file__), "..", "src", "opennpu", "pjrt_c", "libpjrt_npu.so"))


def run_raw(prompt_ids, n_tokens=20):
    """Raw ctypes generation — no torch/jax dependency."""
    plugin = ctypes.CDLL(PLUGIN_PATH)
    plugin.npu_lm_load_params.restype = ctypes.c_int
    plugin.npu_lm_load_params.argtypes = [ctypes.c_char_p]
    plugin.npu_lm_forward.restype = ctypes.c_int
    plugin.npu_lm_forward.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_float)]

    weights_path = os.environ.get("GPT2_WEIGHTS_BIN", "gpt2_w_params.bin")
    if plugin.npu_lm_load_params(weights_path.encode()) != 0:
        print(f"Failed to load weights from {weights_path}")
        print("Run: python3 scripts/extract_gpt2_w.py")
        sys.exit(1)

    token_ids = list(prompt_ids)
    logits = (ctypes.c_float * 50257)()

    print(f"Generating {n_tokens} tokens on NPU...", flush=True)
    t0 = time.perf_counter()
    for step in range(n_tokens):
        ids_arr = (ctypes.c_int * len(token_ids))(*token_ids)
        ret = plugin.npu_lm_forward(ids_arr, len(token_ids), logits)
        if ret != 0:
            print(f"Forward failed: {ret}"); break
        next_token = int(np.argmax(np.array(logits[:])))
        token_ids.append(next_token)
    dt = (time.perf_counter() - t0) / n_tokens * 1000
    print(f"  {dt:.0f} ms/token ({1000/dt:.1f} tok/s)")
    return token_ids


def run_torch(prompt, n_tokens=20):
    """PyTorch + HuggingFace — loads from transformers, generates on NPU."""
    from opennpu.torch_npu import NPULM
    from transformers import GPT2Tokenizer

    print("Loading GPT-2 from HuggingFace + NPU...")
    lm = NPULM.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Prompt: '{prompt}'")
    print(f"Generating {n_tokens} tokens on NPU...", flush=True)

    t0 = time.perf_counter()
    gen_ids = lm.generate(prompt, n_tokens=n_tokens, temperature=0.8, top_k=40)
    dt = (time.perf_counter() - t0) / n_tokens * 1000

    ids = tokenizer.encode(prompt) + gen_ids
    text = tokenizer.decode(ids)
    print(f"  {dt:.0f} ms/token ({1000/dt:.1f} tok/s)")
    print(f"  Output: {text}")
    lm.close()
    return ids


def main():
    parser = argparse.ArgumentParser(description="GPT-2 on RK3588 NPU")
    parser.add_argument("--framework", choices=["torch", "raw"], default="torch")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--n-tokens", type=int, default=20)
    args = parser.parse_args()

    if args.framework == "torch":
        run_torch(args.prompt, args.n_tokens)
    else:
        # Simple token IDs for raw mode (no tokenizer needed)
        # "The robot is a machine." in GPT-2 BPE
        prompt_ids = [464, 3459, 287, 257, 6292, 13]
        run_raw(prompt_ids, args.n_tokens)


if __name__ == "__main__":
    main()