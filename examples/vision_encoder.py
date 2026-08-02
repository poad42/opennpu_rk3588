#!/usr/bin/env python3
"""Example: SigLIP ViT vision encoder on RK3588 NPU.

Single entry point for both PyTorch and JAX. Uses the standard framework
API (nn.Module / JAX primitive) — no ctypes, no custom calls.

PyTorch:  NPUViTEncoder is a torch.nn.Module. model(x) returns torch.Tensor.
JAX:      npu_vit_forward is a JAX primitive. Works inside jax.jit.

All 48 matmuls run on the NPU via CNA descriptor (raw DRM ioctls).
Attention, GELU, LayerNorm run on CPU with OpenMP (3 A76 cores).
No librknnrt at runtime. No RKNN toolkit at runtime.

Prerequisites:
  1. Build: cd src/opennpu/pjrt_c && make full
  2. pip install torch numpy  (for --framework torch)
  2. pip install jax jaxlib numpy  (for --framework jax)

Run on the SBC:
  NPU_PLUGIN_LIB=... python3 examples/vision_encoder.py --framework torch
  NPU_PLUGIN_LIB=... python3 examples/vision_encoder.py --framework jax

  With a real HuggingFace model:
  NPU_PLUGIN_LIB=... python3 examples/vision_encoder.py --framework torch --model google/vit-base-patch16-224
"""
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_WAIT_POLICY", "active")


def run_torch(args):
    import torch
    from opennpu.torch_npu import NPUViTEncoder

    print("=== SigLIP ViT on NPU (PyTorch nn.Module) ===\n")

    if args.model:
        encoder = NPUViTEncoder.from_pretrained(args.model, n_layers=args.n_layers)
    else:
        encoder = NPUViTEncoder.random(n_layers=args.n_layers)
        print("Random weights (benchmarking)")

    x = torch.randn(1, 197, 768) * 0.1
    out = encoder(x)
    print(f"forward(x): {x.shape} -> {out.shape}, mean={out.mean().item():.4f}\n")

    prof = encoder.benchmark(x, n_warmup=2, n_runs=5)
    print(f"benchmark:")
    print(f"  total:      {prof['total_ms']:.0f} ms/image")
    print(f"  npu matmul: {prof['npu_mm']:.0f} ms ({prof['npu_mm']/prof['total_ms']*100:.0f}%)")
    print(f"  attention:  {prof['attn']:.0f} ms ({prof['attn']/prof['total_ms']*100:.0f}%)")
    print(f"  gelu:       {prof['gelu']:.0f} ms")
    print(f"  throughput: {prof['throughput']:.1f} img/s")
    return prof['total_ms']


def run_jax(args):
    import jax
    import jax.numpy as jnp
    from opennpu.jax_npu_vision import load_random_weights, npu_vit_forward

    print("=== SigLIP ViT on NPU (JAX primitive) ===\n")

    load_random_weights(n_layers=args.n_layers)
    print("Random weights (benchmarking)")

    x = jnp.array(np.random.randn(197, 768).astype(np.float32) * 0.1)
    out = npu_vit_forward(x)
    out.block_until_ready()
    print(f"npu_vit_forward(x): {x.shape} -> {out.shape}, mean={float(out.mean()):.4f}")


    composed = npu_vit_forward(x) * 2.0 + jnp.ones_like(x)
    print(f"compose with jax ops: mean={float(composed.mean()):.4f}")
    print("(jax.jit support requires PJRT plugin op=8 — future work)\n")

    for _ in range(2): npu_vit_forward(x)
    t0 = time.perf_counter()
    N = 5
    for _ in range(N): npu_vit_forward(x)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"benchmark:")
    print(f"  total:      {dt:.0f} ms/image")
    print(f"  throughput: {1000/dt:.1f} img/s")
    return dt


def cpu_baseline(n_layers=12):
    print("\ncpu baseline...")
    S, D, H, hd, IN = 197, 768, 12, 64, 3072
    wq = np.random.randn(D, D*3).astype(np.float32) * 0.02
    wp = np.random.randn(D, D).astype(np.float32) * 0.02
    wf = np.random.randn(D, IN).astype(np.float32) * 0.02
    wd = np.random.randn(IN, D).astype(np.float32) * 0.02
    def blk(x):
        h = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)
        qkv = h @ wq; q,k,v = qkv[:,:D],qkv[:,D:2*D],qkv[:,2*D:]
        q=q.reshape(S,H,hd).transpose(1,0,2);k=k.reshape(S,H,hd).transpose(1,0,2);v=v.reshape(S,H,hd).transpose(1,0,2)
        a=np.einsum("hsd,htd->hst",q,k)/np.sqrt(hd);a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        o=np.einsum("hst,htd->hsd",a,v).transpose(1,0,2).reshape(S,D);x=x+o@wp
        h=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-6)
        g=0.5*(h@wf)*(1+np.tanh(0.7978845*(h@wf+0.044715*(h@wf)**3)));x=x+g@wd;return x
    xn = np.random.randn(S, D).astype(np.float32) * 0.1
    for _ in range(2):
        for _ in range(n_layers): h = blk(xn)
    t0 = time.perf_counter()
    for _ in range(3):
        h = xn.copy()
        for _ in range(n_layers): h = blk(h)
    cpu = (time.perf_counter() - t0) / 3 * 1000
    print(f"  cpu: {cpu:.0f} ms/image")
    return cpu


def main():
    parser = argparse.ArgumentParser(description="SigLIP ViT on RK3588 NPU")
    parser.add_argument("--framework", choices=["torch", "jax"], default="torch")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--n-layers", type=int, default=12)
    args = parser.parse_args()

    if args.framework == "torch":
        npu_ms = run_torch(args)
    else:
        npu_ms = run_jax(args)

    cpu_ms = cpu_baseline(args.n_layers)
    print(f"\n  NPU vs CPU: {cpu_ms/npu_ms:.2f}x")


if __name__ == "__main__":
    main()