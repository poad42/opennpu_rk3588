# OpenNPU Documentation

Open-source compiler + runtime stack for the RK3588 NPU.
No proprietary libraries at runtime. No vendor toolkits at runtime. Pure open stack.

## Start Here

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete system architecture: NPU hardware, development methodology, codegen, runtime, PJRT plugin, GPT-2 LM, vision encoder, performance, hardware limits. **Read this first.**
- **[CNA_DESCRIPTOR_GUIDE.md](CNA_DESCRIPTOR_GUIDE.md)** — How to implement bespoke NPU ops using the CNA descriptor API, and wire them into PyTorch and JAX. Covers the 5 BOs, register task, memory layouts, hardware constraints (M must be multiple of 4, K>768 tiling), verification methodology, and troubleshooting.
- **[OP_LOWERING.md](OP_LOWERING.md)** — How each ONNX/HLO op is lowered to NPU hardware: LayerNorm (CPU reduce + affine), Erf (tanh approximation), Attention (CNA matmuls + CPU softmax), Shape (compile-time), Slice/Gather (CPU fallback, hardware limitations).
- **[JAX_NPU_PLUGIN.md](JAX_NPU_PLUGIN.md)** — Deep dive on the PJRT C plugin: ABI pinning, stablehlo bytecode parser, buffer management, matmul execution, GELU overlap, thread layout.
- **[NPU_REGCMD_REFERENCE.md](NPU_REGCMD_REFERENCE.md)** — Register command format reference: entry format, block structure, task structs, ALU selectors, DMA patching, register map, ACTION ioctl.

## Reference Docs (docs/ref/)

| Document | Topic |
|---|---|
| [NPU_REGISTER_INVESTIGATION.md](ref/NPU_REGISTER_INVESTIGATION.md) | RK3588 vs RK3576 NPU register differences |
| [QOS_TEST_RESULTS.md](ref/QOS_TEST_RESULTS.md) | QoS priority test (no improvement for large transfers) |
| [VISION_ENCODER_NPU_VS_CPU.md](ref/VISION_ENCODER_NPU_VS_CPU.md) | SigLIP vision encoder: NPU 98% load, 1.6x faster than CPU |

## Key Results

| Achievement | Detail |
|---|---|
| GPT-2 124M on NPU (KV cache) | 28ms/token (36 tok/s), 1.6x faster than prior art, coherent text |
| GPT-2 124M on NPU (CNA M=64) | 74ms/token (13.5 tok/s), 2.8x faster than template |
| Vision encoder (SigLIP) on NPU | 98% NPU load, 1.6x faster than CPU per-core, 6.2x with 3 cores |
| JAX on NPU | `jax.devices()` returns `[npu:0]`, `jax.jit` executes on NPU |
| Ops verified | 40 op/precision combinations (fp16, int8, int16) |
| Multi-core | 2.45x data-parallel speedup (3 cores) |
| DMA throughput | ~1 GB/s (NPU hardware limit, proven via cross-shape analysis) |

## Key Insight: Compute-to-Data Ratio

The NPU wins when the compute-to-data ratio (FLOP/byte) is high:

| Workload | FLOP/byte | NPU Load | Winner |
|---|---|---|---|
| Vision (1024 patches) | ~11.0 | 98% | **NPU** |
| LLM prefill (250 tok) | ~1.7 | 42% | NPU |
| LLM decode (1 token) | ~0.007 | 0% | **CPU** |

NPU: ~1-2 TFLOPS compute, ~1 GB/s DMA.
CPU: ~40 GFLOPS compute, ~15-21 GB/s DDR4.

NPU wins when compute dominates (vision). CPU wins when DMA dominates (LLM decode).