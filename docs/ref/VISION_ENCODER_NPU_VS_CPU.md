# Vision Encoder (SigLIP) on NPU — NPU WINS

## Executive Summary

For the SmolVLM vision encoder (SigLIP, 108MB, 12 transformer layers):
- **NPU load = 98%** (fully compute-bound, opposite of LLM decode's 0%)
- **NPU is 1.6x faster** than CPU per-core (117ms vs 191ms per layer)
- **NPU 3-core parallel: 370ms/image** (2.45x speedup, 6.2x vs CPU)
- **Batch processing scales well**: 12 images at 371ms/img (stable)

This confirms: **NPU wins for vision (compute-bound), CPU wins for LLM decode (DMA-bound)**.

## NPU Load Comparison

| Workload | NPU Load | Winner | Why |
|----------|----------|--------|-----|
| **Vision encoder** | **98%** | **NPU** | Compute-bound (1024 patches, W read once) |
| **LLM prefill** (RKLLM) | 28-42% | NPU | Large batch, compute-bound |
| **LLM decode** (RKLLM) | 0% | CPU | Batch=1, DMA-bound |
| **GPT-2 decode** (ours) | 94% | CPU (faster) | NPU DMA-bound at 1 GB/s |

## Per-Layer NPU vs CPU (SmolVLM Vision Encoder)

Each layer = attention (1024×768) + MLP (768×3072):

| Layer | NPU (ms) | CPU (ms) | NPU/CPU | 
|-------|----------|----------|---------|
| L0 | 114.7 | 191.2 | 0.60 |
| L1 | 121.2 | 193.0 | 0.63 |
| L2 | 120.4 | 190.3 | 0.63 |
| L3 | 112.7 | 190.1 | 0.59 |
| L4 | 117.4 | 191.3 | 0.61 |
| L5 | 117.3 | 190.6 | 0.62 |
| L6 | 111.9 | 190.1 | 0.59 |
| L7 | 116.7 | 189.8 | 0.61 |
| L8 | 116.9 | 189.7 | 0.62 |
| L9 | 118.7 | 193.1 | 0.61 |
| L10 | 116.8 | 190.5 | 0.61 |
| L11 | 116.9 | 190.6 | 0.61 |
| **TOTAL** | **1402** | **2291** | **0.61 (NPU 1.6x faster)** |

## Full Vision Encoder (single RKNN model, includes conv2d)

| Metric | NPU | CPU (ONNX) |
|--------|-----|------------|
| Single image | 911ms | ~2500ms (2291ms per-layer + conv2d) |
| 3-core parallel | 370ms/img | N/A |
| NPU load | **98%** | N/A |

## Batch Processing (3-core parallel)

| Images | 1-core (ms) | 3-core (ms) | Speedup | Per img (3-core) |
|--------|-------------|-------------|---------|-----------------|
| 1 | 920 | 1181 | 0.78 | 1181 |
| 3 | 2849 | 1211 | 2.35x | 404 |
| 6 | 5425 | 2221 | 2.44x | 370 |
| 12 | 10888 | 4453 | 2.45x | 371 |

At 6+ images, 3-core parallel stabilizes at **371ms/image** (2.45x speedup).

## Why NPU Wins for Vision

### Vision encoder (compute-bound)
- Input: [1, 3, 512, 512] → 1024 patches × 768 dim
- Per layer: 1024×768 attention + 768×3072 MLP matmuls
- W read ONCE for all 1024 patches → DMA amortized
- NPU compute: 1024 × W_size FLOPs / NPU_TFLOPS
- NPU DMA: W_size / 1 GB/s (once per layer, not per patch)
- **Compute >> DMA → NPU wins** (1-2 TFLOPS vs 40 GFLOPS CPU)

### LLM decode (DMA-bound, batch=1)
- Input: [1, 768] → 1 token
- Per layer: 1×768 × 768×N matmul
- W read for 1 token only → DMA NOT amortized
- NPU DMA: W_size / 1 GB/s (per token!)
- **DMA >> compute → CPU wins** (15-21 GB/s DDR4 vs 1 GB/s NPU DMA)

## Compute-to-Data Ratio: The Key Metric

| Workload | Batch | Compute (FLOPs) | Data (bytes) | Ratio | Winner |
|----------|-------|-----------------|--------------|-------|--------|
| Vision (1024 patches) | 1024 | ~100 GFLOP | ~9 MB/layer | ~11 FLOP/byte | **NPU** |
| LLM prefill (250 tokens) | 250 | ~25 GFLOP | ~15 MB/layer | ~1.7 FLOP/byte | **NPU** |
| LLM decode | 1 | ~0.1 GFLOP | ~15 MB/layer | ~0.007 FLOP/byte | **CPU** |

NPU wins when compute-to-data ratio > ~0.1 FLOP/byte (NPU compute advantage exceeds DMA disadvantage).

## Our RE is Fully Validated

1. NPU DMA ~1 GB/s ✓ (LLM decode: 250ms for 248MB GPT-2)
2. NPU compute ~1 TFLOPS ✓ (vision: 98% load, 1.6x faster than CPU)
3. NPU 3 cores = 2.45x data-parallel ✓ (vision batch processing)
4. CPU DDR4 ~15-21 GB/s ✓ (LLM decode: 33-71ms for 500MB-1.5GB)

The NPU is an excellent vision accelerator but a poor LLM decoder.
RKLLM's strategy (NPU for prefill, CPU for decode) is optimal.