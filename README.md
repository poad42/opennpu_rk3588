# OpenNPU — Open-Source NPU Runtime for RK3588

An open-source compiler + runtime stack that executes ONNX and JAX graphs on
the Rockchip RK3588 NPU via **raw DRM ioctls** — no proprietary libraries at
runtime, no proprietary toolkits at runtime.

## What This Does

- **Compiles** ONNX / JAX (HLO) graphs to native NPU register commands
- **Executes** on the NPU through raw DRM ioctls (no proprietary blobs)
- **Runs** GPT-2 124M generating text at 28ms/token (36 tok/s) with KV caching
- **Runs** SigLIP vision encoder at 98% NPU load (1.6x faster than CPU per-core)
- **Integrates** with JAX as a first-class device (`jax.devices()` → `[npu:0]`)

## Quick Start

### Build the PJRT C plugin (on the SBC)

```bash
cd src/opennpu/pjrt_c
python3 gen_plugin.py          # regenerate stub from API header
gcc -shared -fPIC -O3 -ffast-math -fopenmp \
    -o libpjrt_npu.so pjrt_npu.c -I. -lpthread
```

For full GPT-2 support (CNA matmul + KV-cached forward):

```bash
make full   # concatenates pjrt_npu_impl.c + cna_matmul.c + lm_forward.c
```

### Run JAX on the NPU

```bash
export JAX_PLATFORMS=npu,cpu
export PJRT_NAMES_AND_LIBRARY_PATHS=npu:./src/opennpu/pjrt_c/libpjrt_npu.so

python3 -c '
import jax, jax.numpy as jnp
print(jax.devices())  # [npu:0]

x = jnp.ones((1, 64, 768), dtype=jnp.float16)
y = jnp.ones((1, 64, 768), dtype=jnp.float16)
print(jax.jit(lambda a, b: a + b)(x, y))  # executes on NPU
'
```

### Run GPT-2 on the NPU

```bash
pip install torch transformers

# Via PyTorch wrapper (NPULM):
python3 examples/gpt2_generate.py --framework torch --prompt "The future of AI is"
# 28ms/token (36 tok/s) with KV caching, coherent text output

# Via raw ctypes (no torch needed):
python3 examples/gpt2_generate.py --framework raw --prompt "The future of AI is"
```

### Run SigLIP ViT on the NPU (PyTorch)

```bash
export NPU_PLUGIN_LIB=./src/opennpu/pjrt_c/libpjrt_npu.so
export OMP_WAIT_POLICY=active

python3 examples/vision_encoder.py --framework torch
# 995ms/image, 7.11x vs CPU (12-layer ViT, all matmuls on NPU)
```

### Run SigLIP ViT on the NPU (JAX)

```bash
python3 examples/vision_encoder.py --framework jax
# 1007ms/image, 3.95x vs CPU (same C backend, JAX wrapper)
```

### Verify coherence against HuggingFace

```bash
pip install transformers sentencepiece
python3 examples/siglip_coherence.py
# cos sim 0.9999 vs CPU reference (with bias + sandwich quantization)
```

## Requirements

- **Hardware**: RK3588 SBC (OrangePi 5 Max, Radxa Rock 5, etc.)
- **Kernel**: 6.1.x vendor kernel with rknpu driver (built into kernel)
- **NPU device**: `/dev/dri/card1` (RKNPU v2, 3 cores)
- **Software**: Python 3.11+, numpy, gcc with OpenMP
- **For JAX**: jax 0.10.2, jaxlib 0.10.2
- **For GPT-2**: torch, transformers (weight extraction only)
- **For PyTorch vision**: torch, transformers (loads weights from HuggingFace)
- **For ONNX runner**: onnx, onnxruntime (optional, for CPU fallback)

### Kernel patches (optional, for multi-core + fence support)

The stock rknpu driver works for single-core NPU execution. For
multi-core parallel dispatch and fence-based async completion, apply
the patches in `kernel-patches/` to the kernel source and rebuild:

- **Batch submit ioctl**: submit up to 3 jobs in one syscall (reduces
  ioctl overhead for multi-core dispatch)
- **IOMMU lock-free fast path**: skip `domain_lock` mutex when the
  same IOMMU domain is already active
- **kmem_cache for jobs**: pre-allocate `rknpu_job` structs from a
  slab cache instead of `kzalloc` per submit
- **CONFIG_ROCKCHIP_RKNPU_FENCE=y**: enables fence fds for non-blocking
  submit completion (needed for 3-core parallel execution)

See `kernel-patches/README.md` for instructions. The patches are
against the Rockchip `develop-6.1` branch and have been tested on
Orange Pi 5 Max (kernel 6.1.141).

## Architecture

```
JAX / jax.jit ──────► PJRT C Plugin ──────► Raw DRM ioctls ──────► NPU
                        │                      │
  ONNX file ──► ONNXRunner ──► npu_worker ──► runtime.py
                        │                      │
  PyTorch ──► torch_npu ─────────────────────►  ↑
                        │                      │
  Raw C ──────► npu_lm_forward() ─────────────► ↑
                                               │
                        codegen_synthesize.py (ONNX op → regcmd bytes)
```

The PJRT C plugin (`libpjrt_npu.so`) is the core: it parses JAX's stablehlo
bytecode, synthesizes NPU register commands, and submits them via raw DRM
ioctls. No proprietary libraries are loaded at runtime.

## Supported Operations

40 op/precision combinations verified on hardware:

| Precision | Ops |
|---|---|
| float16 | Add, Sub, Mul, Div, ReLU, Sigmoid, Tanh, Clip, LeakyReLU, MatMul, Softmax, LayerNorm, GELU, Concat, Transpose |
| int8 (w8a8) | Same as float16 (except Div) |
| int16 (w16a16i) | Same as int8 |

MatMul supports 5 GPT-2 transformer shapes (768×768, 768×3072, 3072×768, 768×64, 64×768).

## Key Results

**GPT-2 124M on NPU: 36 tok/s (28ms/token)** with KV caching.

**SigLIP ViT on NPU: 995ms/image (7.11x vs CPU)** — all 48 matmuls on NPU
via CNA descriptor, attention/GELU/LN on CPU with OpenMP. Coherent output
(cos sim 0.9999 vs HuggingFace reference). Works from both PyTorch
(`NPUViTEncoder` as `nn.Module`) and JAX (`npu_vit_forward`).

The key optimization: KV caching reduces decode matmuls from M=64 (full context
reprocessing) to M=1 (single token), cutting I/O DMA by 20MB/token. Combined
with the hybrid matmul fusion (fuse QKV+fc, decompose proj for GELU overlap),
this achieves 84 NPU matmuls at 0.32ms each = 21ms matmul time per token.

| Workload | NPU | CPU | Winner |
|---|---|---|---|
| GPT-2 decode (KV cache, M=1) | **28ms/token (36 tok/s)** | 33ms/token (RKLLM) | **NPU 1.4×** |
| GPT-2 decode (CNA, M=64) | 74ms/token | 33ms/token (RKLLM) | CPU (DMA-bound) |
| GPT-2 decode (template, M=64) | 250ms/token | 33ms/token (RKLLM) | CPU (DMA-bound) |
| Vision encoder (batch=1) | 911ms | ~2500ms | **NPU 1.6x** (compute-bound) |
| Vision encoder (3-core) | 370ms/img | ~2500ms | **NPU 6.2x** |
| NPU load (vision) | **98%** | — | NPU fully utilized |
| NPU load (LLM decode) | 94% matmul | — | NPU at hardware floor |
| NPU DMA throughput (CNA) | **3.0 GB/s** (L3 snoop) | — | Hardware ceiling |

The NPU wins for **compute-bound** workloads (vision, prefill) and loses for
**DMA-bound** workloads (LLM decode, batch=1). The key metric is
compute-to-data ratio (FLOP/byte): NPU wins above ~0.1 FLOP/byte.

## Documentation

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Complete system architecture (read first) |
| [docs/JAX_NPU_PLUGIN.md](docs/JAX_NPU_PLUGIN.md) | PJRT C plugin deep dive |
| [docs/NPU_REGCMD_REFERENCE.md](docs/NPU_REGCMD_REFERENCE.md) | Register command format reference |
| [docs/ref/](docs/ref/) | Investigation reports (SRAM, QoS, registers, etc.) |

## Repository Structure

```
opennpu-rk3588/
├── src/opennpu/
│   ├── runtime.py              Raw DRM ioctl runtime
│   ├── codegen_synthesize.py   ONNX → regcmd byte synthesis
│   ├── pjrt.py                 Python PJRT client
│   ├── jax_npu.py              JAX custom op (npu_cached_mm)
│   ├── onnx_runner.py          ONNX → NPU/CPU dispatch
│   ├── npu_worker.py           Per-op subprocess worker
│   ├── pjrt_c/                 PJRT C plugin for JAX
│   │   ├── pjrt_c_api.h        PJRT API v0.112 (Apache 2.0, OpenXLA)
│   │   ├── pjrt_npu_impl.c     Plugin implementation (148KB)
│   │   ├── cna_matmul.c        CNA descriptor matmul (variable K/N/M, weight cache)
│   │   ├── lm_forward.c        GPT-2 forward with KV caching (M=1 decode)
│   │   ├── matmul_tmpl.h       Matmul regcmd templates (5 shapes)
│   │   ├── relu_tmpl.h         ReLU regcmd template
│   │   ├── tanh_tmpl.h         Tanh regcmd template
│   │   └── gen_plugin.py       Plugin stub generator
├── kernel-patches/             NPU driver patches (fence, batch submit, kmem_cache)
├── scripts/                    Weight extraction, GPT-2 generation
├── docs/                       Architecture docs + investigation reports
├── examples/                   Usage examples
│   ├── vision_encoder.py      SigLIP ViT on NPU (--framework torch|jax)
│   ├── gpt2_generate.py       GPT-2 text generation (--framework torch|raw)
│   ├── jax_add.py             JAX elementwise add on NPU
│   ├── jax_matmul.py          JAX matmul on NPU
│   ├── onnx_inference.py      ONNX model inference
│   └── raw_runtime.py         Raw DRM ioctl runtime
```

## Provenance

The NPU register command format, task chain structure, and DMA encoding
were discovered through systematic analysis of the public DRM ioctl
interface and observation of hardware behavior.

- **PJRT API header** (`pjrt_c_api.h`): From the OpenXLA project, Apache 2.0.
- **Regcmd templates** (`*_tmpl.h`): Hardware register values discovered
  from the NPU via the public DRM ioctl interface.
- **No proprietary code** from any vendor is included in this repository.
- **No proprietary libraries** are required at runtime.

## Prior Work & Attribution

This project builds on a foundation of publicly available open-source work:

- **Rockchip rknpu kernel driver** (GPL): The public kernel driver exposes the
  DRM ioctl interface (`MEM_CREATE`, `MEM_MAP`, `MEM_SYNC`, `SUBMIT`, `ACTION`)
  and the NPU register map. All ioctl numbers and register addresses in this
  project come from that public GPL source.
- **OpenXLA / PJRT API** (Apache 2.0): The PJRT C plugin interface is developed
  by the OpenXLA project. This project pins the exact `pjrt_c_api.h` from the
  OpenXLA commit matching jaxlib 0.10.2.
- **CNA descriptor matmul** (GPL v3, Jasbir Matharu / mtx512): The CNA 1x1
  convolution descriptor approach for NPU matmul — using the hardware's
  convolution engine with 1×1 kernels to perform general matrix
  multiplication — was introduced by mtx512/rk3588-npu. This project
  independently reimplemented the CNA descriptor format, weight tiling,
  and register programming from hardware analysis, but the technique of
  using CNA for matmul is attributed to that work.

This project independently developed its own compiler, runtime, PJRT plugin,
multi-op graph executor, and model integrations.

## License

MIT License. See [LICENSE](LICENSE).

The PJRT API header (`src/opennpu/pjrt_c/pjrt_c_api.h`) is also Apache 2.0,
© The OpenXLA Authors.