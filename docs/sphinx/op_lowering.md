# NPU Op Lowering Reference

How each ONNX/HLO op is lowered to RK3588 NPU hardware operations.
This document specifies the decomposition, dispatch strategy, and hardware
constraints for every op the open-source stack supports.

## Lowering Categories

| Category | Description | Examples |
|---|---|---|
| **NPU-native** | Captured .rknn template, DMA-patched at runtime | Add, Sub, Mul, Div, Relu, Softmax |
| **NPU via CNA** | CNA descriptor matmul (arbitrary shapes) | MatMul, Attention QK^T/AV |
| **CPU reduce + NPU affine** | CPU computes reduction, NPU does elementwise | LayerNorm (hybrid) |
| **NPU activation** | Captured activation template, tiled | Erf→tanh, GELU, Sigmoid |
| **Compile-time** | Resolved during graph parsing, no runtime exec | Shape |
| **CPU fallback** | NPU hardware limitation or low ROI | Gather, Slice |

---

## 1. LayerNormalization

### ONNX signature
```
LayerNormalization(X: [*, H], Scale: [H], Bias: [H]) → Y: [*, H]
  Y[i, h] = (X[i, h] - mean(X[i])) / sqrt(var(X[i]) + eps) * Scale[h] + Bias[h]
```

### Lowering: CPU reduce + CPU affine (production path)

```
Step 1 (CPU): mean = sum(X, -1) / H              # reduction
Step 2 (CPU): var  = sum((X-mean)^2, -1) / H     # reduction
Step 3 (CPU): inv  = 1 / sqrt(var + eps)          # elementwise
Step 4 (CPU): Y = (X - mean) * inv * Scale + Bias  # elementwise affine
```

**Why CPU?** The NPU's per-submit overhead (~0.5ms) exceeds the CPU cost of
LayerNorm for H=768 (~0.05ms). The NPU is DMA-bound at ~1 GB/s; the reduction
requires reading H elements per row (768 * 4 = 3KB) and the affine is a
per-element multiply-add. For [1,64,768], the total data is 192KB — the CPU's
DDR4 bandwidth (15-21 GB/s) handles this in ~10μs. The NPU would take ~200μs
just for the DMA roundtrip.

### NPU native template (alternative, limited)

The NPU has a native LayerNormalization op (partial support, fp16 only). A
template was captured at shape [1,64,768]. It performs the full op (mean,
var, sqrt, div, scale, bias) in one NPU submit.

**Limitation**: Scale and Bias are BAKED into the captured template (scattered
across scratch + regcmd, not a patchable IO BO). The default template uses
Scale=1.0, Bias=0.0 (matching torch nn.LayerNorm default init). For trained
models with per-layer Scale/Bias, each layer needs its own captured template.

**Future work**: RE the Scale/Bias layout in the scratch BO to patch them at
runtime, enabling the native NPU path for any trained model.

### C function: `npu_cna_layernorm`

```c
int npu_cna_layernorm(const float* x, const float* scale_w,
                      const float* bias, int M, int H, float eps,
                      float* out);
```

OpenMP-parallelized CPU implementation. Used by `lm_forward.c` for GPT-2.

---

## 2. Erf

### ONNX signature
```
Erf(X) → Y
  Y[i] = erf(X[i])    # error function
```

### Lowering: tanh approximation

```
erf(x) ≈ tanh(1.128379167 * x * (1 + 0.044715 * x²))
```

The NPU supports `tanh` natively (captured activation template). The
approximation has max error ~1.6e-2 vs scipy.erf, sufficient for GELU.

### GELU connection

GELU uses Erf: `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`

With the tanh approximation:
```
gelu(x) ≈ 0.5 * x * (1 + tanh(0.7978845 * (x + 0.044715 * x³)))
```

The NPU has a native `Gelu` op (captured template, tiles to any shape via the
49152-element chunk tiling path). The tanh-approximation GELU is used in
`lm_forward.c` for the FFN activation.

### C function: `npu_erf_approx`

```c
void npu_erf_approx(const float* x, int n, float* out);
```

OpenMP-parallelized tanh approximation. Max error ~1.6e-2 vs math.erf.

---

## 3. Attention

### Decomposition

```
Q [n_heads, M, dh]    K [n_heads, n_kv, dh]    V [n_heads, n_kv, dh]

Step 1: scores = Q @ K^T          # [n_heads, M, n_kv]  — CNA matmul
Step 2: scores *= 1/sqrt(dh)      # elementwise scale   — CPU
Step 3: probs = softmax(scores)   # reduction + exp     — CPU
Step 4: out = probs @ V            # [n_heads, M, dh]   — CNA matmul
```

### Dispatch heuristic

| Condition | Path | Rationale |
|---|---|---|
| M ≤ 4 (decode) | CPU | NPU per-submit overhead > compute |
| M > 4 (prefill) | NPU matmuls + CPU softmax | CNA matmul beneficial |

For M=1 decode: QK^T is [1, 64] @ [64, n_kv] — a tiny matmul. The NPU
per-submit overhead (~0.5ms) far exceeds the CPU cost (~5μs). The CNA
matmul takes 0.22ms for 768×768, but the setup/teardown adds ~0.3ms.

For M=64 prefill: QK^T is [64, 64] @ [64, 64] per head — 12 matmuls of
moderate size. The CNA handles these efficiently.

### CNA weight management

The attention function uses CNA weight slots 126 (K^T) and 127 (V). These
slots are reloaded each step because the KV cache grows:

```c
// QK^T: load K^T as weight [dh, n_kv], run Q @ K^T
npu_cna_cache_load(ATTN_K_SLOT, kt16, dh, n_kv);
npu_cna_cache_run_m(ATTN_K_SLOT, M, q16, scores);

// AV: load V as weight [n_kv, dh], run attn @ V
npu_cna_cache_load(ATTN_V_SLOT, v16, n_kv, dh);
npu_cna_cache_run_m(ATTN_V_SLOT, M, s16, out);
```

The `close(wt_h[slot])` before `npu_cna_cache_load` frees the old GEM handle.
Note: closing the handle does NOT free the IOVA (kernel limitation), so the
IOVA pool grows. For 12 heads × 2 matmuls × n_steps, the pool may exhaust
after many steps. For production, pre-allocate fixed-size weight BOs and
overwrite contents instead of close/reload.

### Causal mask

The causal mask (for autoregressive models) is applied as an elementwise
operation before softmax:

```c
for (int s = pos + 1; s < n_kv; s++)
    scores[s] = -INFINITY;
```

This is a CPU operation — the NPU's `exSoftmaxMask` op (partial support)
could handle this, but the CPU cost is negligible (O(n_kv) per head per row).

### C function: `npu_cna_attention`

```c
int npu_cna_attention(const float* q, const float* k_cache,
                      const float* v_cache, int n_heads, int M,
                      int dh, int n_kv, int use_npu, float* out);
```

- `use_npu = 0`: CPU path (OpenMP, 3 A76 cores)
- `use_npu = 1`: NPU path (CNA matmuls + CPU softmax), requires M ≥ 4

---

## 4. Shape

### ONNX signature
```
Shape(X) → Y: [rank(X)]   # returns the shape of X as a tensor
```

### Lowering: compile-time resolution

Shape is a **compile-time operation** — it produces a constant from the
input's static shape. It is resolved during graph parsing (topological
execution), not dispatched to the NPU.

```python
# In the ONNX runner (onnx_runner.py _cpu_op):
if node_type == "Shape":
    return np.array(inputs[0].shape, dtype=np.int64)
```

Since the runner processes nodes in topological order and stores all
intermediates, the Shape output is available as a constant for downstream
ops (Slice, Concat, Reshape) that consume it.

**No NPU execution is needed or possible** — Shape is a metadata operation.

---

## 5. Slice

### ONNX signature
```
Slice(X, starts, ends, axes, steps) → Y
  Y = X[starts[i]:ends[i]:steps[i]] for each axis i
```

### Lowering: CPU fallback

**NPU hardware limitation**: The NPU's DMA engine uses fixed stride/offset
configuration programmed at submit time. While the reader/writer DMA blocks
have stride fields, the RK3588 NPU's Slice support is "partial" (per the
Rockchip operator list) and requires the RKNN compiler's CPU-side tiling
to set up the scratch BO with the correct sub-range.

Without RE'ing the DMA stride configuration (which encodes the 128-element
tiling schedule), we cannot synthesize Slice regcmd from scratch. The
captured template approach also fails because Slice's regcmd reads from
scratch sub-ranges (like conv2d), not from IO BOs.

### CPU implementation

```python
# In _cpu_op:
if node_type == "Slice":
    x = inputs[0]
    starts = inputs[1].tolist() if len(inputs) > 1 else attrs.get("starts", [])
    ends = inputs[2].tolist() if len(inputs) > 2 else attrs.get("ends", [])
    axes = inputs[3].tolist() if len(inputs) > 3 else attrs.get("axes", list(range(len(starts))))
    steps = inputs[4].tolist() if len(inputs) > 4 else attrs.get("steps", [1]*len(starts))
    slices = [slice(None)] * x.ndim
    for i, ax in enumerate(axes):
        slices[ax] = slice(starts[i], ends[i], steps[i])
    return x[tuple(slices)]
```

### Future NPU path

To move Slice to the NPU, we need to:
1. Capture a Slice template via the LD_PRELOAD dumper
2. RE the reader DMA stride configuration (the 128-element tiling schedule)
3. Patch the DMA strides at runtime for arbitrary start/end/step

This is the same class of RE work as conv2d (scratch-tiled ops with
input-dependent DMA configuration).

---

## 6. Gather

### ONNX signature
```
Gather(X, indices, axis) → Y
  Y[i] = X[indices[i]] along axis
```

### Lowering: CPU fallback (hardware limitation)

**NPU hardware limitation**: Gather is a **data-dependent** operation —
the index values determine which memory locations to read. The NPU's DMA
engine is **address-fixed** (DMA addresses are programmed in the regcmd at
submit time, before the NPU reads any data). The NPU cannot perform
data-dependent memory access.

This is a fundamental hardware constraint, not a software limitation. The
RKNN compiler also implements Gather as a CPU fallback (Gather is not in
the RK3588 supported operator list — it falls in the "not_supported"
category).

### CPU implementation

```python
if node_type == "Gather":
    data, indices = inputs[0], inputs[1].astype(np.int64)
    return np.take(data, indices, axis=attrs.get("axis", 0))
```

### When Gather appears

Gather is used in:
- **Token embedding lookup**: `wte[token_ids]` — this is a Gather operation.
  In the C forward pass (`lm_forward.c`), this is done on CPU:
  `x[j] = s_wte[token_id * H + j] + s_wpe[pos * H + j]`
- **Top-k/top-p sampling**: gathering selected token probabilities

For the embedding lookup, an alternative is to treat it as a matmul: convert
the one-hot token ID vector to a [1, V] tensor and matmul with the [V, H]
embedding matrix. But V=50257 makes this impractical (50257×768 = 38M
elements). The CPU Gather (direct memory copy of H elements) is O(H) and
far more efficient.

---

## 7. CNA Matmul (arbitrary shapes)

### Lowering: CNA 1x1 convolution descriptor

All matmuls in the open-source stack use the CNA descriptor API
(`cna_matmul.c`), which supports arbitrary K and N dimensions (unlike the
shape-pinned templates).

### Hardware constraints

| Constraint | Fix |
|---|---|
| M must be multiple of 4 | Pad M to M4 = max(4, ceil(M/4)*4) |
| N must be multiple of 16 | Pad N to Np = max(16, ceil(N/16)*16) |
| K must be multiple of 32 | Pad K to Kta = ceil(K/32)*32 |
| K > 768 requires tiling | Split into tiles of 768 channels each |

### API

```c
// Setup (once): allocate CNA context with max geometry
int npu_cna_cache_setup(int n_w, int M, int max_K, int max_N);

// Load weight (once per weight): [K, N] fp16 row-major
int npu_cna_cache_load(int w_idx, const void* W_fp16, int K, int N);

// Run matmul: X [M, K] fp16 → Y [M, N] fp32
int npu_cna_cache_run_m(int w_idx, int M, const void* X_fp16, void* Y_fp32);

// Run with setup M
int npu_cna_cache_run(int w_idx, const void* X_fp16, void* Y_fp32);
```

See [`docs/CNA_DESCRIPTOR_GUIDE.md`](CNA_DESCRIPTOR_GUIDE.md) for the full
CNA programming guide.

---

## Summary Table

| Op | NPU Path | CPU Path | Bottleneck |
|---|---|---|---|
| LayerNorm | Native template (scale/B baked) | CPU reduce + affine | CPU faster (reduction) |
| Erf | N/A (not native) | tanh approximation | NPU tanh template (future) |
| Attention QK^T | CNA matmul (M>4) | CPU OpenMP (M≤4) | NPU per-submit overhead |
| Attention softmax | NPU template (seq=64) | CPU (arbitrary seq) | CPU faster (small seq) |
| Attention AV | CNA matmul (M>4) | CPU OpenMP (M≤4) | NPU per-submit overhead |
| Shape | N/A (compile-time) | Resolved at parse | No runtime cost |
| Slice | Partial (DMA stride RE needed) | CPU numpy | DMA stride configuration |
| Gather | Impossible (data-dependent) | CPU numpy | Hardware limitation |
| MatMul | CNA descriptor (any shape) | CPU numpy (fallback) | NPU DMA bandwidth |
| GELU | NPU native Gelu template | CPU tanh-approx | NPU per-submit (small) |
| Softmax | NPU template (seq=64) | CPU (arbitrary) | CPU faster (small seq) |

---

## References

- [CNA Descriptor Programming Guide](CNA_DESCRIPTOR_GUIDE.md)
- [NPU Register Command Reference](NPU_REGCMD_REFERENCE.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Source: cna_matmul.c](../src/opennpu/pjrt_c/cna_matmul.c) — CNA + attention + layernorm
- [Source: lm_forward.c](../src/opennpu/pjrt_c/lm_forward.c) — GPT-2 forward pass
- [Source: onnx_runner.py](../src/opennpu/onnx_runner.py) — ONNX dispatch + lowerings