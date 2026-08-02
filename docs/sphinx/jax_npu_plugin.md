# OpenNPU JAX Integration — PJRT C Plugin Deep Dive

How JAX/OpenXLA talks to the RK3588 NPU through a hand-written PJRT C plugin
that executes via raw DRM ioctls — no proprietary NPU libraries, no vendor toolkits, no closed
binary blobs.

## How JAX Finds the Plugin

JAX discovers PJRT plugins via an environment variable:

```bash
export JAX_PLATFORMS=npu,cpu
export PJRT_NAMES_AND_LIBRARY_PATHS=npu:/path/to/libpjrt_npu.so
```

```python
import jax
print(jax.devices())  # [npu:0]
```

JAX's `jaxlib` dlopens the `.so`, calls `GetPjrtApi()`, and reads the returned
`PJRT_Api` struct. The plugin must export:

```c
// Symbol that jaxlib dlsym's:
PJRT_Api* GetPjrtApi(void);
```

## PJRT API Version Pinning

The most critical detail: **the PJRT API version must match jaxlib exactly**.

jaxlib 0.10.2 was built against XLA commit
`5a9e73cbd92530cac2ac36f4736a774b2412afe2`, which uses PJRT API **v0.112**.
The `PJRT_Api` struct has 138 fields totaling 1144 bytes. If the plugin
declares a different version, jaxlib reads fields at the wrong offsets →
bad_alloc / segfault.

```c
// pjrt_c_api.h — pinned to v0.112
#define PJRT_API_MAJOR_VERSION 0
#define PJRT_API_MINOR_VERSION 112  // NOT 114 (openxla/main)
```

The header `pjrt_c_api.h` is a **frozen copy** of the exact header jaxlib 0.10.2
was compiled with. It must never be updated without recompiling jaxlib.

## Plugin Initialization (`Client_Create`)

When JAX starts, it calls `Client_Create`:

```c
static PJRT_Error* fn_Client_Create(void* a) {
    PJRT_Client_Create_Args* args = (PJRT_Client_Create_Args*)a;
    
    // 1. Open the NPU device
    NpuClient* c = calloc(1, sizeof(NpuClient));
    c->fd = open("/dev/dri/card1", O_RDWR);
    
    // 2. Init NPU (ACTION ioctls: HW version, nice, IOMMU)
    npu_action(c->fd, 0, 0xFFFFFFFF);  // GET_HW_VERSION
    npu_action(c->fd, 19, 0xFFFFFFED); // SET_NICE(-19)
    
    // 3. Pre-load GPT-2 weights if NPU_WEIGHTS_DIR is set
    const char* wdir = getenv("NPU_WEIGHTS_DIR");
    if (wdir) {
        int nwpre = atoi(getenv("NPU_NWPRE") ?: "144");
        npu_mm_cache_setup(nwpre, 64, 768, 768);  // M=64, K=768, N=768
        for (int i = 0; i < nwpre; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/w%03d.fp16", wdir, i);
            FILE* f = fopen(path, "rb");
            if (f) {
                void* buf = malloc(768*768*2);  // 1.1MB fp16
                if (fread(buf, 1, 768*768*2, f) == 768*768*2)
                    npu_mm_cache_load(i, buf);
                free(buf);
                fclose(f);
            }
        }
    }
    
    args->client = &g_client;
    return NULL;  // success
}
```

Key things that must work:
- `Client_Devices` / `AddressableDevices` must return an **array of pointers**
  (`PJRT_Device* arr[1] = {&g_dev}`), not `&g_dev` itself
- `Device_IsAddressable` must set `is_addressable = 1`
- `Buffer_Device` must return `&g_dev`

## Compilation (`Client_Compile`)

When JAX jit-compiles a function, it calls `Client_Compile` with the stablehlo
bytecode:

```c
static PJRT_Error* fn_Client_Compile(void* a) {
    PJRT_Client_Compile_Args* args = (PJRT_Client_Compile_Args*)a;
    
    // The HLO program is in args->program (a serialized MLIR bytecode)
    NpuExecutable* ex = calloc(1, sizeof(NpuExecutable));
    
    // Parse the stablehlo bytecode to extract the op chain
    parse_hlo_bytecode(args->program.data, args->program.size, ex);
    
    // ex->op is now: OP_ADD, OP_MUL, OP_MATMUL, OP_RELU, OP_TANH, etc.
    // ex->is_dag = 1 if multi-op, 0 if single-op
    // ex->w_idx = weight index for npu_cached_mm custom ops
    
    args->executable = (PJRT_LoadedExecutable*)ex;
    return NULL;
}
```

## Stablehlo Bytecode Parser

The plugin includes a **hand-written MLIR stablehlo bytecode parser** (no MLIR
library dependency). It parses bytecode format v6:

```
Bytecode structure:
  [4-byte magic "ML\xR0"]
  [producer info string]
  [version (varint)]
  [string table section]
  [attribute dictionary section]
  [operation list section]
  [region section]
```

The parser extracts:
1. **Op names** from the string table: `stablehlo.add`, `stablehlo.mul`,
   `stablehlo.tanh`, `stablehlo.custom_call @npu_cached_mm`
2. **Constant operands** from DenseElementsAttr (fp16 values)
3. **Operand relationships** to build the DAG

```c
// Simplified parse logic
static int parse_hlo_bytecode(const uint8_t* bc, size_t sz, NpuExecutable* ex) {
    // 1. Read string table
    // 2. Find operation entries
    // 3. Match against known op names:
    if (HAS("add", 3))       ex->op = OP_ADD;
    if (HAS("subtract", 8))  ex->op = OP_SUB;
    if (HAS("multiply", 8))  ex->op = OP_MUL;
    if (HAS("divide", 6))    ex->op = OP_DIV;
    if (HAS("tanh", 4))      ex->op = OP_TANH;
    if (HAS("npu_cached_mm", 13)) ex->op = OP_MATMUL_CACHED;
    // ...
    
    // 4. If multiple ops found, build DAG (topological sort)
    // 5. Extract constant operands (DenseElementsAttr)
}
```

### Supported HLO Ops

| JAX Code | Stablehlo Op | Plugin Op | NPU Execution |
|---|---|---|---|
| `a + b` | `stablehlo.add` | OP_ADD | Template (addtmpl.h) |
| `a - b` | `stablehlo.subtract` | OP_SUB | Template (ALU patched) |
| `a * b` | `stablehlo.multiply` | OP_MUL | Template (ALU patched) |
| `a / b` | `stablehlo.divide` | OP_DIV | Template (ALU patched) |
| `jnp.maximum(x, 0)` | `stablehlo.max` | OP_MAX | Template (relu) |
| `jnp.tanh(x)` | `stablehlo.tanh` | OP_TANH | Template (tanh_tmpl.h) |
| `npu_cached_mm(x, w_idx)` | `stablehlo.custom_call` | OP_MATMUL_CACHED | Matmul template + cached W |

### Unsupported → Fail Loudly

```c
// If no known op found:
return make_err("unsupported HLO: no matching NPU op");
```

The plugin NEVER silently falls back to CPU. Unsupported programs fail with a
clear error message.

## Execution (`LoadedExecutable_Execute`)

```c
static PJRT_Error* fn_LoadedExecutable_Execute(void* a) {
    PJRT_LoadedExecutable_Execute_Args* args = ...;
    NpuExecutable* ex = (NpuExecutable*)args->executable;
    NpuBuffer** inputs = (NpuBuffer**)args->args;
    NpuBuffer** outputs = (NpuBuffer**)args->outputs;
    
    switch (ex->op) {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV: {
        // Elementwise: load add template, patch ALU + DMAs, submit
        uint8_t regcmd[REGCMD_SZ];
        memcpy(regcmd, add_template, sizeof(add_template));
        patch_alu(regcmd, ex->op);    // Set ALU selector for op
        patch_dma(regcmd, inputs[0]->dma, inputs[1]->dma, outputs[0]->dma);
        
        // Create task + regcmd BOs, write, submit, read output
        npu_submit_elementwise(c->fd, regcmd, inputs, outputs);
        break;
    }
    
    case OP_MATMUL_CACHED: {
        // Matmul: use cached weight, no W upload
        uint16_t* Zh = malloc(M * N * 2);
        npu_mm_cache_run(ex->w_idx, inputs[0]->bo.mm, Zh);
        
        // Copy result to output buffer
        memcpy(outputs[0]->bo.mm, Zh, M * N * 2);
        free(Zh);
        break;
    }
    
    case OP_DAG: {
        // Multi-op: execute each node in topological order
        for (int i = 0; i < ex->n_nodes; i++) {
            DagNode* node = &ex->nodes[i];
            // Execute node (elementwise or matmul)
            // Pass output to next node's input
        }
        break;
    }
    }
    
    return NULL;
}
```

## Buffer Management

### `Buffer_ToHostBuffer` (NPU → host)

```c
static PJRT_Error* fn_Buffer_ToHostBuffer(void* a) {
    PJRT_Buffer_ToHostBuffer_Args* args = ...;
    NpuBuffer* buf = (NpuBuffer*)args->src;
    
    // Sync BO from device (invalidate cache, ensure CPU sees NPU writes)
    npu_sync_bo(buf->fd, buf->obj_addr, buf->size, 2); // FROM_DEVICE
    
    // Copy from mmap'd BO to host buffer
    memcpy(args->dst, buf->mm, buf->size);
    
    // Must return a ready event (JAX awaits it)
    return NULL; // Event_OnReady fires immediately
}
```

### `Event_OnReady` (must fire immediately)

```c
static PJRT_Error* fn_Event_OnReady(void* a) {
    PJRT_Event_OnReady_Args* args = ...;
    // Our events are always ready (synchronous execution)
    args->event->on_ready_callback(args->event, args->user_data, NULL);
    return NULL;
}
```

If `Event_OnReady` doesn't call the callback, JAX hangs forever.

## Matmul Execution (`npu_mm_cache_*`)

The matmul path is the most complex. It uses cached weights and a dedicated
NPU fd:

### Setup (`npu_mm_cache_setup`)

```c
int npu_mm_cache_setup(int n_w, int M, int K, int N) {
    // Open a dedicated NPU fd for matmul (avoids IOVA conflicts with elementwise)
    s_wc_fd = open("/dev/dri/card1", O_RDWR);
    
    // Allocate BOs: task, regcmd, scratch, X (input), Z (output)
    npu_create_bo(s_wc_fd, 4096, 0x40b, &s_wc_task_bo);
    npu_create_bo(s_wc_fd, 28672, 0x403, &s_wc_regcmd_bo);
    npu_create_bo(s_wc_fd, SCRATCH_SIZE, 0x403, &s_wc_scratch_bo);
    npu_create_bo(s_wc_fd, M*K*2, 0x403, &s_wc_x_bo);
    npu_create_bo(s_wc_fd, M*N*2, 0x403, &s_wc_z_bo);
    
    // Write the matmul regcmd template (768×768 shape)
    memcpy(s_wc_regcmd_bo.mm, mm_768x768_template, sizeof(mm_768x768_template));
    
    // Partially zero the scratch (only 64KB needed, not full 3.5MB)
    memset(s_wc_scratch_bo.mm, 0, 65536);
    
    // Setup NPU task structs
    setup_matmul_tasks(&s_wc_task_bo, &s_wc_regcmd_bo, &s_wc_scratch_bo,
                       &s_wc_x_bo, &s_wc_z_bo);
    
    // Pre-allocate weight BO slots
    s_wc_nw = n_w;
    s_wc_w_bos = calloc(n_w, sizeof(NpuBo));
    for (int i = 0; i < n_w; i++) {
        npu_create_bo(s_wc_fd, K*N*2, 0x403, &s_wc_w_bos[i]);
    }
    
    return 2; // returns n_io1 (number of I/O BOs besides X)
}
```

### Weight Loading (`npu_mm_cache_load`)

```c
int npu_mm_cache_load(int w_idx, const void* Wh) {
    if (w_idx < 0 || w_idx >= s_wc_nw) return -1;
    
    // Copy W bytes into the pre-allocated weight BO
    memcpy(s_wc_w_bos[w_idx].mm, Wh, K * N * 2);
    npu_sync_bo(s_wc_fd, s_wc_w_bos[w_idx].obj_addr, K*N*2, 1); // TO_DEVICE
    
    // Patch the regcmd's W DMA entry to point to this weight's IOVA
    patch_w_dma(s_wc_regcmd_bo.mm, s_wc_w_bos[w_idx].dma_addr);
    
    return 0;
}
```

### Execution (`npu_mm_cache_run`)

```c
int npu_mm_cache_run(int w_idx, const void* Xh, void* Zh) {
    // 1. Write X (input) into X BO
    memcpy(s_wc_x_bo.mm, Xh, M * K * 2);
    npu_sync_bo(s_wc_fd, s_wc_x_bo.obj_addr, M*K*2, 1);
    
    // 2. Patch regcmd W DMA to point to weight w_idx
    // (weights are already loaded and synced)
    patch_w_dma(s_wc_regcmd_bo.mm, s_wc_w_bos[w_idx].dma_addr);
    npu_sync_bo(s_wc_fd, s_wc_regcmd_bo.obj_addr, 28672, 1);
    
    // 3. Submit the matmul task chain (72 tasks, 1 submit)
    npu_submit(s_wc_fd, s_wc_task_bo.obj_addr, 72, flags=5, core_mask=1);
    
    // 4. Wait for completion (blocking — NPU DMA takes ~1.6ms)
    // The kernel's blocking submit waits for the IRQ
    
    // 5. Sync Z (output) from device
    npu_sync_bo(s_wc_fd, s_wc_z_bo.obj_addr, M*N*2, 2);
    memcpy(Zh, s_wc_z_bo.mm, M * N * 2);
    
    return 0;
}
```

### Fused Bias + Residual (`npu_mm_cache_run_bias_res`)

```c
int npu_mm_cache_run_bias_res(int w_idx, const void* Xh,
                                const float* bh, const float* resh,
                                float* Zh32) {
    // Run matmul (fp16 output)
    uint16_t* Zh_fp16 = ...;
    npu_mm_cache_run(w_idx, Xh, Zh_fp16);
    
    // Add bias + residual on CPU (fused, avoids extra NPU round-trip)
    for (int i = 0; i < M * N; i++) {
        float z = fp16_to_f32(Zh_fp16[i]);
        Zh32[i] = z + bh[i % N] + resh[i];
    }
    return 0;
}
```

This fusion is critical for GPT-2 — it eliminates 144 separate bias-add NPU
submissions.

## Full LM Forward (`npu_lm_forward`)

The fastest path: one C call for the entire GPT-2 forward pass:

```python
import ctypes

lib = ctypes.CDLL("./libpjrt_npu.so")

# Load weights (one-time)
lib.npu_lm_load_params(b"/tmp/gpt2_w_params.bin")

# Forward pass (per token)
ids = ctypes.c_int * 64
input_ids = ids(*token_list)
logits = (ctypes.c_float * 50257)()

lib.npu_lm_forward(
    input_ids,  # token IDs
    64,         # sequence length
    logits      # output logits
)

# logits is now filled — argmax to get next token
next_token = np.argmax(np.array(logits[:]))
```

### What `npu_lm_forward` Does Internally

```
1. Embedding: wte[id] + wpe[pos]                    (CPU, fp32)
2. For each of 12 layers:
   a. LayerNorm1                                    (CPU, fp32)
   b. Q = NPU_matmul(h, w_q)                        (NPU, fp16)
   c. K = NPU_matmul(h, w_k)                        (NPU, fp16)
   d. V = NPU_matmul(h, w_v)                        (NPU, fp16)
   e. Q += q_bias, K += k_bias, V += v_bias         (CPU, fp32)
   f. Attention (causal, 12 heads, 64 dim)           (CPU, OpenMP 3-core)
   g. O = NPU_matmul(av, w_o)                       (NPU, fp16)
   h. h += O + o_bias                               (CPU, fused)
   i. LayerNorm2                                    (CPU, fp32)
   j. FFN with GELU overlap:
      - For 4 tiles b=0..3:
        fc[b] = NPU_matmul(h, w_fc[b])              (NPU, fp16)
        post fc_done[b] semaphore
        GELU worker: gelu[b] = gelu(fc[b] + fc_bias)  (CPU, core 5)
        wait gelu_done[b]
        pj[b] = NPU_matmul(gelu[b], w_pj[b])         (NPU, fp16)
      h += sum(pj) + pj_bias                        (CPU, fused)
3. LayerNorm_f                                      (CPU, fp32)
4. lm_head: last_token × vocab_matrix               (CPU, OpenMP 3-core)
5. Return logits
```

### Thread Layout

| Thread | Pinned to | Role |
|---|---|---|
| Main (NPU submit) | A76 cores 4,6,7 | NPU matmul submit + CPU orchestration |
| GELU worker | A76 core 5 | GELU computation (overlapped with NPU) |
| OpenMP workers | A76 cores 4,6,7 | Attention (12 heads) + lm_head (50257 vocab) |

### CPU Affinity (RK3588 Core Layout)

```
Cores 0-3: A55 (little, 1.8 GHz) — NOT used for LM
Cores 4-7: A76 (big, 2.25-2.35 GHz) — used for LM
  Core 4: Main thread (NPU submit)
  Core 5: GELU worker thread
  Core 6: OpenMP worker
  Core 7: OpenMP worker
```

CPU pinning uses raw `sched_setaffinity` with bitmask manipulation because
`CPU_ZERO`/`CPU_SET` macros are not available on this glibc:

```c
static void pin_cpu(int cpu) {
    cpu_set_t cs;
    memset(&cs, 0, sizeof(cs));
    unsigned char* p = (unsigned char*)&cs;
    p[cpu / 8] |= (1u << (cpu % 8));
    sched_setaffinity(0, sizeof(cs), &cs);
}
```

## JAX Custom Op (`npu_cached_mm`)

The Python side defines a JAX Primitive for the cached matmul:

```python
# src/opennpu/jax_npu.py
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import xla, mlir

npu_mm_p = core.Primitive("npu_cached_mm")

@jax.custom_op
def npu_cached_mm(x, w_idx):
    """Matmul on NPU with pre-loaded cached weight."""
    return jnp.matmul(x, np.ones((768, 768)))  # shape inference only

# Register with the PJRT plugin via stablehlo.custom_call
npu_mm_p.def_impl(lambda x, w_idx: x @ jnp.ones((768, 768)))
npu_mm_p.def_abstract_eval(lambda x, w_idx: core.ShapedArray(x.shape[:-1] + (768,), x.dtype))

# MLIR lowering → stablehlo.custom_call @npu_cached_mm
def npu_cached_mm_lowering(ctx, x, *, w_idx):
    return mlir.custom_call(
        "npu_cached_mm",
        result_types=[ctx.avals_out[0].dtype],
        operands=[x],
        backend_config=str(w_idx),
    )
```

The plugin recognizes `stablehlo.custom_call @npu_cached_mm` in the bytecode and
dispatches to `OP_MATMUL_CACHED` with the weight index from `backend_config`.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `JAX_PLATFORMS` | (unset) | Set to `npu,cpu` to enable NPU |
| `PJRT_NAMES_AND_LIBRARY_PATHS` | (unset) | `npu:/path/to/libpjrt_npu.so` |
| `NPU_WEIGHTS_DIR` | (unset) | Directory with `w000.fp16`..`w143.fp16` weight files |
| `NPU_NWPRE` | 144 | Number of pre-loaded weights |
| `NPU_ZSZ` | 3637248 | Scratch zero size (65536 for LM = partial zero) |
| `NPU_1COPY` | 0 | Set to 1 for single-copy matmul (no speedup, correct) |

## Performance Comparison: JAX vs Raw C

| Path | Time/token | Overhead |
|---|---|---|
| JAX `jax.jit(npu_cached_mm)` | 0.59s | JAX dispatch + buffer transfer |
| Raw ctypes `npu_lm_forward` | 0.25s | Direct C call, no Python per-token |

JAX adds ~0.34s/token of overhead (buffer transfer, dispatch, Python
orchestration). For maximum performance, use the raw C path.

The JAX path is useful for:
- Prototyping and debugging (JAX handles autodiff, vmap, etc.)
- Running arbitrary JAX functions (not just GPT-2)
- Demonstrating that the NPU works as a first-class JAX device

The raw C path is useful for:
- Maximum throughput (no Python overhead)
- Production inference (one ctypes call per token)
- Embedded deployment (no JAX dependency)