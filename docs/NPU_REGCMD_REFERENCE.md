# NPU Register Command (regcmd) Reference

The complete reference for the RK3588 NPU's register
command format, task chain structure, and DMA encoding.

## Entry Format (8 bytes)

Every regcmd entry is 8 bytes, little-endian:

```
Offset  Size  Field   Description
  0      2    reg      NPU register address (u16)
  2      2    val      Value to write (u16)
  4      4    tag      Tag: [core_id(16) | param(16)]
                      core_id: 0x1001 = CORE0, 0x2001 = CORE1
                      param: DMA high 16 bits or packed parameters
```

### DMA Address Encoding

DMA addresses are 32-bit but split across two fields:

```python
dma_addr = ((tag32 & 0xFFFF) << 16) | val16

# To patch a DMA address in a regcmd entry:
entry[2:4] = struct.pack('<H', dma_addr & 0xFFFF)        # val = low 16 bits
entry[4:8] = struct.pack('<I', (tag32 & 0xFFFF0000) |    # preserve tag_hi
                          ((dma_addr >> 16) & 0xFFFF))   # tag_lo = high 16 bits
```

## Block Structure

Regcmd data is organized into blocks of 80 entries (640 bytes), but only
69 entries are used (entries 0-68). Entry 69 is the chain pointer.

```
Block layout (640 bytes, 80 entries × 8 bytes):
  Entry 0-4:   Init sequence (op enable, mode)
  Entry 5:     DMA base address (patched at runtime)
  Entry 6-22:  DMA configuration (strides, sizes, mode)
  Entry 23:    ALU selector 1 (for compute blocks)
  Entry 24-27: ALU config
  Entry 28:    ALU selector 2
  Entry 29-62: Additional configuration
  Entry 63:    ALU mode selector
  Entry 64-68: Post-config
  Entry 69:    Chain pointer (reg=0x0010, cid=0x0101, val=next_offset)
               val=0 → chain end (last block)
               val>0 → offset of next block in BO[1]
```

## Task Struct (40 bytes, packed)

```c
struct rknpu_task {  // 40 bytes
    uint32_t flags;          // @0:  job flags (PC=1, NONBLOCK=2, PINGPONG=4)
    uint32_t op_idx;         // @4:  operation index (1=reader, 2=reader, 3=reader, 4=compute, 5=writer)
    uint32_t enable_mask;    // @8:  PC enable mask
    uint32_t int_mask;       // @12: interrupt mask
    uint32_t int_clear;      // @16: interrupt clear
    uint32_t int_status;     // @20: interrupt status (kernel-updated)
    uint32_t regcfg_amount;  // @24: number of regcmd entries for this task
    uint32_t regcfg_offset;  // @28: offset within regcmd BO
    uint64_t regcmd_addr;    // @32: DMA address of regcmd block in BO[1]
};
```

## Submit Struct (104 bytes)

```c
struct rknpu_submit {  // 104 bytes
    uint32_t flags;           // @0:  job flags
    uint32_t timeout;         // @4:  timeout in ms
    uint32_t task_start;      // @8:  first task index
    uint32_t task_number;     // @12: number of tasks
    uint32_t task_counter;    // @16: kernel-updated counter
    // ... (padding) ...
    uint64_t task_obj_addr;   // @24: DMA addr of task BO
    // ... (padding) ...
    uint64_t task_base_addr;  // @40: PC_DMA_BASE_ADDR (0 = default)
    // ... (padding) ...
    uint32_t core_mask;       // @56: bit 0=CORE0, 1=CORE1, 2=CORE2
    int32_t  fence_fd;        // @60: -1 = no fence
    struct {                  // @64: subcore task routing (5 subcores)
        uint32_t start;
        uint32_t num;
    } subcore_tasks[5];       // @64..103: 5 × 8 = 40 bytes
};
```

## Binary Elementwise (Add/Sub/Mul/Div)

### Task Chain: [2, 3, 4, 5] — 4 tasks, 1 submit

| Task | op_idx | Role | Scratch Region |
|------|--------|------|---------------|
| 0 | 2 | Reader 1 | DMA input1 → scratch @ 0x18000 |
| 1 | 3 | Reader 2 | DMA input2 → scratch @ 0x00000 |
| 2 | 4 | Compute | ALU: scratch[0x18000] OP scratch[0x00000] → scratch @ 0x30000 |
| 3 | 5 | Writer | DMA scratch @ 0x30000 → output BO |

### BO Layout

| BO | Size | Flags | Role |
|----|------|-------|------|
| BO[0] | 4096 | 0x40b | Task structs (4 × 40 bytes) |
| BO[1] | 8192 | 0x403 | Regcmd data (4 blocks × 640 bytes) |
| BO[2] | 393216 | 0x403 | Scratch (NPU internal memory) |
| BO[3] | 98304 | 0x403 | Input 1 (64 × 768 × 2 bytes = 98304) |
| BO[4] | 98304 | 0x403 | Input 2 |
| BO[5] | 98304 | 0x403 | Output |

### ALU Selectors (Compute Block, op_idx=4)

Located at entries 23, 28, 63 of the compute block (byte offset 0xC0 + 0x500):

| Op | e23_val | e23_tag_lo | e28_tag_lo | e63_val | ALU Mode |
|----|---------|------------|------------|---------|----------|
| add | 0x02c0 | 0x1082 | 0x0001 | 0x7849 | FP16 add |
| sub | 0x02c0 | 0x1084 | 0x0001 | 0x7849 | FP16 sub |
| mul | 0x03c4 | 0x1080 | 0x0001 | 0x7849 | FP16 mul |
| div | 0x03c0 | 0x1083 | 0x0000 | 0x7841 | FP16 div |

Entry format: `reg(u16) + val(u16) + tag(u32)`. To patch:

```c
#define SETALU(entry, val, tag) do { \
    (entry)[2] = (val) & 0xFF;       \
    (entry)[3] = ((val) >> 8) & 0xFF; \
    (entry)[4] = (tag) & 0xFF;       \
    (entry)[5] = ((tag) >> 8) & 0xFF; \
    (entry)[6] = ((tag) >> 16) & 0xFF; \
    (entry)[7] = ((tag) >> 24) & 0xFF; \
} while(0)

// Patch add → mul:
SETALU(&regcmd[COMPUTE_OFFSET + 23*8], 0x03c4, 0x1001<<16 | 0x1080);
SETALU(&regcmd[COMPUTE_OFFSET + 63*8], 0x7849, 0x1001<<16);
```

### DMA Entry Locations

| Block | Entry 5 | Points to |
|-------|---------|-----------|
| Reader 1 (op=2) | DMA of BO[3] | Input 1 |
| Reader 2 (op=3) | DMA of BO[4] | Input 2 |
| Writer (op=5) | DMA of BO[5] | Output |

Patch these at runtime to actual BO IOVAs.

## Unary Activation (ReLU/Sigmoid/Tanh/Clip/LeakyReLU)

### Task Chain: [1, 2, 3, 1, 2, 3, 2, 1, 2] — 9 tasks, 1 submit (ReLU/Clip/LeakyReLU)

| Task | op_idx | Role |
|------|--------|------|
| 0 | 1 | Reader: DMA input → scratch @ 0x30000 |
| 1 | 2 | Compute: ALU transform in-place |
| 2 | 3 | Writer: DMA scratch @ 0x30000 → output |
| 3-8 | (repeat for multi-pass) | Additional compute passes |

### Sigmoid/Tanh: 3 submits (3+6+3 tasks)

Sigmoid and tanh require more computation passes, so they're split across
3 submits with different core routing:

| Submit | Tasks | Subcore routing |
|--------|-------|----------------|
| 0 | 3 | subcore 0 |
| 1 | 6 | subcore 1 |
| 2 | 3 | subcore 3 |

### BO Layout (ReLU)

| BO | Size | Flags | Role |
|----|------|-------|------|
| BO[0] | 4096 | 0x40b | Task structs |
| BO[1] | 8192 | 0x403 | Regcmd data |
| BO[2] | 294912 | 0x403 | Scratch |
| BO[3] | 98304 | 0x403 | Input |
| BO[4] | 98304 | 0x403 | Output |

### Activation ALU Selectors

| Op | e23_val | e23_tag_lo | e63_val |
|----|---------|------------|---------|
| ReLU | (max(x, 0)) | — | — |
| Sigmoid | (1/(1+exp(-x))) | — | — |
| Tanh | (tanh(x)) | — | — |
| Clip | (clip(x, min, max)) | — | — |
| LeakyReLU | (max(x, α*x)) | — | — |

The activation selector is in the compute block at a different offset than
binary ops. The exact values are baked into the captured templates
(`relu_tmpl.h`, `tanh_tmpl.h`).

## MatMul

### Task Chain: 72 tasks, 1 submit

The matmul is a single 28KB regcmd block (position-locked in BO[1]).
The essential header (0..0x40c0, 16.6KB) configures the NPU's matrix engine.

### BO Layout (768×768)

| BO | Size | Flags | Role |
|----|------|-------|------|
| BO[0] | 4096 | 0x40b | Task structs (72 × 40 bytes) |
| BO[1] | 28672 | 0x403 | Regcmd data (28KB block) |
| BO[2] | 3637248 | 0x403 | Scratch (3.47MB) |
| BO[3] | 98304 | 0x403 | X (input, 64 × 768 × 2) |
| BO[4] | 1179648 | 0x403 | W (weight, 768 × 768 × 2 = 1.1MB) |
| BO[5] | 98304 | 0x403 | Z (output, 64 × 768 × 2) |

### Regcmd Header Analysis

The header (0..0x40c0, 139 non-zero bytes) contains:
- ~135 bytes: fixed 0x3c pattern at regular intervals (NPU init sequence)
- 4 positions that encode matmul dimensions:
  - 0x08: M dimension (768)
  - 0x18: K dimension (3072 or 768)
  - 0x50: N dimension (64 or 768)
  - 0x98: total elements

No DMA burst/bus-width/throughput configuration exists in the header.

### W is Raw fp16 (No Weight Transform)

Unlike some NPU accelerators, the RK3588 NPU reads W directly as fp16 — no
reordering or quantization is needed. This simplifies the runtime: just write
W bytes into the weight BO and submit.

### Shape Switching

Each shape requires a different regcmd template. The 5 GPT-2 shapes are:
- mm_up: [1, 64, 768] × [768, 3072]
- mm_qkv: [1, 64, 768] × [768, 768]
- mm_down: [1, 64, 3072] × [3072, 768]
- mm_qkt: [1, 64, 768] × [768, 64]
- mm_atv: [1, 64, 64] × [64, 768]

All are decomposed to the 768×768 template (mm_qkv) to eliminate shape switches.
The larger matmuls (up/down) are tiled into 4 × 768×768 blocks.

## Subcore Routing

Tasks are routed to NPU subcores via the `subcore_tasks` field in the submit
struct:

```python
# Single-core (all tasks on subcore 0):
subcore_tasks = [(0, n_tasks), (0, 0), (0, 0), (0, 0), (0, 0)]

# Data-parallel 3-core (4 tasks per core for elementwise):
subcore_tasks = [(0, 4), (4, 4), (8, 4), (0, 0), (0, 0)]
# core_mask = 7 (all 3 cores)
```

The kernel dispatches one `rknpu_job_subcore_commit` per set bit in `core_mask`
and waits on the lowest-numbered core.

## DMA Patching at Runtime

When BOs are allocated at different IOVAs than the original capture, all DMA
entries must be patched. The general algorithm:

```python
def patch_dmas(regcmd_data, task_data, orig_dmas, orig_sizes, new_dmas):
    # 1. Patch regcmd_addr in task structs (points into BO[1])
    for ti in range(n_tasks):
        off = ti * 40 + 32  # regcmd_addr field
        orig_addr = struct.unpack_from('<Q', task_data, off)[0]
        for bi in range(len(orig_dmas)):
            if orig_dmas[bi] <= orig_addr < orig_dmas[bi] + orig_sizes[bi]:
                new_addr = new_dmas[bi] + (orig_addr - orig_dmas[bi])
                struct.pack_into('<Q', task_data, off, new_addr)
                break
    
    # 2. Patch embedded DMAs in regcmd data (scan all entries)
    for off in range(0, len(regcmd_data) - 8, 8):
        val16 = struct.unpack_from('<H', regcmd_data, off + 2)[0]
        tag32 = struct.unpack_from('<I', regcmd_data, off + 4)[0]
        dma = ((tag32 & 0xFFFF) << 16) | val16
        for bi in range(len(orig_dmas)):
            if orig_dmas[bi] <= dma < orig_dmas[bi] + orig_sizes[bi]:
                new_dma = new_dmas[bi] + (dma - orig_dmas[bi])
                struct.pack_into('<H', regcmd_data, off + 2, new_dma & 0xFFFF)
                new_tag = (tag32 & 0xFFFF0000) | ((new_dma >> 16) & 0xFFFF)
                struct.pack_into('<I', regcmd_data, off + 4, new_tag)
                break
```

## NPU Register Map

### Per-Core Registers (base + offset)

| Register | Offset | Description |
|----------|--------|-------------|
| VERSION | 0x000 | Hardware version ("FIRE" = 0x46495245) |
| PC_OP_EN | 0x008 | PC operation enable |
| PC_DATA_ADDR | 0x010 | PC data address |
| PC_DATA_AMOUNT | 0x014 | PC data amount |
| INT_MASK | 0x020 | Interrupt mask |
| INT_CLEAR | 0x024 | Interrupt clear |
| INT_STATUS | 0x028 | Interrupt status |
| INT_RAW_STATUS | 0x02c | Raw interrupt status |
| PC_TASK_CONTROL | 0x030 | Task control |
| PC_DMA_BASE_ADDR | 0x034 | DMA base address |
| ENABLE_MASK | 0xf008 | Enable mask |

### Undocumented Registers

| Register | Offset | RK3588 | RK3576 |
|----------|--------|--------|--------|
| Core routing | 0x1004 | 0x0e (bits 1-3, writable) | 0x1e (bits 1-4, writable) |
| State init | 0x1024 | 0x003f0300 (NPU-updated) | 0x80000000 (driver writes) |
| Internal mem enable | bit 4 of 0x1004 | NOT writable | Writable (0x1e) |

### Core Base Addresses

| Core | Base | Size |
|------|------|------|
| CORE0 | 0xfdab0000 | 64KB |
| CORE1 | 0xfdac0000 | 64KB |
| CORE2 | 0xfdad0000 | 64KB |

## ACTION Ioctl (0xC0086440)

```c
struct rknpu_action {
    uint32_t flags;  // action ID
    uint32_t value;  // action parameter
};
```

| Action | ID | Description |
|--------|-----|-------------|
| GET_HW_VERSION | 0 | Returns "FIRE" (0x46495245) |
| GET_DRV_VERSION | 1 | Driver version |
| GET_FREQ | 2 | NPU clock (1000 MHz) |
| SET_FREQ | 3 | Set clock |
| GET_VOLT | 4 | Voltage (825 mV) |
| ACT_RESET | 6 | Soft reset (no SBC reboot) |
| GET_BW_PRIORITY | 7 | QoS priority |
| SET_BW_PRIORITY | 8 | Set QoS priority |
| CLR_RW_AMOUNT | 13 | Clear DMA counters |
| GET_IOMMU_EN | 18 | IOMMU enable status |
| SET_NICE | 19 | CPU nice priority |
| POWER_ON | 20 | Power on NPU |
| POWER_OFF | 21 | Power off NPU |
| GET_SRAM_TOTAL | 22 | Total SRAM (0 on RK3588) |
| GET_IOMMU_DOMAIN_ID | 24 | Current IOMMU domain |
| SET_IOMMU_DOMAIN_ID | 25 | Switch domain (fresh IOVA pool) |

## Debugfs

```
/sys/kernel/debug/rknpu/
  power      — echo on/off
  freq       — NPU clock frequency
  reset      — echo 1 to reset
  load       — NPU core load (Core0/1/2 percentages)
  volt       — voltage
  delayms    — auto power-off delay (3000ms default; set to 999999 to disable)
  version    — driver/hardware version
```

Requires root access. Set `delayms` high before running to prevent the NPU
from powering off between inferences.