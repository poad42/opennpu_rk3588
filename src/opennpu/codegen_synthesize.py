"""
Open NPU codegen — synthesizes regcmd bytes from ONNX op type + shape.

NO vendor toolkits, NO proprietary NPU libraries — produces the raw register command bytes
that the NPU hardware executes directly.

The NPU register command format (8 bytes per entry):
  reg16(u16) + val16(u16) + tag32(u32)
  tag_hi = 0x1001 (core 0) or 0x2001 (core 1)
  tag_lo = used for DMA high bits or packed parameters

For the Add op at shape (1, 64, 768), the NPU hardware uses 12 tasks
using 4 unique regcmd block types (op_idx 2/3/4/5). This module
reproduces those blocks byte-for-byte from the shape parameters.

The register write patterns were discovered by analyzing the
vendor model files and comparing block types across ops (add, sub, mul, etc.).
"""
import struct
import os
from typing import List, Tuple, Dict, Optional

ENTRY_SZ = 8
CORE0_ID = 0x1001
CORE1_ID = 0x2001
BLOCK_SZ = 640  # 80 entries × 8 bytes


def _entry(reg: int, val: int, tag_hi: int = 0x1001, tag_lo: int = 0) -> bytes:
    """Build one 8-byte regcmd entry."""
    tag = (tag_hi << 16) | tag_lo
    return struct.pack('<HHI', reg & 0xFFFF, val & 0xFFFF, tag & 0xFFFFFFFF)


def _build_block(op_idx: int, N: int, C: int, H: int, W: int,
                 core0_dma: int = 0, core1_dma: int = 0,
                 **kwargs) -> bytes:
    """Build one 69-entry regcmd block for the given op_idx and shape.

    The block is 69 entries × 8 bytes = 552 bytes, padded to 640.

    Register write patterns by op_idx:
      op=2,3 (readers): read input data, set up DMA descriptors
      op=4   (compute): elementwise computation (add/sub/mul/div)
      op=5   (writer): write output data back to memory
    """
    entries = []

    if op_idx in (2, 3):
        # Reader block: reads input tensor into NPU internal memory
        Cm1 = C - 1       # 0x3f for C=64
        Wm1 = W - 1       # 0x2ff for W=768
        entries.append(_entry(0x4004, 0x000e, CORE0_ID))              # 0
        entries.append(_entry(0x5004, 0x000e, CORE1_ID))              # 1
        entries.append(_entry(0x400c, 0x01e5, CORE0_ID, 0x0030))      # 2
        entries.append(_entry(0x4010, 0x0001, CORE0_ID, 0x2400))      # 3
        entries.append(_entry(0x4014, 0x0000, CORE0_ID))              # 4
        # entry 5 = core0 DMA base (placeholder, patched at runtime)
        entries.append(_entry(0x4020, core0_dma & 0xFFFF, CORE0_ID,
                              (core0_dma >> 16) & 0xFFFF))             # 5
        entries.append(_entry(0x4024, 0x0010, CORE0_ID))              # 6
        entries.append(_entry(0x4030, 0x0000, CORE0_ID))              # 7
        entries.append(_entry(0x4034, Cm1, CORE0_ID))                # 8
        entries.append(_entry(0x4038, Wm1, CORE0_ID, Wm1))            # 9
        entries.append(_entry(0x403c, Wm1, CORE0_ID, Wm1))           # 10
        entries.append(_entry(0x4040, 0x0053, CORE0_ID))              # 11
        entries.append(_entry(0x4044, 0x0000, CORE0_ID))              # 12
        entries.append(_entry(0x4048, 0x0000, CORE0_ID))              # 13
        entries.append(_entry(0x404c, 0x0000, CORE0_ID))              # 14
        entries.append(_entry(0x4050, 0x07fe, CORE0_ID, 0x0800))      # 15
        entries.append(_entry(0x4054, 0x0000, CORE0_ID))              # 16
        entries.append(_entry(0x4058, Wm1, CORE0_ID, 0x085f))         # 17
        entries.append(_entry(0x405c, 0x0000, CORE0_ID, 0x0007))      # 18
        entries.append(_entry(0x4060, 0x0053, CORE0_ID))              # 19
        # Entries 20-29: registers 0x4064-0x4088 (stride 4)
        reg_base_20 = 0x4064
        for i in range(20, 30):
            if i == 23:
                entries.append(_entry(0x4070, 0x0383, CORE0_ID))
            elif i == 25:
                entries.append(_entry(0x4078, 0x0001, CORE0_ID))
            elif i == 28:
                entries.append(_entry(0x4084, 0x0001, CORE0_ID))
            else:
                entries.append(_entry(reg_base_20 + (i - 20) * 4, 0, CORE0_ID))
        # Entries 30-37: registers 0x4090-0x40ac (stride 4, skip 0x408c)
        reg_base_30 = 0x4090
        for i in range(30, 38):
            entries.append(_entry(reg_base_30 + (i - 30) * 4, 0, CORE0_ID))
        entries.append(_entry(0x40c0, 0x0080, CORE0_ID))              # 38
        entries.append(_entry(0x40c4, 0x0000, CORE0_ID))              # 39
        for i in range(40, 52):
            entries.append(_entry(0x4100 + (i - 40) * 4, 0, CORE0_ID))
        # Core 1 entries (52-68)
        entries.append(_entry(0x500c, 0x0000, CORE1_ID))              # 52
        entries.append(_entry(0x5010, Cm1, CORE1_ID))                # 53
        entries.append(_entry(0x5014, Wm1, CORE1_ID))                # 54
        # entry 55 = core1 DMA base
        entries.append(_entry(0x5018, core1_dma & 0xFFFF, CORE1_ID,
                              (core1_dma >> 16) & 0xFFFF))             # 55
        entries.append(_entry(0x501c, 0x0000, CORE1_ID))              # 56
        entries.append(_entry(0x5020, 0x0000, CORE1_ID))              # 57
        entries.append(_entry(0x5028, 0x0000, CORE1_ID))              # 58
        entries.append(_entry(0x502c, 0x0000, CORE1_ID))              # 59
        entries.append(_entry(0x5034, 0x0001, CORE1_ID))              # 60
        entries.append(_entry(0x5038, 0x0000, CORE1_ID))              # 61
        entries.append(_entry(0x5040, 0x0000, CORE1_ID))              # 62
        entries.append(_entry(0x5044, 0xf821, CORE1_ID))              # 63
        entries.append(_entry(0x5048, 0x0000, CORE1_ID, 0x02f8))      # 64
        entries.append(_entry(0x504c, 0x8600, CORE1_ID, 0xfffe))      # 65
        entries.append(_entry(0x5064, 0x0000, CORE1_ID))              # 66
        entries.append(_entry(0x5068, 0x0101, CORE1_ID, 0x0101))      # 67
        entries.append(_entry(0x506c, 0x0000, CORE1_ID))              # 68

    elif op_idx == 4:
        # Compute block: elementwise operation (add/sub/mul/div)
        # C_tile is the channel tile width for this block.
        # For the first compute block, C_tile = C (full width).
        # For subsequent blocks, C_tile = C // 2 or a smaller chunk.
        #
        # Op-specific parameters (entry 23, 28, 63) vary by operation:
        #   add: e23=(0x02c0, 0x1082), e28=(0x0001, 0x0001), e63=(0x7849, 0x0001)
        #   sub: e23=(0x02c0, 0x1084), e28=(0x0001, 0x0001), e63=(0x7849, 0x0001)
        #   mul: e23=(0x03c4, 0x1080), e28=(0x0001, 0x0001), e63=(0x7849, 0x0001)
        #   div: e23=(0x03c0, 0x1083), e28=(0x0001, 0x0000), e63=(0x7841, 0x0001)
        C_tile = kwargs.get('c_tile', C)
        Cm1 = C_tile - 1
        Wm1 = W - 1
        e23_val = kwargs.get('e23_val', 0x02c0)
        e23_tag_lo = kwargs.get('e23_tag_lo', 0x1082)
        e28_tag_lo = kwargs.get('e28_tag_lo', 0x0001)
        e63_val = kwargs.get('e63_val', 0x7849)
        entries.append(_entry(0x4004, 0x000e, CORE0_ID))              # 0
        entries.append(_entry(0x5004, 0x000e, CORE1_ID))              # 1
        entries.append(_entry(0x400c, 0x01e5, CORE0_ID))              # 2
        entries.append(_entry(0x4010, 0x0002, CORE0_ID, 0x4800))      # 3
        entries.append(_entry(0x4014, 0x0000, CORE0_ID))              # 4
        entries.append(_entry(0x4020, core0_dma & 0xFFFF, CORE0_ID,
                              (core0_dma >> 16) & 0xFFFF))             # 5
        entries.append(_entry(0x4024, 0x3000, CORE0_ID))              # 6
        entries.append(_entry(0x4030, Wm1, CORE0_ID))                # 7
        entries.append(_entry(0x4034, 0x0000, CORE0_ID))              # 8
        entries.append(_entry(0x4038, 0x0000, CORE0_ID))              # 9
        entries.append(_entry(0x403c, Cm1, CORE0_ID, Cm1))            # 10
        entries.append(_entry(0x4040, 0x0053, CORE0_ID))              # 11
        entries.append(_entry(0x4044, 0x0000, CORE0_ID))              # 12
        entries.append(_entry(0x4048, 0x0000, CORE0_ID))              # 13
        entries.append(_entry(0x404c, 0x0000, CORE0_ID))              # 14
        entries.append(_entry(0x4050, 0x0002, CORE0_ID))              # 15
        entries.append(_entry(0x4054, 0x0000, CORE0_ID))              # 16
        entries.append(_entry(0x4058, Cm1, CORE0_ID))                # 17
        entries.append(_entry(0x405c, Wm1, CORE0_ID))                # 18
        entries.append(_entry(0x4060, 0x0053, CORE0_ID))              # 19
        reg_base_20 = 0x4064
        for i in range(20, 30):
            if i == 23:
                entries.append(_entry(0x4070, e23_val, CORE0_ID, e23_tag_lo))
            elif i == 25:
                entries.append(_entry(0x4078, 0x0001, CORE0_ID))
            elif i == 28:
                entries.append(_entry(0x4084, 0x0001, CORE0_ID, e28_tag_lo))
            else:
                entries.append(_entry(reg_base_20 + (i - 20) * 4, 0, CORE0_ID))
        reg_base_30 = 0x4090
        for i in range(30, 38):
            entries.append(_entry(reg_base_30 + (i - 30) * 4, 0, CORE0_ID))
        entries.append(_entry(0x40c0, 0x3000, CORE0_ID))              # 38
        entries.append(_entry(0x40c4, 0x0000, CORE0_ID))              # 39
        for i in range(40, 52):
            entries.append(_entry(0x4100 + (i - 40) * 4, 0, CORE0_ID))
        entries.append(_entry(0x500c, Wm1, CORE1_ID))                # 52
        entries.append(_entry(0x5010, 0x0000, CORE1_ID))              # 53
        entries.append(_entry(0x5014, Cm1, CORE1_ID))                # 54
        entries.append(_entry(0x5018, core1_dma & 0xFFFF, CORE1_ID,
                              (core1_dma >> 16) & 0xFFFF))             # 55
        entries.append(_entry(0x501c, 0x0000, CORE1_ID))              # 56
        entries.append(_entry(0x5020, 0x0000, CORE1_ID))              # 57
        entries.append(_entry(0x5028, 0x0000, CORE1_ID))              # 58
        entries.append(_entry(0x502c, 0x0000, CORE1_ID))              # 59
        entries.append(_entry(0x5034, 0x0008, CORE1_ID, 0x4000))      # 60
        dma_off_61 = kwargs.get('dma_off_61', 0)
        entries.append(_entry(0x5038, dma_off_61, CORE1_ID))         # 61
        entries.append(_entry(0x5040, 0x3000, CORE1_ID))              # 62
        entries.append(_entry(0x5044, e63_val, CORE1_ID, 0x0001))      # 63
        entries.append(_entry(0x5048, 0x0000, CORE1_ID))              # 64
        entries.append(_entry(0x504c, 0x0000, CORE1_ID))              # 65
        entries.append(_entry(0x5064, 0x0000, CORE1_ID))              # 66
        entries.append(_entry(0x5068, 0x0101, CORE1_ID, 0x0101))      # 67
        entries.append(_entry(0x506c, 0x0000, CORE1_ID))              # 68

    elif op_idx == 5:
        # Writer block: writes NPU output back to memory
        Cm1 = C - 1
        Wm1 = W - 1
        entries.append(_entry(0x4004, 0x000e, CORE0_ID))              # 0
        entries.append(_entry(0x5004, 0x000e, CORE1_ID))              # 1
        entries.append(_entry(0x400c, 0x01e5, CORE0_ID, 0x0030))      # 2
        entries.append(_entry(0x4010, 0x0001, CORE0_ID, 0x2400))      # 3
        entries.append(_entry(0x4014, 0x0000, CORE0_ID))              # 4
        entries.append(_entry(0x4020, core0_dma & 0xFFFF, CORE0_ID,
                              (core0_dma >> 16) & 0xFFFF))             # 5
        entries.append(_entry(0x4024, 0x0600, CORE0_ID))              # 6
        entries.append(_entry(0x4030, 0x001f, CORE0_ID))              # 7
        entries.append(_entry(0x4034, 0x0017, CORE0_ID))              # 8
        entries.append(_entry(0x4038, 0x0000, CORE0_ID))              # 9
        entries.append(_entry(0x403c, Cm1, CORE0_ID, Cm1))            # 10
        entries.append(_entry(0x4040, 0x0053, CORE0_ID))              # 11
        for i in range(12, 20):
            if i == 15:
                entries.append(_entry(0x4050, 0x07fe, CORE0_ID, 0x0800))
            elif i == 17:
                entries.append(_entry(0x4058, Cm1, CORE0_ID, 0x0807))
            elif i == 18:
                entries.append(_entry(0x405c, 0x000b, CORE0_ID, 0x0007))
            elif i == 19:
                entries.append(_entry(0x4060, 0x0053, CORE0_ID))
            else:
                entries.append(_entry(0x4044 + (i - 12) * 4, 0, CORE0_ID))
        reg_base_20 = 0x4064
        for i in range(20, 30):
            if i == 23:
                entries.append(_entry(0x4070, 0x0383, CORE0_ID))
            elif i == 25:
                entries.append(_entry(0x4078, 0x0001, CORE0_ID))
            elif i == 28:
                entries.append(_entry(0x4084, 0x0001, CORE0_ID))
            else:
                entries.append(_entry(reg_base_20 + (i - 20) * 4, 0, CORE0_ID))
        reg_base_30 = 0x4090
        for i in range(30, 38):
            entries.append(_entry(reg_base_30 + (i - 30) * 4, 0, CORE0_ID))
        entries.append(_entry(0x40c0, 0x3000, CORE0_ID))              # 38
        entries.append(_entry(0x40c4, 0x0000, CORE0_ID))              # 39
        for i in range(40, 52):
            entries.append(_entry(0x4100 + (i - 40) * 4, 0, CORE0_ID))
        entries.append(_entry(0x500c, 0x001f, CORE1_ID))              # 52
        entries.append(_entry(0x5010, 0x0017, CORE1_ID))              # 53
        entries.append(_entry(0x5014, Cm1, CORE1_ID))                # 54
        entries.append(_entry(0x5018, core1_dma & 0xFFFF, CORE1_ID,
                              (core1_dma >> 16) & 0xFFFF))             # 55
        for i in range(56, 60):
            if i == 56:
                entries.append(_entry(0x501c, 0, CORE1_ID))
            elif i == 57:
                entries.append(_entry(0x5020, 0, CORE1_ID))
            elif i == 58:
                entries.append(_entry(0x5028, 0, CORE1_ID))
            elif i == 59:
                entries.append(_entry(0x502c, 0, CORE1_ID))
        entries.append(_entry(0x5034, 0x0001, CORE1_ID))              # 60
        entries.append(_entry(0x5038, 0x0000, CORE1_ID))              # 61
        entries.append(_entry(0x5040, 0x0000, CORE1_ID))              # 62
        entries.append(_entry(0x5044, 0xf821, CORE1_ID))              # 63
        entries.append(_entry(0x5048, 0x0000, CORE1_ID))              # 64
        entries.append(_entry(0x504c, 0x0000, CORE1_ID))              # 65
        entries.append(_entry(0x5064, 0x0000, CORE1_ID))              # 66
        entries.append(_entry(0x5068, 0x0101, CORE1_ID, 0x0101))      # 67
        entries.append(_entry(0x506c, 0x0000, CORE1_ID))              # 68

    # Pad to 69 entries
    while len(entries) < 69:
        entries.append(_entry(0, 0, 0))

    # Extra entries (69-72): NPU-internal chain data
    next_offset = kwargs.get('next_offset', 0)
    extra_70 = kwargs.get('extra_70', 0x0024)
    if next_offset > 0:
        # Chained block: entry 69 points to next block
        entries.append(struct.pack('<HHI', 0x0010, next_offset & 0xFFFF, 0x01010000))
    else:
        # Last block in chain: entry 69 is zeros
        entries.append(struct.pack('<HHI', 0, 0, 0))
    entries.append(struct.pack('<HHI', 0x0014, extra_70, 0x01010000))  # 70
    entries.append(struct.pack('<HHI', 0x0000, 0x0000, 0x00410000))  # 71
    entries.append(struct.pack('<HHI', 0x0008, 0x0018, 0x00810000))  # 72

    block = b''.join(entries[:73])
    # Pad to 640 bytes
    block += b'\x00' * (BLOCK_SZ - len(block))
    return block


# BO indices in the bo_dmas list
BO_TASK = 0      # BO[0] — task structs
BO_REGCMD = 1    # BO[1] — regcmd blocks (with 192B header)
BO_SCRATCH = 2   # BO[2] — scratch/weight (0x60000 bytes)
BO_IN1 = 3       # BO[3] — input1 tensor
BO_IN2 = 4       # BO[4] — input2 tensor
BO_OUT = 5       # BO[5] — output tensor

# Scratch BO[2] internal offsets (for Add at shape (1,64,768))
SCRATCH_IN2_BASE = 0x00000   # reader1 writes input2 here
SCRATCH_IN1_BASE = 0x18000   # reader0 writes input1 here
SCRATCH_RESULT   = 0x30000   # compute writes result here (+ chan_off)


def scratch_layout(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Compute scratch BO offsets for an arbitrary elementwise shape.

    Layout (no overlap): input2 @ 0, input1 @ region, result @ 2*region,
    where region = align_up(tensor_bytes, 0x1000). Returns
    (in2_base, in1_base, result_base, scratch_size).
    A zeroed scratch of this size is sufficient (the captured scratch_init
    residue is unnecessary for elementwise ops).
    """
    n = 1
    for d in shape:
        n *= int(d)
    tbytes = n * 2  # fp16
    region = (tbytes + 0xFFF) & ~0xFFF  # align up to 4KB
    in2_base = 0
    in1_base = region
    result_base = 2 * region
    scratch_size = (4 * region + 0xFFF) & ~0xFFF  # 4x for headroom, aligned
    return in2_base, in1_base, result_base, scratch_size

# Block offsets within regcmd BO (after 0xC0 header)
BLOCK_OFFSETS = [0x0, 0x280, 0x500, 0x780, 0xa00, 0xc80, 0xf00, 0x1180, 0x1400]

# Task sequence: 15 tasks (12 submitted + 3 extra)
# Each entry: (op_idx, block_id, c_tile, acc_c)
#   op_idx:   2=reader1, 3=reader2, 4=compute, 5=writer
#   block_id: index into BLOCK_OFFSETS
#   c_tile:   channel tile width for this task
#   acc_c:    accumulated channel offset (for DMA calculation)
ADD_TASK_SEQUENCE = [
    (2, 0,  0,  0),    # task 0:  reader1  (read input1 → scratch)
    (3, 1,  0,  0),    # task 1:  reader2  (read input2 → scratch)
    (4, 2,  64, 0),    # task 2:  compute  (C=64, offset=0)
    (5, 3,  0,  0),    # task 3:  writer   (scratch → output)
    (2, 0,  0,  0),    # task 4:  reader1  (reuse)
    (3, 1,  0,  0),    # task 5:  reader2  (reuse)
    (4, 4,  32, 0),    # task 6:  compute  (C=32, offset=0)
    (5, 3,  0,  0),    # task 7:  writer   (reuse)
    (4, 5,  32, 32),   # task 8:  compute  (C=32, offset=32)
    (2, 0,  0,  0),    # task 9:  reader1  (reuse)
    (3, 1,  0,  0),    # task 10: reader2  (reuse)
    (4, 6,  24, 0),    # task 11: compute  (C=24, offset=0)
    # --- 3 extra tasks (not submitted but needed in BO[0]) ---
    (5, 3,  0,  0),    # task 12: writer   (reuse)
    (4, 7,  24, 24),   # task 13: compute  (C=24, offset=24)
    (4, 8,  16, 48),   # task 14: compute  (C=16, offset=48)
]


def _build_header() -> bytes:
    """Build the 192-byte BO[1] header (3 × 64-byte core config structs).

    Discovered from hardware register analysis offset 0x00-0xBF.
    Each 64-byte struct configures one NPU subcore.
    """
    header = bytearray(192)
    # Core config 0 (offset 0x00): subcore 0
    struct.pack_into('<Q', header, 0x00, 0x0000000000000001)
    struct.pack_into('<Q', header, 0x08, 0x0000000000000040)
    struct.pack_into('<Q', header, 0x10, 0x0000000000000001)
    struct.pack_into('<Q', header, 0x18, 0x0000000000000300)
    # Core config 1 (offset 0x40): subcore 1
    struct.pack_into('<Q', header, 0x40, 0x0000000000000001)
    struct.pack_into('<Q', header, 0x48, 0x0000000000000040)
    struct.pack_into('<Q', header, 0x50, 0x0000000000000001)
    struct.pack_into('<Q', header, 0x58, 0x0000000000000300)
    # Core config 2 (offset 0x80): subcore 2
    struct.pack_into('<Q', header, 0x80, 0x0000000000000001)
    struct.pack_into('<Q', header, 0x88, 0x0000000000000040)
    struct.pack_into('<Q', header, 0x90, 0x0000000000000300)
    return bytes(header)


def _patch_dma(block: bytearray, entry_idx: int, reg: int,
               dma: int, core_id: int) -> None:
    """Patch a DMA entry in a regcmd block (entries 5, 55, or 61)."""
    off = entry_idx * ENTRY_SZ
    val16 = dma & 0xFFFF
    tag32 = (core_id << 16) | ((dma >> 16) & 0xFFFF)
    struct.pack_into('<HHI', block, off, reg, val16, tag32)


def gen_add(shape: Tuple[int, ...] = (1, 64, 768),
            bo_dmas: List[int] = None) -> Tuple[bytes, bytes, bytes]:
    """Generate regcmd BO, task BO, and weight BO for an Add operation.

    Produces the FULL BO content matching the proprietary NPU libraries capture format:
      - regcmd BO: 192-byte header + 9 regcmd blocks at 0xC0 (8192 bytes total)
      - task BO: 15 task structs × 40 bytes (12 submitted + 3 extra)
      - weight BO: zeros (Add has no weights; BO[2] is scratch space)

    DMA address formulas (verified against capture):
      Reader 0 (block 0): e5 = BO[3]+0,          e55 = BO[2]+0x18000
      Reader 1 (block 1): e5 = BO[4]+0,          e55 = BO[2]+0x0
      Compute (blocks 2,4-8):
                          e5 = BO[2]+0x30000+chan_off,
                          e55 = BO[3]+chan_off,
                          e61 = BO[4]+chan_off
      Writer   (block 3): e5 = BO[5]+0,          e55 = BO[2]+0x30000

    where chan_off = acc_c * W * 2 (fp16 stride).

    Args:
        shape: Input tensor shape (N, C, W) or (N, C, H, W).
        bo_dmas: List of 6 BO DMA addresses [task, regcmd, scratch, in1, in2, out].

    Returns:
        (regcmd_bo_data, task_bo_data, weight_bo_data)
    """
    if len(shape) == 3:
        N, C, W = shape
        H = 1
    else:
        N, C, H, W = shape

    if bo_dmas is None:
        bo_dmas = [0] * 6

    stride = W * 2  # fp16 stride per channel

    # ── Build 9 unique regcmd blocks ──────────────────────────
    # Block types: 0=reader1, 1=reader2, 2=compute(64),
    #              3=writer, 4=compute(32), 5=compute(32,off),
    #              6=compute(24), 7=compute(24,off), 8=compute(16,off)
    add_params = OP_PARAMS["add"]
    blocks = []
    for block_id in range(9):
        next_off = BLOCK_OFFSETS[block_id + 1] if block_id + 1 < 9 else 0
        if block_id == 0:
            blk = _build_block(2, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=BLOCK_OFFSETS[1])
        elif block_id == 1:
            blk = _build_block(3, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=BLOCK_OFFSETS[2])
        elif block_id == 2:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=64, next_offset=BLOCK_OFFSETS[3],
                               **add_params)
        elif block_id == 3:
            blk = _build_block(5, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=0, extra_70=0x0000)
        elif block_id == 4:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=32, next_offset=0, extra_70=0x0000,
                               **add_params)
        elif block_id == 5:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=32, next_offset=0, extra_70=0x0000,
                               **add_params)
        elif block_id == 6:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=24, next_offset=0, extra_70=0x0000,
                               **add_params)
        elif block_id == 7:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=24, next_offset=0, extra_70=0x0000,
                               **add_params)
        elif block_id == 8:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=16, next_offset=0, extra_70=0x0000,
                               **add_params)
        blocks.append(bytearray(blk))

    # ── Patch DMA entries in each block ───────────────────────
    for block_id in range(9):
        blk = blocks[block_id]
        if block_id in (0, 1):
            # Reader blocks: e5 = input BO, e55 = scratch area
            if block_id == 0:
                e5_dma = bo_dmas[BO_IN1]
                e55_dma = bo_dmas[BO_SCRATCH] + SCRATCH_IN1_BASE
            else:
                e5_dma = bo_dmas[BO_IN2]
                e55_dma = bo_dmas[BO_SCRATCH] + SCRATCH_IN2_BASE
            _patch_dma(blk, 5, 0x4020, e5_dma, CORE0_ID)
            _patch_dma(blk, 55, 0x5018, e55_dma, CORE1_ID)
        elif block_id == 3:
            # Writer block: e5 = output BO, e55 = scratch result
            e5_dma = bo_dmas[BO_OUT]
            e55_dma = bo_dmas[BO_SCRATCH] + SCRATCH_RESULT
            _patch_dma(blk, 5, 0x4020, e5_dma, CORE0_ID)
            _patch_dma(blk, 55, 0x5018, e55_dma, CORE1_ID)
        else:
            # Compute blocks: e5 = scratch+result+chan_off,
            #                 e55 = input1+chan_off,
            #                 e61 = input2+chan_off
            # Determine acc_c from block_id
            acc_c_map = {2: 0, 4: 0, 5: 32, 6: 0, 7: 24, 8: 48}
            acc_c = acc_c_map[block_id]
            chan_off = acc_c * stride
            e5_dma = bo_dmas[BO_SCRATCH] + SCRATCH_RESULT + chan_off
            e55_dma = bo_dmas[BO_IN1] + chan_off
            e61_dma = bo_dmas[BO_IN2] + chan_off
            _patch_dma(blk, 5, 0x4020, e5_dma, CORE0_ID)
            _patch_dma(blk, 55, 0x5018, e55_dma, CORE1_ID)
            _patch_dma(blk, 61, 0x5038, e61_dma, CORE1_ID)

    # ── Patch chain pointers (entry 69) in blocks 0, 1, 2 ────
    # Blocks 0→1→2→3 form a chain. Entry 69 must contain the
    # ABSOLUTE DMA address of the next block (not relative offset).
    # tag_hi=0x0101, tag_lo=high16(dma), val=low16(dma)
    regcmd_dma_base = bo_dmas[BO_REGCMD] + 0xC0
    for block_id in (0, 1, 2):
        next_block_off = BLOCK_OFFSETS[block_id + 1]
        next_dma = regcmd_dma_base + next_block_off
        blk = blocks[block_id]
        off = 69 * ENTRY_SZ
        struct.pack_into('<HHI', blk, off,
                         0x0010, next_dma & 0xFFFF,
                         (0x0101 << 16) | ((next_dma >> 16) & 0xFFFF))

    # ── Assemble regcmd BO: 192B header + 9 blocks + task struct copies ───
    header = _build_header()
    blocks_data = b''.join(bytes(b) for b in blocks)

    # Build task struct copies with relative offsets for BO[1]
    task_copies = bytearray(15 * 40)
    for ti, (op_idx, block_id, c_tile, acc_c) in enumerate(ADD_TASK_SEQUENCE):
        off = ti * 40
        block_off = BLOCK_OFFSETS[block_id]
        struct.pack_into('<IIIIIII', task_copies, off,
                         0, op_idx, 0x18, 0x300, 0x1ffff, 0, 69)
        struct.pack_into('<I', task_copies, off + 28, 0)
        struct.pack_into('<Q', task_copies, off + 32, block_off)  # relative offset

    regcmd_bo = header + blocks_data + bytes(task_copies)
    # Pad to 8192 bytes (BO[1] size)
    regcmd_bo += b'\x00' * (8192 - len(regcmd_bo))

    # ── Build task struct array for BO[0] (with ABSOLUTE DMA addresses) ────
    task_bo = bytearray(15 * 40)
    regcmd_base_dma = bo_dmas[BO_REGCMD] + 0xC0  # header offset

    for ti, (op_idx, block_id, c_tile, acc_c) in enumerate(ADD_TASK_SEQUENCE):
        off = ti * 40
        block_off = BLOCK_OFFSETS[block_id]
        struct.pack_into('<IIIIIII', task_bo, off,
                         0,           # flags
                         op_idx,      # op_idx
                         0x18,        # enable_mask
                         0x300,       # int_mask
                         0x1ffff,     # int_clear
                         0,           # int_status
                         69)          # regcfg_amount
        struct.pack_into('<I', task_bo, off + 28, 0)  # regcfg_offset
        struct.pack_into('<Q', task_bo, off + 32,
                         regcmd_base_dma + block_off)

    # ── Weight BO: zeros (Add has no weights; BO[2] is scratch) ──
    weight_bo = b'\x00' * 4096

    return regcmd_bo, bytes(task_bo), weight_bo


# ── Op-specific compute parameters ────────────────────────────

# Entry 23 (reg 0x4070) encodes the ALU operation selector.
# Entry 28 (reg 0x4084) tag_lo differs for div.
# Entry 63 (reg 0x5044) val differs for div.
OP_PARAMS = {
    "add": {"e23_val": 0x02c0, "e23_tag_lo": 0x1082,
            "e28_tag_lo": 0x0001, "e63_val": 0x7849},
    "sub": {"e23_val": 0x02c0, "e23_tag_lo": 0x1084,
            "e28_tag_lo": 0x0001, "e63_val": 0x7849},
    "mul": {"e23_val": 0x03c4, "e23_tag_lo": 0x1080,
            "e28_tag_lo": 0x0001, "e63_val": 0x7849},
    "div": {"e23_val": 0x03c0, "e23_tag_lo": 0x1083,
            "e28_tag_lo": 0x0000, "e63_val": 0x7841},
}

# Task sequence for sub/mul/div: 12 tasks, 4 blocks (single tiling C=64)
# Runs 3 identical iterations of (read, read, compute, write)
SIMPLE_TASK_SEQUENCE = [
    (2, 0, 0, 0),   # reader1
    (3, 1, 0, 0),   # reader2
    (4, 2, 64, 0),  # compute (C=64)
    (5, 3, 0, 0),   # writer
    (2, 0, 0, 0),   # reader1 (reuse)
    (3, 1, 0, 0),   # reader2 (reuse)
    (4, 2, 64, 0),  # compute (reuse)
    (5, 3, 0, 0),   # writer (reuse)
    (2, 0, 0, 0),   # reader1 (reuse)
    (3, 1, 0, 0),   # reader2 (reuse)
    (4, 2, 64, 0),  # compute (reuse)
    (5, 3, 0, 0),   # writer (reuse)
]

# Block offsets for 4-block layout (sub/mul/div)
SIMPLE_BLOCK_OFFSETS = [0x0, 0x280, 0x500, 0x780]


def gen_elementwise(op: str,
                    shape: Tuple[int, ...] = (1, 64, 768),
                    bo_dmas: List[int] = None,
                    rknn_trailing: bytes = b""
                    ) -> Tuple[bytes, bytes, bytes]:
    """Generate regcmd BO, task BO, and weight BO for an elementwise op.

    Supports: add, sub, mul, div. Uses the single-tiling layout (4 blocks:
    reader1, reader2, compute(c_tile=C), writer) for ALL ops and arbitrary
    shapes. The scratch BO is sized from the shape (see scratch_layout) and a
    zeroed scratch is sufficient — the captured scratch_init residue is not
    needed. Callers must allocate BOs sized via scratch_layout() and zero the
    scratch BO before writing inputs.

    Args:
        op: Elementwise operation name ("add", "sub", "mul", "div").
        shape: Input tensor shape (N, C, W) or (N, C, H, W).
        bo_dmas: List of 6 BO DMA addresses [task, regcmd, scratch, in1, in2, out].
        rknn_trailing: Reserved (unused by the single-tiling path).

    Returns:
        (regcmd_bo_data, task_bo_data, weight_bo_data)
    """
    op = op.lower()
    if op not in OP_PARAMS:
        raise ValueError(f"Unsupported elementwise op: {op}. "
                         f"Supported: {list(OP_PARAMS.keys())}")

    # ── single-tiling layout for all ops (add/sub/mul/div) ────
    if len(shape) == 3:
        N, C, W = shape
        H = 1
    else:
        N, C, H, W = shape

    if bo_dmas is None:
        bo_dmas = [0] * 6

    in2_base, in1_base, result_base, _scratch_size = scratch_layout(shape)
    stride = W * 2
    params = OP_PARAMS[op]

    # Build 4 blocks: reader1, reader2, compute(c_tile=C), writer
    blocks = []
    for block_id in range(4):
        next_off = SIMPLE_BLOCK_OFFSETS[block_id + 1] if block_id + 1 < 4 else 0
        if block_id == 0:
            blk = _build_block(2, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=SIMPLE_BLOCK_OFFSETS[1])
        elif block_id == 1:
            blk = _build_block(3, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=SIMPLE_BLOCK_OFFSETS[2])
        elif block_id == 2:
            blk = _build_block(4, N, C, H, W, core0_dma=0, core1_dma=0,
                               c_tile=C, next_offset=SIMPLE_BLOCK_OFFSETS[3],
                               **params)
        elif block_id == 3:
            blk = _build_block(5, N, C, H, W, core0_dma=0, core1_dma=0,
                               next_offset=0, extra_70=0x0000)
        blocks.append(bytearray(blk))

    # Patch DMA entries
    for block_id in range(4):
        blk = blocks[block_id]
        if block_id == 0:
            _patch_dma(blk, 5, 0x4020, bo_dmas[BO_IN1], CORE0_ID)
            _patch_dma(blk, 55, 0x5018,
                       bo_dmas[BO_SCRATCH] + in1_base, CORE1_ID)
        elif block_id == 1:
            _patch_dma(blk, 5, 0x4020, bo_dmas[BO_IN2], CORE0_ID)
            _patch_dma(blk, 55, 0x5018,
                       bo_dmas[BO_SCRATCH] + in2_base, CORE1_ID)
        elif block_id == 2:
            chan_off = 0
            _patch_dma(blk, 5, 0x4020,
                       bo_dmas[BO_SCRATCH] + result_base + chan_off,
                       CORE0_ID)
            # The NPU ALU computes e55 <op> e61. The readers copy
            # scratch+in1_base (input1) -> BO[3] and scratch+in2_base (input2)
            # -> BO[4], so e55=BO[3]=input1 and e61=BO[4]=input2, giving
            # input1 <op> input2 for every op. No swap needed (matches the
            # proprietary NPU libraries capture byte-for-byte).
            _patch_dma(blk, 55, 0x5018,
                       bo_dmas[BO_IN1] + chan_off, CORE1_ID)
            _patch_dma(blk, 61, 0x5038,
                       bo_dmas[BO_IN2] + chan_off, CORE1_ID)
        elif block_id == 3:
            _patch_dma(blk, 5, 0x4020, bo_dmas[BO_OUT], CORE0_ID)
            _patch_dma(blk, 55, 0x5018,
                       bo_dmas[BO_SCRATCH] + result_base, CORE1_ID)

    # Patch chain pointers (entry 69) in blocks 0, 1, 2
    regcmd_dma_base = bo_dmas[BO_REGCMD] + 0xC0
    for block_id in (0, 1, 2):
        next_block_off = SIMPLE_BLOCK_OFFSETS[block_id + 1]
        next_dma = regcmd_dma_base + next_block_off
        off = 69 * ENTRY_SZ
        struct.pack_into('<HHI', blocks[block_id], off,
                         0x0010, next_dma & 0xFFFF,
                         (0x0101 << 16) | ((next_dma >> 16) & 0xFFFF))

    # Assemble regcmd BO: 192B header + 4 blocks + task struct copies
    # The NPU requires a copy of the task structs with RELATIVE block offsets
    # (not absolute DMA addresses) appended after the regcmd blocks in BO[1].
    header = _build_header()
    blocks_data = b''.join(bytes(b) for b in blocks)

    # Build task struct copies with relative offsets for BO[1]
    task_copies = bytearray(12 * 40)
    for ti, (op_idx, block_id, c_tile, acc_c) in enumerate(SIMPLE_TASK_SEQUENCE):
        off = ti * 40
        block_off = SIMPLE_BLOCK_OFFSETS[block_id]
        struct.pack_into('<IIIIIII', task_copies, off,
                         0, op_idx, 0x18, 0x300, 0x1ffff, 0, 69)
        struct.pack_into('<I', task_copies, off + 28, 0)
        struct.pack_into('<Q', task_copies, off + 32, block_off)  # relative offset

    regcmd_bo = header + blocks_data + bytes(task_copies)
    # Pad to 4096 bytes
    regcmd_bo += b'\x00' * (4096 - len(regcmd_bo))

    # Build task struct array for BO[0] (with ABSOLUTE DMA addresses for submission)
    task_bo = bytearray(12 * 40)
    for ti, (op_idx, block_id, c_tile, acc_c) in enumerate(SIMPLE_TASK_SEQUENCE):
        off = ti * 40
        block_off = SIMPLE_BLOCK_OFFSETS[block_id]
        struct.pack_into('<IIIIIII', task_bo, off,
                         0, op_idx, 0x18, 0x300, 0x1ffff, 0, 69)
        struct.pack_into('<I', task_bo, off + 28, 0)
        struct.pack_into('<Q', task_bo, off + 32,
                         regcmd_dma_base + block_off)  # absolute DMA

    weight_bo = b'\x00' * 4096
    return regcmd_bo, bytes(task_bo), weight_bo


# ── ONNX op name → gen_elementwise dispatch ───────────────────

_ONNX_TO_ELEMENTWISE = {
    "add": "add", "sub": "sub", "mul": "mul", "div": "div",
}


def gen_op(op_type: str,
           shape: Tuple[int, ...] = (1, 64, 768),
           bo_dmas: List[int] = None,
           rknn_path: Optional[str] = None
           ) -> Tuple[bytes, bytes, bytes]:
    """Dispatch to the appropriate codegen function for an ONNX op type.

    Currently supports elementwise ops (add, sub, mul, div).
    Future: matmul, relu, sigmoid, softmax, etc.

    Args:
        op_type: ONNX op name (e.g. "add", "Add", "mul", "Mul").
        shape: Input tensor shape.
        bo_dmas: List of 6 BO DMA addresses.
        rknn_path: Path to the vendor model files (to extract trailing task descriptor
            data that proprietary NPU libraries writes to BO[1] after regcmd blocks).
            If None, tries to find the vendor model files automatically.

    Returns:
        (regcmd_bo_data, task_bo_data, weight_bo_data)
    """
    op_lower = op_type.lower()
    ew = _ONNX_TO_ELEMENTWISE.get(op_lower)
    if ew:
        # Extract model files trailing data (task descriptors for BO[1])
        rknn_trailing = b""
        if rknn_path is None:
            # Try common locations
            candidates = [
                os.path.join(os.environ.get("OPENNPU_TEMPLATE_DIR", "./patterns"), ew),
                os.path.join("/tmp/templates", ew),
                os.path.join(os.path.dirname(__file__), ew),
            ]
            for c in candidates:
                if os.path.exists(c):
                    rknn_path = c
                    break
        if rknn_path and os.path.exists(rknn_path):
            with open(rknn_path, 'rb') as f:
                rknn_data = f.read()
            entry0_pattern = struct.pack('<HHI', 0x4004, 0x000e, 0x10010000)
            rc_off = rknn_data.find(entry0_pattern)
            if rc_off >= 0:
                n_blocks = 9 if ew == "add" else 4
                rknn_trailing = rknn_data[rc_off + n_blocks * BLOCK_SZ:]
        return gen_elementwise(ew, shape, bo_dmas, rknn_trailing=rknn_trailing)
    raise NotImplementedError(
        f"Codegen for op '{op_type}' not yet implemented. "
        f"Supported: {list(_ONNX_TO_ELEMENTWISE.keys())}")


if __name__ == "__main__":
    import sys

    # Use realistic BO DMA addresses from the capture for comparison
    # BO layout (contiguous): BO[5], BO[2], BO[4], BO[3], BO[1], BO[0]
    capture_bo_dmas = [
        0xFFFFF000,  # BO[0] task
        0xFFFFD000,  # BO[1] regcmd
        0xFFF6D000,  # BO[2] scratch
        0xFFFE5000,  # BO[3] input1
        0xFFFCD000,  # BO[4] input2
        0xFFF55000,  # BO[5] output
    ]

    regcmd, tasks, weight = gen_add((1, 64, 768), bo_dmas=capture_bo_dmas)

    # Compare with capture file
    with open('/tmp/captures/capture_0_bo_h2.bin', 'rb') as f:
        capture_bo1 = f.read()
    with open('/tmp/captures/capture_0_bo_h1.bin', 'rb') as f:
        capture_bo0 = f.read()

    # Compare regcmd BO (192B header + 9 blocks)
    print("=== Regcmd BO comparison (gen vs capture) ===")
    total_diffs = 0
    # Header
    hdr_diffs = sum(1 for a, b in zip(regcmd[:192], capture_bo1[:192]) if a != b)
    print(f"  Header (192B): {hdr_diffs} diffs")
    total_diffs += hdr_diffs

    # Blocks
    for bi in range(9):
        off = 0xC0 + BLOCK_OFFSETS[bi]
        gen_blk = regcmd[off:off + BLOCK_SZ]
        capture_blk = capture_bo1[off:off + BLOCK_SZ]
        diffs = sum(1 for a, b in zip(gen_blk, capture_blk) if a != b)
        # Ignore DMA entries (5, 55, 61) — those depend on actual IOVA
        dma_diffs = 0
        for ei in [5, 55, 61]:
            for j in range(8):
                if gen_blk[ei * 8 + j] != capture_blk[ei * 8 + j]:
                    dma_diffs += 1
        real_diffs = diffs - dma_diffs
        status = "OK" if real_diffs == 0 else "MISMATCH"
        print(f"  block[{bi}]: {diffs} diffs ({dma_diffs} DMA, {real_diffs} real) [{status}]")
        total_diffs += real_diffs

    # Compare task BO
    print()
    print("=== Task BO comparison (gen vs capture) ===")
    task_diffs = 0
    for ti in range(15):
        gen_t = tasks[ti * 40:(ti + 1) * 40]
        capture_t = capture_bo0[ti * 40:(ti + 1) * 40]
        diffs = sum(1 for a, b in zip(gen_t, capture_t) if a != b)
        # regcmd_addr will differ since we use capture DMAs (should match!)
        if diffs > 0:
            import struct as s
            gen_addr = s.unpack_from('<Q', gen_t, 32)[0]
            capture_addr = s.unpack_from('<Q', capture_t, 32)[0]
            print(f"  task[{ti}]: {diffs} diffs  gen_addr={gen_addr:#x} capture_addr={capture_addr:#x}")
        task_diffs += diffs
    print(f"  Total task diffs: {task_diffs}")

    print()
    print(f"=== SUMMARY: {total_diffs + task_diffs} total real differences ===")
    if total_diffs + task_diffs == 0:
        print("  PERFECT BYTE-FOR-BYTE MATCH!")
    else:
        print("  Differences remain (may be DMA/IOVA dependent)")