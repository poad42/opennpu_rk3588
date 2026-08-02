# NPU Register Investigation — RK3588 vs RK3576

## Goal
Determine if the RK3588 NPU has internal SRAM/Nbuf (like the RK3576) that could
be used for weight caching to eliminate the W-DMA bottleneck.

## Method
Safe kernel module probing NPU CORE0 registers via ioremap(0xfdab0000).

## Key Findings

### 1. REG_1024 (0x1024) — EXISTS on RK3588
- Default: 0x00000000
- Writable: YES — writing 0x80000000 (RK3576 state_init value) takes effect
- BUT: NPU hardware overwrites it to 0x003f0300 during job execution
- The register is used by the NPU internally, not just for state_init

### 2. REG_1004 (0x1004) — BIT 4 NOT WRITABLE on RK3588
- Default: 0x00000000 (idle), 0x0000000e (during submit)
- Writing 0x1e (RK3576 state_init value) reads back 0x0e
- **Bit 4 (0x10) is NOT writable on RK3588**
- RK3576 state_init writes 0x1e (bits 1-4). RK3588 only accepts bits 1-3 (0xe).
- Bit 4 is likely the "enable internal memory" bit — NOT available on RK3588

### 3. Performance Counters — DON'T EXIST on RK3588
- 0x2210-0x223c (top-level): kernel oops (DECERR — register not decoded)
- 0x8000-0x803c (old-style): BUS HANG (SBC freezes, watchdog reboot)
- RK3576 has amount_top=&rknpu_top_amount (offsets 0x2210+)
- RK3588 has amount_top=NULL — registers not implemented

### 4. 0x3fe80000 (RK3576 nbuf address) — BUS HANG on RK3588
- CPU /dev/mem access crashes SBC
- Kernel module ioremap + readl hangs kernel
- No memory device at this address on RK3588

### 5. 0x1000-0x1030 — Full Register Scan (ALL readable)
| Offset  | Value        | Notes                              |
|---------|-------------|-------------------------------------|
| 0x1000  | 0x00000000  |                                     |
| 0x1004  | 0x0000000e  | Routing (bits 1-3 only, bit 4=NO)  |
| 0x1008  | 0x00000000  |                                     |
| 0x100c  | 0x20000120  | Config?                             |
| 0x1010  | 0x00000020  |                                     |
| 0x1014  | 0x00000009  |                                     |
| 0x1018  | 0x00000000  |                                     |
| 0x101c  | 0x00000000  |                                     |
| 0x1020  | 0x00540001  |                                     |
| 0x1024  | 0x003f0300  | State (overwritten by NPU during job)|
| 0x1028  | 0x00000054  |                                     |
| 0x102c  | 0x00000054  |                                     |
| 0x1030  | 0x00018000  |                                     |

### 6. 0x0-0x40 — Core Control (all readable)
- VERSION (0x0): 0x46495245 = "FIRE"
- VERSION_NUM (0x4): 0x00000000
- All standard registers (PC_OP_EN, PC_DATA_ADDR, etc.) readable

## RK3588 vs RK3576 NPU Config Comparison
| Feature              | RK3588              | RK3576              |
|---------------------|---------------------|---------------------|
| nbuf_phyaddr        | 0                   | 0x3fe80000          |
| nbuf_size           | 0                   | 1MB                 |
| state_init          | NULL                | rk3576_state_init   |
| cache_sgt_init      | NULL                | rk3576_cache_sgt_init|
| amount_top          | NULL                | &rknpu_top_amount   |
| amount_core         | NULL                | &rknpu_core_amount |
| 0x1004 bit 4        | NOT writable        | Writable (state_init)|
| 0x2210 perf ctrs    | DON'T exist (oops)  | Exist               |
| 0x3fe80000 nbuf     | BUS HANG            | 1MB internal memory |
| Core count          | 3                   | 2                   |

## Conclusion
The RK3588 NPU appears to be a DIFFERENT revision from the RK3576:
1. Internal memory enable bit (0x1004 bit 4) is NOT writable
2. Performance counters (0x2210+) don't exist
3. nbuf at 0x3fe80000 is not accessible
4. Driver config explicitly sets nbuf_size=0, state_init=NULL

The RK3576 (newer SoC) added internal memory support. The RK3588 (older)
does not have this feature. The NPU compute cores may be the same "FIRE"
architecture, but the memory/cache subsystem differs.

## Kernel Module Safety
- Reading NPU registers via ioremap + readl is SAFE for known offsets
- Reading non-existent registers causes either:
  - Kernel oops (DECERR — contained, SBC survives): 0x2210
  - Bus hang (no response — SBC freezes): 0x8000, 0x3fe80000
- The watchdog (10s timeout) auto-reboots the SBC on bus hang
