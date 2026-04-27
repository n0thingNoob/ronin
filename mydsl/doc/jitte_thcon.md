# Jitte ThCon emulator — change log

This document tracks every modification made under the Jitte ThCon emulator
v0 effort. Each stage is a small, independently-verifiable PR-sized step.
The plan (charter, decisions, file-level change list, op semantics) lives at:
`~/.claude/plans/functional-prancing-wand.md`.

Charter clarification: the mydsl README says "does not modify
tanto/jitte/algo/yari" — that clause referred to mydsl-internal work. The
ThCon emulator is explicit infrastructure work *in jitte*, agreed in chat.
Tanto is untouched.

---

## Stage A — plumbing only — **DONE 2026-04-26**

**Goal:** prove the entire interception chain works end-to-end:
linker → JALR hook → BuiltinHandler dispatch → INSTRUCTION_WORD redef.

### Changes

**Jitte (new):**
- `src/device/riscv/builtin_tensix.{hpp,cpp}` — tensix builtin domain. ID
  range 5120-6143; v0 uses 5120 only (`jitte_tensix_exec`).
- `src/device/riscv/tensix_handler.{hpp,cpp}` — `TensixHandler`. Stage A
  body was just `printf+abort`.
- `home/tt_metal/emulator/kernels/jitte_thcon.h` — kernel-facing header.
  When `__JITTE__` is defined, redefines `INSTRUCTION_WORD(x)` from
  `__asm__(".ttinsn ...")` to `jitte_tensix_exec((x))`.
- `home/tt_metal/emulator/kernels/ckernel_ops.h` — copied verbatim from
  `~/tt-metal/.../tt_llk_wormhole_b0/common/inc/ckernel_ops.h`. Jitte does
  not vendor tt-llk; this copy is the bridge. Pure macros, no #include / no
  namespace.

**Jitte (modified):**
- `src/device/riscv/builtin_handler.{hpp,cpp}` — added
  `TensixHandler m_tensix_handler;` member + 6th lookup branch in `call()`.
- `src/device/api/kernel_builder.cpp` — register tensix builtins on both
  compute and dataflow paths.
- `src/tt_metal/jit_build/build.cpp` — `g_cpp_cmd_base` now includes
  `-D__JITTE__`. Required to activate the `jitte_thcon.h` redefinition.

**mydsl (new):**
- `src/handwritten/thcon_scalar_add/{device/metal,host,test}/` — minimal
  smoke demo. Kernel: `void kernel_main() { TTI_SETDMAREG(0,42,0,0); }`.
  Host: dispatches via Tanto `core::Kernel(KernelKind::READER,
  KernelFormat::METAL)`; no pipes, no readback.
- `jitte/prj/handwritten/thcon_scalar_add/{deploy_jitte.sh,build_test_tanto.sh}`.

### Verification

```
[tensix] insn=0x45002a00
Aborted
```
Opcode `0x45` = SETDMAREG, payload `0x002a` = 42. Seeing the print is PASS.

### Surprises encountered

1. **`ckernel_ops.h` is not in Jitte's include tree.** The plan assumed
   it would be reachable through the existing `tt_metal/emulator/kernels`
   include path; it isn't. Resolved by a one-shot copy from upstream tt-llk.

2. **`-D__JITTE__` was not defined for kernel builds.** The kernel build
   command is constructed in `jit_build/build.cpp`'s `g_cpp_cmd_base`. Plan
   assumed Jitte already had a kernel-side `__JITTE__` define; it didn't.
   One-line edit added it.

---

## Stage B — GPR file + L1 R/W (3 ops) — **DONE 2026-04-27**

**Goal:** real ThCon GPR file and the SETDMAREG/LOADIND/STOREIND decoders.
Replace Stage A's printf+abort with a real dispatcher.

### Changes

**Jitte (modified):**
- `src/device/riscv/tensix_handler.hpp` — added `struct ThConState`
  (per-hart `std::array<uint32_t, 64>`); `TensixHandler` now owns
  `unordered_map<Riscv32Core *, ThConState>` keyed by issuing hart.
- `src/device/riscv/tensix_handler.cpp` — full v0 dispatcher:
  - `do_setdmareg` decodes opcode `0x45`. RegIndex16b is a 7-bit
    half-word address: `gpr_idx = RegIndex16b >> 1`,
    `high_half = RegIndex16b & 1`. Mode != 0 (signal-select) and
    SigSelSize != 0 (multi-cycle wide imm) panic.
  - `do_loadind` decodes opcode `0x49`. Reads u8/u16/u32 from
    `L1[Th[AddrRegIndex]]` into `Th[DataRegIndex]`. AutoIncSpec/OffsetIndex
    panic if non-zero (defer RWC-counter modes).
  - `do_storeind` decodes opcode `0x66`. Symmetric. MemHierSel != 0
    panics (remote-L1 routing is a v1 question — ATINCGET will hit it
    again).
  - L1 host pointer obtained via `Machine::get_worker_l1()->map_addr(addr)`.
    `m_curr_tensix` (the issuing core's L1) is set by the scheduler before
    the kernel runs, so this returns the right L1.
  - Diagnostic `[tensix]` prints for every decoded op (toggle via
    `DIAG_TENSIX_TRACE`).

**mydsl (modified):**
- `src/handwritten/thcon_scalar_add/device/metal/thcon_scalar_add_kernel.cpp`
  — real ThCon program: load 42 into Th[3], load slot addr `0x40000` into
  Th[0], `STOREIND L1[Th[0]] <- Th[3]` (u32), `LOADIND Th[5] <- L1[Th[0]]`.
- `src/handwritten/thcon_scalar_add/test/main.cpp` — kernel runs to
  completion; PASS reported via `[tensix]` round-trip prints.

### Verification

```
[tensix] SETDMAREG  Th[3].lo = 0x002a  (Th[3]=0x0000002a)
[tensix] SETDMAREG  Th[3].hi = 0x0000
[tensix] SETDMAREG  Th[0].lo = 0x0000
[tensix] SETDMAREG  Th[0].hi = 0x0004  (Th[0]=0x00040000)
[tensix] STOREIND   L1[0x00040000] <- Th[3]=0x0000002a  (u32)
[tensix] LOADIND    Th[5] <- L1[0x00040000] = 0x0000002a  (u32)
RESULT: PASS
```

### What this validates

- ThCon GPR file is real (per-hart `Th[64]`, keyed by `Riscv32Core*`).
- `Machine::get_worker_l1()` reliably returns the issuing core's L1 from
  inside the BuiltinHandler call chain.
- Kernel can write a value to L1 via STOREIND and read it back via LOADIND
  with bit-exact round-trip — i.e. ThCon is a usable scalar PE on Jitte
  for any computation that fits the SETDMAREG/LOADIND/STOREIND triangle.

### Encoding details captured

(from `tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_ops.h`)

| Op | OpCode | Layout |
|---|---|---|
| SETDMAREG | 0x45 | `[23:22] SigSelSize` (2b) `[21:8] SigSel` (14b!) `[7] Mode` `[6:0] RegIndex16b` (7b) |
| LOADIND   | 0x49 | `[23:22] SizeSel` `[21:14] OffsetIndex` `[13:12] AutoIncSpec` `[11:6] DataRegIndex` `[5:0] AddrRegIndex` |
| STOREIND  | 0x66 | `[23] MemHierSel` `[22] SizeSel` `[21] RegSizeSel` `[20:14] OffsetIndex` `[13:12] AutoIncSpec` `[11:6] DataRegIndex` `[5:0] AddrRegIndex` |

**SigSel is 14 bits, not 16.** `LO_16(N)=2*N`, `HI_16(N)=2*N+1` (ThCon
GPRs are addressed as 16-bit halves; 64 GPRs × 32-bit = 128 × 16-bit slots,
hence the 7-bit RegIndex16b). For values whose halves exceed 14 bits, real
LLK uses multiple SETDMAREG instructions or wide-imm modes. v0 panics on
those modes.

### Plan deviation

Stage B verification per the plan was "host seeds value via runtime arg,
host reads back L1 slot." That requires an RV32-arg → ThCon-GPR bridge
(real silicon uses `ckernel::instrn_buffer` + `TT_SETDMAREG`); we don't
have that bridge yet. **Stage B substitutes hardcoded value + diagnostic
prints for round-trip verification.** The bridge is a follow-up; print-
based verification is sufficient to prove correctness of the GPR/L1 path.

### Opcode encodings vs plan

The plan listed LOADIND=0x40, STOREIND=0x41 — those were wrong. Real
encodings are LOADIND=0x49, STOREIND=0x66 (verified from
`ckernel_ops.h`). Plan corrected in this doc.

---

## Stage C — arithmetic (+2 ops) — **PENDING**

Adds ADDDMAREG (0x58), MULDMAREG (0x5A). Updates kernel to the full
hello-world: load `a` and `b` from L1, add, store `c`.

## Stage D — atomic (+1 op + per-core mutex) — **PENDING**

Adds ATINCGET (0x61) with a `std::mutex` per `(soc_x, soc_y)`. Adds a
second smoke kernel `thcon_atomic_inc_kernel.cpp` exercising ATINCGET in
a loop.

---

## File index for v0

**Modified (Jitte):**
- `src/device/riscv/builtin_handler.hpp`
- `src/device/riscv/builtin_handler.cpp`
- `src/device/api/kernel_builder.cpp`
- `src/tt_metal/jit_build/build.cpp`

**Added (Jitte):**
- `src/device/riscv/builtin_tensix.{hpp,cpp}`
- `src/device/riscv/tensix_handler.{hpp,cpp}`
- `home/tt_metal/emulator/kernels/jitte_thcon.h`
- `home/tt_metal/emulator/kernels/ckernel_ops.h` (vendored from tt-llk)

**Added (mydsl demo):**
- `src/handwritten/thcon_scalar_add/{device/metal,host,test}/*`
- `jitte/prj/handwritten/thcon_scalar_add/{deploy_jitte.sh,build_test_tanto.sh}`

**Untouched:** `tanto/`, `algo/`, `yari/`, all other
`mydsl/src/handwritten/*` demos.

## Build/run reference

```sh
# 1. Build Jitte
cd ~/ronin/jitte/prj && bash build_all.sh

# 2. Deploy + build the demo
cd ~/ronin/mydsl/jitte/prj/handwritten/thcon_scalar_add
./deploy_jitte.sh
./build_test_tanto.sh

# 3. Run
cd ~/ronin/mydsl/jitte
TT_METAL_HOME=~/ronin/jitte/home TT_ARCH=wormhole_b0 \
    ./prj/bin/handwritten/thcon_scalar_add/test_tanto
```
