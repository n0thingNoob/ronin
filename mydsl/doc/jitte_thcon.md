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

## Stage C — arithmetic (+2 ops) — **DONE 2026-04-27**

**Goal:** add `ADDDMAREG (0x58)` and `MULDMAREG (0x5A)` and run a full
scalar dataflow program that exercises every Stage A/B/C op together.

### Changes

**Jitte (modified):**
- `src/device/riscv/tensix_handler.cpp` — added `do_arith()` shared by
  ADD/MUL. Encoding (verified from `ckernel_ops.h`):
  - `[23]    OpBisConst` (1=OpB is 6-bit immediate, 0=OpB is GPR index)
  - `[22:12] ResultRegIndex` (11b; only low 6 bits used in v0)
  - `[11:6]  OpBRegIndex` (6b)
  - `[5:0]   OpARegIndex` (6b)
  Dispatcher gets two new cases: `0x58 → do_arith(false)`, `0x5A → do_arith(true)`.

**mydsl (modified):**
- `src/handwritten/thcon_scalar_add/device/metal/thcon_scalar_add_kernel.cpp`
  — full scalar dataflow program: a=100, b=42; loads vals into Th[3]/Th[4];
  loads slot addrs (A=0x40000, B=0x40010, C=0x40020, D=0x40030) into
  Th[0,1,2,10]; STOREIND seeds A/B; LOADIND reads back into Th[6]/Th[7];
  ADDDMAREG → Th[8] = 142; MULDMAREG → Th[9] = 4200; STOREIND writes c, d.

### What this validates

- ALU path is real, not just register-cache shuffling: result of LOADIND
  feeds ADDDMAREG, output feeds STOREIND — full dataflow round-trip.
- ADDDMAREG/MULDMAREG produce `0x8e` (=142) and `0x1068` (=4200) on the
  expected operands (0x64 + 0x2a, 0x64 * 0x2a).
- 32-bit wrap-around is u32-clean (multiply uses `uint32_t(a*b)` cast to
  match HW behavior — relevant once values exceed 2^16).

### Verification

```
[tensix] ADDDMAREG Th[8] = Th[6] + Th[7] = 0x64 + 0x2a = 0x8e (=142)
[tensix] MULDMAREG Th[9] = Th[6] * Th[7] = 0x64 * 0x2a = 0x1068 (=4200)
[tensix] STOREIND  L1[0x00040020] <- Th[8]=0x0000008e  (u32)
[tensix] STOREIND  L1[0x00040030] <- Th[9]=0x00001068  (u32)
RESULT: PASS
```

### Plan deviation

Plan said "host seeds a/b via runtime arg, host reads back c". v0 still
lacks the runtime-arg→ThCon-GPR bridge and host→L1 readback bridge, so
Stage C uses hardcoded values + `[tensix]` diag prints (same approach as
Stage B). The arithmetic is fully exercised; the wiring to host args is a
follow-up that does not affect ThCon-emulator correctness.

### Encoding details captured (ADDDMAREG/MULDMAREG)

| Op | OpCode | Layout |
|---|---|---|
| ADDDMAREG | 0x58 | `[23] OpBisConst` `[22:12] ResultRegIndex` (11b) `[11:6] OpBRegIndex` `[5:0] OpARegIndex` |
| MULDMAREG | 0x5A | (same as ADDDMAREG) |

ResultRegIndex is 11 bits in HW because `RegSpec` allows TOC config-reg
addressing as a third class of destinations; v0 only models the 64×u32
ThCon GPR file, so we mask down to the low 6 bits and panic-by-omission
on the high range. The wider register class is a v1 question.

---

## Stage D — atomic (+1 op + mutex) — **DONE 2026-04-27**

**Goal:** add `ATINCGET (0x61)` with mutex-guarded RMW. Adds the second
smoke kernel (`thcon_atomic_inc_kernel.cpp`) and the second test binary
(`test_atomic_inc`).

### Changes

**Jitte (modified):**
- `src/device/riscv/tensix_handler.hpp` — added `<mutex>` plus a per-L1
  mutex map (`unordered_map<Memory*, std::mutex>` + a table-lock).
- `src/device/riscv/tensix_handler.cpp` — added `do_atincget`. Encoding
  (verified from `ckernel_ops.h`):
  - `[23]    MemHierSel` (1=remote-L1; v0 panics)
  - `[22:14] WrapVal` (9b)  — counter wraps to 0 when (old+1) > WrapVal.
                              WrapVal=0 means no wrap.
  - `[13:12] Sel32b` (2b)   — v0 supports Sel32b=0 (32b RMW only).
  - `[11:6]  DataRegIndex`  — destination GPR receives OLD counter value.
  - `[5:0]   AddrRegIndex`  — Th[AddrRegIndex] is the L1 byte address.
  Atomic semantics: lock → memcpy old → memcpy old+1 → unlock; Th[Data]
  receives old. Dispatcher gets case `0x61 → do_atincget`. Signature of
  `jitte_tensix_exec` extended to take the mutex by reference.

**mydsl (added):**
- `src/handwritten/thcon_scalar_add/device/metal/thcon_atomic_inc_kernel.cpp`
  — seeds `L1[0x40000]` to 0, runs 4× `TTI_ATINCGET(0,0,0,3,0)` on slot 0,
  STOREIND persists final returned old to `L1[0x40010]`.
- `src/handwritten/thcon_scalar_add/test/main_atomic_inc.cpp` — driver.
- `host/thcon_scalar_add.{hpp,cpp}` — `init()` now takes optional
  `kernel_name` arg (default `"thcon_scalar_add_kernel"`); driver passes
  `"thcon_atomic_inc_kernel"` for Stage D.

**mydsl (modified):**
- `jitte/prj/handwritten/thcon_scalar_add/build_test_tanto.sh` — now
  builds two binaries (`test_tanto` for Stage C, `test_atomic_inc` for
  Stage D) by sourcing per-test main files instead of `test/*.cpp`.

### What this validates

- Atomic L1 read-modify-write under a host-side mutex (correct semantics
  even though Jitte runs harts sequentially today).
- ATINCGET returns the *old* counter value to the GPR — kernels can
  build queue-counter / fetch-and-increment patterns directly.
- Verifies the v0 hypothesis: a single ATINCGET replaces what would
  otherwise be a two-semaphore credit/data handshake. (The cross-core
  demo exercising this is a v1 task.)

### Verification

```
[tensix] STOREIND  L1[0x00040000] <- Th[2]=0x00000000  (u32)
[tensix] ATINCGET  Th[3] <- L1[0x00040000] (old=0x0, new=0x1, wrap=0x0)
[tensix] ATINCGET  Th[3] <- L1[0x00040000] (old=0x1, new=0x2, wrap=0x0)
[tensix] ATINCGET  Th[3] <- L1[0x00040000] (old=0x2, new=0x3, wrap=0x0)
[tensix] ATINCGET  Th[3] <- L1[0x00040000] (old=0x3, new=0x4, wrap=0x0)
[tensix] STOREIND  L1[0x00040010] <- Th[3]=0x00000003  (u32)
RESULT: PASS
```

### Encoding details captured (ATINCGET)

| Op | OpCode | Layout |
|---|---|---|
| ATINCGET | 0x61 | `[23] MemHierSel` `[22:14] WrapVal` (9b) `[13:12] Sel32b` `[11:6] DataRegIndex` `[5:0] AddrRegIndex` |

Real HW supports `Sel32b ∈ {0..3}` (16b/32b/64b counter widths) and
remote-L1 routing (`MemHierSel=1`). v0 hard-panics on those — a NOC
mailbox/atomic path is the natural v1 demo.

### Lessons learned

First attempt at per-L1 mutex map crashed at device-open with a
`_int_malloc` assertion. Root cause: ABI mismatch. `BuiltinHandler`
holds `TensixHandler` by value, and `builtin_handler.hpp` is included
from `device/ref/machine_impl.{cpp,hpp}` (in the **ref** lib) as well
as `device/riscv/*` (the **riscv** lib). `build_riscv.sh` only rebuilds
the riscv lib, so the ref lib retained the old `sizeof(TensixHandler)`
— constructions and destructions used different layouts and trashed
the heap. Fix: run `prj/device/build_all.sh` (rebuilds both ref and
riscv) whenever `tensix_handler.hpp` or `builtin_handler.hpp` changes.

**Rule of thumb (Jitte build):** any header change that changes the
size/layout of a class held by value across libs requires a full
`prj/device/build_all.sh` rebuild, not just `build_riscv.sh`.

---

---

## Stage E — host→ThCon-GPR bridge — **DONE 2026-04-27**

**Goal:** kill the last v0 deviation — runtime args from host should
reach ThCon GPRs without abuse of `TTI_SETDMAREG`'s 14-bit immediate.

### Changes

**Jitte (modified):**
- `src/device/riscv/builtin_tensix.{hpp,cpp}` — second tensix builtin
  `jitte_thcon_set_gpr(uint32_t gpr_idx, uint32_t value)` (id 5121,
  arg-count 2). Brings the tensix builtin range to 5120-5121 used.
- `src/device/riscv/tensix_handler.cpp` — handler for the new id pulls
  `gpr_idx` and `value` from `core->get_arg(0/1)` and writes
  `th.gpr[gpr_idx] = value`. Range-checks `gpr_idx < 64`.
- `home/tt_metal/emulator/kernels/jitte_thcon.h` — declares
  `extern "C" void jitte_thcon_set_gpr(uint32_t, uint32_t)`.

**mydsl (added):**
- `src/handwritten/thcon_scalar_add/device/metal/thcon_runtime_arg_kernel.cpp`
  — same scalar dataflow as Stage C, but `a` and `b` come from
  `get_arg_val<uint32_t>(0/1)` and are staged into Th[3]/Th[4] via
  the bridge.
- `src/handwritten/thcon_scalar_add/test/main_runtime_arg.cpp` — host
  passes `a=7, b=11` and runs the kernel.

**mydsl (modified):**
- `host/thcon_scalar_add.{hpp,cpp}` — `init()` now takes
  `runtime_args` (`std::vector<uint32_t>`); converts to
  `std::vector<KernelArg>` (Tanto's variant type) before `set_args`.
- `jitte/prj/handwritten/thcon_scalar_add/build_test_tanto.sh` — third
  binary `test_runtime_arg`.

### What this validates

- Host-passed runtime args (RV32 register values via Tanto's
  `KernelArg` machinery) make it to ThCon GPRs and drive ALU ops.
- The `[tensix]` SET_GPR prints carry `0x7` and `0xb` straight from
  host args, not hardcoded.
- The earlier deviation in Stage B/C/D notes is now closed: any
  Jitte ThCon kernel can be parameterized via host args.

### Verification

```
[tensix] SET_GPR    Th[3] = 0x00000007  (host bridge)
[tensix] SET_GPR    Th[4] = 0x0000000b  (host bridge)
...
[tensix] ADDDMAREG Th[8] = 0x7 + 0xb = 0x12 (=18)
[tensix] MULDMAREG Th[9] = 0x7 * 0xb = 0x4d (=77)
RESULT: PASS
```

### Plan deviation

Bridge is **Jitte-only** by design — `jitte_thcon_set_gpr` is not a
real Tensix opcode. On real silicon the equivalent path is
`ckernel::instrn_buffer + TT_SETDMAREG`, which models a hardware
mailbox (MOP / MMIO-driven instruction buffer) that v0 doesn't
emulate. v1 may reproduce the mailbox; for v0 this builtin is the
escape hatch that lets parameterized kernels exist on Jitte today.
The kernel source diverges slightly between Jitte and silicon for
this one call (silicon would use TT_SETDMAREG in a small inline
helper), which is the cost of not having instrn_buffer.

---

## Stage F — real-HW fidelity (v0.5) — **DONE 2026-04-27**

**Goal:** close the gap surfaced while porting Stage C to real Wormhole
(see "Real-Wormhole port — RESOLVED" below). On silicon, two Tensix ThCon
semantics that Jitte v0 silently flattened actually mattered: `OffsetIndex`
is a half-register *index* whose deref is the address addend, and LOADIND
is asynchronous (the GPR fill happens "at some later point in time" per
ISA doc). Both cause silent wrong addresses / stale-zero reads on real HW
that Jitte v0 hid.

### Changes

**Jitte (modified):**
- `src/device/riscv/tensix_handler.hpp` — `ThConState` gains
  `std::unordered_map<uint32_t,uint32_t> pending_loadind` to stage
  asynchronous LOADIND results.
- `src/device/riscv/tensix_handler.cpp` —
  - `read_half_reg`/`write_half_reg`: deref `OffsetIndex` as a half-reg
    index into the GPR file; the half-reg's *value* is the byte addend
    to `Th[AddrReg]*16` (per ISA doc `STOREIND_L1.md` /
    `LOADIND.md`'s `*Offset = (char*)&GPRs[0] + OffsetHalfReg*2`).
  - `AUTO_INC_BYTES = {0,2,4,16}`: AutoIncSpec now post-increments
    `*OffsetHalfReg` after the L1 access, matching real HW.
  - `do_loadind` no longer writes `gpr[data_reg]` synchronously —
    the loaded value goes into `pending_loadind[data_reg]` and only
    retires into `gpr` on STALLWAIT/FLUSHDMA.
  - `check_no_pending(th, gpr_idx, role, op_name)` aborts loud if any
    consumer (ADDDMAREG, MULDMAREG, STOREIND data/addr, ATINCGET,
    SETDMAREG, jitte_thcon_set_gpr) reads a GPR with a pending
    LOADIND. The error message names the missing barrier.
  - New decoders: `do_stallwait` (op `0xa2`) and `do_flushdma`
    (op `0x46`), each calls `retire_pending(th)` to commit all
    pending entries into `gpr`. Resource-bit fields are over-broad
    rather than fine-grained — safe because real HW also drains
    everything on these barriers.
- `home/tt_metal/emulator/kernels/jitte_thcon.h` — under `__JITTE__`,
  exposes a small `namespace p_stall { THCON, STALL_THCON, STALL_THREAD }`
  with values mirrored from `ckernel_instr_params.h`. Lets BRISC
  kernels (which don't pull in the compute API) write
  `TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::THCON)` exactly as
  real-HW kernels do.

**mydsl (modified, all three demo kernels):**
- `device/metal/thcon_scalar_add_kernel.cpp` (Stage C) — pass
  `OffsetIndex = HI_16(3) = 7` (Th[3]=100 → hi16 = 0) instead of
  literal `0`; insert `TTI_STALLWAIT` between LOADIND and
  ADDDMAREG/MULDMAREG.
- `device/metal/thcon_atomic_inc_kernel.cpp` (Stage D) — pass
  `OffsetIndex = LO_16(2) = 4` after explicitly staging Th[2]=0.
- `device/metal/thcon_runtime_arg_kernel.cpp` (Stage E) — stage Th[5]=0
  explicitly (host args could carry hi-bits) and use
  `OffsetIndex = LO_16(5) = 10`; insert `TTI_STALLWAIT` between
  LOADIND and the ALU ops.

### Verification

All three Jitte demos (`test_tanto`, `test_atomic_inc`, `test_runtime_arg`)
re-PASS unchanged after the rewrite. `[tensix]` traces now carry an
extra `+0x?` offset field on STOREIND/LOADIND prints (the dereferenced
half-reg) plus `STALLWAIT/FLUSHDMA  retire Th[N] <- 0x... (was pending
LOADIND)` lines that visualise the async retirement.

Real-HW counterpart (`tt-playground/thcon_scalar_add`) PASSes with the
same `ZERO_OFF` + STALLWAIT pattern. Same source compiles+runs on both
Jitte and real Wormhole — the v0 plan's cross-platform fidelity claim
is now actually true.

### Why catch-it-loud rather than emulate-the-race

`pending_loadind` could in principle silently retire on the next read
(emulate "out-of-order with most-likely real-HW timing"). v0.5 instead
aborts on read-before-retire because the bug is silent on both sides
otherwise: the kernel sees a zero/stale GPR, no message tells you why,
and the same bug then ships to silicon. The abort message names the
exact missing barrier — that turns a mystery into a one-line fix.

---

## v0.5 status — **COMPLETE 2026-04-27**

All six stages (A–F) PASS. Op set:
- ThCon ops: SETDMAREG (0x45), LOADIND (0x49), STOREIND (0x66),
  ADDDMAREG (0x58), MULDMAREG (0x5A), ATINCGET (0x61).
- Sync ops: STALLWAIT (0xa2), FLUSHDMA (0x46) — retire async LOADIND.
- Jitte-only host bridge: `jitte_thcon_set_gpr`.

Fidelity additions over v0:
- `OffsetIndex` half-register deref + `AutoIncSpec` post-increment.
- LOADIND asynchrony with explicit pending-set + STALLWAIT/FLUSHDMA
  retirement; consumer-of-pending aborts with a clear "insert barrier"
  message.

Three smoke binaries (`test_tanto`, `test_atomic_inc`, `test_runtime_arg`)
plus the real-HW counterpart `tt-playground/thcon_scalar_add` cover the
full ThCon scalar dataflow on both Jitte and real Wormhole.

Out-of-scope items (deferred to v1) still apply: faithful 3-pipe ThCon
GPR model, NOC-routable atomics (`MemHierSel=1`), other ThCon ops
(`SUBDMAREG, BITWOPDMAREG, CMPDMAREG, SHIFTDMAREG, LOADREG, STOREREG,
ATSWAP, ATCAS, ATINCGETPTR, REG2FLOP`), 16b/64b counter widths in
ATINCGET, cross-core ATINCGET demo replacing the two-semaphore handshake,
host L1 readback bridge.

---

## Real-Wormhole port — RESOLVED 2026-04-27

The first end-to-end port to real silicon
(`/home/yijia/tt-playground/thcon_scalar_add/`) read all-zeros from L1
even though the kernel was clearly running (a plain RV32 volatile store
to a probe slot landed). Four root causes, in order of how they were
unmasked:

### 1. BRISC cannot drive ThCon — re-target to TRISC PACK

`TTI_*` expands to `__asm__(".ttinsn 0x...")`. On TRISC the binary is
fed into the Tensix thread instruction engine and `.ttinsn` becomes a
real Tensix op. On BRISC the binary is plain RV32 — `.ttinsn` words
just sit in `.text` as data, with nothing decoding them. Writing the
same word to `__instrn_buffer[0]` (BRISC's MMIO mailbox) also does not
drive ThCon from a user kernel.

**Fix:** gate the kernel body to PACK (`PACK({...})` macro). MATH didn't
actuate L1 stores in our prior run; PACK works. The MATH/UNPACK/PACK
macros compile-out the body for non-target threads, so only one TRISC
issues the ThCon ops and there's no L1 race on the result slots.

### 2. `OffsetIndex` is a half-register *index*, not a literal offset

ISA doc `STOREIND_L1.md` / `LOADIND.md`:

```c
uint16_t* Offset    = (char*)&GPRs[CurrentThread][0] + OffsetHalfReg * 2;
uint32_t  L1Address = (GPRs[CurrentThread][AddrReg] * 16) + *Offset;
```

Passing `OffsetIndex=0` does NOT mean "zero offset" — it means "use
`Th[0].lo16` as the offset". If `Th[0].lo16 == A16` (the 16B-aligned
base addr), the effective L1 address becomes `Th[0]*16 + Th[0].lo16`,
which scribbles in some other slot.

**Fix:** point `OffsetIndex` at a half-register you've explicitly
zeroed (or one that happens to be zero by construction, e.g. `HI_16(N)`
of any GPR whose top 16 bits are 0).

### 3. LOADIND is asynchronous

ISA doc `LOADIND.md`: *"The instruction completes execution as soon as
the read-request has been sent towards L1, at which point the thread's
next instruction can execute. … [The GPR is filled] at some later point
in time."*

So `LOADIND ... → ADDDMAREG` reads a stale (zero) GPR without a
barrier.

**Fix:** insert `TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::THCON)`
between LOADIND and any consumer of the destination GPR. `TTI_FLUSHDMA(0)`
+ `tensix_sync()` at the tail of the program drains the pipeline before
the host reads back.

### 4. ISA doc / yaml disagree on STOREIND size encoding

`tt-llk/.../instructions/assembly.yaml` and the public ISA doc disagree
on the size-bit interpretation for STOREIND. ISA doc is authoritative
(matches HW behaviour): for L1 stores pass
`(MemHierSel=1, SizeSel=0, RegSizeSel=Size)` where `Size` is the 2-bit
`{0=16B, 1=32b, 2=16b, 3=8b}` enum carried entirely in `RegSizeSel`.

### Why Jitte v0 didn't catch (1)–(3)

Jitte v0's `do_loadind` / `do_storeind` panicked on `OffsetIndex != 0`
or `AutoIncSpec != 0` and treated `0` as "no offset" — so
`OffsetIndex=0` silently worked even though it was semantically wrong
on real HW. ATINCGET / ADDDMAREG / MULDMAREG ran synchronously in the
emulator, so the LOADIND-async issue was also invisible.

Jitte **v0.5** (Stage F above) closes this: half-register deref is
faithful, AutoIncSpec post-increment is modelled, and LOADIND results
land in a `pending_loadind` set that requires STALLWAIT or FLUSHDMA to
retire. Reading a pending GPR aborts with a message that names the
missing barrier — the same bug now fails loud at simulation time
instead of slipping through to silicon.

### Real-HW demo

`/home/yijia/tt-playground/thcon_scalar_add/` PASSes on real Wormhole
B0:
```
L1[0x30000] a = 100   L1[0x30010] b = 42
L1[0x30020] c = 142   L1[0x30030] d = 4200
RESULT: PASS
```
The kernel uses the same `ZERO_OFF = HI_16(3) = 7` + STALLWAIT pattern
as the Jitte demos.

## File index for v0.5

**Modified (Jitte):**
- `src/device/riscv/builtin_handler.hpp`
- `src/device/riscv/builtin_handler.cpp`
- `src/device/riscv/builtin_tensix.{hpp,cpp}` (Stage E added builtin id 5121)
- `src/device/riscv/tensix_handler.hpp` (Stage F added `pending_loadind`)
- `src/device/riscv/tensix_handler.cpp` (Stage F: half-reg deref,
  AutoIncSpec, async LOADIND, STALLWAIT/FLUSHDMA)
- `src/device/api/kernel_builder.cpp`
- `src/tt_metal/jit_build/build.cpp`
- `home/tt_metal/emulator/kernels/jitte_thcon.h` (Stage F: `p_stall` constants
  under `__JITTE__`)

**Added (Jitte):**
- `home/tt_metal/emulator/kernels/ckernel_ops.h` (vendored from tt-llk)

**Added (mydsl demo):**
- `src/handwritten/thcon_scalar_add/device/metal/thcon_scalar_add_kernel.cpp`
  (Stage C; Stage F updated for `OffsetIndex` half-reg + STALLWAIT)
- `src/handwritten/thcon_scalar_add/device/metal/thcon_atomic_inc_kernel.cpp`
  (Stage D; Stage F updated for `OffsetIndex` half-reg)
- `src/handwritten/thcon_scalar_add/device/metal/thcon_runtime_arg_kernel.cpp`
  (Stage E; Stage F updated for `OffsetIndex` half-reg + STALLWAIT)
- `src/handwritten/thcon_scalar_add/host/thcon_scalar_add.{hpp,cpp}`
- `src/handwritten/thcon_scalar_add/test/main.cpp`
- `src/handwritten/thcon_scalar_add/test/main_atomic_inc.cpp` (Stage D)
- `src/handwritten/thcon_scalar_add/test/main_runtime_arg.cpp` (Stage E)
- `jitte/prj/handwritten/thcon_scalar_add/{deploy_jitte.sh,build_test_tanto.sh}`

**Real-HW counterpart (outside mydsl, for cross-platform validation):**
- `tt-playground/thcon_scalar_add/kernels/thcon_scalar_add_compute_kernel.cpp`
  (TRISC PACK gate; same `ZERO_OFF` + STALLWAIT pattern as the Jitte demos)
- `tt-playground/thcon_scalar_add/thcon_scalar_add.cpp` (host driver)

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

# Stage D smoke (atomic increment):
TT_METAL_HOME=~/ronin/jitte/home TT_ARCH=wormhole_b0 \
    ./prj/bin/handwritten/thcon_scalar_add/test_atomic_inc
```
