// thcon_scalar_add_kernel.cpp — Stage C smoke kernel.
//
// Stage C exercises the full scalar dataflow pipeline through ThCon:
//   1. SETDMAREG to load `a` (100) and `b` (42) into Th GPRs.
//   2. STOREIND  to seed L1 slots A_SLOT/B_SLOT (proves write path).
//   3. LOADIND   to read those slots back into different GPRs (proves
//                read path is not just a register-cache illusion).
//   4. ADDDMAREG to compute `c = a + b`         → 142
//   5. MULDMAREG to compute `d = a * b`         → 4200
//   6. STOREIND  to write c, d to C_SLOT/D_SLOT.
//
// Verification (Stage C): the TensixHandler diagnostic prints show
// `ADDDMAREG ... = 0x8e (=142)` and `MULDMAREG ... = 0x1068 (=4200)`,
// followed by the two STOREINDs writing those exact values to L1. No abort.
//
// Values fit easily under the 14-bit SETDMAREG immediate limit.
//
// Real-Wormhole semantics (matched by Jitte v0.5 as of 2026-04-27):
//   * Th[AddrReg] holds a 16B-aligned address (byte_addr >> 4), not a
//     byte address. Slots are spaced 16 bytes apart.
//   * OffsetIndex (4th macro arg of TT_STOREIND, 2nd of TT_LOADIND) is a
//     HALF-REGISTER INDEX — the half-register's *current value* becomes
//     the byte addend to (Th[AddrReg]*16). It is NOT a literal offset.
//     Pass an index pointing at a half-register known to be 0; e.g.
//     HI_16(3)=7 since A_VAL=100 leaves Th[3].hi16 zero.
//   * LOADIND is asynchronous: GPR fill happens later. Insert a
//     TTI_STALLWAIT(STALL_THREAD, THCON) before any consumer of the
//     LOADIND destination GPR.
//   * LOADIND  SizeSel=1 selects 32b.
//   * STOREIND args: (MemHierSel, SizeSel, RegSizeSel, OffsetIndex,
//                     AutoIncSpec, DataRegIndex, AddrRegIndex)
//     L1 + 32b store: MemHierSel=1, SizeSel=1, RegSizeSel=0.

#include "dataflow_api.h"
#include "jitte_thcon.h"
#include "ckernel_ops.h"

namespace {

constexpr uint32_t A_VAL  = 100;
constexpr uint32_t B_VAL  = 42;

// Byte addresses (16B-aligned).
//
// SETDMAREG's 14-bit immediate caps the LO half at 0x3FFF, so the 16B-
// aligned form (byte_addr >> 4) must fit in 14 bits, i.e. byte_addr <
// 0x40000. We pick 0x10000-base; A16=0x1000 fits cleanly.
constexpr uint32_t A_SLOT = 0x10000;
constexpr uint32_t B_SLOT = 0x10010;
constexpr uint32_t C_SLOT = 0x10020;
constexpr uint32_t D_SLOT = 0x10030;

// 16B-aligned forms — what the ThCon address GPRs actually hold.
constexpr uint32_t A16 = A_SLOT >> 4;
constexpr uint32_t B16 = B_SLOT >> 4;
constexpr uint32_t C16 = C_SLOT >> 4;
constexpr uint32_t D16 = D_SLOT >> 4;

constexpr uint32_t LO_16(uint32_t n) { return 2 * n; }
constexpr uint32_t HI_16(uint32_t n) { return 2 * n + 1; }

} // namespace

void kernel_main() {
    // ---- Th[3] = A_VAL, Th[4] = B_VAL ----
    TTI_SETDMAREG(0, A_VAL & 0xFFFFu, 0, LO_16(3));
    TTI_SETDMAREG(0, (A_VAL >> 16) & 0x3FFFu, 0, HI_16(3));
    TTI_SETDMAREG(0, B_VAL & 0xFFFFu, 0, LO_16(4));
    TTI_SETDMAREG(0, (B_VAL >> 16) & 0x3FFFu, 0, HI_16(4));

    // ---- Th[0..2,10] = 16B-aligned slot addresses ----
    TTI_SETDMAREG(0, A16 & 0xFFFFu, 0, LO_16(0));
    TTI_SETDMAREG(0, (A16 >> 16) & 0x3FFFu, 0, HI_16(0));
    TTI_SETDMAREG(0, B16 & 0xFFFFu, 0, LO_16(1));
    TTI_SETDMAREG(0, (B16 >> 16) & 0x3FFFu, 0, HI_16(1));
    TTI_SETDMAREG(0, C16 & 0xFFFFu, 0, LO_16(2));
    TTI_SETDMAREG(0, (C16 >> 16) & 0x3FFFu, 0, HI_16(2));
    TTI_SETDMAREG(0, D16 & 0xFFFFu, 0, LO_16(10));
    TTI_SETDMAREG(0, (D16 >> 16) & 0x3FFFu, 0, HI_16(10));

    // OffsetIndex must point at a half-reg that holds 0. A_VAL=100 fits in
    // the low 16 bits of Th[3], so Th[3].hi16 (= half-reg HI_16(3) = 7) is
    // a guaranteed zero. With OffsetHalfReg=ZERO_OFF, the address addend
    // is 0 and the effective L1 address is just (Th[AddrReg] * 16).
    constexpr uint32_t ZERO_OFF = HI_16(3);

    // ---- Seed L1: A_SLOT <- a, B_SLOT <- b ----
    // (MemHierSel=1, SizeSel=1, RegSizeSel=0) → 32b L1 store.
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/3, /*Addr*/0);
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/4, /*Addr*/1);

    // ---- Read back into Th[6], Th[7] (SizeSel=1 → 32b) ----
    TTI_LOADIND(1, ZERO_OFF, 0, /*Data*/6, /*Addr*/0);
    TTI_LOADIND(1, ZERO_OFF, 0, /*Data*/7, /*Addr*/1);

    // LOADIND is async — drain ThCon before consuming Th[6]/Th[7].
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::THCON);

    // ---- c = a + b   → Th[8] ----
    TTI_ADDDMAREG(/*OpBisConst*/0, /*Result*/8, /*OpB*/7, /*OpA*/6);
    // ---- d = a * b   → Th[9] ----
    TTI_MULDMAREG(/*OpBisConst*/0, /*Result*/9, /*OpB*/7, /*OpA*/6);

    // ---- Store results: C_SLOT <- c, D_SLOT <- d ----
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/8, /*Addr*/2);
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/9, /*Addr*/10);
}
