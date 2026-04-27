// thcon_atomic_inc_kernel.cpp — Stage D smoke kernel.
//
// Exercises ATINCGET in a small loop:
//   1. Seed L1[COUNTER_SLOT] = 0 via STOREIND.
//   2. Run N iterations of ATINCGET on COUNTER_SLOT — each adds Th[3]
//      into the L1 cell (mod 1<<WrapVal) and overwrites Th[3] with the
//      pre-increment value. Th[3] must therefore be re-staked to 1
//      before every iteration.
//   3. STOREIND the final returned old value to RESULT_SLOT.
//
// Verification (Stage D): after N=4 iterations, the four ATINCGET prints
// show old=0,1,2,3 in order; L1[COUNTER_SLOT] ends at 4; L1[RESULT_SLOT]
// holds the last-returned old=3. No abort.
//
// Real-Wormhole semantics (matched by Jitte v0.5 as of 2026-04-27):
//   * Th[AddrReg] holds a 16B-aligned address (byte_addr >> 4).
//   * STOREIND L1 32b store: (MemHierSel=1, SizeSel=1, RegSizeSel=0).
//   * STOREIND OffsetIndex is a HALF-REGISTER INDEX (the half-reg's value
//     is the addend), not a literal offset. Use a half-reg known to be
//     zero — Th[2] is set to 0 here, so LO_16(2)=4 works.
//   * ATINCGET (MemHierSel=1=L1):
//       inc = Th[Data]; old = *L1; new = (old+inc) & ((1<<WrapVal)-1);
//       *L1 = new; Th[Data] = old;
//     WrapVal=31 gives a near-32b counter (mask=0x7FFFFFFF). WrapVal=0
//     means mask=0 → always zeroes the cell; never use that here.

#include "dataflow_api.h"
#include "jitte_thcon.h"
#include "ckernel_ops.h"

namespace {

// Byte addresses (16B-aligned).
//
// SETDMAREG's 14-bit immediate forces byte_addr < 0x40000 once we shift
// down by 4 to feed the ThCon address GPRs. Use 0x10000-base.
constexpr uint32_t COUNTER_SLOT = 0x10000;
constexpr uint32_t RESULT_SLOT  = 0x10010;
constexpr uint32_t ITERATIONS   = 4;

constexpr uint32_t COUNTER16 = COUNTER_SLOT >> 4;
constexpr uint32_t RESULT16  = RESULT_SLOT  >> 4;

constexpr uint32_t LO_16(uint32_t n) { return 2 * n; }
constexpr uint32_t HI_16(uint32_t n) { return 2 * n + 1; }

} // namespace

void kernel_main() {
    // ---- Th[0] = COUNTER16, Th[1] = RESULT16 (16B-aligned) ----
    TTI_SETDMAREG(0, COUNTER16 & 0xFFFFu, 0, LO_16(0));
    TTI_SETDMAREG(0, (COUNTER16 >> 16) & 0x3FFFu, 0, HI_16(0));
    TTI_SETDMAREG(0, RESULT16 & 0xFFFFu, 0, LO_16(1));
    TTI_SETDMAREG(0, (RESULT16 >> 16) & 0x3FFFu, 0, HI_16(1));

    // ---- Th[2] = 0 (seed value AND zero half-reg pool) ----
    TTI_SETDMAREG(0, 0, 0, LO_16(2));
    TTI_SETDMAREG(0, 0, 0, HI_16(2));

    // OffsetIndex points at LO_16(2) — Th[2] is now fully zero, so the
    // address addend is 0 and effective L1 address = Th[AddrReg]*16.
    constexpr uint32_t ZERO_OFF = LO_16(2);

    // ---- Seed L1: COUNTER_SLOT <- 0 (32b L1 store) ----
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/2, /*Addr*/0);

    // ---- N x { Th[3] = 1; ATINCGET } ----
    //
    // ATINCGET consumes Th[3] as the increment AND overwrites it with the
    // old counter value, so we must re-stake Th[3]=1 before each iteration.
    // Macro args: TTI_ATINCGET(MemHierSel, WrapVal, Sel32b, DataReg, AddrReg).
    // WrapVal=31 → counter wraps at 2^31, fine for ITERATIONS=4.
    TTI_SETDMAREG(0, 1, 0, LO_16(3));
    TTI_SETDMAREG(0, 0, 0, HI_16(3));
    TTI_ATINCGET(/*MemHier*/1, /*Wrap*/31, /*Sel32b*/0, /*Data*/3, /*Addr*/0);

    TTI_SETDMAREG(0, 1, 0, LO_16(3));
    TTI_SETDMAREG(0, 0, 0, HI_16(3));
    TTI_ATINCGET(/*MemHier*/1, /*Wrap*/31, /*Sel32b*/0, /*Data*/3, /*Addr*/0);

    TTI_SETDMAREG(0, 1, 0, LO_16(3));
    TTI_SETDMAREG(0, 0, 0, HI_16(3));
    TTI_ATINCGET(/*MemHier*/1, /*Wrap*/31, /*Sel32b*/0, /*Data*/3, /*Addr*/0);

    TTI_SETDMAREG(0, 1, 0, LO_16(3));
    TTI_SETDMAREG(0, 0, 0, HI_16(3));
    TTI_ATINCGET(/*MemHier*/1, /*Wrap*/31, /*Sel32b*/0, /*Data*/3, /*Addr*/0);
    static_assert(ITERATIONS == 4, "loop unrolled to 4");

    // ---- Persist the final returned old value (3) to RESULT_SLOT ----
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/3, /*Addr*/1);
}
