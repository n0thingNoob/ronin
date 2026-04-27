// thcon_runtime_arg_kernel.cpp — Stage E: host→ThCon-GPR bridge demo.
//
// Same scalar dataflow as Stage C, but `a` and `b` are no longer
// hardcoded: host passes them as runtime args. The kernel pulls them
// via `get_arg_val<uint32_t>` and stages them into Th[3] / Th[4] using
// the Jitte-only `jitte_thcon_set_gpr` builtin. The rest of the
// program (ADDDMAREG / MULDMAREG / STOREIND) is unchanged from Stage C.
//
// Verification: with host args (a=7, b=11), expect c=18 (0x12) and
// d=77 (0x4d) in the [tensix] prints.
//
// Real-Wormhole semantics (matched by Jitte v0.5 as of 2026-04-27):
//   * Th[AddrReg] holds a 16B-aligned address (byte_addr >> 4).
//   * LOADIND  SizeSel=1 → 32b.
//   * LOADIND is asynchronous: insert TTI_STALLWAIT before consumers.
//   * STOREIND L1 32b store: (MemHierSel=1, SizeSel=1, RegSizeSel=0).
//   * STOREIND/LOADIND OffsetIndex is a HALF-REGISTER INDEX, not a literal
//     offset. Pass an index pointing at a known-zero half-reg (here we
//     stage Th[5]=0 explicitly and use LO_16(5)=10).

#include "dataflow_api.h"
#include "jitte_thcon.h"
#include "ckernel_ops.h"

namespace {

// Byte addresses (16B-aligned).
//
// SETDMAREG's 14-bit immediate forces byte_addr < 0x40000 once we shift
// down by 4 to feed the ThCon address GPRs. Use 0x10000-base.
constexpr uint32_t A_SLOT = 0x10000;
constexpr uint32_t B_SLOT = 0x10010;
constexpr uint32_t C_SLOT = 0x10020;
constexpr uint32_t D_SLOT = 0x10030;

constexpr uint32_t A16 = A_SLOT >> 4;
constexpr uint32_t B16 = B_SLOT >> 4;
constexpr uint32_t C16 = C_SLOT >> 4;
constexpr uint32_t D16 = D_SLOT >> 4;

constexpr uint32_t LO_16(uint32_t n) { return 2 * n; }
constexpr uint32_t HI_16(uint32_t n) { return 2 * n + 1; }

} // namespace

void kernel_main() {
    // ---- Pull a, b from host runtime args ----
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);

    // ---- Bridge: stage a, b into Th[3], Th[4] ----
    jitte_thcon_set_gpr(3, a);
    jitte_thcon_set_gpr(4, b);

    // ---- 16B-aligned slot addresses via SETDMAREG (compile-time consts) ----
    TTI_SETDMAREG(0, A16 & 0xFFFFu, 0, LO_16(0));
    TTI_SETDMAREG(0, (A16 >> 16) & 0x3FFFu, 0, HI_16(0));
    TTI_SETDMAREG(0, B16 & 0xFFFFu, 0, LO_16(1));
    TTI_SETDMAREG(0, (B16 >> 16) & 0x3FFFu, 0, HI_16(1));
    TTI_SETDMAREG(0, C16 & 0xFFFFu, 0, LO_16(2));
    TTI_SETDMAREG(0, (C16 >> 16) & 0x3FFFu, 0, HI_16(2));
    TTI_SETDMAREG(0, D16 & 0xFFFFu, 0, LO_16(10));
    TTI_SETDMAREG(0, (D16 >> 16) & 0x3FFFu, 0, HI_16(10));

    // ---- Th[5] = 0 (zero half-reg pool — Th[3]/Th[4] could carry hi-bits
    //      from arbitrary host args, so don't rely on their hi halves). ----
    TTI_SETDMAREG(0, 0, 0, LO_16(5));
    TTI_SETDMAREG(0, 0, 0, HI_16(5));
    constexpr uint32_t ZERO_OFF = LO_16(5);

    // ---- Seed L1, then run ALU pipeline same as Stage C ----
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/3, /*Addr*/0);
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/4, /*Addr*/1);
    TTI_LOADIND(1, ZERO_OFF, 0, /*Data*/6, /*Addr*/0);
    TTI_LOADIND(1, ZERO_OFF, 0, /*Data*/7, /*Addr*/1);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::THCON);
    TTI_ADDDMAREG(0, /*Result*/8, /*OpB*/7, /*OpA*/6);
    TTI_MULDMAREG(0, /*Result*/9, /*OpB*/7, /*OpA*/6);
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/8, /*Addr*/2);
    TTI_STOREIND(1, 1, 0, ZERO_OFF, 0, /*Data*/9, /*Addr*/10);
}
