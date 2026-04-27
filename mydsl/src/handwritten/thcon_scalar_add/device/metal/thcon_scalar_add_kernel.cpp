// thcon_scalar_add_kernel.cpp — Stage B smoke kernel.
//
// Stage B exercises the ThCon GPR file + L1 R/W path:
//   1. SETDMAREG to load a value (42) into Th[3]
//   2. SETDMAREG to load an L1 scratch slot address into Th[0]
//   3. STOREIND to write Th[3] into L1[Th[0]]
//   4. LOADIND  to read it back into Th[5]
//
// Verification (Stage B): the TensixHandler diagnostic prints show a
// matching round-trip — the value stored at L1[slot] equals the value
// loaded back into Th[5]. No abort.
//
// Notes:
// - LO_16(N) = 2*N, HI_16(N) = 2*N+1 (ThCon GPR half-word addressing).
// - SETDMAREG immediate is 14 bits (per ckernel_ops.h encoding); 42 and
//   the 16-bit halves of a 0x4xxxx address fit fine.
// - L1 scratch slot at 0x40000 sits comfortably above Jitte's
//   L1_UNRESERVED_BASE (120 KiB).

#include "dataflow_api.h"
#include "jitte_thcon.h"
#include "ckernel_ops.h"

namespace {

constexpr uint32_t SCRATCH_SLOT = 0x40000;
constexpr uint32_t LOAD_VALUE   = 42;

constexpr uint32_t LO_16(uint32_t n) { return 2 * n; }
constexpr uint32_t HI_16(uint32_t n) { return 2 * n + 1; }

} // namespace

void kernel_main() {
    // Th[3] = LOAD_VALUE (low half only; high half stays 0 from init)
    TTI_SETDMAREG(0, LOAD_VALUE & 0xFFFFu, 0, LO_16(3));
    TTI_SETDMAREG(0, (LOAD_VALUE >> 16) & 0x3FFFu, 0, HI_16(3));

    // Th[0] = SCRATCH_SLOT
    TTI_SETDMAREG(0, SCRATCH_SLOT & 0xFFFFu, 0, LO_16(0));
    TTI_SETDMAREG(0, (SCRATCH_SLOT >> 16) & 0x3FFFu, 0, HI_16(0));

    // L1[Th[0]] <- Th[3]   (32-bit store: SizeSel=0, RegSizeSel=0)
    TTI_STOREIND(/*MemHierSel*/0, /*SizeSel*/0, /*RegSizeSel*/0,
                 /*OffsetIndex*/0, /*AutoIncSpec*/0,
                 /*DataRegIndex*/3, /*AddrRegIndex*/0);

    // Th[5] <- L1[Th[0]]   (32-bit load: SizeSel=2)
    TTI_LOADIND(/*SizeSel*/2, /*OffsetIndex*/0, /*AutoIncSpec*/0,
                /*DataRegIndex*/5, /*AddrRegIndex*/0);
}
