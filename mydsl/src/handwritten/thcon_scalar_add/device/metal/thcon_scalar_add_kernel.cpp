// thcon_scalar_add_kernel.cpp — Stage A smoke kernel.
//
// Stage A goal: prove the entire ThCon interception chain works end-to-end:
//   linker → JALR hook → BuiltinHandler dispatch → INSTRUCTION_WORD redef.
//
// On Jitte, jitte_thcon.h redefines INSTRUCTION_WORD(x) to call the
// jitte_tensix_exec(x) builtin instead of emitting `.ttinsn`. The Stage A
// TensixHandler body just prints the encoded instruction word and aborts.
//
// Expected Jitte output:
//   [tensix] insn=0x45XXXXXX
//   Aborted

#include "dataflow_api.h"
#include "jitte_thcon.h"
#include "ckernel_ops.h"

void kernel_main() {
    TTI_SETDMAREG(0, 42, 0, 0);
}
