
#include "tanto/dataflow.h"

#define T bfloat16

// head_reader.cpp — phase 2 step 3 (true MPMD)
//
// Head core's reader (NCRISC). No-op: head's writer does both DRAM read and
// NOC mcast in one kernel. Stub exists because the Tanto/Metal kernel
// triple (reader/writer/math) is bound per Grid; leaving NCRISC unbound
// would change how the Program dispatches.

void kernel() {}
void kernel_main() { kernel(); }

