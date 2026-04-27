// main.cpp — Stage A smoke test driver.
//
// Expected (Jitte): kernel emits TTI_SETDMAREG, TensixHandler prints
//   [tensix] insn=0x45XXXXXX
// then aborts. Process exit code is non-zero. PASS iff that printf is seen.

#include <cstdio>

#include "host/core/api.hpp"

#include "host/thcon_scalar_add.hpp"

namespace core = ronin::tanto::host;
using namespace mydsl::handwritten;

int main() {
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);

    ThConScalarAdd demo;
    demo.init(device);
    demo.run();

    device.close();

    printf("thcon_scalar_add: dispatched (Stage A expects abort before this line)\n");
    return 0;
}
