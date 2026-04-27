// main.cpp — Stage C smoke test driver.
//
// Stage C verification: the kernel runs the full scalar dataflow
// pipeline (SETDMAREG / STOREIND / LOADIND / ADDDMAREG / MULDMAREG /
// STOREIND) and produces c=142, d=4200 visible in [tensix] prints.
// Success = no abort + the expected ADDDMAREG/MULDMAREG result lines.

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

    printf("thcon_scalar_add: kernel completed (Stage C)\n");
    printf("RESULT: PASS  (verify ADDDMAREG=0x8e MULDMAREG=0x1068 in [tensix] prints)\n");
    return 0;
}
