// main.cpp — Stage B smoke test driver.
//
// Stage B verification: the kernel writes LOAD_VALUE (42) to an L1
// scratch slot via STOREIND and reads it back via LOADIND. Success is
// the TensixHandler diagnostic prints showing a round-trip match (no
// abort). Process exits 0.

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

    printf("thcon_scalar_add: kernel completed (Stage B)\n");
    printf("RESULT: PASS  (verify round-trip via [tensix] STOREIND/LOADIND prints)\n");
    return 0;
}
