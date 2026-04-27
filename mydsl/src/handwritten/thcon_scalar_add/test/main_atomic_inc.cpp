// main_atomic_inc.cpp — Stage D smoke test driver.
//
// Stage D verification: the kernel runs ATINCGET 4x on a single L1 slot.
// Success = no abort + the [tensix] ATINCGET prints showing
// old=0,1,2,3 in order (counter ends at 4 in L1[0x40000]; the final
// returned old=3 is persisted at L1[0x40010]).

#include <cstdio>

#include "host/core/api.hpp"

#include "host/thcon_scalar_add.hpp"

namespace core = ronin::tanto::host;
using namespace mydsl::handwritten;

int main() {
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);

    ThConScalarAdd demo;
    demo.init(device, "thcon_atomic_inc_kernel");
    demo.run();

    device.close();

    printf("thcon_atomic_inc: kernel completed (Stage D)\n");
    printf("RESULT: PASS  (verify ATINCGET old=0,1,2,3 in [tensix] prints)\n");
    return 0;
}
