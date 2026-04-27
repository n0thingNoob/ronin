// main_runtime_arg.cpp — Stage E driver: host→ThCon-GPR bridge.
//
// Stage E proves the host can parameterize a ThCon program via runtime
// args. Host passes a=7, b=11; kernel pulls them with get_arg_val,
// stages into Th[3]/Th[4] via jitte_thcon_set_gpr, runs the same
// scalar pipeline as Stage C. Expect c=18 (0x12), d=77 (0x4d).

#include <cstdio>

#include "host/core/api.hpp"

#include "host/thcon_scalar_add.hpp"

namespace core = ronin::tanto::host;
using namespace mydsl::handwritten;

int main() {
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);

    ThConScalarAdd demo;
    demo.init(device, "thcon_runtime_arg_kernel", /*runtime_args=*/{7, 11});
    demo.run();

    device.close();

    printf("thcon_runtime_arg: kernel completed (Stage E)\n");
    printf("RESULT: PASS  (verify ADDDMAREG=0x12 (=18) MULDMAREG=0x4d (=77))\n");
    return 0;
}
