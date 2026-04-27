// main.cpp — phase 1 v0 test driver for scalar_add
//
// Self-contained: no algo/test/util/* dependency. Just generates a
// deterministic A, runs the device path, computes a CPU reference, and
// reports max diff. Returns 0 on PASS.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#include "host/core/api.hpp"

#include "host/scalar_add.hpp"

namespace core = ronin::tanto::host;
using namespace mydsl::handwritten;

namespace {

union U32 {
    float f;
    uint32_t i;
};

std::vector<uint16_t> float_to_bf16(const std::vector<float> &x) {
    U32 u;
    std::vector<uint16_t> y(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        u.f = x[i];
        y[i] = uint16_t(u.i >> 16);
    }
    return y;
}

std::vector<float> bf16_to_float(const std::vector<uint16_t> &x) {
    U32 u;
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        u.i = uint32_t(x[i]) << 16;
        y[i] = u.f;
    }
    return y;
}

} // namespace

int main() {
    int N = 4 * 1024; // 4 tiles
    float scalar = 3.0f;

    std::vector<float> a(N);
    for (int i = 0; i < N; i++) {
        a[i] = float(i) * 0.01f;
    }

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) {
        expected[i] = a[i] + scalar;
    }
    // round expected through bf16 so comparison is fair
    std::vector<float> expected_bf16 = bf16_to_float(float_to_bf16(expected));

    std::vector<uint16_t> ta = float_to_bf16(a);
    std::vector<uint16_t> tc(N);

    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);

    ScalarAdd solver;
    solver.init(device, N, scalar);
    solver.run(ta.data(), tc.data());

    core::Queue queue(device, 0);
    queue.finish();
    device.close();

    std::vector<float> c = bf16_to_float(tc);

    float max_diff = 0.0f;
    int num_bad = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(c[i] - expected_bf16[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1.0e-2f) num_bad++;
    }

    printf("scalar_add: N=%d, scalar=%g\n", N, scalar);
    printf("  a[0..4]   = %g %g %g %g %g\n", a[0], a[1], a[2], a[3], a[4]);
    printf("  c[0..4]   = %g %g %g %g %g\n", c[0], c[1], c[2], c[3], c[4]);
    printf("  exp[0..4] = %g %g %g %g %g\n",
        expected_bf16[0], expected_bf16[1], expected_bf16[2],
        expected_bf16[3], expected_bf16[4]);
    printf("  max_diff=%g  num_bad=%d/%d\n", max_diff, num_bad, N);

    bool pass = (num_bad == 0);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
