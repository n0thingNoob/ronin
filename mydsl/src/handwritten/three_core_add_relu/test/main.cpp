// main.cpp — phase 2 step 2 test driver for three_core_add_relu
//
// 3-core dataflow:
//   head reads A, B from DRAM; middle adds T = A+B; tail does C = relu(T) and
//   writes C back to DRAM.
//
// CPU reference: C = max(0, A + B). Uses bf16-clean inputs (mix of small
// positives and negatives that round-trip through bf16) so the float
// reference matches the bf16-domain device path bit-for-bit.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#include "host/core/api.hpp"

#include "host/three_core_add_relu.hpp"

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

    std::vector<float> a(N);
    std::vector<float> b(N);
    for (int i = 0; i < N; i++) {
        // Mix of positive and negative integer-like values, exactly
        // representable in bf16. Some sums will be negative (relu zeroes them),
        // some positive (relu passes through).
        a[i] = float((i % 16) - 8);          // -8..7
        b[i] = float(((i * 3) % 16) - 8);    // -8..7
    }

    std::vector<float> expected(N);
    for (int i = 0; i < N; i++) {
        float t = a[i] + b[i];
        expected[i] = (t > 0.0f) ? t : 0.0f;
    }
    std::vector<float> expected_bf16 = bf16_to_float(float_to_bf16(expected));

    std::vector<uint16_t> ta = float_to_bf16(a);
    std::vector<uint16_t> tb = float_to_bf16(b);
    std::vector<uint16_t> tc(N);

    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);

    ThreeCoreAddRelu solver;
    solver.init(device, N);
    solver.run(ta.data(), tb.data(), tc.data());

    core::Queue queue(device, 0);
    queue.finish();
    device.close();

    std::vector<float> c = bf16_to_float(tc);

    float max_diff = 0.0f;
    int num_bad = 0;
    int first_bad = -1;
    for (int i = 0; i < N; i++) {
        float diff = std::fabs(c[i] - expected_bf16[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1.0e-2f) {
            if (first_bad < 0) first_bad = i;
            num_bad++;
        }
    }
    if (first_bad >= 0) {
        printf("  first_bad: i=%d c=%g exp=%g a=%g b=%g\n",
            first_bad, c[first_bad], expected_bf16[first_bad],
            a[first_bad], b[first_bad]);
        int bad_per_tile[4] = {0, 0, 0, 0};
        for (int i = 0; i < N; i++) {
            float diff = std::fabs(c[i] - expected_bf16[i]);
            if (diff > 1.0e-2f) bad_per_tile[i / 1024]++;
        }
        printf("  bad per tile: %d %d %d %d\n",
            bad_per_tile[0], bad_per_tile[1],
            bad_per_tile[2], bad_per_tile[3]);
    }

    printf("three_core_add_relu: N=%d\n", N);
    printf("  a[0..4]   = %g %g %g %g %g\n", a[0], a[1], a[2], a[3], a[4]);
    printf("  b[0..4]   = %g %g %g %g %g\n", b[0], b[1], b[2], b[3], b[4]);
    printf("  c[0..4]   = %g %g %g %g %g\n", c[0], c[1], c[2], c[3], c[4]);
    printf("  exp[0..4] = %g %g %g %g %g\n",
        expected_bf16[0], expected_bf16[1], expected_bf16[2],
        expected_bf16[3], expected_bf16[4]);
    printf("  max_diff=%g  num_bad=%d/%d\n", max_diff, num_bad, N);

    bool pass = (num_bad == 0);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
