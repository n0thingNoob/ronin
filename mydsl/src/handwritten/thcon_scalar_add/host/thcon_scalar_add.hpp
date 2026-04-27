// thcon_scalar_add.hpp — Stage A smoke host.
//
// Single core (0,0), single BRISC kernel, no pipes, no readback. Used to
// smoke-test that a TTI_* instruction issued from a Jitte kernel reaches
// the TensixHandler.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

namespace mydsl {
namespace handwritten {

namespace core = ronin::tanto::host;

class ThConScalarAdd {
public:
    ThConScalarAdd();
    ~ThConScalarAdd();
public:
    // kernel_name: file stem under device/metal/, e.g. "thcon_scalar_add_kernel"
    // or "thcon_atomic_inc_kernel". Default keeps Stage A/B/C behavior.
    void init(
        const core::Device &device,
        const std::string &kernel_name = "thcon_scalar_add_kernel",
        const std::vector<uint32_t> &runtime_args = {});
    void run();
private:
    core::Device m_device;
    core::Program m_program;
    core::Grid m_grid;
    core::Kernel m_kernel;
};

} // namespace handwritten
} // namespace mydsl
