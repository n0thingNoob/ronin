// thcon_scalar_add.hpp — Stage A smoke host.
//
// Single core (0,0), single BRISC kernel, no pipes, no readback. Used to
// smoke-test that a TTI_* instruction issued from a Jitte kernel reaches
// the TensixHandler.

#pragma once

#include "host/core/api.hpp"

namespace mydsl {
namespace handwritten {

namespace core = ronin::tanto::host;

class ThConScalarAdd {
public:
    ThConScalarAdd();
    ~ThConScalarAdd();
public:
    void init(const core::Device &device);
    void run();
private:
    core::Device m_device;
    core::Program m_program;
    core::Grid m_grid;
    core::Kernel m_kernel;
};

} // namespace handwritten
} // namespace mydsl
