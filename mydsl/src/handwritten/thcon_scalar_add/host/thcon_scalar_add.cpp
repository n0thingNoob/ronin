// thcon_scalar_add.cpp — Stage A smoke host.

#include <string>

#include "host/core/api.hpp"

#include "host/thcon_scalar_add.hpp"

namespace mydsl {
namespace handwritten {

ThConScalarAdd::ThConScalarAdd() { }
ThConScalarAdd::~ThConScalarAdd() { }

void ThConScalarAdd::init(const core::Device &device) {
    m_device = device;
    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    std::string path = "mydsl/handwritten/thcon_scalar_add/device/metal/thcon_scalar_add_kernel.cpp";
    m_kernel =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path,
            {},
            {});
    m_kernel.set_args(m_grid, {});
}

void ThConScalarAdd::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
    queue.finish();
}

} // namespace handwritten
} // namespace mydsl
