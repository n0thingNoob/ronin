// thcon_scalar_add.cpp — Stage A smoke host.

#include <string>

#include "host/core/api.hpp"

#include "host/thcon_scalar_add.hpp"

namespace mydsl {
namespace handwritten {

ThConScalarAdd::ThConScalarAdd() { }
ThConScalarAdd::~ThConScalarAdd() { }

void ThConScalarAdd::init(
        const core::Device &device,
        const std::string &kernel_name,
        const std::vector<uint32_t> &runtime_args) {
    m_device = device;
    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    std::string path =
        "mydsl/handwritten/thcon_scalar_add/device/metal/"
        + kernel_name + ".cpp";
    m_kernel =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path,
            {},
            {});
    std::vector<core::KernelArg> kargs;
    kargs.reserve(runtime_args.size());
    for (uint32_t v : runtime_args) {
        kargs.emplace_back(v);
    }
    m_kernel.set_args(m_grid, kargs);
}

void ThConScalarAdd::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
    queue.finish();
}

} // namespace handwritten
} // namespace mydsl
