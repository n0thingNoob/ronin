// scalar_add.cpp — phase 1 v0 host wrapper
//
// Derived from algo/src/basic/host/tanto/eltwise_binary.cpp.

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/scalar_add.hpp"

namespace mydsl {
namespace handwritten {

namespace {

union U32 {
    float f;
    uint32_t i;
};

uint16_t float_to_bf16_bits(float f) {
    U32 u;
    u.f = f;
    return uint16_t(u.i >> 16);
}

} // namespace

ScalarAdd::ScalarAdd() { }
ScalarAdd::~ScalarAdd() { }

void ScalarAdd::init(
        const core::Device &device,
        int N,
        float scalar) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_N = uint32_t(N);
    m_scalar = scalar;
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "mydsl/handwritten/scalar_add/device/metal";
    m_defines = {{"T", "bfloat16"}};

    uint16_t s_bits = float_to_bf16_bits(m_scalar);
    m_b_filled.assign(m_N, s_bits);

    create_globals();
    create_pipes();
    create_kernels();
}

void ScalarAdd::run(
        const void *a,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, m_b_filled.data(), false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void ScalarAdd::create_globals() {
    uint32_t log2_page_size = 10; // 2^10 = 1024 (one tile in elements)
    m_ga = core::Global(m_device, T, m_N, log2_page_size);
    m_gb = core::Global(m_device, T, m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_N, log2_page_size);
}

void ScalarAdd::create_pipes() {
    m_pa =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pc =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void ScalarAdd::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void ScalarAdd::create_reader() {
    std::string path = m_kernel_base_path + "/scalar_add_reader.cpp";
    m_reader =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        m_num_blocks,
        m_block_tiles
    };
    m_reader.set_args(m_grid, args);
}

void ScalarAdd::create_writer() {
    std::string path = m_kernel_base_path + "/scalar_add_writer.cpp";
    m_writer =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::WRITER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        m_num_blocks,
        m_block_tiles
    };
    m_writer.set_args(m_grid, args);
}

void ScalarAdd::create_math() {
    std::string path = m_kernel_base_path + "/scalar_add_math.cpp";
    m_math =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::MATH,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        m_num_blocks,
        m_block_tiles
    };
    m_math.set_args(m_grid, args);
}

} // namespace handwritten
} // namespace mydsl
