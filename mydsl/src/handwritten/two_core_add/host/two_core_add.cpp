// two_core_add.cpp — phase 2 step 1 host wrapper

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/two_core_add.hpp"

namespace mydsl {
namespace handwritten {

TwoCoreAdd::TwoCoreAdd() { }
TwoCoreAdd::~TwoCoreAdd() { }

void TwoCoreAdd::init(
        const core::Device &device,
        int N) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_N = uint32_t(N);
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    // 1x2 grid: producer at (0,0), consumer at (0,1).
    m_grid = core::Grid(m_program, 0, 0, 0, 1);

    m_kernel_base_path = "mydsl/handwritten/two_core_add/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_semaphores();
    create_kernels();
}

void TwoCoreAdd::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void TwoCoreAdd::create_globals() {
    uint32_t log2_page_size = 10; // 2^10 = 1024 (one tile in elements)
    m_ga = core::Global(m_device, T, m_N, log2_page_size);
    m_gb = core::Global(m_device, T, m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_N, log2_page_size);
}

void TwoCoreAdd::create_pipes() {
    // pa, pb: cross-core data — declared as INPUT on the whole grid.
    // L1 layout is identical on both cores; producer NOC-writes into
    // consumer's pa/pb region.
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
    // pc: only consumer (core 1) uses it. Declared on full grid for layout
    // uniformity; producer's pc is unused.
    m_pc =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void TwoCoreAdd::create_semaphores() {
    m_sem_credit = core::Semaphore(m_program, m_grid, 0);
    m_sem_data = core::Semaphore(m_program, m_grid, 0);
}

void TwoCoreAdd::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void TwoCoreAdd::create_reader() {
    std::string path = m_kernel_base_path + "/two_core_add_reader.cpp";
    m_reader =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    uint32_t prod_x_phy, prod_y_phy;
    m_device.worker_core_from_logical_core(0, 0, prod_x_phy, prod_y_phy);
    /*
    void kernel(
            pipe<T> pa,
            pipe<T> pb,
            semaphore sem_credit,
            semaphore sem_data,
            uint32 prod_x,
            uint32 prod_y,
            uint32 num_blocks,
            uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_sem_credit,
        m_sem_data,
        prod_x_phy,
        prod_y_phy,
        m_num_blocks,
        m_block_tiles,
        uint32_t(0)
    };
    args[8] = uint32_t(0);
    m_reader.set_args(0, 0, args);
    args[8] = uint32_t(1);
    m_reader.set_args(0, 1, args);
}

void TwoCoreAdd::create_writer() {
    std::string path = m_kernel_base_path + "/two_core_add_writer.cpp";
    m_writer =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::WRITER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    uint32_t cons_x_phy, cons_y_phy;
    m_device.worker_core_from_logical_core(0, 1, cons_x_phy, cons_y_phy);
    /*
    void kernel(
            global<T> ga,
            global<T> gb,
            global<T> gc,
            pipe<T> pa,
            pipe<T> pb,
            pipe<T> pc,
            semaphore sem_credit,
            semaphore sem_data,
            uint32 ga_pos,
            uint32 gb_pos,
            uint32 gc_pos,
            uint32 cons_x,
            uint32 cons_y,
            uint32 num_blocks,
            uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_gc,
        m_pa,
        m_pb,
        m_pc,
        m_sem_credit,
        m_sem_data,
        uint32_t(0),     // ga_pos
        uint32_t(0),     // gb_pos
        uint32_t(0),     // gc_pos
        cons_x_phy,
        cons_y_phy,
        m_num_blocks,
        m_block_tiles,
        uint32_t(0)
    };
    args[15] = uint32_t(0);
    m_writer.set_args(0, 0, args);
    args[15] = uint32_t(1);
    m_writer.set_args(0, 1, args);
}

void TwoCoreAdd::create_math() {
    std::string path = m_kernel_base_path + "/two_core_add_math.cpp";
    m_math =
        core::Kernel(
            m_program,
            m_grid,
            core::KernelKind::MATH,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    /*
    void kernel(
            pipe<T> pa,
            pipe<T> pb,
            pipe<T> pc,
            uint32 num_blocks,
            uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        m_num_blocks,
        m_block_tiles,
        uint32_t(0)      // core_id
    };
    args[5] = uint32_t(0);
    m_math.set_args(0, 0, args);
    args[5] = uint32_t(1);
    m_math.set_args(0, 1, args);
}

} // namespace handwritten
} // namespace mydsl
