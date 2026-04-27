// mpmd_two_core_add.cpp — phase 2 step 3 (true MPMD)

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/mpmd_two_core_add.hpp"

namespace mydsl {
namespace handwritten {

MpmdTwoCoreAdd::MpmdTwoCoreAdd() { }
MpmdTwoCoreAdd::~MpmdTwoCoreAdd() { }

void MpmdTwoCoreAdd::init(
        const core::Device &device,
        int N) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_N = uint32_t(N);
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    // Union grid: covers both cores. Used to declare pipes/semaphores so
    // L1 layout matches on (0,0) and (0,1).
    m_grid_full = core::Grid(m_program, 0, 0, 0, 1);
    // Per-core grids: each kernel binary is bound to its own sub-grid.
    m_grid_head = core::Grid(m_program, 0, 0, 0, 0);
    m_grid_tail = core::Grid(m_program, 0, 1, 0, 1);

    m_kernel_base_path = "mydsl/handwritten/mpmd_two_core_add/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_semaphores();
    create_kernels();
}

void MpmdTwoCoreAdd::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void MpmdTwoCoreAdd::create_globals() {
    uint32_t log2_page_size = 10; // 2^10 = 1024 (one tile in elements)
    m_ga = core::Global(m_device, T, m_N, log2_page_size);
    m_gb = core::Global(m_device, T, m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_N, log2_page_size);
}

void MpmdTwoCoreAdd::create_pipes() {
    // pa, pb, pc all on union grid: same CB id, same L1 offset on both
    // cores. pc's slot on head's L1 is allocated but never touched.
    m_pa =
        core::Pipe(
            m_program,
            m_grid_full,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid_full,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pc =
        core::Pipe(
            m_program,
            m_grid_full,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void MpmdTwoCoreAdd::create_semaphores() {
    m_sem_credit = core::Semaphore(m_program, m_grid_full, 0);
    m_sem_data = core::Semaphore(m_program, m_grid_full, 0);
}

void MpmdTwoCoreAdd::create_kernels() {
    create_head_reader();
    create_head_writer();
    create_head_math();
    create_tail_reader();
    create_tail_writer();
    create_tail_math();
}

void MpmdTwoCoreAdd::create_head_reader() {
    std::string path = m_kernel_base_path + "/head_reader.cpp";
    m_head_reader =
        core::Kernel(
            m_program,
            m_grid_head,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    /* void kernel() */
    std::vector<core::KernelArg> args{};
    m_head_reader.set_args(0, 0, args);
}

void MpmdTwoCoreAdd::create_head_writer() {
    std::string path = m_kernel_base_path + "/head_writer.cpp";
    m_head_writer =
        core::Kernel(
            m_program,
            m_grid_head,
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
            pipe<T> pa,
            pipe<T> pb,
            semaphore sem_credit,
            semaphore sem_data,
            uint32 ga_pos,
            uint32 gb_pos,
            uint32 cons_x,
            uint32 cons_y,
            uint32 num_blocks,
            uint32 block_tiles)
    */
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        m_sem_credit,
        m_sem_data,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        cons_x_phy,
        cons_y_phy,
        m_num_blocks,
        m_block_tiles
    };
    m_head_writer.set_args(0, 0, args);
}

void MpmdTwoCoreAdd::create_head_math() {
    std::string path = m_kernel_base_path + "/head_math.cpp";
    m_head_math =
        core::Kernel(
            m_program,
            m_grid_head,
            core::KernelKind::MATH,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    /*
    void kernel(
            pipe<T> pa,
            pipe<T> pb,
            uint32 num_blocks,
            uint32 block_tiles)
    */
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_num_blocks,
        m_block_tiles
    };
    m_head_math.set_args(0, 0, args);
}

void MpmdTwoCoreAdd::create_tail_reader() {
    std::string path = m_kernel_base_path + "/tail_reader.cpp";
    m_tail_reader =
        core::Kernel(
            m_program,
            m_grid_tail,
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
            uint32 block_tiles)
    */
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_sem_credit,
        m_sem_data,
        prod_x_phy,
        prod_y_phy,
        m_num_blocks,
        m_block_tiles
    };
    m_tail_reader.set_args(0, 1, args);
}

void MpmdTwoCoreAdd::create_tail_writer() {
    std::string path = m_kernel_base_path + "/tail_writer.cpp";
    m_tail_writer =
        core::Kernel(
            m_program,
            m_grid_tail,
            core::KernelKind::WRITER,
            core::KernelFormat::METAL,
            path,
            {},
            m_defines);
    /*
    void kernel(
            global<T> gc,
            pipe<T> pc,
            uint32 gc_pos,
            uint32 num_blocks,
            uint32 block_tiles)
    */
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        m_num_blocks,
        m_block_tiles
    };
    m_tail_writer.set_args(0, 1, args);
}

void MpmdTwoCoreAdd::create_tail_math() {
    std::string path = m_kernel_base_path + "/tail_math.cpp";
    m_tail_math =
        core::Kernel(
            m_program,
            m_grid_tail,
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
            uint32 block_tiles)
    */
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        m_num_blocks,
        m_block_tiles
    };
    m_tail_math.set_args(0, 1, args);
}

} // namespace handwritten
} // namespace mydsl
