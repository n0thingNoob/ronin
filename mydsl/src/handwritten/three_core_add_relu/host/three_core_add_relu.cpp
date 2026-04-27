// three_core_add_relu.cpp — phase 2 step 2 host wrapper

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/three_core_add_relu.hpp"

namespace mydsl {
namespace handwritten {

ThreeCoreAddRelu::ThreeCoreAddRelu() { }
ThreeCoreAddRelu::~ThreeCoreAddRelu() { }

void ThreeCoreAddRelu::init(
        const core::Device &device,
        int N) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_N = uint32_t(N);
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    // 1x3 grid: head (0,0), middle (0,1), tail (0,2).
    m_grid = core::Grid(m_program, 0, 0, 0, 2);

    m_kernel_base_path = "mydsl/handwritten/three_core_add_relu/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_semaphores();
    create_kernels();
}

void ThreeCoreAddRelu::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void ThreeCoreAddRelu::create_globals() {
    uint32_t log2_page_size = 10; // 2^10 = 1024 (one tile in elements)
    m_ga = core::Global(m_device, T, m_N, log2_page_size);
    m_gb = core::Global(m_device, T, m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_N, log2_page_size);
}

void ThreeCoreAddRelu::create_pipes() {
    // pa, pb: cross-core 0 -> 1 (INPUT, declared on full grid for layout uniformity).
    m_pa =
        core::Pipe(
            m_program, m_grid, core::PipeKind::INPUT,
            T, m_pipe_frame_size * 2, m_pipe_frame_size);
    m_pb =
        core::Pipe(
            m_program, m_grid, core::PipeKind::INPUT,
            T, m_pipe_frame_size * 2, m_pipe_frame_size);
    // pt: cross-core 1 -> 2 (INPUT). Middle reserves+pushes lockstep with tail
    // so middle's get_write_ptr(pt) tracks tail's. Source for the NOC write is
    // pt_local; pt is just the receive-side cursor on tail (and tracking-only
    // cursor on middle).
    m_pt =
        core::Pipe(
            m_program, m_grid, core::PipeKind::INPUT,
            T, m_pipe_frame_size * 2, m_pipe_frame_size);
    // pt_local: per-core helper on middle. INTERMED so it has no input/output
    // role from host's perspective. Math packs into it; writer reads from it
    // and NOC-forwards to tail's pt. (Also exists on head/tail but unused.)
    m_pt_local =
        core::Pipe(
            m_program, m_grid, core::PipeKind::INTERMED,
            T, m_pipe_frame_size * 2, m_pipe_frame_size);
    // pc: only tail uses it.
    m_pc =
        core::Pipe(
            m_program, m_grid, core::PipeKind::OUTPUT,
            T, m_pipe_frame_size * 2, m_pipe_frame_size);
}

void ThreeCoreAddRelu::create_semaphores() {
    m_sem_credit_ab = core::Semaphore(m_program, m_grid, 0);
    m_sem_data_ab = core::Semaphore(m_program, m_grid, 0);
    m_sem_credit_t = core::Semaphore(m_program, m_grid, 0);
    m_sem_data_t = core::Semaphore(m_program, m_grid, 0);
}

void ThreeCoreAddRelu::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void ThreeCoreAddRelu::create_reader() {
    std::string path = m_kernel_base_path + "/three_core_add_relu_reader.cpp";
    m_reader =
        core::Kernel(
            m_program, m_grid,
            core::KernelKind::READER,
            core::KernelFormat::METAL,
            path, {}, m_defines);
    uint32_t head_x_phy, head_y_phy;
    uint32_t mid_x_phy, mid_y_phy;
    m_device.worker_core_from_logical_core(0, 0, head_x_phy, head_y_phy);
    m_device.worker_core_from_logical_core(0, 1, mid_x_phy, mid_y_phy);
    /*
    void kernel(
            pipe<T> pa, pipe<T> pb, pipe<T> pt,
            semaphore sem_credit_ab, semaphore sem_data_ab,
            semaphore sem_credit_t, semaphore sem_data_t,
            uint32 head_x, uint32 head_y,
            uint32 mid_x, uint32 mid_y,
            uint32 num_blocks, uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_pa, m_pb, m_pt,
        m_sem_credit_ab, m_sem_data_ab,
        m_sem_credit_t, m_sem_data_t,
        head_x_phy, head_y_phy,
        mid_x_phy, mid_y_phy,
        m_num_blocks, m_block_tiles,
        uint32_t(0)
    };
    args[13] = uint32_t(0);
    m_reader.set_args(0, 0, args);
    args[13] = uint32_t(1);
    m_reader.set_args(0, 1, args);
    args[13] = uint32_t(2);
    m_reader.set_args(0, 2, args);
}

void ThreeCoreAddRelu::create_writer() {
    std::string path = m_kernel_base_path + "/three_core_add_relu_writer.cpp";
    m_writer =
        core::Kernel(
            m_program, m_grid,
            core::KernelKind::WRITER,
            core::KernelFormat::METAL,
            path, {}, m_defines);
    uint32_t mid_x_phy, mid_y_phy;
    uint32_t tail_x_phy, tail_y_phy;
    m_device.worker_core_from_logical_core(0, 1, mid_x_phy, mid_y_phy);
    m_device.worker_core_from_logical_core(0, 2, tail_x_phy, tail_y_phy);
    /*
    void kernel(
            global<T> ga, global<T> gb, global<T> gc,
            pipe<T> pa, pipe<T> pb, pipe<T> pt, pipe<T> pt_local, pipe<T> pc,
            semaphore sem_credit_ab, semaphore sem_data_ab,
            semaphore sem_credit_t, semaphore sem_data_t,
            uint32 ga_pos, uint32 gb_pos, uint32 gc_pos,
            uint32 mid_x, uint32 mid_y,
            uint32 tail_x, uint32 tail_y,
            uint32 num_blocks, uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_ga, m_gb, m_gc,
        m_pa, m_pb, m_pt, m_pt_local, m_pc,
        m_sem_credit_ab, m_sem_data_ab,
        m_sem_credit_t, m_sem_data_t,
        uint32_t(0),     // ga_pos
        uint32_t(0),     // gb_pos
        uint32_t(0),     // gc_pos
        mid_x_phy, mid_y_phy,
        tail_x_phy, tail_y_phy,
        m_num_blocks, m_block_tiles,
        uint32_t(0)
    };
    args[21] = uint32_t(0);
    m_writer.set_args(0, 0, args);
    args[21] = uint32_t(1);
    m_writer.set_args(0, 1, args);
    args[21] = uint32_t(2);
    m_writer.set_args(0, 2, args);
}

void ThreeCoreAddRelu::create_math() {
    std::string path = m_kernel_base_path + "/three_core_add_relu_math.cpp";
    m_math =
        core::Kernel(
            m_program, m_grid,
            core::KernelKind::MATH,
            core::KernelFormat::METAL,
            path, {}, m_defines);
    /*
    void kernel(
            pipe<T> pa, pipe<T> pb, pipe<T> pt, pipe<T> pt_local, pipe<T> pc,
            uint32 num_blocks, uint32 block_tiles,
            uint32 core_id)
    */
    std::vector<core::KernelArg> args{
        m_pa, m_pb, m_pt, m_pt_local, m_pc,
        m_num_blocks, m_block_tiles,
        uint32_t(0)
    };
    args[7] = uint32_t(0);
    m_math.set_args(0, 0, args);
    args[7] = uint32_t(1);
    m_math.set_args(0, 1, args);
    args[7] = uint32_t(2);
    m_math.set_args(0, 2, args);
}

} // namespace handwritten
} // namespace mydsl
