// mpmd_two_core_add.hpp — phase 2 step 3 (true MPMD)
//
// 2 cores, **separate kernel binaries per core** (no core_id branching).
//   core (0,0) head: head_reader (no-op), head_writer (DRAM read + mcast),
//                    head_math (drain pa, pb)
//   core (0,1) tail: tail_reader (handshake), tail_writer (drain pc to DRAM),
//                    tail_math (pc = pa + pb)
//
// Tanto host API design lets a single Program own multiple Grid + Kernel
// objects. Pipes are declared on the union grid (1x2) so L1 layout is
// identical on both cores; kernels are bound to per-core sub-grids.
//
// Cross-core protocol: see ../../../../doc/notes.md.

#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "host/core/api.hpp"

namespace mydsl {
namespace handwritten {

namespace core = ronin::tanto::host;

class MpmdTwoCoreAdd {
public:
    MpmdTwoCoreAdd();
    ~MpmdTwoCoreAdd();
public:
    void init(
        const core::Device &device,
        int N);
    void run(
        const void *a,
        const void *b,
        void *c);
private:
    void create_globals();
    void create_pipes();
    void create_semaphores();
    void create_kernels();
    void create_head_reader();
    void create_head_writer();
    void create_head_math();
    void create_tail_reader();
    void create_tail_writer();
    void create_tail_math();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t TILE_SIZE = 1024;
private:
    core::Device m_device;
    uint32_t m_N = 0;
    uint32_t m_pipe_frame_size = 0;
    uint32_t m_block_tiles = 0;
    uint32_t m_num_blocks = 0;
    core::Program m_program;
    // Union grid (full): owns pipes and semaphores so both cores see the
    // same L1 layout. Sub-grids own their kernels.
    core::Grid m_grid_full;
    core::Grid m_grid_head;
    core::Grid m_grid_tail;
    core::Global m_ga;
    core::Global m_gb;
    core::Global m_gc;
    core::Pipe m_pa;
    core::Pipe m_pb;
    core::Pipe m_pc;
    core::Semaphore m_sem_credit;
    core::Semaphore m_sem_data;
    core::Kernel m_head_reader;
    core::Kernel m_head_writer;
    core::Kernel m_head_math;
    core::Kernel m_tail_reader;
    core::Kernel m_tail_writer;
    core::Kernel m_tail_math;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace handwritten
} // namespace mydsl
