// two_core_add.hpp — phase 2 step 1 host wrapper
//
// 2 cores wired as producer/consumer dataflow:
//   core (0,0): reads A,B from DRAM, NOC-mcasts pa, pb to (0,1)
//   core (0,1): receives pa, pb via NOC, computes pc = pa + pb, drains pc to DRAM
//
// Cross-core protocol: see ../../../doc/notes.md.

#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "host/core/api.hpp"

namespace mydsl {
namespace handwritten {

namespace core = ronin::tanto::host;

class TwoCoreAdd {
public:
    TwoCoreAdd();
    ~TwoCoreAdd();
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
    void create_reader();
    void create_writer();
    void create_math();
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
    core::Grid m_grid;
    core::Global m_ga;
    core::Global m_gb;
    core::Global m_gc;
    core::Pipe m_pa;
    core::Pipe m_pb;
    core::Pipe m_pc;
    core::Semaphore m_sem_credit;
    core::Semaphore m_sem_data;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace handwritten
} // namespace mydsl
