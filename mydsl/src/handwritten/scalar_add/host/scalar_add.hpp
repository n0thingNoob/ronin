// scalar_add.hpp — phase 1 v0 host wrapper
//
// Derived from algo/src/basic/host/tanto/eltwise_binary.hpp.
// Difference from EltwiseBinary: no `op` selection (always add), and the
// caller passes only A; B is filled internally with the scalar value.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "host/core/api.hpp"

namespace mydsl {
namespace handwritten {

namespace core = ronin::tanto::host;

class ScalarAdd {
public:
    ScalarAdd();
    ~ScalarAdd();
public:
    void init(
        const core::Device &device,
        int N,
        float scalar);
    void run(
        const void *a,
        void *c);
private:
    void create_globals();
    void create_pipes();
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
    float m_scalar = 0.0f;
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
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
    std::vector<uint16_t> m_b_filled;
};

} // namespace handwritten
} // namespace mydsl
