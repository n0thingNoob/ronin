// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"
#include "core/memory.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

// ThCon GPR file (one 32-bit register per index, 64 entries).
// Real HW has three pipes shared between threads; v0 ships a per-hart
// simplification keyed by the issuing Riscv32Core pointer.
//
// `pending_loadind` models the asynchronous LOADIND result: per ISA doc,
// LOADIND retires and frees the issuing thread before the GPR is filled.
// v0.5 captures the loaded value here and only commits it into `gpr` on
// STALLWAIT/FLUSHDMA. Any access to a GPR with a pending entry before
// retirement aborts loudly — that catches the missing-STALLWAIT bug at
// simulation time instead of letting it slip through to silicon.
struct ThConState {
    std::array<uint32_t, 64> gpr{};
    std::unordered_map<uint32_t, uint32_t> pending_loadind;
};

class TensixHandler {
public:
    TensixHandler(Machine *machine);
    ~TensixHandler();
public:
    void call(Riscv32Core *core, int id);
private:
    ThConState &state_for(Riscv32Core *core);
    std::mutex &mutex_for(Memory *l1);
private:
    Machine *m_machine;
    std::unordered_map<Riscv32Core *, ThConState> m_state;
    // Per-tensix mutex for ATINCGET's L1 read-modify-write. Keyed by the
    // L1 Memory* pointer because each tensix owns its own L1 buffer
    // (different L1 = different tensix = independent atomic domain).
    // m_mutex_table_lock guards lookups/insertions into m_l1_mutex itself.
    std::mutex m_mutex_table_lock;
    std::unordered_map<Memory *, std::mutex> m_l1_mutex;
};

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt
