// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

// ThCon GPR file (one 32-bit register per index, 64 entries).
// Real HW has three pipes shared between threads; v0 ships a per-hart
// simplification keyed by the issuing Riscv32Core pointer.
struct ThConState {
    std::array<uint32_t, 64> gpr{};
};

class TensixHandler {
public:
    TensixHandler(Machine *machine);
    ~TensixHandler();
public:
    void call(Riscv32Core *core, int id);
private:
    ThConState &state_for(Riscv32Core *core);
private:
    Machine *m_machine;
    std::unordered_map<Riscv32Core *, ThConState> m_state;
};

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt
