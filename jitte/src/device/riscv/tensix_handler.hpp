// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

class TensixHandler {
public:
    TensixHandler(Machine *machine);
    ~TensixHandler();
public:
    void call(Riscv32Core *core, int id);
private:
    Machine *m_machine;
};

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt
