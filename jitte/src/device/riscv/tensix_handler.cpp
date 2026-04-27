// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"

#include "riscv/builtin_tensix.hpp"
#include "riscv/tensix_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

namespace {

void jitte_tensix_exec(Machine *machine, Riscv32Core *core) {
    uint32_t insn = core->get_arg(0);
    fprintf(stderr, "[tensix] insn=0x%08x\n", insn);
    std::abort();
}

} // namespace

//
//    TensixHandler
//

TensixHandler::TensixHandler(Machine *machine):
        m_machine(machine) { }

TensixHandler::~TensixHandler() { }

#define DECL_BUILTIN(name, count) \
    case TensixBuiltinId::name: \
        name(m_machine, core); \
        break;

void TensixHandler::call(Riscv32Core *core, int id) {
    switch (TensixBuiltinId(id)) {
TENSIX_BUILTINS
    default:
        assert(false);
        break;
    }
}

#undef DECL_BUILTIN

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt
