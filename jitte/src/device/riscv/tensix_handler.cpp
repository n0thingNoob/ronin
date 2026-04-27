// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"
#include "core/memory.hpp"

#include "riscv/builtin_tensix.hpp"
#include "riscv/tensix_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

namespace {

// Diagnostic prints for each decoded ThCon op. Flip to false to silence.
constexpr bool DIAG_TENSIX_TRACE = true;

//
// Opcode decoders. Encodings come from
// tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_ops.h:
//
//   SETDMAREG  (0x45) bits[23:22]=SigSelSize, [21:8]=SigSel(14b),
//                     [7]=Mode, [6:0]=RegIndex16b
//   LOADIND    (0x49) bits[23:22]=SizeSel, [21:14]=OffsetIndex,
//                     [13:12]=AutoIncSpec, [11:6]=DataRegIndex,
//                     [5:0]=AddrRegIndex
//   STOREIND   (0x66) bits[23]=MemHierSel, [22]=SizeSel, [21]=RegSizeSel,
//                     [20:14]=OffsetIndex, [13:12]=AutoIncSpec,
//                     [11:6]=DataRegIndex, [5:0]=AddrRegIndex
//
// ThCon GPRs are 32-bit but addressed as 16-bit halves: LO_16(N)=2*N
// (low half of Th[N]) and HI_16(N)=2*N+1 (high half).

void do_setdmareg(ThConState &th, uint32_t insn) {
    uint32_t reg_index_16b = (insn >> 0) & 0x7Fu;
    uint32_t mode = (insn >> 7) & 0x1u;
    uint32_t sig_sel = (insn >> 8) & 0x3FFFu;
    uint32_t sig_sel_size = (insn >> 22) & 0x3u;

    if (mode != 0) {
        // MODE_IMMEDIATE=0; signal-select mode unsupported in v0.
        fprintf(stderr,
            "[tensix v0] SETDMAREG: signal-select mode (Mode=%u) not "
            "supported (insn=0x%08x)\n",
            mode, insn);
        std::abort();
    }
    if (sig_sel_size != 0) {
        // Multi-cycle wide-immediate forms not supported in v0.
        fprintf(stderr,
            "[tensix v0] SETDMAREG: SigSelSize=%u not supported "
            "(insn=0x%08x)\n",
            sig_sel_size, insn);
        std::abort();
    }

    uint32_t gpr_idx = reg_index_16b >> 1;
    bool high_half = (reg_index_16b & 1u) != 0;
    uint32_t mask = high_half ? 0x0000FFFFu : 0xFFFF0000u;
    uint32_t shifted =
        high_half ? (sig_sel << 16) : (sig_sel & 0xFFFFu);
    uint32_t old_val = th.gpr[gpr_idx];
    th.gpr[gpr_idx] = (old_val & mask) | shifted;

    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] SETDMAREG  Th[%u].%s = 0x%04x  (Th[%u]=0x%08x)\n",
            gpr_idx, high_half ? "hi" : "lo", sig_sel,
            gpr_idx, th.gpr[gpr_idx]);
    }
}

uint8_t *l1_ptr(Machine *machine, uint32_t addr) {
    Memory *l1 = machine->get_worker_l1();
    if (l1 == nullptr) {
        fprintf(stderr, "[tensix] L1 unavailable (no current tensix)\n");
        std::abort();
    }
    if (addr >= l1->size()) {
        fprintf(stderr,
            "[tensix] L1 address 0x%08x out of range (size=0x%08x)\n",
            addr, l1->size());
        std::abort();
    }
    return l1->map_addr(addr);
}

void do_loadind(Machine *machine, ThConState &th, uint32_t insn) {
    uint32_t addr_reg = (insn >> 0) & 0x3Fu;
    uint32_t data_reg = (insn >> 6) & 0x3Fu;
    uint32_t auto_inc = (insn >> 12) & 0x3u;
    uint32_t off_idx = (insn >> 14) & 0xFFu;
    uint32_t size_sel = (insn >> 22) & 0x3u;

    if (auto_inc != 0 || off_idx != 0) {
        // v0 ignores OffsetIndex/AutoIncSpec — base-addr-only addressing.
        // Real HW uses these to drive RWC-style auto-increment counters.
        fprintf(stderr,
            "[tensix v0] LOADIND: AutoIncSpec=%u OffsetIndex=%u not "
            "supported yet (insn=0x%08x)\n",
            auto_inc, off_idx, insn);
        std::abort();
    }

    uint32_t addr = th.gpr[addr_reg];
    uint8_t *p = l1_ptr(machine, addr);
    uint32_t value = 0;
    const char *size_str = "?";
    switch (size_sel) {
    case 0:
        value = uint32_t(*p);
        size_str = "u8";
        break;
    case 1: {
        uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        value = uint32_t(v);
        size_str = "u16";
        break;
    }
    case 2: {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        value = v;
        size_str = "u32";
        break;
    }
    default:
        fprintf(stderr,
            "[tensix v0] LOADIND: SizeSel=%u not supported (insn=0x%08x)\n",
            size_sel, insn);
        std::abort();
    }

    th.gpr[data_reg] = value;
    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] LOADIND   Th[%u] <- L1[0x%08x] = 0x%08x  (%s)\n",
            data_reg, addr, value, size_str);
    }
}

void do_storeind(Machine *machine, ThConState &th, uint32_t insn) {
    uint32_t addr_reg = (insn >> 0) & 0x3Fu;
    uint32_t data_reg = (insn >> 6) & 0x3Fu;
    uint32_t auto_inc = (insn >> 12) & 0x3u;
    uint32_t off_idx = (insn >> 14) & 0x7Fu;
    uint32_t reg_size_sel = (insn >> 21) & 0x1u;
    uint32_t size_sel = (insn >> 22) & 0x1u;
    uint32_t mem_hier_sel = (insn >> 23) & 0x1u;

    if (auto_inc != 0 || off_idx != 0) {
        fprintf(stderr,
            "[tensix v0] STOREIND: AutoIncSpec=%u OffsetIndex=%u not "
            "supported yet (insn=0x%08x)\n",
            auto_inc, off_idx, insn);
        std::abort();
    }
    if (mem_hier_sel != 0) {
        // v0 only supports local-L1 stores; remote-L1 / memory hierarchy
        // routing is a v1 question that ATINCGET will surface again.
        fprintf(stderr,
            "[tensix v0] STOREIND: MemHierSel=%u not supported (insn=0x%08x)\n",
            mem_hier_sel, insn);
        std::abort();
    }

    uint32_t addr = th.gpr[addr_reg];
    uint8_t *p = l1_ptr(machine, addr);

    // STOREIND has separate SizeSel (mem width) and RegSizeSel (reg width).
    // v0: treat them together — SizeSel selects 16b vs 32b store width;
    // RegSizeSel selects which half of the source GPR to use for 16b stores.
    uint32_t value = th.gpr[data_reg];
    const char *size_str = "?";
    if (size_sel == 0) {
        // 32-bit store
        std::memcpy(p, &value, sizeof(value));
        size_str = "u32";
    } else {
        // 16-bit store
        uint16_t v =
            reg_size_sel ? uint16_t(value >> 16) : uint16_t(value & 0xFFFFu);
        std::memcpy(p, &v, sizeof(v));
        size_str = reg_size_sel ? "u16(hi)" : "u16(lo)";
        value = v;
    }

    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] STOREIND  L1[0x%08x] <- Th[%u]=0x%08x  (%s)\n",
            addr, data_reg, value, size_str);
    }
}

void jitte_tensix_exec(Machine *machine, ThConState &th, uint32_t insn) {
    uint8_t op = uint8_t((insn >> 24) & 0xFFu);
    switch (op) {
    case 0x45:
        do_setdmareg(th, insn);
        break;
    case 0x49:
        do_loadind(machine, th, insn);
        break;
    case 0x66:
        do_storeind(machine, th, insn);
        break;
    default:
        fprintf(stderr,
            "[tensix v0] unsupported opcode 0x%02x (insn=0x%08x)\n",
            op, insn);
        std::abort();
    }
}

} // namespace

//
//    TensixHandler
//

TensixHandler::TensixHandler(Machine *machine):
        m_machine(machine) { }

TensixHandler::~TensixHandler() { }

ThConState &TensixHandler::state_for(Riscv32Core *core) {
    return m_state[core];
}

void TensixHandler::call(Riscv32Core *core, int id) {
    ThConState &th = state_for(core);
    switch (TensixBuiltinId(id)) {
    case TensixBuiltinId::jitte_tensix_exec: {
        uint32_t insn = core->get_arg(0);
        jitte_tensix_exec(m_machine, th, insn);
        break;
    }
    default:
        assert(false);
        break;
    }
}

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt
