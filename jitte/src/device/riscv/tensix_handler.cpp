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
// Opcode decoders. Field layouts come from
// tt-llk/tt_llk_wormhole_b0/instructions/assembly.yaml (authoritative)
// and ckernel_ops.h (macro shapes):
//
//   SETDMAREG  (0x45) bits[23:22]=SigSelSize, [21:8]=SigSel(14b),
//                     [7]=Mode, [6:0]=RegIndex16b
//   LOADIND    (0x49) bits[23:22]=SizeSel, [21:14]=OffsetIndex,
//                     [13:12]=AutoIncSpec, [11:6]=DataRegIndex,
//                     [5:0]=AddrRegIndex
//                     SizeSel: 0=16B, 1=32b, 2=16b, 3=8b
//                     Th[AddrRegIndex] holds a 16B-aligned L1 address.
//   STOREIND   (0x66) bits[23]=MemHierSel, [22]=SizeSel, [21]=RegSizeSel,
//                     [20:14]=OffsetIndex, [13:12]=AutoIncSpec,
//                     [11:6]=DataRegIndex, [5:0]=AddrRegIndex
//                     MemHierSel: 0=SRC regfile, 1=L1.
//                     When MemHierSel=1, bits[22:21] = (SizeSel,RegSizeSel)
//                     compose size: 0=16B, 1=16b, 2=32b, 3=8b.
//                     Th[AddrRegIndex] holds a 16B-aligned L1 address.
//   ATINCGET   (0x61) bits[23]=MemHierSel, [22:14]=WrapVal[8:0],
//                     [13:12]=Sel32b, [11:6]=DataRegIndex,
//                     [5:0]=AddrRegIndex
//                     MemHierSel: 0=DataRam, 1=L1.
//                     Sel32b selects which 32b word in the 16B block.
//                     WrapVal[4:0] is a power-of-two cap: counter wraps
//                     at (1<<WrapVal). Th[DataRegIndex] supplies the
//                     increment value before the op, and receives the
//                     old counter value after.
//
// ThCon GPRs are 32-bit but addressed as 16-bit halves: LO_16(N)=2*N
// (low half of Th[N]) and HI_16(N)=2*N+1 (high half).

// Per ISA doc STOREIND_L1.md / LOADIND.md: OffsetIndex is a HALF-REGISTER
// INDEX into the GPR file, and that half-register's *current value* is
// the byte offset added to (Th[AddrReg] * 16). It is NOT a literal offset.
// Real-HW formula: `uint16_t* Offset = (char*)&GPRs[0] + OffsetHalfReg*2;`
// Little-endian: half-reg 2N = low half of Th[N], 2N+1 = high half.
uint16_t read_half_reg(ThConState &th, uint32_t off_idx) {
    if (off_idx >= 128u) {
        fprintf(stderr,
            "[tensix v0.5] OffsetIndex=%u out of range (max 127)\n", off_idx);
        std::abort();
    }
    uint32_t gpr_idx = off_idx >> 1;
    bool high = (off_idx & 1u) != 0;
    uint32_t v = th.gpr[gpr_idx];
    return uint16_t(high ? (v >> 16) : (v & 0xFFFFu));
}

void write_half_reg(ThConState &th, uint32_t off_idx, uint16_t value) {
    uint32_t gpr_idx = off_idx >> 1;
    bool high = (off_idx & 1u) != 0;
    uint32_t old = th.gpr[gpr_idx];
    if (high) {
        th.gpr[gpr_idx] = (old & 0x0000FFFFu) | (uint32_t(value) << 16);
    } else {
        th.gpr[gpr_idx] = (old & 0xFFFF0000u) | uint32_t(value);
    }
}

// AutoIncSpec post-increment of *Offset (the half-register selected by
// OffsetIndex) — applied AFTER the L1 access. Encoding: 0=+0, 1=+2, 2=+4,
// 3=+16 bytes. Wraps modulo 2^16 since *Offset is uint16_t.
constexpr uint16_t AUTO_INC_BYTES[4] = {0, 2, 4, 16};

// Abort if `gpr_idx` has an outstanding LOADIND that hasn't been retired
// by a STALLWAIT/FLUSHDMA yet. On real HW, reading a pending GPR yields
// stale data (LOADIND fills the GPR "at some later point in time"); the
// bug typically manifests as zero results downstream. v0.5 surfaces it
// at simulation time.
void check_no_pending(
        ThConState &th, uint32_t gpr_idx, const char *role,
        const char *op_name) {
    auto it = th.pending_loadind.find(gpr_idx);
    if (it == th.pending_loadind.end()) {
        return;
    }
    fprintf(stderr,
        "[tensix v0.5] %s: %s Th[%u] has a pending LOADIND result (loaded "
        "value 0x%08x has not retired). Insert TTI_STALLWAIT(STALL_THREAD, "
        "THCON) or TTI_FLUSHDMA between the LOADIND and this consumer.\n",
        op_name, role, gpr_idx, it->second);
    std::abort();
}

// Commit all pending LOADIND results into the GPR file. Called by
// STALLWAIT (when waiting on THCON) and FLUSHDMA. v0.5 commits all
// pending entries unconditionally — STALLWAIT's resource bits are
// over-broad rather than fine-grained, but the over-commit is safe
// because real HW also drains everything on these barriers.
void retire_pending(ThConState &th, const char *op_name) {
    if (th.pending_loadind.empty()) {
        return;
    }
    if (DIAG_TENSIX_TRACE) {
        for (const auto &kv : th.pending_loadind) {
            fprintf(stderr,
                "[tensix] %s   retire Th[%u] <- 0x%08x (was pending LOADIND)\n",
                op_name, kv.first, kv.second);
        }
    }
    for (const auto &kv : th.pending_loadind) {
        th.gpr[kv.first] = kv.second;
    }
    th.pending_loadind.clear();
}

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
    // Race against pending LOADIND: on real HW SETDMAREG and the deferred
    // LOADIND fill both target the same GPR with no defined ordering. v0.5
    // refuses to model the race and aborts.
    check_no_pending(th, gpr_idx, "Dst", "SETDMAREG");
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

    // OffsetIndex picks a 16-bit half-register; that half-register's
    // current value is the byte addend to (Th[addr_reg] * 16).
    check_no_pending(th, addr_reg, "AddrReg", "LOADIND");
    uint16_t off = read_half_reg(th, off_idx);
    uint32_t addr = (th.gpr[addr_reg] << 4) + uint32_t(off);
    uint8_t *p = l1_ptr(machine, addr);
    uint32_t value = 0;
    const char *size_str = "?";
    switch (size_sel) {
    case 0:
        // 16B load: not modeled in v0 (would target SRC regfile / 16B GPR
        // pair). Panic loud rather than silently truncate.
        fprintf(stderr,
            "[tensix v0] LOADIND: SizeSel=0 (16B) not supported (insn=0x%08x)\n",
            insn);
        std::abort();
    case 1: {
        // 32b load
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        value = v;
        size_str = "u32";
        break;
    }
    case 2: {
        // 16b load
        uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        value = uint32_t(v);
        size_str = "u16";
        break;
    }
    case 3:
        // 8b load
        value = uint32_t(*p);
        size_str = "u8";
        break;
    }

    // LOADIND is asynchronous on real HW: the GPR fill happens "at some
    // later point in time" relative to the issuing thread. v0.5 stages the
    // result in pending_loadind; STALLWAIT or FLUSHDMA retires it into gpr.
    th.pending_loadind[data_reg] = value;

    // AutoIncSpec post-increments *Offset (the selected half-register).
    // Skip if the half-reg lives inside Th[addr_reg] or Th[data_reg]
    // (degenerate case nothing legitimate would do; abort-loud).
    if (auto_inc != 0) {
        uint32_t off_gpr = off_idx >> 1;
        if (off_gpr == data_reg) {
            fprintf(stderr,
                "[tensix v0.5] LOADIND: OffsetIndex half-reg lives inside "
                "DataReg Th[%u] — undefined ordering with the deferred "
                "result write (insn=0x%08x)\n", data_reg, insn);
            std::abort();
        }
        uint16_t new_off = uint16_t(off + AUTO_INC_BYTES[auto_inc]);
        write_half_reg(th, off_idx, new_off);
    }

    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] LOADIND   Th[%u] <- L1[0x%08x (16B@0x%x +0x%x)] = "
            "0x%08x  (%s, async — pending)\n",
            data_reg, addr, th.gpr[addr_reg], off, value, size_str);
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

    if (mem_hier_sel != 1) {
        // Real-HW: MemHierSel=0 routes the store into the SRC A/B regfile;
        // v0 only models L1 (MemHierSel=1).
        fprintf(stderr,
            "[tensix v0] STOREIND: MemHierSel=%u (SRC regfile) not supported "
            "(insn=0x%08x)\n",
            mem_hier_sel, insn);
        std::abort();
    }

    // Same OffsetIndex semantics as LOADIND: half-register index, not a
    // literal offset. STOREIND's OffsetIndex is 7-bit (vs 8-bit for LOADIND)
    // — fits all 128 half-regs.
    check_no_pending(th, addr_reg, "AddrReg", "STOREIND");
    check_no_pending(th, data_reg, "DataReg", "STOREIND");
    uint16_t off = read_half_reg(th, off_idx);
    uint32_t addr = (th.gpr[addr_reg] << 4) + uint32_t(off);
    uint8_t *p = l1_ptr(machine, addr);

    // When MemHierSel=1, bits[22:21]=(SizeSel,RegSizeSel) compose store
    // width: 0=16B, 1=16b, 2=32b, 3=8b.
    uint32_t size_code = (size_sel << 1) | reg_size_sel;
    uint32_t value = th.gpr[data_reg];
    const char *size_str = "?";
    switch (size_code) {
    case 0:
        // 16B store: not modeled in v0 (would consume 4 consecutive GPRs).
        fprintf(stderr,
            "[tensix v0] STOREIND: 16B store not supported (insn=0x%08x)\n",
            insn);
        std::abort();
    case 1: {
        // 16-bit store (low half of GPR)
        uint16_t v = uint16_t(value & 0xFFFFu);
        std::memcpy(p, &v, sizeof(v));
        size_str = "u16";
        break;
    }
    case 2:
        // 32-bit store
        std::memcpy(p, &value, sizeof(value));
        size_str = "u32";
        break;
    case 3:
        // 8-bit store
        *p = uint8_t(value & 0xFFu);
        size_str = "u8";
        break;
    }

    if (auto_inc != 0) {
        uint16_t new_off = uint16_t(off + AUTO_INC_BYTES[auto_inc]);
        write_half_reg(th, off_idx, new_off);
    }

    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] STOREIND  L1[0x%08x (16B@0x%x +0x%x)] <- Th[%u]=0x%08x  (%s)\n",
            addr, th.gpr[addr_reg], off, data_reg, value, size_str);
    }
}

// ATINCGET (0x61):
//   bits[23]    MemHierSel (0=DataRam; v0 only models 1=L1)
//   bits[22:14] WrapVal (9b)  — only low 5 bits used: counter wraps at
//                               (1<<WrapVal[4:0]). WrapVal=0 means a
//                               1-bit counter (mask=0).
//   bits[13:12] Sel32b (2b)   — selects which 32b lane in the 16B block
//                               (byte offset = sel32b*4).
//   bits[11:6]  DataRegIndex  — Th[DataReg] supplies the increment value
//                               BEFORE the op, and receives the old
//                               counter value AFTER (read-modify-write).
//   bits[5:0]   AddrRegIndex  — Th[AddrRegIndex] holds a 16B-aligned
//                               base address.
//
// Semantics (atomic): inc = Th[DataReg];
//                     old = *L1;
//                     new = (old + inc) & ((1u << WrapVal[4:0]) - 1);
//                     *L1 = new;
//                     Th[DataReg] = old;
void do_atincget(
        Machine *machine,
        ThConState &th,
        std::mutex &mtx,
        uint32_t insn) {
    uint32_t addr_reg = (insn >> 0) & 0x3Fu;
    uint32_t data_reg = (insn >> 6) & 0x3Fu;
    uint32_t sel32b = (insn >> 12) & 0x3u;
    uint32_t wrap_val_field = (insn >> 14) & 0x1FFu;
    uint32_t mem_hier_sel = (insn >> 23) & 0x1u;

    if (mem_hier_sel != 1) {
        // Real-HW: MemHierSel=0 routes to DataRam (TDMA); v0 models L1.
        fprintf(stderr,
            "[tensix v0] ATINCGET: MemHierSel=%u (DataRam) not supported "
            "(insn=0x%08x)\n",
            mem_hier_sel, insn);
        std::abort();
    }

    check_no_pending(th, addr_reg, "AddrReg", "ATINCGET");
    check_no_pending(th, data_reg, "DataReg", "ATINCGET");

    uint32_t wrap_bits = wrap_val_field & 0x1Fu;
    uint32_t addr = (th.gpr[addr_reg] << 4) + (sel32b * 4u);
    uint8_t *p = l1_ptr(machine, addr);

    uint32_t inc = th.gpr[data_reg];
    uint32_t old_val;
    uint32_t new_val;
    {
        std::lock_guard<std::mutex> guard(mtx);
        std::memcpy(&old_val, p, sizeof(old_val));
        // wrap_bits=0 → mask=0 (1-bit-counter modulo 1, i.e. always 0).
        // wrap_bits=32 is unrepresentable in 5 bits; max useful is 31.
        uint32_t mask =
            (wrap_bits == 0) ? 0u : ((1u << wrap_bits) - 1u);
        new_val = (old_val + inc) & mask;
        std::memcpy(p, &new_val, sizeof(new_val));
    }
    th.gpr[data_reg] = old_val;

    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] ATINCGET  L1[0x%08x (16B@0x%x +%u)]: old=0x%x inc=0x%x"
            " new=0x%x wrap=1<<%u; Th[%u] <- 0x%x\n",
            addr, th.gpr[addr_reg], sel32b * 4u,
            old_val, inc, new_val, wrap_bits,
            data_reg, old_val);
    }
}

// ADDDMAREG (0x58) and MULDMAREG (0x5A) share encoding:
//   bits[23]    OpBisConst (1=OpB is 6-bit immediate, 0=OpB is GPR index)
//   bits[22:12] ResultRegIndex (11b; only low 6 bits used in v0)
//   bits[11:6]  OpBRegIndex (6b)
//   bits[5:0]   OpARegIndex (6b)
void do_arith(ThConState &th, uint32_t insn, bool is_mul) {
    uint32_t a_idx = (insn >> 0) & 0x3Fu;
    uint32_t b_field = (insn >> 6) & 0x3Fu;
    uint32_t r_idx = (insn >> 12) & 0x3Fu; // low 6 of 11-bit field
    uint32_t op_b_is_const = (insn >> 23) & 0x1u;

    const char *op_name = is_mul ? "MULDMAREG" : "ADDDMAREG";
    check_no_pending(th, a_idx, "OpA", op_name);
    if (!op_b_is_const) {
        check_no_pending(th, b_field, "OpB", op_name);
    }
    check_no_pending(th, r_idx, "Dst", op_name);
    uint32_t a = th.gpr[a_idx];
    uint32_t b = op_b_is_const ? b_field : th.gpr[b_field];
    uint32_t result = is_mul ? uint32_t(a * b) : uint32_t(a + b);
    th.gpr[r_idx] = result;

    if (DIAG_TENSIX_TRACE) {
        const char *name = is_mul ? "MULDMAREG" : "ADDDMAREG";
        const char *sym = is_mul ? "*" : "+";
        if (op_b_is_const) {
            fprintf(stderr,
                "[tensix] %s Th[%u] = Th[%u] %s 0x%x = 0x%x %s 0x%x = 0x%x (=%u)\n",
                name, r_idx, a_idx, sym, b_field, a, sym, b, result, result);
        } else {
            fprintf(stderr,
                "[tensix] %s Th[%u] = Th[%u] %s Th[%u] = 0x%x %s 0x%x = 0x%x (=%u)\n",
                name, r_idx, a_idx, sym, b_field, a, sym, b, result, result);
        }
    }
}

// STALLWAIT (0xa2) — synchronisation barrier. Macro form
// TTI_STALLWAIT(stall_res, wait_res). v0.5 treats any STALLWAIT as a
// retire point for pending LOADIND results. The fine-grained resource
// bits (THCON, TDMA, etc.) are over-broad here but the over-commit is
// safe because real HW also drains everything on these barriers.
void do_stallwait(ThConState &th, uint32_t insn) {
    uint32_t wait_res = insn & 0x7FFFu;
    uint32_t stall_res = (insn >> 15) & 0x1FFu;
    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] STALLWAIT  stall_res=0x%03x wait_res=0x%04x\n",
            stall_res, wait_res);
    }
    retire_pending(th, "STALLWAIT");
}

// FLUSHDMA (0x46) — drain all in-flight DMA/ThCon work. v0.5 retires
// every pending LOADIND result.
void do_flushdma(ThConState &th, uint32_t insn) {
    uint32_t flush_spec = insn & 0xFFFFFFu;
    if (DIAG_TENSIX_TRACE) {
        fprintf(stderr,
            "[tensix] FLUSHDMA   flush_spec=0x%06x\n", flush_spec);
    }
    retire_pending(th, "FLUSHDMA");
}

void jitte_tensix_exec(
        Machine *machine,
        ThConState &th,
        std::mutex &l1_mutex,
        uint32_t insn) {
    uint8_t op = uint8_t((insn >> 24) & 0xFFu);
    switch (op) {
    case 0x45:
        do_setdmareg(th, insn);
        break;
    case 0x46:
        do_flushdma(th, insn);
        break;
    case 0x49:
        do_loadind(machine, th, insn);
        break;
    case 0x58:
        do_arith(th, insn, /*is_mul=*/false);
        break;
    case 0x5A:
        do_arith(th, insn, /*is_mul=*/true);
        break;
    case 0x61:
        do_atincget(machine, th, l1_mutex, insn);
        break;
    case 0x66:
        do_storeind(machine, th, insn);
        break;
    case 0xA2:
        do_stallwait(th, insn);
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

std::mutex &TensixHandler::mutex_for(Memory *l1) {
    std::lock_guard<std::mutex> guard(m_mutex_table_lock);
    // unordered_map is node-based: nodes don't move on rehash, so the
    // returned mutex reference stays stable. operator[] default-constructs
    // a std::mutex in place on first access.
    return m_l1_mutex[l1];
}

void TensixHandler::call(Riscv32Core *core, int id) {
    ThConState &th = state_for(core);
    Memory *l1 = m_machine->get_worker_l1();
    std::mutex &mtx = mutex_for(l1);
    switch (TensixBuiltinId(id)) {
    case TensixBuiltinId::jitte_tensix_exec: {
        uint32_t insn = core->get_arg(0);
        jitte_tensix_exec(m_machine, th, mtx, insn);
        break;
    }
    case TensixBuiltinId::jitte_thcon_set_gpr: {
        uint32_t gpr_idx = core->get_arg(0);
        uint32_t value = core->get_arg(1);
        if (gpr_idx >= th.gpr.size()) {
            fprintf(stderr,
                "[tensix v0] jitte_thcon_set_gpr: gpr_idx=%u out of range\n",
                gpr_idx);
            std::abort();
        }
        check_no_pending(th, gpr_idx, "Dst", "jitte_thcon_set_gpr");
        th.gpr[gpr_idx] = value;
        if (DIAG_TENSIX_TRACE) {
            fprintf(stderr,
                "[tensix] SET_GPR    Th[%u] = 0x%08x  (host bridge)\n",
                gpr_idx, value);
        }
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
