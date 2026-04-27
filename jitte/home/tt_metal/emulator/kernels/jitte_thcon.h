// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Jitte-side ThCon (Tensix Scalar Unit) intercept header.
//
// On real silicon, ckernel_ops.h's INSTRUCTION_WORD(x) emits an inline-asm
// `.ttinsn` directive that the RV32 core dispatches to the Tensix
// coprocessor. Jitte runs kernels on the Whisper RV32 ISS and has no
// `.ttinsn` decoder — so we redefine INSTRUCTION_WORD to call a builtin
// (jitte_tensix_exec) that the Jitte BuiltinHandler intercepts.
//
// The TTI_* macros in upstream ckernel_ops.h are unmodified; only
// INSTRUCTION_WORD is overridden, so the same kernel source works on
// both real Wormhole and Jitte.

#pragma once

#include <cstdint>

#ifndef API
#define API extern "C"
#endif

API void jitte_tensix_exec(uint32_t insn_word);

// Jitte-only host→ThCon-GPR bridge. Kernels can pull a runtime arg via
// `get_arg_val<uint32_t>(i)` and stage it into Th[gpr_idx] without the
// 14-bit SETDMAREG immediate limit. No-op / undefined on real silicon.
API void jitte_thcon_set_gpr(uint32_t gpr_idx, uint32_t value);

#include "ckernel_ops.h"

#ifdef __JITTE__
#undef INSTRUCTION_WORD
#define INSTRUCTION_WORD(x) jitte_tensix_exec((x))

// Subset of ckernel_instr_params.h's `p_stall` for Jitte BRISC kernels
// (which include dataflow_api.h, not the compute API where p_stall lives).
// Defined only under __JITTE__ so we never collide with the real-HW
// definition. Values must match ckernel_instr_params.h exactly.
namespace p_stall {
    constexpr uint32_t THCON       = 0x1;
    constexpr uint32_t STALL_THCON = 0x20;
    constexpr uint32_t STALL_THREAD = 0x1ff;
}
#endif
