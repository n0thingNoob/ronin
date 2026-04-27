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

#include "ckernel_ops.h"

#ifdef __JITTE__
#undef INSTRUCTION_WORD
#define INSTRUCTION_WORD(x) jitte_tensix_exec((x))
#endif
