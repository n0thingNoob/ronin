// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

//
//    Generic list of tensix builtins (ThCon emulator + future units)
//

// `jitte_thcon_set_gpr(gpr_idx, value)` is a Jitte-only escape hatch:
// it copies a runtime RV32 register value directly into ThCon Th[gpr_idx].
// On real HW the same effect would go through `ckernel::instrn_buffer +
// TT_SETDMAREG`; modelling that mailbox is a v1 question. This builtin
// lets kernels parameterize ThCon programs from `get_arg_val<>(...)`
// instead of hardcoding values via `TTI_SETDMAREG` immediates.
#define TENSIX_BUILTINS \
    DECL_BUILTIN(jitte_tensix_exec, 1) \
    DECL_BUILTIN(jitte_thcon_set_gpr, 2)

//
//    Tensix builtin enumeration.
//    ID range 5120-6143 reserved (stdlib uses 2048+, dataflow_tanto uses 4096+).
//

#define DECL_BUILTIN(name, count) name,

enum class TensixBuiltinId {
    START = 5120,
TENSIX_BUILTINS
};

#undef DECL_BUILTIN

// public functions

std::unordered_map<TensixBuiltinId, std::pair<std::string, int>> &get_tensix_builtin_map();
