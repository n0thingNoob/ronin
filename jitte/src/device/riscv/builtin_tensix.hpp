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

#define TENSIX_BUILTINS \
    DECL_BUILTIN(jitte_tensix_exec, 1)

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
